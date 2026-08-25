"""Env-free diffusion planner: depth + command -> z_g. Single-robot RolloutPlanner."""

from __future__ import annotations

import math

import torch

from motion_diffusion.features.window_adapter import WindowAdapter
from motion_diffusion.planner.core import load_planner, sample_actions_from_unified


class DeployPlanner:
    def __init__(self, model_path, cfg_path, device, *, sampler="ddim",
                 num_inference_steps=12, action_horizon=2, num_dof=23, blend_frames=8):
        self.bundle, self.model, self.kind = load_planner(model_path, cfg_path, device)
        if not self.bundle.use_ego_depth:
            raise ValueError("Deploy planner requires a depth-conditioned checkpoint.")
        self.device = device
        self.state_in_mode = str(self.bundle.state_in_mode)
        self.adapter = WindowAdapter(
            self.bundle, int(self.bundle.H), int(self.bundle.F),
            int(self.bundle.condition_dim), int(self.bundle.action_dim),
        )
        self.exec_action = torch.zeros(1, int(self.bundle.H), int(num_dof), device=device)
        self.sampler = str(sampler)
        self.num_inference_steps = int(num_inference_steps)
        self.blend_frames = int(blend_frames)
        # the plan feeds its own z_g back; keep it on the sqrt(dim) sphere the teacher put it on (as sim)
        self.zg_radius = float(self.bundle.action_dim) ** 0.5 if str(self.bundle.action_stream) == "zg" else 0.0
        self.action_horizon = int(action_horizon)
        self.H = int(self.bundle.H)
        self.F = int(self.bundle.F)
        self.action_dim = int(self.bundle.action_dim)
        self.hist_action = torch.zeros(1, self.H, self.action_dim, device=device)
        self.plan = None
        self.cursor = 0
        self.start_yaw = None

    def reset(self, yaw_now: float) -> None:
        self.hist_action.zero_()
        self.exec_action.zero_()
        self.plan = None
        self.cursor = 0
        self.start_yaw = float(yaw_now)

    def push_exec_action(self, actions: torch.Tensor) -> None:
        self.exec_action = torch.roll(self.exec_action, shifts=1, dims=1)
        self.exec_action[:, 0] = actions.to(self.device).reshape(1, -1)

    def _encode_state(self, w: dict) -> torch.Tensor:
        if self.state_in_mode == "proprio":
            return self.adapter.encode_proprio_window({
                "body_rot": w["body_rot"], "body_vel": w["body_vel"],
                "body_ang_vel": w["body_ang_vel"], "dof_pos": w["dof_pos"],
                "dof_vel": w["dof_vel"], "exec_action": self.exec_action,
            })
        if self.state_in_mode == "proprio_with_pos_obs":
            return self.bundle.encode_pos_obs_window(
                w["body_pos"], w["body_rot"], w["dof_pos"], ref_frame_idx=0,
            )
        raise ValueError(f"unsupported state_in_mode {self.state_in_mode!r}")

    def _condition(self, depth_flat: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        chunks = []
        depth = depth_flat.reshape(1, 1, self.bundle.depth_height, self.bundle.depth_width)
        chunks.append(self.model.encode_depth(depth))
        if self.model.goal_encoder is not None:
            chunks.append(self.model.encode_goal(command.to(self.device).reshape(1, -1), None))
        cond = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)
        if cond.shape[-1] != self.bundle.condition_dim:
            raise RuntimeError(
                f"condition dim {cond.shape[-1]} != {self.bundle.condition_dim}"
            )
        return cond

    def command_vector(self, yaw_command_rad: float, yaw_now: float) -> torch.Tensor:
        """[cos/sin(cmd), cos/sin(prog), cos/sin(rem)] from the start heading; prog from IMU yaw."""
        cmd = _wrap_pi(yaw_command_rad)
        prog = _wrap_pi(yaw_now - self.start_yaw)
        rem = _wrap_pi(cmd - prog)
        return torch.tensor(
            [[math.cos(cmd), math.sin(cmd), math.cos(prog), math.sin(prog),
              math.cos(rem), math.sin(rem)]], device=self.device, dtype=torch.float32,
        )

    def _project(self, a: torch.Tensor) -> torch.Tensor:
        if self.zg_radius <= 0.0:
            return a
        return a * (self.zg_radius / a.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    def install_plan(self, plan: torch.Tensor, cursor: int = 0) -> None:
        """Sim's _install_plan_actions: project, then ramp in over blend_frames from the old plan's
        continuation. `cursor` = ticks elapsed since the state the plan was computed from."""
        plan.record_stream(torch.cuda.current_stream(plan.device))  # produced on the planner stream
        new = self._project(plan.clone())
        start = min(int(cursor), new.shape[1] - 1)
        n = min(self.blend_frames, new.shape[1] - start)
        if self.plan is not None and n > 0:
            idx = (self.cursor + torch.arange(n, device=new.device)).clamp(max=self.plan.shape[1] - 1)
            old = self.plan[:, idx]
            w = (torch.arange(1, n + 1, device=new.device, dtype=new.dtype) / n).view(1, n, 1)
            new[:, start:start + n] = self._project(old * (1.0 - w) + new[:, start:start + n] * w)
        self.plan = new
        self.cursor = start
    def advance(self) -> torch.Tensor:
        """Next z_g of the installed plan (holds the last frame past its end)."""
        zg = self.plan[:, min(self.cursor, self.plan.shape[1] - 1)]
        self.cursor += 1
        self.hist_action = torch.roll(self.hist_action, shifts=1, dims=1)
        self.hist_action[:, 0] = zg
        return zg
    @torch.no_grad()
    def step(self, state_window: dict, depth_flat: torch.Tensor,
             command: torch.Tensor) -> torch.Tensor:
        if self.plan is None or self.cursor >= self.action_horizon:  # sim: execute action_horizon frames, then replan
            self._replan(state_window, depth_flat, command)
        return self.advance()

    @torch.no_grad()
    def plan_from(self, state_window: dict, depth_flat, command) -> torch.Tensor:
        return sample_actions_from_unified(
            self.model, self._encode_state(state_window), self.hist_action,
            self._condition(depth_flat, command),
            num_plans=1, sampler=self.sampler,
            num_inference_steps=self.num_inference_steps,
            horizon=self.F, mode="full",  # decode the full chunk (sim horizon_steps = F); action_horizon is the cadence
        )

    def _replan(self, state_window, depth_flat, command) -> None:
        self.install_plan(self.plan_from(state_window, depth_flat, command))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
