"""Env-free diffusion planner: depth + command -> z_g. Single-robot RolloutPlanner."""

from __future__ import annotations

import math
import threading
import time

import torch

from motion_diffusion.features.window_adapter import WindowAdapter
from motion_diffusion.planner.core import load_planner, sample_actions_from_unified


class DeployPlanner:
    def __init__(self, model_path, cfg_path, device, *, sampler="ddim",
                 num_inference_steps=12, action_horizon=2, num_dof=23):
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

    @torch.no_grad()
    def step(self, state_window: dict, depth_flat: torch.Tensor,
             command: torch.Tensor) -> torch.Tensor:
        if self.plan is None or self.cursor >= self.plan.shape[1]:
            self._replan(state_window, depth_flat, command)
        zg = self.plan[:, self.cursor]
        self.cursor += 1
        self.hist_action = torch.roll(self.hist_action, shifts=1, dims=1)
        self.hist_action[:, 0] = zg
        return zg

    @torch.no_grad()
    def plan_from(self, state_window: dict, depth_flat, command) -> torch.Tensor:
        return sample_actions_from_unified(
            self.model, self._encode_state(state_window), self.hist_action,
            self._condition(depth_flat, command),
            num_plans=1, sampler=self.sampler,
            num_inference_steps=self.num_inference_steps,
            horizon=self.action_horizon or self.F, mode="full",
        )

    def _replan(self, state_window, depth_flat, command) -> None:
        self.plan = self.plan_from(state_window, depth_flat, command)
        self.cursor = 0


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AsyncPlanner(threading.Thread):
    """Sampling off the control loop; newest plan published by reference, no locks."""

    def __init__(self, planner: DeployPlanner):
        super().__init__(daemon=True)
        self.planner = planner
        self.running = True
        self._request = None
        self.latest_plan = None
        self.latest_seq = 0
        self.plan_count = 0
        self.last_latency_ms = 0.0
        self.error = None

    def submit(self, state_window: dict, depth_flat, command) -> None:
        self._request = (
            {k: v.clone() for k, v in state_window.items()}, depth_flat.clone(), command.clone(),
        )

    def run(self) -> None:
        while self.running:
            req, self._request = self._request, None
            if req is None:
                time.sleep(0.001)
                continue
            try:
                t0 = time.perf_counter()
                with torch.no_grad():
                    plan = self.planner.plan_from(*req)
                self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
                self.latest_plan = plan
                self.latest_seq += 1
                self.plan_count += 1
            except Exception as exc:
                self.error = exc
                self.running = False

    def stop(self) -> None:
        self.running = False
