"""Fall-recovery deploy: planner -> z_g -> tracker. Drop-in for PolicyRunner; no SONIC code."""

from __future__ import annotations

import contextlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HV2_ROOT = Path(__file__).resolve().parents[3]
if str(HV2_ROOT) not in sys.path:
    sys.path.insert(0, str(HV2_ROOT))

from policy.fallrec.planner import AsyncPlanner, DeployPlanner  # noqa: E402
from policy.fallrec.state import DeployStateHistory  # noqa: E402
from policy.fallrec.tracker import ACTOR_OBS_KEYS, DeployTracker  # noqa: E402


@contextlib.contextmanager
def _in_repo_root():
    """Training configs store repo-relative asset paths."""
    prev = os.getcwd()
    os.chdir(HV2_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


class FallRecRunner:
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        with _in_repo_root():
            self._build(config, device)

    def _build(self, config, device) -> None:
        self.tracker = DeployTracker(config.tracker_checkpoint, device)
        self.planner = DeployPlanner(
            config.planner_model, config.planner_cfg, device,
            sampler=config.planner_sampler,
            num_inference_steps=config.planner_inference_steps,
            action_horizon=config.planner_action_horizon,
            num_dof=self.tracker.num_actions,
        )
        tcfg = self.tracker.config
        self.dof_names = [str(n) for n in tcfg.robot.dof_names]
        self.num_actions = self.tracker.num_actions
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.act_count = 0

        from motion_diffusion.terrain.robot_mesh import (
            build_pyroki_fk_model, load_fk_urdf_for_mesh,
        )

        asset = tcfg.robot.asset
        urdf = str(asset.urdf_file).replace(
            "${robot.asset.robot_type}", str(asset.robot_type)
        )
        urdf_path = HV2_ROOT / "humanoidverse2/data/robots" / urdf
        fk = build_pyroki_fk_model(load_fk_urdf_for_mesh(urdf_path), self.dof_names)
        self.state = DeployStateHistory(
            self.planner.bundle, fk, self.dof_names,
            [str(n) for n in self.planner.bundle.body_names], device,
        )
        self.depth_h = int(self.planner.bundle.depth_height)
        self.depth_w = int(self.planner.bundle.depth_width)
        self.min_d = float(config.depth_min_d)
        self.max_d = float(config.depth_max_d)
        self.yaw_command_rad = math.radians(float(config.yaw_command_deg))
        self.depth_source = str(config.depth_source)
        if self.depth_source not in ("full", "square"):
            raise ValueError(f"depth_source must be full or square, got {self.depth_source!r}")
        self._started = False
        self._plan_seq = 0
        self.plan_age = 0
        self.async_planner = None
        if bool(config.planner_async):
            self.async_planner = AsyncPlanner(self.planner)
            self.async_planner.start()

    def verify_robot_config(self, robot_config) -> None:
        """The robot yaml must carry the gains the tracker was trained with."""
        tcfg = self.tracker.config
        train_scale = np.asarray(tcfg.robot.control.action_scale, dtype=np.float32)
        deploy_scale = np.asarray(robot_config.action_scale, dtype=np.float32)
        if train_scale.shape != deploy_scale.shape or not np.allclose(
            train_scale, deploy_scale, atol=1e-6
        ):
            raise ValueError(
                f"action_scale mismatch: robot yaml {deploy_scale.reshape(-1)[:4]}... vs "
                f"training {train_scale.reshape(-1)[:4]}..."
            )
        train_default = np.array(
            [float(tcfg.robot.init_state.default_joint_angles[n]) for n in self.dof_names],
            dtype=np.float32,
        )
        deploy_default = np.asarray(robot_config.physical_default_angles, dtype=np.float32)
        if not np.allclose(train_default, deploy_default, atol=1e-4):
            bad = int(np.argmax(np.abs(train_default - deploy_default)))
            raise ValueError(
                f"default angle mismatch at {self.dof_names[bad]}: robot yaml "
                f"{deploy_default[bad]:.4f} vs training {train_default[bad]:.4f}"
            )

    def reset(self, yaw_now: float = 0.0) -> None:
        self.state.reset()
        self.planner.reset(yaw_now)
        self.action[:] = 0.0
        self.act_count = 0
        self._started = False
        self._plan_seq = 0
        self.plan_age = 0

    def close(self) -> None:
        if self.async_planner is not None:
            self.async_planner.stop()

    def compute_action(self, base_ang_vel, projected_gravity, dof_pos, dof_vel,
                       depth_raw=None, base_quat_wxyz=None, depth_full=None) -> np.ndarray:
        depth = depth_full if self.depth_source == "full" else depth_raw
        if depth is None:
            raise ValueError(
                f"depth_source={self.depth_source!r} needs a "
                f"{'depth_full' if self.depth_source == 'full' else 'depth_raw'} frame."
            )
        if base_quat_wxyz is None:
            raise ValueError("Fall recovery needs the IMU quaternion for FK placement.")
        if self.async_planner is not None and self.async_planner.error is not None:
            raise RuntimeError("Planner thread died") from self.async_planner.error

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32,  # noqa: E731
                                      device=self.device).reshape(1, -1)
        dof_pos_t, dof_vel_t = t(dof_pos), t(dof_vel)
        quat_wxyz = t(base_quat_wxyz)
        quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, :1]], dim=-1)
        yaw_now = _yaw_from_quat_xyzw(quat_xyzw)
        if not self._started:
            self.planner.reset(yaw_now)
            self._started = True

        self.state.push(dof_pos_t, dof_vel_t, quat_xyzw, t(base_ang_vel))
        if not self.state.ready:
            return np.zeros_like(self.action)

        depth_flat = self.normalize_depth(depth)
        command = self.planner.command_vector(self.yaw_command_rad, yaw_now)
        zg = (
            self._zg_async(depth_flat, command) if self.async_planner is not None
            else self.planner.step(self.state.window(), depth_flat, command)
        )
        if not bool(torch.isfinite(zg).all()):
            raise FloatingPointError(f"Planner produced non-finite z_g at tick {self.act_count}")

        proprio = {
            "actions": t(self.action),
            "base_ang_vel": t(base_ang_vel),
            "dof_pos": dof_pos_t,
            "dof_vel": dof_vel_t,
            "projected_gravity": t(projected_gravity),
        }
        assert set(proprio) == set(ACTOR_OBS_KEYS)
        action = self.tracker.act(proprio, zg).squeeze(0).cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Tracker produced non-finite actions at tick {self.act_count}")
        self.action = action.astype(np.float32)
        self.planner.push_exec_action(torch.from_numpy(self.action))
        self.act_count += 1
        return self.action

    def _zg_async(self, depth_flat, command) -> torch.Tensor:
        if self.act_count % self.planner.action_horizon == 0:
            self.async_planner.submit(self.state.window(), depth_flat, command)
        if self.async_planner.latest_seq != self._plan_seq:
            self._plan_seq = self.async_planner.latest_seq
            self.planner.plan = self.async_planner.latest_plan
            self.planner.cursor = 0
            self.plan_age = 0
        if self.planner.plan is None:
            raise RuntimeError(
                f"No plan after {self.act_count} ticks; the planner thread has not produced one."
            )
        self.plan_age += 1
        idx = min(self.planner.cursor, self.planner.plan.shape[1] - 1)
        zg = self.planner.plan[:, idx]
        self.planner.cursor += 1
        self.planner.hist_action = torch.roll(self.planner.hist_action, shifts=1, dims=1)
        self.planner.hist_action[:, 0] = zg
        return zg

    def normalize_depth(self, depth_m) -> torch.Tensor:
        """full = whole frame, matches sim; square = student-era centre crop."""
        from common.depth_client import depth_to_obs_vector_match_sim

        vec = depth_to_obs_vector_match_sim(
            depth_m, self.depth_h, self.depth_w, self.min_d, self.max_d
        )
        return torch.as_tensor(vec, dtype=torch.float32, device=self.device).reshape(1, -1)


def _yaw_from_quat_xyzw(q: torch.Tensor) -> float:
    x, y, z, w = (float(v) for v in q.reshape(-1))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
