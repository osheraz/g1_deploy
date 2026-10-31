"""Fall-recovery deploy: planner -> z_g -> tracker. Drop-in for PolicyRunner; no SONIC code."""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HV2_ROOT = Path(__file__).resolve().parents[3]
if str(HV2_ROOT) not in sys.path:
    sys.path.insert(0, str(HV2_ROOT))

from deprecated.vigor2.planner import DeployPlanner  # noqa: E402
from deprecated.vigor2.planner_process import PlannerProcess  # noqa: E402
from deprecated.vigor2.state import DeployStateHistory  # noqa: E402
from deprecated.vigor2.lidar import DeployLidarHistory  # noqa: E402
from deprecated.vigor2.tracker import ACTOR_OBS_KEYS, DeployTracker  # noqa: E402


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
        if not config.tracker_checkpoint:
            raise ValueError("tracker_checkpoint must point to the paired 29-DoF tracker")
        if not config.planner_model:
            raise ValueError(
                "planner_model is unset; train the pinned 29-DoF LiDAR config and "
                "set this to its model.pt before deployment"
            )
        self.tracker = DeployTracker(config.tracker_checkpoint, device)
        self._planner_kwargs = dict(
            model_path=config.planner_model, cfg_path=config.planner_cfg, device=device,
            sampler=config.planner_sampler,
            num_inference_steps=config.planner_inference_steps,
            seed=config.planner_seed,
            action_horizon=config.planner_action_horizon,
            num_dof=self.tracker.num_actions,
            blend_frames=config.planner_blend_frames,
        )
        self.planner = DeployPlanner(**self._planner_kwargs)
        tcfg = self.tracker.config
        self.dof_names = [str(n) for n in tcfg.robot.dof_names]
        self.num_actions = self.tracker.num_actions
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.act_count = 0

        from humanoidverse2.simulator.sensors.robot_mesh import (
            build_pyroki_fk_model, load_fk_urdf_for_mesh,
        )

        asset = tcfg.robot.asset
        urdf = str(asset.urdf_file).replace(
            "${robot.asset.robot_type}", str(asset.robot_type)
        )
        urdf_path = HV2_ROOT / "humanoidverse2/data/robots" / urdf
        self.fk_urdf_path = urdf_path
        fk = build_pyroki_fk_model(load_fk_urdf_for_mesh(urdf_path), self.dof_names)
        self.state = DeployStateHistory(
            self.planner.bundle, fk, self.dof_names,
            [str(n) for n in self.planner.bundle.body_names], device,
        )
        self.use_depth = bool(self.planner.bundle.use_ego_depth)
        self.use_lidar = bool(self.planner.bundle.use_lidar)
        self.depth_h = int(self.planner.bundle.depth_height)
        self.depth_w = int(self.planner.bundle.depth_width)
        self.min_d = float(config.depth_min_d)
        self.max_d = float(config.depth_max_d)
        self.crop_px = tuple(int(v) for v in config.depth_crop_px)
        self.yaw_command_rad = math.radians(float(config.yaw_command_deg))
        self.depth_source = str(config.depth_source)
        if self.use_depth and self.depth_source not in ("full", "square"):
            raise ValueError(f"depth_source must be full or square, got {self.depth_source!r}")
        self.lidar = None
        if self.use_lidar:
            fk_names = list(self.state.fk.link_names)
            self.lidar = DeployLidarHistory(
                self.planner.bundle.lidar_config, fk_names, device
            )
            parent = str(
                self.planner.bundle.lidar_config.get("parent_body", "torso_link")
            )
            root = str(
                self.planner.bundle.body_names[self.planner.bundle.anchor_body_id]
            )
            self._lidar_torso_index = fk_names.index(parent)
            self._lidar_root_index = fk_names.index(root)
        self._started = False
        self._plan_seq = 0
        self._last_submit_tick = -10**9
        self.plan_age = 0
        self.stand_takeover = bool(config.stand_takeover)
        self.stand_up_threshold = float(config.stand_up_threshold)
        self.stand_held = False
        self.stand_held_tick = 0
        self.async_planner = None
        self.first_plan_timeout_s = 5.0  # the first plan includes the child's CUDA warm-up
        if bool(config.planner_async):
            w = self.state.window()
            self.async_planner = PlannerProcess(
                str(HV2_ROOT), self._planner_kwargs, {k: tuple(v.shape) for k, v in w.items()},
                self.planner.vision_dim, 6, self.planner.H, self.planner.F,
                self.planner.action_dim, self.num_actions, int(self.planner.bundle.state_out_dim), device,
            )

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
        if self.use_lidar:
            if not bool(robot_config.require_torso_imu):
                raise ValueError("LiDAR checkpoint requires the physical torso IMU")
            expected = int(self.lidar.spec.num_points)
            if int(robot_config.livox_max_points) != expected:
                raise ValueError(
                    f"Livox point count {robot_config.livox_max_points} != "
                    f"checkpoint contract {expected}"
                )
            expected_duration = float(
                self.planner.bundle.lidar_config.get("scan_duration_s", 0.10)
            )
            if not math.isclose(
                float(robot_config.livox_scan_duration_s), expected_duration,
                rel_tol=0.0, abs_tol=1.0e-6,
            ):
                raise ValueError(
                    f"Livox window {robot_config.livox_scan_duration_s:.3f} s != "
                    f"checkpoint contract {expected_duration:.3f} s"
                )

    def warmup(
        self, base_ang_vel, projected_gravity, dof_pos, dof_vel,
        vision, base_quat_wxyz, **sensor_kwargs,
    ) -> float:
        """Compile FK and run one plan on real data before the loop (first ticks otherwise take ~0.9 s); returns seconds."""
        t0 = time.perf_counter()
        for _ in range(int(self.planner.bundle.H) + 1):
            self.compute_action(base_ang_vel, projected_gravity, dof_pos, dof_vel,
                                depth_raw=vision, depth_full=vision,
                                base_quat_wxyz=base_quat_wxyz, **sensor_kwargs)
        self.reset()
        return time.perf_counter() - t0
    def reset(self, yaw_now: float = 0.0) -> None:
        self.state.reset()
        self.planner.reset(yaw_now)
        self.action[:] = 0.0
        self.act_count = 0
        self._started = False
        self._plan_seq = 0
        self._last_submit_tick = -10**9
        self.plan_age = 0
        self.stand_held = False
        self.stand_held_tick = 0
        if self.lidar is not None:
            self.lidar.reset()

    def close(self) -> None:
        if self.async_planner is not None:
            self.async_planner.stop()

    def compute_action(self, base_ang_vel, projected_gravity, dof_pos, dof_vel,
                       depth_raw=None, base_quat_wxyz=None, depth_full=None,
                       lidar_scan=None, lidar_seq=None, lidar_age_s=None,
                       torso_quat_wxyz=None, torso_gyro=None,
                       physical_dof_pos=None, physical_dof_vel=None,
                       physical_base_ang_vel=None,
                       lidar_physical_dof_pos=None) -> np.ndarray:
        depth = depth_full if self.depth_source == "full" else depth_raw
        if self.use_depth and depth is None:
            raise ValueError(
                f"depth_source={self.depth_source!r} needs a "
                f"{'depth_full' if self.depth_source == 'full' else 'depth_raw'} frame."
            )
        if base_quat_wxyz is None:
            raise ValueError("Fall recovery needs the IMU quaternion for FK placement.")
        if self.async_planner is not None:
            self.async_planner.check()

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32,  # noqa: E731
                                      device=self.device).reshape(1, -1)
        dof_pos_t, dof_vel_t = t(dof_pos), t(dof_vel)
        quat_wxyz = t(base_quat_wxyz)
        quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, :1]], dim=-1)
        yaw_now = _yaw_from_quat_xyzw(quat_xyzw)
        if not self._started:
            self.planner.reset(yaw_now)
            self._started = True

        physical_dof_pos_t = t(dof_pos if physical_dof_pos is None else physical_dof_pos)
        physical_dof_vel_t = t(dof_vel if physical_dof_vel is None else physical_dof_vel)
        physical_ang_vel_t = t(
            base_ang_vel if physical_base_ang_vel is None else physical_base_ang_vel
        )
        self.state.push(
            physical_dof_pos_t, physical_dof_vel_t, quat_xyzw, physical_ang_vel_t
        )
        if not self.state.ready:
            return np.zeros_like(self.action)

        # sim's stand_takeover criterion, latched: stop planning, hold the last z_g, fade the action to 0
        if self.stand_takeover and not self.stand_held and -float(projected_gravity[2]) > self.stand_up_threshold:
            self.stand_held = True
            self.stand_held_tick = self.act_count
        if self.use_depth:
            vision_flat = self.normalize_depth(depth)
        else:
            if any(value is None for value in (
                lidar_scan, lidar_seq, lidar_age_s, torso_quat_wxyz, torso_gyro
            )):
                raise ValueError(
                    "LiDAR planner needs scan, sequence, age, torso quaternion, and torso gyro."
                )
            lidar_q = (
                physical_dof_pos
                if lidar_physical_dof_pos is None
                else lidar_physical_dof_pos
            )
            fk_pos, fk_rot = self.state.fk_poses(t(lidar_q))
            self.lidar.update(
                seq=int(lidar_seq),
                raw_scan=np.asarray(lidar_scan),
                scan_age_s=float(lidar_age_s),
                torso_quat_wxyz=np.asarray(torso_quat_wxyz),
                torso_gyro=np.asarray(torso_gyro),
                fk_pos=fk_pos[0].detach().cpu().numpy(),
                fk_quat_xyzw=fk_rot[0].detach().cpu().numpy(),
                torso_index=self._lidar_torso_index,
                root_index=self._lidar_root_index,
                now=time.monotonic(),
            )
            vision_flat = self.lidar.flatten(time.monotonic())
        command = self.planner.command_vector(self.yaw_command_rad, yaw_now)
        if self.stand_held and self.planner.plan is not None:
            zg = self.planner.advance()
        elif self.async_planner is not None:
            zg = self._zg_async(vision_flat, command)
        else:
            zg = self.planner.step(self.state.window(), vision_flat, command)
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
        if self.stand_held:  # action 0 = physical default angles; the launcher ramps onto it
            action = np.zeros_like(action)
        self.action = action.astype(np.float32)
        self.planner.push_exec_action(torch.from_numpy(self.action))
        self.act_count += 1
        return self.action

    def _install_latest(self) -> None:
        self._plan_seq = self.async_planner.latest_seq
        age = self.act_count - self.async_planner.latest_tick  # ticks since the plan's origin state
        self.planner.install_plan(self.async_planner.latest_plan, cursor=age)
        self.plan_age = age
    def _zg_async(self, vision_flat, command) -> torch.Tensor:
        # replan as soon as the planner is free, at most every action_horizon ticks (SONIC: replan ahead, never wait)
        if self.async_planner.poll():
            self._install_latest()
        if self.async_planner.idle and self.act_count - self._last_submit_tick >= self.planner.action_horizon:
            self.async_planner.submit(self.state.window(), vision_flat, command,
                                      self.planner.hist_action, self.planner.exec_action, self.act_count)
            self._last_submit_tick = self.act_count
        if self.planner.plan is None:  # first plan only: wait for it, bounded, then raise
            deadline = time.perf_counter() + self.first_plan_timeout_s
            while not self.async_planner.poll():
                self.async_planner.check()
                if time.perf_counter() > deadline:
                    raise RuntimeError(
                        f"No plan after {self.first_plan_timeout_s:.1f}s; planner process stalled."
                    )
                time.sleep(0.001)
            self._install_latest()
        self.plan_age += 1
        return self.planner.advance()

    def normalize_depth(self, depth_m) -> torch.Tensor:
        """full = whole frame, matches sim; square = student-era centre crop."""
        from core.common.depth_client import depth_to_obs_vector_match_sim

        vec = depth_to_obs_vector_match_sim(
            depth_m, self.depth_h, self.depth_w, self.min_d, self.max_d, self.crop_px
        )
        return torch.as_tensor(vec, dtype=torch.float32, device=self.device).reshape(1, -1)


def _yaw_from_quat_xyzw(q: torch.Tensor) -> float:
    x, y, z, w = (float(v) for v in q.reshape(-1))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
