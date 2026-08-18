"""Reference-window sources for the SONIC tracker.

The encoder needs, every tick, a window describing what the robot should be
doing: 10 future frames of joint targets plus current body poses. Two sources,
matching the two deploy modes:

    MotionReferenceWindow             replay an exported clip (standalone)
    policy/pico_reference.py          live 3-point VR teleop

Both return the dict ``sonic.policy.obs.build_encoder_obs`` expects, batched
with a leading dimension of 1. Term shapes and the future stride are taken from
``sonic/motion/bank.py`` in the research repo and must stay identical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Wrist IsaacLab dof indices used by the smpl term (== obs.WRIST_IL_IDX).
WRIST_IL_IDX = np.array([23, 24, 25, 26, 27, 28])


def _yaw(q) -> float:
    w, x, y, z = np.asarray(q, np.float64)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class MotionReferenceWindow:
    """Replay an exported motion clip.

    Expects an .npz with the arrays ``MotionBank`` holds, already in IsaacLab
    order and resampled to 50 fps:

        dof_il              (T,29)
        dof_vel_il          (T,29)
        body_pos_il         (T,14,3)
        body_quat_wxyz_il   (T,14,4)

    Export one with ``scripts/export_sonic_motion.py`` -- that step needs the
    research repo's training stack, which is why it is offline and this class
    is not.
    """

    def __init__(self, clip_path, loop: bool = True, *, model_profile="default", robot_profile="g1_23dof"):
        from policy.sonic.model_profile import get_model_profile
        from policy.sonic.robot_profile import get_robot_profile

        self.model_profile = get_model_profile(model_profile)
        self.robot_profile = get_robot_profile(robot_profile)
        path = Path(clip_path).expanduser()
        if not path.exists():
            raise SystemExit(
                f"sonic motion clip not found: {path}\n"
                "Export one first:  python scripts/export_sonic_motion.py "
                "--pack <pack.pt> --out <clip.npz>"
            )
        d = np.load(path)
        missing = [k for k in ("dof_il", "dof_vel_il", "body_pos_il",
                               "body_quat_wxyz_il") if k not in d.files]
        if missing:
            raise SystemExit(f"{path} is missing {missing}; re-export it.")

        self.dof_il = d["dof_il"].astype(np.float64)
        self.dof_vel_il = d["dof_vel_il"].astype(np.float64)
        self.body_pos_il = d["body_pos_il"].astype(np.float64)
        self.body_quat_wxyz_il = d["body_quat_wxyz_il"].astype(np.float64)
        expected_shapes = {
            "dof_il": (29,),
            "dof_vel_il": (29,),
            "body_pos_il": (14, 3),
            "body_quat_wxyz_il": (14, 4),
        }
        arrays = {
            "dof_il": self.dof_il,
            "dof_vel_il": self.dof_vel_il,
            "body_pos_il": self.body_pos_il,
            "body_quat_wxyz_il": self.body_quat_wxyz_il,
        }
        lengths = set()
        for name, array in arrays.items():
            if array.ndim < 1 or tuple(array.shape[1:]) != expected_shapes[name]:
                raise SystemExit(
                    f"{path}: {name} has shape {array.shape}; expected "
                    f"(T,{','.join(map(str, expected_shapes[name]))})"
                )
            lengths.add(array.shape[0])
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
            raise SystemExit(
                f"{path}: motion arrays must have one common length of at least 2; "
                f"got {sorted(lengths)}"
            )
        if "fps" in d.files and not np.allclose(d["fps"], 50.0):
            raise SystemExit(f"{path} must be resampled to 50 Hz; fps={d['fps']}")
        if not all(np.isfinite(x).all() for x in (
            self.dof_il, self.dof_vel_il,
            self.body_pos_il, self.body_quat_wxyz_il,
        )):
            raise SystemExit(f"{path} contains NaN or infinity")
        quat_norm = np.linalg.norm(self.body_quat_wxyz_il, axis=-1)
        if not np.allclose(quat_norm, 1.0, atol=2e-3, rtol=2e-3):
            raise SystemExit(f"{path} contains non-unit body quaternions")
        self._body_pos_source = self.body_pos_il.copy()
        self._body_quat_source = self.body_quat_wxyz_il.copy()
        self.T = len(self.dof_il)
        self.loop = loop
        self.frame = 0
        self.done = False
        self.initialized = False
        self.sonic_mode = "g1"        # clip tracking
        print(f"SONIC reference: motion clip {path.name}, {self.T} frames "
              f"({self.T / 50.0:.1f}s @50fps){' looping' if loop else ''}")

    @property
    def initial_joint_pose(self) -> np.ndarray:
        """Frame-0 physical pose used by the real deployment handoff."""
        return self.robot_profile.isaaclab_to_physical(
            self.dof_il[0]
        ).astype(np.float32).copy()

    @property
    def initial_joint_pose_23(self) -> np.ndarray:
        """Compatibility alias for older deployment callers."""
        return self.initial_joint_pose

    def reset(self) -> None:
        self.frame = 0
        self.done = False

    def reference_diagnostics(self) -> dict:
        """Same row schema the planner reference logs, for the eval gate."""
        ci = min(self.frame, self.T - 1)
        return {
            "reference_q_il": self.dof_il[ci].astype(np.float32),
            "reference_dq_il": self.dof_vel_il[ci].astype(np.float32),
            "reference_root_position": self.body_pos_il[ci, 0].astype(np.float32),
            "reference_root_z": np.float32(self.body_pos_il[ci, 0, 2]),
            "idle_hold_phase": np.int8(0),
            "stand_capture_pending": np.bool_(False),
            "stand_capture_count": np.int32(0),
        }

    def step(self) -> None:
        self.frame += 1
        if self.frame >= self.T:
            if self.loop:
                self.frame = 0
            else:
                self.frame = self.T - 1
                self.done = True

    def initialize_from_robot(self, base_quat_wxyz, dof_pos) -> None:
        """Rotate the clip about z so its heading starts at the robot's.

        The encoder compares the robot quat against rootq_fut ABSOLUTELY, and
        clips carry a baked-in world yaw. Without this the policy spins the
        robot to the clip's yaw on the first step.
        """
        # Always restart from the file data so reset/reinitialize is idempotent.
        self.body_pos_il = self._body_pos_source.copy()
        self.body_quat_wxyz_il = self._body_quat_source.copy()
        d = _yaw(base_quat_wxyz) - _yaw(self.body_quat_wxyz_il[0, 0])
        c, s = np.cos(d), np.sin(d)
        off = np.array([np.cos(d / 2), 0.0, 0.0, np.sin(d / 2)])

        q = self.body_quat_wxyz_il
        w0, x0, y0, z0 = off
        w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        self.body_quat_wxyz_il = np.stack([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ], axis=-1)

        x, y = self.body_pos_il[..., 0].copy(), self.body_pos_il[..., 1].copy()
        self.body_pos_il[..., 0] = c * x - s * y
        self.body_pos_il[..., 1] = s * x + c * y

        self.initialized = True
        print(f"clip yaw-aligned to robot: {np.degrees(d):+.1f} deg")

    def teleop_fields(self) -> dict:
        """No teleop command in clip replay."""
        return {}

    def _idx(self, n: int, step: int) -> np.ndarray:
        return np.clip(self.frame + np.arange(n) * step, 0, self.T - 1)

    def window(self) -> dict:
        gi = self._idx(
            self.model_profile.g1_future_frames,
            self.model_profile.g1_future_stride,
        )
        si = self._idx(
            self.model_profile.smpl_future_frames,
            self.model_profile.smpl_future_stride,
        )
        ci = min(self.frame, self.T - 1)
        return {
            "dof_fut": self.dof_il[gi][None],                       # (1,10,29)
            "dvel_fut": self.dof_vel_il[gi][None],                  # (1,10,29)
            "rootq_fut": self.body_quat_wxyz_il[gi, 0][None],       # (1,10,4)
            "rootz_fut": self.body_pos_il[gi, 0, 2][None],          # (1,10)
            "body_pos_cur": self.body_pos_il[ci][None],             # (1,14,3)
            "body_quat_cur": self.body_quat_wxyz_il[ci][None],      # (1,14,4)
            "rootz_cur": np.array([self.body_pos_il[ci, 0, 2]]),    # (1,)
            "wrist_dof_smpl": self.dof_il[si][:, WRIST_IL_IDX][None],  # (1,10,6)
        }


# Live teleop lives in policy/pico_reference.py -- it needs the SONIC kinematic
# planner, whose ONNX is ~774 MB and therefore is not vendored here.


def build_reference(config, mode: str, *, gated: bool = True, idle_hold: bool = False):
    """Build a motion, dynamic-PICO, or forced-PLANNER/G1 reference."""
    if mode == "motion":
        return MotionReferenceWindow(
            config.sonic_motion_clip,
            loop=config.sonic_motion_loop,
            model_profile=config.sonic_model_variant,
            robot_profile=config.sonic_robot_profile,
        )
    if mode in ("pico", "planner", "smpl"):
        from common.pico_client import PicoClient
        from policy.pico_reference import PicoReferenceWindow

        client = PicoClient(
            config.pico_server_ip,
            config.pico_server_port,
        )
        client.start()
        print(f"teleop:  tcp://{client.server_address}:{client.port}")
        return PicoReferenceWindow(
            config,
            client,
            gated=gated,
            force_g1=(mode == "planner"),
            force_smpl=(mode == "smpl"),
            idle_hold=idle_hold,
        )
    raise SystemExit(
        f"unknown sonic mode {mode!r}; expected motion, pico, planner, or smpl"
    )
