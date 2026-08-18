"""Python/TensorRT mirror of the official SONIC kinematic planner.

This module implements the public behavior of NVIDIA's
``LocalMotionPlannerBase`` and production ``LocalMotionPlannerTensorRT`` in
``GR00T-WholeBodyControl``.  It intentionally has no dependency on the
unreleased ``sonic.planner.kinematic`` research wrapper that older versions of
this deployment imported.

Planner coordinates are an abstract world frame whose initial heading is +X.
The deployment reference provider applies the robot's initial heading when it
builds encoder orientation observations, matching the official C++ heading
state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from . import robot_g1
from .inference_backend import create_session


PLANNER_FPS = 30.0
CONTROL_FPS = 50.0
CONTEXT_FRAMES = 4
QPOS_SIZE = 7 + robot_g1.NUM_DOF
DEFAULT_HEIGHT = 0.788740
DEFAULT_RANDOM_SEED = 1234
MOTION_LOOKAHEAD_STEPS = 2
BLEND_FRAMES = 8

# Exact active TensorRT deployment setting.  Entries correspond to generated
# token counts 6..16; the release permits 9, 10, or 11 tokens.
ALLOWED_PRED_NUM_TOKENS = np.array(
    [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64
)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("planner produced a zero-length quaternion")
    return q / n


def quat_slerp(q0: np.ndarray, q1: np.ndarray, amount) -> np.ndarray:
    """Shortest-path WXYZ SLERP with broadcasting."""
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    t = np.asarray(amount, dtype=np.float64)
    while t.ndim < q0.ndim:
        t = t[..., None]

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    linear = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe = np.where(np.abs(sin_theta) < 1e-8, 1.0, sin_theta)
    curved = (
        np.sin((1.0 - t) * theta) / safe * q0
        + np.sin(t * theta) / safe * q1
    )
    lerped = (1.0 - t) * q0 + t * q1
    return _normalize_quat(np.where(linear, lerped, curved))


@dataclass(frozen=True)
class PlannerMotion:
    """One resampled planner trajectory, all arrays in IsaacLab convention."""

    joint_pos_il: np.ndarray       # (T, 29)
    joint_vel_il: np.ndarray       # (T, 29)
    root_pos: np.ndarray           # (T, 3)
    root_quat_wxyz: np.ndarray     # (T, 4)

    @property
    def timesteps(self) -> int:
        return int(self.joint_pos_il.shape[0])


class SonicKinematicPlanner:
    """ONNX implementation of the official V0/V1/V2 planner pipeline."""

    def __init__(
        self,
        model_path,
        *,
        version: int | None = None,
        default_height: float = DEFAULT_HEIGHT,
        random_seed: int = DEFAULT_RANDOM_SEED,
        backend: str = "auto",
        cache_dir=None,
        fp16: bool = False,
        allow_tf32: bool = False,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"SONIC planner model not found: {self.model_path}")

        if version is None:
            name_upper = str(self.model_path).upper()
            version = 2 if "V2" in name_upper else 1 if "V1" in name_upper else 0
        self.version = int(version)
        if self.version not in (0, 1, 2):
            raise ValueError(f"unsupported planner version {self.version}; expected 0, 1, or 2")
        self.max_mode = {0: 4, 1: 20, 2: 27}[self.version]
        self.default_height = float(default_height)
        self.random_seed = int(random_seed)
        self.motion_look_ahead_steps = MOTION_LOOKAHEAD_STEPS

        self._session = create_session(
            self.model_path,
            backend=backend,
            cache_dir=cache_dir,
            fp16=fp16,
            allow_tf32=allow_tf32,
        )
        self.providers = tuple(self._session.get_providers())
        self.engine_path = getattr(self._session, "engine_path", None)
        self._inputs = {node.name: node for node in self._session.get_inputs()}
        self._output_names = [node.name for node in self._session.get_outputs()]
        required = {
            "context_mujoco_qpos", "target_vel", "mode",
            "movement_direction", "facing_direction", "random_seed",
        }
        if self.version >= 1:
            required |= {
                "height", "has_specific_target", "specific_target_positions",
                "specific_target_headings", "allowed_pred_num_tokens",
            }
        missing = sorted(required - self._inputs.keys())
        if missing:
            raise RuntimeError(f"planner ONNX missing required inputs: {missing}")
        if not {"mujoco_qpos", "num_pred_frames"}.issubset(self._output_names):
            raise RuntimeError(
                f"planner ONNX outputs are {self._output_names}; expected "
                "mujoco_qpos and num_pred_frames"
            )

        self.current_motion: PlannerMotion | None = None
        self.current_frame = 0
        self.last_inference_ms = np.nan
        self.last_total_ms = np.nan
        print(f"SONIC planner providers: {self.providers}")

    def _reshape_input(self, name: str, value, dtype) -> np.ndarray:
        node = self._inputs[name]
        shape = tuple(1 if not isinstance(d, int) or d < 1 else d for d in node.shape)
        arr = np.asarray(value, dtype=dtype)
        return arr.reshape(shape)

    def _feed(
        self,
        context: np.ndarray,
        *,
        mode: int,
        target_vel: float,
        height: float,
        movement_direction,
        facing_direction,
    ) -> dict[str, np.ndarray]:
        if not 0 <= int(mode) < self.max_mode:
            raise ValueError(
                f"locomotion mode {mode} is invalid for planner V{self.version}; "
                f"expected [0, {self.max_mode - 1}]"
            )
        feed = {
            "context_mujoco_qpos": self._reshape_input(
                "context_mujoco_qpos", context, np.float32
            ),
            "target_vel": self._reshape_input("target_vel", [target_vel], np.float32),
            "mode": self._reshape_input("mode", [mode], np.int64),
            "movement_direction": self._reshape_input(
                "movement_direction", movement_direction, np.float32
            ),
            "facing_direction": self._reshape_input(
                "facing_direction", facing_direction, np.float32
            ),
            "random_seed": self._reshape_input(
                "random_seed", [self.random_seed], np.int64
            ),
        }
        if self.version >= 1:
            feed.update({
                "height": self._reshape_input("height", [height], np.float32),
                "has_specific_target": self._reshape_input(
                    "has_specific_target", [0], np.int64
                ),
                "specific_target_positions": self._reshape_input(
                    "specific_target_positions", np.zeros((1, 4, 3)), np.float32
                ),
                "specific_target_headings": self._reshape_input(
                    "specific_target_headings", np.zeros((1, 4)), np.float32
                ),
                "allowed_pred_num_tokens": self._reshape_input(
                    "allowed_pred_num_tokens", ALLOWED_PRED_NUM_TOKENS, np.int64
                ),
            })
        return feed

    def _run_planner(self, context: np.ndarray, **command) -> PlannerMotion:
        start = time.perf_counter()
        feed = self._feed(context, **command)
        infer_start = time.perf_counter()
        outputs = self._session.run(
            ["mujoco_qpos", "num_pred_frames"], feed
        )
        self.last_inference_ms = (time.perf_counter() - infer_start) * 1e3

        qpos = np.asarray(outputs[0], dtype=np.float64)
        if qpos.ndim == 3:
            qpos = qpos[0]
        qpos = qpos.reshape(-1, QPOS_SIZE)
        n_valid = int(np.asarray(outputs[1]).reshape(-1)[0])
        if not 2 <= n_valid <= min(64, len(qpos)):
            raise RuntimeError(
                f"planner returned invalid num_pred_frames={n_valid} "
                f"for output shape {qpos.shape}"
            )
        qpos = qpos[:n_valid]
        if not np.isfinite(qpos).all():
            raise RuntimeError("planner output contains NaN or infinity")
        motion = self._resample_50hz(qpos)
        self.last_total_ms = (time.perf_counter() - start) * 1e3
        return motion

    @staticmethod
    def _resample_50hz(qpos_mj: np.ndarray) -> PlannerMotion:
        """Exact C++ sampling grid: 30 Hz output to floor(seconds*50) frames."""
        n30 = len(qpos_mj)
        n50 = int(np.floor(n30 / PLANNER_FPS * CONTROL_FPS))
        if n50 < 2:
            raise RuntimeError(f"planner motion is too short after resampling: {n50}")

        f30 = np.arange(n50, dtype=np.float64) / CONTROL_FPS * PLANNER_FPS
        f0 = np.floor(f30).astype(np.int64)
        f1 = np.minimum(f0 + 1, n30 - 1)
        w = f30 - f0

        root_pos = (1.0 - w[:, None]) * qpos_mj[f0, :3] + w[:, None] * qpos_mj[f1, :3]
        root_quat = quat_slerp(qpos_mj[f0, 3:7], qpos_mj[f1, 3:7], w)
        joint_mj = (
            (1.0 - w[:, None]) * qpos_mj[f0, 7:]
            + w[:, None] * qpos_mj[f1, 7:]
        )
        joint_il = robot_g1.mj_to_il(joint_mj)
        joint_vel = np.empty_like(joint_il)
        joint_vel[:-1] = np.diff(joint_il, axis=0) * CONTROL_FPS
        joint_vel[-1] = joint_vel[-2]
        return PlannerMotion(joint_il, joint_vel, root_pos, root_quat)

    def initialize(self, base_quat_wxyz, joint_positions_mj) -> None:
        """Initialize with the measured joints, as official C++ ``Initialize``."""
        del base_quat_wxyz  # C++ initializes the normalized planner frame at zero yaw.
        joints = np.asarray(joint_positions_mj, dtype=np.float32).reshape(robot_g1.NUM_DOF)
        context = np.zeros((1, CONTEXT_FRAMES, QPOS_SIZE), dtype=np.float32)
        context[0, :, 2] = self.default_height
        context[0, :, 3] = 1.0
        context[0, :, 7:] = joints
        self.current_motion = self._run_planner(
            context,
            mode=0,
            target_vel=-1.0,
            height=-1.0,
            movement_direction=np.array([[0.0, 0.0, 0.0]], np.float32),
            facing_direction=np.array([[1.0, 0.0, 0.0]], np.float32),
        )
        self.current_frame = 0

    def prepare_replan(self) -> tuple[int, np.ndarray]:
        """Snapshot the official lookahead context before inference runs unlocked."""
        if self.current_motion is None:
            raise RuntimeError("planner is not initialized")
        gen_frame = self.current_frame + self.motion_look_ahead_steps
        return gen_frame, self._context_from_motion(self.current_motion, gen_frame)

    @staticmethod
    def _context_from_motion(motion: PlannerMotion, gen_frame: int) -> np.ndarray:
        context = np.empty((1, CONTEXT_FRAMES, QPOS_SIZE), dtype=np.float32)
        for n in range(CONTEXT_FRAMES):
            f50 = (gen_frame / CONTROL_FPS + n / PLANNER_FPS) * CONTROL_FPS
            f0 = min(int(np.floor(f50)), motion.timesteps - 1)
            f1 = min(f0 + 1, motion.timesteps - 1)
            w = f50 - np.floor(f50)
            context[0, n, :3] = (
                (1.0 - w) * motion.root_pos[f0] + w * motion.root_pos[f1]
            )
            context[0, n, 3:7] = quat_slerp(
                motion.root_quat_wxyz[f0], motion.root_quat_wxyz[f1], w
            )
            joint_il = (
                (1.0 - w) * motion.joint_pos_il[f0]
                + w * motion.joint_pos_il[f1]
            )
            context[0, n, 7:] = robot_g1.il_to_mj(joint_il)
        return context

    def infer_replan(self, context: np.ndarray, **command) -> PlannerMotion:
        return self._run_planner(context, **command)

    def install_replan(self, generated: PlannerMotion, gen_frame: int) -> None:
        """Install a completed replan with the official eight-frame cross-fade."""
        old = self.current_motion
        if old is None:
            self.current_motion = generated
            self.current_frame = 0
            return

        current = self.current_frame
        new_length = gen_frame - current + generated.timesteps
        if new_length <= 0:
            raise RuntimeError(
                "planner result arrived after its entire generated horizon"
            )
        f = np.arange(new_length, dtype=np.int64)
        f_old = np.clip(f + current, 0, old.timesteps - 1)
        f_new = np.clip(f_old - gen_frame, 0, generated.timesteps - 1)
        blend_start = max(0, gen_frame - current)
        w_new = np.clip((f - blend_start) / float(BLEND_FRAMES), 0.0, 1.0)
        w_old = 1.0 - w_new

        q = w_old[:, None] * old.joint_pos_il[f_old] + w_new[:, None] * generated.joint_pos_il[f_new]
        # Do not blend q and dq independently: the changing cross-fade weight
        # adds a derivative term that the weighted source velocities omit.
        # Recompute the exact velocity of the trajectory the tracker receives.
        dq = np.zeros_like(q)
        if len(q) > 1:
            dq[:-1] = np.diff(q, axis=0) * CONTROL_FPS
            dq[-1] = dq[-2]
        pos = w_old[:, None] * old.root_pos[f_old] + w_new[:, None] * generated.root_pos[f_new]
        quat = quat_slerp(old.root_quat_wxyz[f_old], generated.root_quat_wxyz[f_new], w_new)
        self.current_motion = PlannerMotion(q, dq, pos, quat)
        self.current_frame = 0

    def advance(self, frames: int = 1) -> None:
        if self.current_motion is not None:
            self.current_frame = min(
                self.current_frame + int(frames), self.current_motion.timesteps - 1
            )

    def gather(self, future_frames: int, future_stride: int) -> dict[str, np.ndarray]:
        if self.current_motion is None:
            raise RuntimeError("planner is not initialized")
        motion = self.current_motion
        idx = np.minimum(
            self.current_frame + np.arange(future_frames) * future_stride,
            motion.timesteps - 1,
        )
        cur = min(self.current_frame, motion.timesteps - 1)
        # frames clamped past the plan end are a hold: their velocity is zero,
        # otherwise an exhausted idle plan commands perpetual motion
        dvel = motion.joint_vel_il[idx].copy()
        dvel[idx >= motion.timesteps - 1] = 0.0
        return {
            "dof_fut": motion.joint_pos_il[idx],
            "dvel_fut": dvel,
            "rootq_fut": motion.root_quat_wxyz[idx],
            "rootz_fut": motion.root_pos[idx, 2],
            "root_pos_cur": motion.root_pos[cur],
            "root_quat_cur": motion.root_quat_wxyz[cur],
            "rootz_cur": motion.root_pos[cur, 2],
        }
