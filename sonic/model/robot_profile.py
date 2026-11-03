"""Explicit physical-robot adaptations for the official 29-DoF SONIC model.

The released SONIC encoder, decoder, and planner always use the official
29-joint IsaacLab/MuJoCo layouts.  This module is the only place where a
physical G1 embodiment is projected into or out of that model layout.

Keeping this boundary separate is intentional: :mod:`robot_g1` remains a
small copy of NVIDIA's joint-order definitions, while 23-DoF support can be
reviewed, tested, and disabled without modifying the model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonic.model import robot_g1

WRIST_IL = np.array([23, 24, 25, 26, 27, 28], dtype=np.int64)
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class RobotProfile:
    """Mapping between physical motor order and SONIC's 29-D model order."""

    name: str
    physical_motor_indices: np.ndarray
    isaaclab_from_physical: np.ndarray
    missing_isaaclab: np.ndarray
    missing_motor: np.ndarray
    missing_names: tuple[str, ...]

    def __post_init__(self) -> None:
        motor = np.asarray(self.physical_motor_indices, dtype=np.int64)
        il = np.asarray(self.isaaclab_from_physical, dtype=np.int64)
        missing_il = np.asarray(self.missing_isaaclab, dtype=np.int64)
        missing_motor = np.asarray(self.missing_motor, dtype=np.int64)
        if motor.ndim != 1 or il.shape != motor.shape:
            raise ValueError(f"{self.name}: physical/IsaacLab maps must be 1-D and equal")
        if len(np.unique(motor)) != len(motor) or np.any((motor < 0) | (motor >= 29)):
            raise ValueError(f"{self.name}: invalid physical motor mapping")
        if len(np.unique(il)) != len(il) or np.any((il < 0) | (il >= 29)):
            raise ValueError(f"{self.name}: invalid IsaacLab mapping")
        if set(il.tolist()) | set(missing_il.tolist()) != set(range(29)):
            raise ValueError(f"{self.name}: present and missing IsaacLab joints must partition 29")
        if set(motor.tolist()) | set(missing_motor.tolist()) != set(range(29)):
            raise ValueError(f"{self.name}: present and missing motor joints must partition 29")
        if len(missing_il) != len(self.missing_names):
            raise ValueError(f"{self.name}: missing joint names do not match indices")
        object.__setattr__(self, "physical_motor_indices", motor)
        object.__setattr__(self, "isaaclab_from_physical", il)
        object.__setattr__(self, "missing_isaaclab", missing_il)
        object.__setattr__(self, "missing_motor", missing_motor)

    @property
    def physical_dof(self) -> int:
        return int(self.physical_motor_indices.size)

    @property
    def is_reduced(self) -> bool:
        return bool(self.missing_isaaclab.size)

    @property
    def physical_joint_names(self) -> tuple[str, ...]:
        return tuple(MUJOCO_JOINT_NAMES[i] for i in self.physical_motor_indices)

    @property
    def wrist_missing_mask(self) -> np.ndarray:
        """Mask in the official six-value wrist-reference order."""
        return np.isin(WRIST_IL, self.missing_isaaclab)

    def matches_mode_machine(self, mode_machine: int) -> bool:
        """Whether a live Unitree model code has this exact motor topology."""
        mode = int(mode_machine)
        if self.name == "g1_23dof":
            return mode in (1, 4)
        if self.name == "g1_29dof":
            return mode in (2, 5)
        return False

    def require_mode_machine(self, mode_machine: int) -> None:
        if not self.matches_mode_machine(mode_machine):
            raise RuntimeError(
                f"selected {self.name} does not match LowState mode_machine="
                f"{int(mode_machine)} (23-DoF requires 1/4; 29-DoF requires "
                "2/5; locked-waist 3/6 is unsupported)"
            )

    def validate_physical(self, values, *, name: str = "values") -> np.ndarray:
        arr = np.asarray(values)
        if arr.shape[-1:] != (self.physical_dof,):
            raise ValueError(
                f"{self.name}: {name} must end in {self.physical_dof}, got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{self.name}: {name} contains NaN or infinity")
        return arr

    def physical_to_isaaclab(self, values, *, fill: float = 0.0) -> np.ndarray:
        arr = self.validate_physical(values)
        out = np.full(arr.shape[:-1] + (29,), fill, dtype=arr.dtype)
        out[..., self.isaaclab_from_physical] = arr
        return out

    def physical_to_mujoco(self, values, *, fill: float = 0.0) -> np.ndarray:
        arr = self.validate_physical(values)
        out = np.full(arr.shape[:-1] + (29,), fill, dtype=arr.dtype)
        out[..., self.physical_motor_indices] = arr
        return out

    def isaaclab_to_physical(self, values) -> np.ndarray:
        arr = np.asarray(values)
        if arr.shape[-1:] != (29,):
            raise ValueError(f"{self.name}: SONIC values must end in 29, got {arr.shape}")
        return arr[..., self.isaaclab_from_physical]

    def mask_isaaclab_reference(self, values, *, copy: bool = True) -> np.ndarray:
        arr = np.array(values, copy=copy)
        if arr.shape[-1:] != (29,):
            raise ValueError(f"{self.name}: reference must end in 29, got {arr.shape}")
        if self.is_reduced:
            arr[..., self.missing_isaaclab] = 0.0
        return arr

    def mask_wrist_reference(self, values, *, copy: bool = True) -> np.ndarray:
        arr = np.array(values, copy=copy)
        if arr.shape[-1:] != (6,):
            raise ValueError(f"{self.name}: wrist reference must end in 6, got {arr.shape}")
        if self.is_reduced:
            arr[..., self.wrist_missing_mask] = 0.0
        return arr


G1_23DOF = RobotProfile(
    name="g1_23dof",
    physical_motor_indices=np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26],
        dtype=np.int64,
    ),
    isaaclab_from_physical=np.array(
        [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 11, 15, 19, 21, 23, 12, 16, 20, 22, 24],
        dtype=np.int64,
    ),
    missing_isaaclab=np.array([5, 8, 25, 26, 27, 28], dtype=np.int64),
    missing_motor=np.array([13, 14, 20, 21, 27, 28], dtype=np.int64),
    missing_names=(
        "waist_roll",
        "waist_pitch",
        "left_wrist_pitch",
        "right_wrist_pitch",
        "left_wrist_yaw",
        "right_wrist_yaw",
    ),
)

G1_29DOF = RobotProfile(
    name="g1_29dof",
    physical_motor_indices=np.arange(29, dtype=np.int64),
    isaaclab_from_physical=robot_g1.ISAACLAB_TO_MUJOCO_DOF.copy(),
    missing_isaaclab=np.empty(0, dtype=np.int64),
    missing_motor=np.empty(0, dtype=np.int64),
    missing_names=(),
)

PROFILES = {p.name: p for p in (G1_23DOF, G1_29DOF)}


def get_robot_profile(value) -> RobotProfile:
    """Resolve a profile name, DoF count, config object, or profile instance."""
    if isinstance(value, RobotProfile):
        return value
    if hasattr(value, "sonic_robot_profile"):
        value = value.sonic_robot_profile
    if isinstance(value, (int, np.integer)):
        value = f"g1_{int(value)}dof"
    key = str(value).strip().lower().replace("-", "_")
    aliases = {"23": "g1_23dof", "29": "g1_29dof", "g1_23": "g1_23dof", "g1_29": "g1_29dof"}
    key = aliases.get(key, key)
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown SONIC robot profile {value!r}; choose {sorted(PROFILES)}"
        ) from exc
