"""Training-free adaptations applied at the SONIC tracker-reference boundary.

The planner owns a recurrent 29-DoF trajectory.  Adaptations in this module
must therefore operate on a copy returned to the encoder; modifying the
planner-owned trajectory would feed altered state back into later replans.
"""

from __future__ import annotations

import numpy as np

# Physical 23-DoF sagittal stride joints: hip pitch, knee, ankle pitch.
SAGITTAL_PHYSICAL_IDX = (0, 3, 4, 6, 9, 10)
# IsaacLab indices whose positive deviation from stand is flexion / clearance.
FLEXION_POSITIVE_IL = (0, 1, 9, 10, 13, 14)
# Same sagittal joints the stride scale uses, in IsaacLab order.
LEFT_SAGITTAL_IL = (0, 1, 7)
RIGHT_SAGITTAL_IL = (2, 3, 9)
LEFT_KNEE_IL = 1
RIGHT_KNEE_IL = 3

# Encoder-facing healthy-plant recipe. Official Slow Walk stays 0.4.
# C++ constants in sonic_cpp/.../reference_adaptation.hpp must match.
SLOW_GAIT_SAGITTAL_SCALE = 0.28
SLOW_GAIT_KNEE_OFFSET_DEG = 8.0
SLOW_GAIT_YAW_COMMAND_SCALE = 0.15
SLOW_GAIT_YAW_INPLACE_SCALE = 0.35


def _spec_group(spec: str) -> str:
    return str(spec).split("=", 1)[0].strip().lower()


def expand_slow_gait_recipe(
    *,
    joint_scale_specs=(),
    joint_offset_specs=(),
    yaw_command_scale=None,
    yaw_inplace_scale=None,
):
    """Expand ``--slow-gait`` the same way the MuJoCo and C++ launchers do.

    Sagittal 0.28 + gait-only knee +8° + walk yaw 0.15 / in-place 0.35.
    An explicit knee offset or yaw scale is kept. Sagittal or hip-pitch
    scale specs are rejected so the recipe cannot be mixed with those flags.
    """
    scale_groups = {_spec_group(spec) for spec in joint_scale_specs}
    if scale_groups & {"sagittal", "hip_pitch"}:
        raise ValueError("--slow-gait is mutually exclusive with sagittal/hip-pitch scale")
    scales = [str(spec) for spec in joint_scale_specs]
    scales.append(f"sagittal={SLOW_GAIT_SAGITTAL_SCALE:g}")
    offsets = [str(spec) for spec in joint_offset_specs]
    if "knee" not in {_spec_group(spec) for spec in offsets}:
        offsets.append(f"knee={SLOW_GAIT_KNEE_OFFSET_DEG:g}")
    walk_yaw = (
        SLOW_GAIT_YAW_COMMAND_SCALE if yaw_command_scale is None else float(yaw_command_scale)
    )
    inplace_yaw = (
        SLOW_GAIT_YAW_INPLACE_SCALE if yaw_inplace_scale is None else float(yaw_inplace_scale)
    )
    return scales, offsets, walk_yaw, inplace_yaw


def gait_yaw_command_scale(reference_joint_scale) -> float:
    """Commanded heading rate that stays consistent with a smaller stride.

    Hip roll/yaw stay unscaled in the tracker. Walk-plus-turn jitter comes
    from feeding the original stick yaw rate into a shortened gait, so the
    teleop angular velocity uses the smallest sagittal reference scale.
    """
    scale = np.asarray(reference_joint_scale, dtype=np.float64).reshape(-1)
    if scale.size < 12:
        return 1.0
    values = scale[list(SAGITTAL_PHYSICAL_IDX)]
    if not np.isfinite(values).all() or np.any(values <= 0.0) or np.any(values > 2.0):
        raise ValueError("sagittal reference scales must be finite and in (0, 2]")
    return float(np.min(values))


class GaitReferenceAdapter:
    """Reduce locomotion amplitude around a captured standing reference.

    This is deliberately different from output action scaling.  The tracker
    sees a smaller, internally consistent desired gait and retains authority to
    generate unscaled recovery actions when the measured robot deviates from
    that gait.
    """

    def __init__(
        self,
        *,
        leg_idx,
        leg_scale: float = 1.0,
        root_height_scale: float = 1.0,
        default_stand_q_il,
        joint_scale_il=None,
        joint_dvel_scale_il=None,
        joint_offset_il=None,
        joint_flex_scale_il=None,
    ):
        self.leg_idx = np.asarray(leg_idx, dtype=np.int64).reshape(-1)
        if len(np.unique(self.leg_idx)) != len(self.leg_idx):
            raise ValueError("leg_idx contains duplicates")
        if np.any((self.leg_idx < 0) | (self.leg_idx >= 29)):
            raise ValueError("leg_idx must contain SONIC IsaacLab indices in [0, 29)")
        self.leg_scale = self._scale(leg_scale, "leg_scale")
        self.root_height_scale = self._scale(root_height_scale, "root_height_scale")
        self.default_stand_q_il = np.asarray(default_stand_q_il, dtype=np.float64).reshape(29)
        if not np.isfinite(self.default_stand_q_il).all():
            raise ValueError("default_stand_q_il contains NaN or infinity")
        self.joint_scale_il = np.asarray(
            np.ones(29) if joint_scale_il is None else joint_scale_il,
            dtype=np.float64,
        ).reshape(29)
        if (
            not np.isfinite(self.joint_scale_il).all()
            or np.any(self.joint_scale_il <= 0.0)
            or np.any(self.joint_scale_il > 2.0)
        ):
            raise ValueError("joint_scale_il values must be finite and in (0, 2]")
        self._effective_joint_scale = self.joint_scale_il.copy()
        self._effective_joint_scale[self.leg_idx] *= self.leg_scale
        if np.any(self._effective_joint_scale > 2.0):
            raise ValueError("combined leg and joint reference scale exceeds 2")
        self._independent_dvel = joint_dvel_scale_il is not None
        if joint_dvel_scale_il is None:
            self._effective_dvel_scale = self._effective_joint_scale.copy()
        else:
            self.joint_dvel_scale_il = np.asarray(joint_dvel_scale_il, dtype=np.float64).reshape(29)
            if (
                not np.isfinite(self.joint_dvel_scale_il).all()
                or np.any(self.joint_dvel_scale_il <= 0.0)
                or np.any(self.joint_dvel_scale_il > 2.0)
            ):
                raise ValueError("joint_dvel_scale_il values must be finite and in (0, 2]")
            self._effective_dvel_scale = self.joint_dvel_scale_il.copy()
            self._effective_dvel_scale[self.leg_idx] *= self.leg_scale
            if np.any(self._effective_dvel_scale > 2.0):
                raise ValueError("combined leg and joint dvel scale exceeds 2")
        self.joint_offset_il = np.asarray(
            np.zeros(29) if joint_offset_il is None else joint_offset_il,
            dtype=np.float64,
        ).reshape(29)
        if not np.isfinite(self.joint_offset_il).all():
            raise ValueError("joint_offset_il contains NaN or infinity")
        if np.any(np.abs(self.joint_offset_il) > np.pi / 2):
            raise ValueError("joint_offset_il values must be within ±π/2")
        if joint_flex_scale_il is None:
            self._effective_flex_scale = self._effective_joint_scale.copy()
        else:
            self.joint_flex_scale_il = np.asarray(joint_flex_scale_il, dtype=np.float64).reshape(29)
            if (
                not np.isfinite(self.joint_flex_scale_il).all()
                or np.any(self.joint_flex_scale_il <= 0.0)
                or np.any(self.joint_flex_scale_il > 2.0)
            ):
                raise ValueError("joint_flex_scale_il values must be finite and in (0, 2]")
            self._effective_flex_scale = self.joint_flex_scale_il.copy()
            self._effective_flex_scale[self.leg_idx] *= self.leg_scale
            if np.any(self._effective_flex_scale > 2.0):
                raise ValueError("combined leg and joint flex scale exceeds 2")
        self._flex_sign_il = np.zeros(29, dtype=np.float64)
        self._flex_sign_il[list(FLEXION_POSITIVE_IL)] = 1.0
        self.last_active = False

    @staticmethod
    def _scale(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or not 0.0 < value <= 2.0:
            raise ValueError(f"{name} must be finite and in (0, 2]")
        return value

    @property
    def enabled(self) -> bool:
        return (
            np.any(self._effective_joint_scale != 1.0)
            or np.any(self._effective_dvel_scale != 1.0)
            or np.any(self._effective_flex_scale != self._effective_joint_scale)
            or np.any(self.joint_offset_il != 0.0)
            or self.root_height_scale != 1.0
        )

    def apply(self, reference: dict) -> dict:
        """Return an encoder-facing copy, leaving provider data untouched."""
        active = bool(reference.get("reference_gait_active", False))
        self.last_active = self.enabled and active
        if not self.last_active:
            return reference

        out = dict(reference)
        q = np.asarray(reference["dof_fut"], dtype=np.float64).copy()
        dq = np.asarray(reference["dvel_fut"], dtype=np.float64).copy()
        if q.ndim != 3 or q.shape[0] != 1 or q.shape[-1] != 29:
            raise ValueError(f"dof_fut must have shape (1,T,29), got {q.shape}")
        if dq.shape != q.shape:
            raise ValueError(f"dvel_fut must match dof_fut, got {dq.shape}")
        stand = np.asarray(
            reference.get("stable_stand_q_il", self.default_stand_q_il),
            dtype=np.float64,
        ).reshape(29)
        if not np.isfinite(stand).all():
            raise ValueError("stable_stand_q_il contains NaN or infinity")
        idx = np.flatnonzero(
            (self._effective_joint_scale != 1.0)
            | (self._effective_dvel_scale != 1.0)
            | (self._effective_flex_scale != self._effective_joint_scale)
            | (self.joint_offset_il != 0.0)
        )
        delta = q[..., idx] - stand[idx]
        ext_scale = self._effective_joint_scale[idx]
        flex_scale = self._effective_flex_scale[idx]
        flex_side = (delta * self._flex_sign_il[idx]) > 0.0
        q_scale = np.where(flex_side, flex_scale, ext_scale)
        q[..., idx] = stand[idx] + q_scale * delta + self.joint_offset_il[idx]
        if not self._independent_dvel and np.any(
            self._effective_flex_scale != self._effective_joint_scale
        ):
            dq[..., idx] *= q_scale
        else:
            dq[..., idx] *= self._effective_dvel_scale[idx]
        out["dof_fut"] = q.astype(np.asarray(reference["dof_fut"]).dtype)
        out["dvel_fut"] = dq.astype(np.asarray(reference["dvel_fut"]).dtype)

        if self.root_height_scale != 1.0:
            rootz_fut = np.asarray(reference["rootz_fut"], dtype=np.float64).copy()
            rootz_cur = np.asarray(reference["rootz_cur"], dtype=np.float64).copy()
            stand_z = float(reference.get("stable_stand_root_z", rootz_cur.reshape(-1)[0]))
            if not np.isfinite(stand_z):
                raise ValueError("stable_stand_root_z is not finite")
            out["rootz_fut"] = (stand_z + self.root_height_scale * (rootz_fut - stand_z)).astype(
                np.asarray(reference["rootz_fut"]).dtype
            )
            out["rootz_cur"] = (stand_z + self.root_height_scale * (rootz_cur - stand_z)).astype(
                np.asarray(reference["rootz_cur"]).dtype
            )
        return out
