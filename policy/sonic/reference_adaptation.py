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
        self.root_height_scale = self._scale(
            root_height_scale, "root_height_scale"
        )
        self.default_stand_q_il = np.asarray(
            default_stand_q_il, dtype=np.float64
        ).reshape(29)
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
            self.joint_dvel_scale_il = np.asarray(
                joint_dvel_scale_il, dtype=np.float64
            ).reshape(29)
            if (
                not np.isfinite(self.joint_dvel_scale_il).all()
                or np.any(self.joint_dvel_scale_il <= 0.0)
                or np.any(self.joint_dvel_scale_il > 2.0)
            ):
                raise ValueError(
                    "joint_dvel_scale_il values must be finite and in (0, 2]"
                )
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
            self.joint_flex_scale_il = np.asarray(
                joint_flex_scale_il, dtype=np.float64
            ).reshape(29)
            if (
                not np.isfinite(self.joint_flex_scale_il).all()
                or np.any(self.joint_flex_scale_il <= 0.0)
                or np.any(self.joint_flex_scale_il > 2.0)
            ):
                raise ValueError(
                    "joint_flex_scale_il values must be finite and in (0, 2]"
                )
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
        if (
            not self._independent_dvel
            and np.any(self._effective_flex_scale != self._effective_joint_scale)
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
            out["rootz_fut"] = (
                stand_z
                + self.root_height_scale * (rootz_fut - stand_z)
            ).astype(np.asarray(reference["rootz_fut"]).dtype)
            out["rootz_cur"] = (
                stand_z
                + self.root_height_scale * (rootz_cur - stand_z)
            ).astype(np.asarray(reference["rootz_cur"]).dtype)
        return out


class LandingHoldAdapter:
    """Legacy landing-hold attempt: suppress sagittal swing after touchdown.

    Withdrawn. Do not use this as a slow-walk method. Pose freeze destroys
    swing. A 160 ms sagittal ``dvel_fut`` zero is gait-neutral and does not
    beat ``--slow-gait``. Official Slow Walk stays 0.4; the current recipe
    is encoder-facing sagittal scale only.

    Same joint set as ``GaitReferenceAdapter`` sagittal scale (hip-pitch /
    knee / ankle-pitch, encoder copy only). Kept so ``--landing-hold`` and
    ``--landing-hold-dvel`` can reproduce the 2026-08-19 study.

    Plant contact is optional. When it is absent the adapter uses measured
    foot height and a knee-flexion proxy so the same code can run from
    proprioception alone.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = "pose",
        confirm_down_steps: int = 3,
        min_swing_steps: int = 4,
        confirm_release_steps: int = 3,
        max_hold_steps: int = 8,
        land_z_m: float = 0.020,
        swing_z_m: float = 0.035,
        clip_knee_margin_rad: float = 0.08,
    ):
        if confirm_down_steps < 1 or min_swing_steps < 1 or confirm_release_steps < 1:
            raise ValueError("landing-hold debounce counts must be >= 1")
        if max_hold_steps < 1:
            raise ValueError("max_hold_steps must be >= 1")
        if not 0.0 < land_z_m < swing_z_m < 0.20:
            raise ValueError("landing-hold height gates must satisfy 0 < land < swing < 0.2")
        if not 0.0 < clip_knee_margin_rad < 0.5:
            raise ValueError("clip_knee_margin_rad must be in (0, 0.5)")
        mode = str(mode).strip().lower()
        if mode not in ("pose", "dvel"):
            raise ValueError("landing-hold mode must be pose or dvel")
        self.enabled = bool(enabled)
        self.mode = mode
        self.confirm_down_steps = int(confirm_down_steps)
        self.min_swing_steps = int(min_swing_steps)
        self.confirm_release_steps = int(confirm_release_steps)
        self.max_hold_steps = int(max_hold_steps)
        self.land_z_m = float(land_z_m)
        self.swing_z_m = float(swing_z_m)
        self.clip_knee_margin_rad = float(clip_knee_margin_rad)
        self._down_count = np.zeros(2, dtype=np.int32)
        self._swing_count = np.zeros(2, dtype=np.int32)
        self._release_count = np.zeros(2, dtype=np.int32)
        self._holding = np.zeros(2, dtype=bool)
        self._hold_q = np.zeros((2, 3), dtype=np.float64)
        self._hold_age = np.zeros(2, dtype=np.int32)
        self.last_holding = np.zeros(2, dtype=bool)

    def reset(self) -> None:
        self._down_count[:] = 0
        self._swing_count[:] = 0
        self._release_count[:] = 0
        self._holding[:] = False
        self._hold_q[:] = 0.0
        self._hold_age[:] = 0
        self.last_holding[:] = False

    def apply(
        self,
        reference: dict,
        *,
        measured_q_il,
        raw_dof_fut=None,
        foot_z=None,
        foot_contact=None,
    ) -> dict:
        """Return an encoder-facing copy. Provider data is not modified."""
        if not self.enabled or not bool(reference.get("reference_gait_active", False)):
            self.reset()
            return reference

        q = np.asarray(reference["dof_fut"], dtype=np.float64)
        dq = np.asarray(reference["dvel_fut"], dtype=np.float64)
        if q.ndim != 3 or q.shape[0] != 1 or q.shape[-1] != 29:
            raise ValueError(f"dof_fut must have shape (1,T,29), got {q.shape}")
        if dq.shape != q.shape:
            raise ValueError(f"dvel_fut must match dof_fut, got {dq.shape}")
        measured = np.asarray(measured_q_il, dtype=np.float64).reshape(29)
        if not np.isfinite(measured).all():
            raise ValueError("measured_q_il contains NaN or infinity")
        phase = np.asarray(
            reference["dof_fut"] if raw_dof_fut is None else raw_dof_fut,
            dtype=np.float64,
        )
        if phase.shape[-1] != 29:
            raise ValueError("raw_dof_fut must end in 29")
        phase_now = phase.reshape(-1, 29)[0]
        down = self._plant_down(measured, foot_z, foot_contact)
        clip_swing = self._clip_swing(phase_now)

        out = None
        for side, idx in enumerate((LEFT_SAGITTAL_IL, RIGHT_SAGITTAL_IL)):
            if down[side]:
                self._down_count[side] += 1
            else:
                self._down_count[side] = 0
                self._swing_count[side] += 1

            if self._holding[side]:
                self._hold_age[side] += 1
                timed_out = self._hold_age[side] >= self.max_hold_steps
                if timed_out:
                    self._holding[side] = False
                    self._release_count[side] = 0
                    self._hold_age[side] = 0
                elif not clip_swing[side] or not down[side]:
                    self._release_count[side] += 1
                    if self._release_count[side] >= self.confirm_release_steps:
                        self._holding[side] = False
                        self._release_count[side] = 0
                        self._hold_age[side] = 0
                else:
                    self._release_count[side] = 0
            elif (
                down[side]
                and clip_swing[side]
                and self._down_count[side] >= self.confirm_down_steps
                and self._swing_count[side] >= self.min_swing_steps
            ):
                self._holding[side] = True
                self._hold_q[side] = measured[list(idx)]
                self._swing_count[side] = 0
                self._release_count[side] = 0
                self._hold_age[side] = 0

            if self._holding[side]:
                if out is None:
                    out = dict(reference)
                    out["dof_fut"] = q.copy()
                    out["dvel_fut"] = dq.copy()
                    q = out["dof_fut"]
                    dq = out["dvel_fut"]
                dq[..., idx] = 0.0
                if self.mode == "pose":
                    q[..., idx] = self._hold_q[side]

        self.last_holding = self._holding.copy()
        return reference if out is None else out

    def _plant_down(self, measured, foot_z, foot_contact) -> np.ndarray:
        down = np.zeros(2, dtype=bool)
        if foot_contact is not None:
            contact = np.asarray(foot_contact, dtype=bool).reshape(2)
            down[:] = contact
        if foot_z is not None:
            height = np.asarray(foot_z, dtype=np.float64).reshape(2)
            height_down = np.isfinite(height) & (height <= self.land_z_m)
            if foot_contact is None:
                down[:] = height_down
            else:
                down &= height_down | ~np.isfinite(height)
        if foot_contact is None and foot_z is None:
            left = float(measured[LEFT_KNEE_IL])
            right = float(measured[RIGHT_KNEE_IL])
            down[0] = left <= right + self.clip_knee_margin_rad
            down[1] = right <= left + self.clip_knee_margin_rad
        return down

    def _clip_swing(self, phase_now) -> np.ndarray:
        left = float(phase_now[LEFT_KNEE_IL])
        right = float(phase_now[RIGHT_KNEE_IL])
        return np.array(
            [
                left > right + self.clip_knee_margin_rad,
                right > left + self.clip_knee_margin_rad,
            ],
            dtype=bool,
        )
