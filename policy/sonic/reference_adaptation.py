"""Training-free adaptations applied at the SONIC tracker-reference boundary.

The planner owns a recurrent 29-DoF trajectory.  Adaptations in this module
must therefore operate on a copy returned to the encoder; modifying the
planner-owned trajectory would feed altered state back into later replans.
"""
from __future__ import annotations

import numpy as np


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
        idx = np.flatnonzero(self._effective_joint_scale != 1.0)
        scale = self._effective_joint_scale[idx]
        q[..., idx] = stand[idx] + scale * (q[..., idx] - stand[idx])
        dq[..., idx] *= scale
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
