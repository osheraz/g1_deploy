"""Piecewise, direction-aware inverse maps at the joint-PD boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from sonic.calibration.protocol import MOTOR_NAMES, PRESENT_23

SCHEMA = "e_skin.joint_calibration.v1"


def _strictly_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median duplicate x values and make the interpolation domain strict."""
    order = np.argsort(x)
    x = np.asarray(x, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order]
    unique, inverse = np.unique(np.round(x, 9), return_inverse=True)
    out = np.empty(len(unique), dtype=np.float64)
    for k in range(len(unique)):
        out[k] = float(np.median(y[inverse == k]))
    return unique, out


@dataclass(frozen=True)
class JointCurve:
    model_knots_rad: np.ndarray
    hardware_command_increasing_rad: np.ndarray
    hardware_command_decreasing_rad: np.ndarray
    max_abs_correction_rad: float

    def __post_init__(self) -> None:
        x = np.asarray(self.model_knots_rad, dtype=np.float64)
        inc = np.asarray(self.hardware_command_increasing_rad, dtype=np.float64)
        dec = np.asarray(self.hardware_command_decreasing_rad, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2 or inc.shape != x.shape or dec.shape != x.shape:
            raise ValueError("joint curve needs equal one-dimensional arrays with >=2 knots")
        if not (np.isfinite(x).all() and np.isfinite(inc).all() and np.isfinite(dec).all()):
            raise ValueError("joint curve contains NaN or infinity")
        if np.any(np.diff(x) <= 0):
            raise ValueError("model knots must be strictly increasing")
        if np.any(np.diff(inc) < -1e-8) or np.any(np.diff(dec) < -1e-8):
            raise ValueError("hardware command maps must be monotone")
        if not np.isfinite(self.max_abs_correction_rad) or self.max_abs_correction_rad <= 0:
            raise ValueError("max_abs_correction_rad must be positive")
        object.__setattr__(self, "model_knots_rad", x)
        object.__setattr__(self, "hardware_command_increasing_rad", inc)
        object.__setattr__(self, "hardware_command_decreasing_rad", dec)

    def apply(self, model_target_rad: np.ndarray | float, direction: np.ndarray | float = 0):
        q = np.asarray(model_target_rad, dtype=np.float64)
        sign = np.asarray(direction, dtype=np.float64)
        # Interpolate the *correction*, not the absolute hardware command.
        # np.interp clamps outside its knot domain; clamping an absolute
        # command would incorrectly pull every out-of-range model target back
        # toward the screening endpoint. Constant endpoint residuals are the
        # conservative bounded extrapolation until a wider run is approved.
        inc_residual = self.hardware_command_increasing_rad - self.model_knots_rad
        dec_residual = self.hardware_command_decreasing_rad - self.model_knots_rad
        inc = q + np.interp(q, self.model_knots_rad, inc_residual)
        dec = q + np.interp(q, self.model_knots_rad, dec_residual)
        mapped = np.where(sign < 0, dec, inc)
        correction = np.clip(mapped - q, -self.max_abs_correction_rad, self.max_abs_correction_rad)
        out = q + correction
        return float(out) if out.ndim == 0 else out

    def to_json(self) -> dict[str, Any]:
        return {
            "model_knots_rad": self.model_knots_rad.tolist(),
            "hardware_command_increasing_rad": self.hardware_command_increasing_rad.tolist(),
            "hardware_command_decreasing_rad": self.hardware_command_decreasing_rad.tolist(),
            "max_abs_correction_rad": float(self.max_abs_correction_rad),
        }


@dataclass(frozen=True)
class JointCalibrationProfile:
    embodiment: str
    coordinate_basis: str
    approved_for_deployment: bool
    joints: dict[int, JointCurve]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.embodiment != "g1_23dof":
            raise ValueError("only g1_23dof calibration is currently defined")
        if self.coordinate_basis not in {"mujoco_truth", "external_truth", "encoder_only"}:
            raise ValueError("invalid coordinate_basis")
        if self.approved_for_deployment and self.coordinate_basis == "encoder_only":
            raise ValueError("encoder-only calibration cannot be approved for deployment")
        if any(int(index) not in PRESENT_23 for index in self.joints):
            raise ValueError("profile contains an absent 23-DoF motor")

    def apply_motor(self, motor: int, model_target_rad, direction=0):
        curve = self.joints.get(int(motor))
        return model_target_rad if curve is None else curve.apply(model_target_rad, direction)

    def apply_29(self, model_target: np.ndarray, direction: np.ndarray | None = None) -> np.ndarray:
        q = np.asarray(model_target, dtype=np.float64)
        if q.shape[-1:] != (29,):
            raise ValueError(f"expected a final dimension of 29, got {q.shape}")
        sign = np.zeros_like(q) if direction is None else np.asarray(direction, dtype=np.float64)
        if sign.shape != q.shape:
            sign = np.broadcast_to(sign, q.shape)
        out = q.copy()
        for motor, curve in self.joints.items():
            out[..., motor] = curve.apply(q[..., motor], sign[..., motor])
        return out

    def save(self, path: Path | str) -> Path:
        path = Path(path).expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "embodiment": self.embodiment,
            "coordinate_basis": self.coordinate_basis,
            "approved_for_deployment": self.approved_for_deployment,
            "metadata": self.metadata,
            "joints": {
                str(index): {"name": MOTOR_NAMES[index], **curve.to_json()}
                for index, curve in sorted(self.joints.items())
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "JointCalibrationProfile":
        path = Path(path).expanduser().absolute()
        payload = json.loads(path.read_text())
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"unsupported calibration profile schema {payload.get('schema')!r}")
        joints = {}
        for key, value in payload["joints"].items():
            joints[int(key)] = JointCurve(
                value["model_knots_rad"],
                value["hardware_command_increasing_rad"],
                value["hardware_command_decreasing_rad"],
                value["max_abs_correction_rad"],
            )
        return cls(
            embodiment=payload["embodiment"],
            coordinate_basis=payload["coordinate_basis"],
            approved_for_deployment=bool(payload["approved_for_deployment"]),
            joints=joints,
            metadata=dict(payload.get("metadata", {})),
        )


def monotone_inverse_curve(
    commanded: np.ndarray,
    measured: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return measured-position knots and commands using isotonic PAVA.

    PAVA is applied to measured response as a function of increasing command;
    the fitted forward map is then inverted.  This keeps the eventual command
    correction monotone even when individual dwell medians contain noise.
    """
    x, y = _strictly_increasing(commanded, measured)
    if len(x) < 2:
        raise ValueError("at least two distinct command levels are required")
    blocks = [[float(y[i]), 1.0, i, i] for i in range(len(y))]
    k = 0
    while k < len(blocks) - 1:
        if blocks[k][0] <= blocks[k + 1][0] + 1e-12:
            k += 1
            continue
        left, right = blocks[k], blocks[k + 1]
        weight = left[1] + right[1]
        mean = (left[0] * left[1] + right[0] * right[1]) / weight
        blocks[k : k + 2] = [[mean, weight, left[2], right[3]]]
        k = max(0, k - 1)
    fitted = np.empty_like(y)
    for mean, _, lo, hi in blocks:
        fitted[lo : hi + 1] = mean
    model_knots, hardware = _strictly_increasing(fitted, x)
    if len(model_knots) < 2 or model_knots[-1] - model_knots[0] < 1e-3:
        raise ValueError("measured response is not sufficiently monotone")
    # Non-strict command plateaus are valid for interpolation, but numerical
    # noise must never create a decreasing inverse command.
    hardware = np.maximum.accumulate(hardware)
    return model_knots, hardware
