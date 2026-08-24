"""Causal, opt-in motor-command shaping for reduced SONIC embodiments.

SONIC emits normalized action deltas around the configured default pose.  The
official deployment sends those deltas directly.  This adapter keeps that
path bit-for-bit unchanged when all parameters are at their neutral values,
while providing a single testable boundary for training-free 23-DoF studies.

All vectors use the selected robot's *physical* joint order.  Rate limits are
specified in target-joint radians/second and are converted through SONIC's
per-joint action scale, so a limit has the same meaning for every motor.
"""
from __future__ import annotations

import numpy as np


class MotorActionAdapter:
    """Apply scale, absolute clipping, EMA, and target-angle slew limits."""

    def __init__(
        self,
        *,
        action_scale,
        control_dt: float,
        scale=None,
        abs_limit=None,
        ema=None,
        target_rate_limit=None,
    ) -> None:
        self.action_scale = self._vector(action_scale, "action_scale")
        self.n = len(self.action_scale)
        if not np.all(self.action_scale > 0):
            raise ValueError("action_scale must be strictly positive")
        self.control_dt = float(control_dt)
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        self.scale = self._parameter(scale, 1.0, "scale")
        self.abs_limit = self._parameter(abs_limit, np.inf, "abs_limit")
        self.ema = self._parameter(ema, 1.0, "ema")
        self.target_rate_limit = self._parameter(
            target_rate_limit, np.inf, "target_rate_limit"
        )
        if np.any(self.scale <= 0):
            raise ValueError("motor action scale factors must be positive")
        if np.any(self.abs_limit <= 0):
            raise ValueError("motor action absolute limits must be positive")
        if np.any((self.ema <= 0) | (self.ema > 1)):
            raise ValueError("motor action EMA coefficients must be in (0, 1]")
        if np.any(self.target_rate_limit <= 0):
            raise ValueError("motor target-rate limits must be positive")
        self.previous = np.zeros(self.n, np.float64)

    def _vector(self, value, name: str) -> np.ndarray:
        out = np.asarray(value, np.float64)
        if out.ndim != 1 or not np.isfinite(out).all():
            raise ValueError(f"{name} must be a finite 1-D vector")
        return out.copy()

    def _parameter(self, value, neutral: float, name: str) -> np.ndarray:
        if value is None:
            return np.full(self.n, neutral, np.float64)
        out = np.asarray(value, np.float64)
        if out.ndim == 0:
            out = np.full(self.n, float(out), np.float64)
        if out.shape != (self.n,):
            raise ValueError(f"{name} must have shape ({self.n},), got {out.shape}")
        if np.isnan(out).any():
            raise ValueError(f"{name} contains NaN")
        return out.copy()

    @property
    def enabled(self) -> bool:
        return bool(
            np.any(self.scale != 1.0)
            or np.any(np.isfinite(self.abs_limit))
            or np.any(self.ema != 1.0)
            or np.any(np.isfinite(self.target_rate_limit))
        )

    def reset(self) -> None:
        self.previous.fill(0.0)

    def apply(self, raw_action) -> np.ndarray:
        raw = np.asarray(raw_action, np.float64)
        if raw.shape != (self.n,) or not np.isfinite(raw).all():
            raise ValueError(
                f"raw motor action must be finite with shape ({self.n},), "
                f"got {raw.shape}"
            )
        shaped = np.clip(raw * self.scale, -self.abs_limit, self.abs_limit)
        shaped = self.ema * shaped + (1.0 - self.ema) * self.previous
        max_delta = (
            self.target_rate_limit * self.control_dt / self.action_scale
        )
        shaped = self.previous + np.clip(
            shaped - self.previous, -max_delta, max_delta
        )
        self.previous = shaped.copy()
        return shaped.astype(np.float32)

