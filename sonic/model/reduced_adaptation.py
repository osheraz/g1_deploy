"""Training-free observation adapters for a reduced SONIC embodiment.

The released decoder always observes 29 joint positions, velocities, and prior
actions.  A 23-DoF G1 has no measurements for six of those channels.  This
module keeps that choice explicit and causal so it can be ablated without
changing planner data, encoder dimensions, decoder dimensions, or motor output
projection.
"""
from __future__ import annotations

import numpy as np


STRATEGIES = ("zero", "reference", "shadow")


class ReducedProprioceptionAdapter:
    """Fill only unavailable 29-DoF proprioception channels.

    ``zero`` is the current physical-parity behavior: an unavailable joint did
    not move. ``reference`` is a diagnostic perfect-reference observer.
    ``shadow`` is a causal, critically damped virtual plant driven by the prior
    raw decoder target. The shadow affects decoder observations only; no
    virtual command is sent to hardware and no physical action is clipped.
    """

    def __init__(
        self,
        *,
        strategy: str,
        missing_idx,
        default_il,
        action_scale_il,
        control_dt: float,
        settling_time_s: float,
    ) -> None:
        strategy = str(strategy).strip().lower()
        if strategy not in STRATEGIES:
            raise ValueError(f"reduced proprioception strategy must be one of {STRATEGIES}")
        self.strategy = strategy
        self.missing_idx = np.asarray(missing_idx, np.int64)
        self.default_il = np.asarray(default_il, np.float64).reshape(29)
        self.action_scale_il = np.asarray(action_scale_il, np.float64).reshape(29)
        self.dt = float(control_dt)
        self.settling_time_s = float(settling_time_s)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if not np.isfinite(self.settling_time_s) or self.settling_time_s <= 0:
            raise ValueError("shadow settling time must be finite and positive")
        self.q_il = self.default_il.copy()
        self.dq_il = np.zeros(29, np.float64)

    def reset(self) -> None:
        self.q_il[:] = self.default_il
        self.dq_il.fill(0.0)

    def fill(
        self,
        q_il: np.ndarray,
        dq_il: np.ndarray,
        *,
        reference_q_il: np.ndarray,
        reference_dq_il: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q_il, np.float64).copy()
        dq = np.asarray(dq_il, np.float64).copy()
        if not len(self.missing_idx):
            return q, dq
        if self.strategy == "reference":
            reference_q = np.asarray(reference_q_il, np.float64).reshape(29)
            reference_dq = np.asarray(reference_dq_il, np.float64).reshape(29)
            q[..., self.missing_idx] = reference_q[self.missing_idx]
            dq[..., self.missing_idx] = reference_dq[self.missing_idx]
        elif self.strategy == "shadow":
            q[..., self.missing_idx] = self.q_il[self.missing_idx]
            dq[..., self.missing_idx] = self.dq_il[self.missing_idx]
        # zero leaves the profile's zero-filled missing channels unchanged.
        return q, dq

    def advance(self, raw_action_il: np.ndarray) -> None:
        """Advance the virtual plant for the next observation (shadow only)."""
        if self.strategy != "shadow" or not len(self.missing_idx):
            return
        action = np.asarray(raw_action_il, np.float64).reshape(29)
        target = self.default_il + self.action_scale_il * action
        idx = self.missing_idx
        error = self.q_il[idx] - target[idx]
        velocity = self.dq_il[idx]
        # Exact constant-target solution for a critically damped second-order
        # system. omega=4/Ts makes settling_time_s approximately the 2% time.
        omega = 4.0 / self.settling_time_s
        c2 = velocity + omega * error
        decay = np.exp(-omega * self.dt)
        next_error = (error + c2 * self.dt) * decay
        next_velocity = (velocity - omega * c2 * self.dt) * decay
        self.q_il[idx] = target[idx] + next_error
        self.dq_il[idx] = next_velocity

    def diagnostics(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.q_il[self.missing_idx].astype(np.float32).copy(),
            self.dq_il[self.missing_idx].astype(np.float32).copy(),
        )
