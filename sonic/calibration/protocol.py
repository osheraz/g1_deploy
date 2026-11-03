"""Deterministic, one-joint-at-a-time suspended calibration protocol.

The plan contains *offsets* from a pose captured at run time.  This prevents a
calibration launch from first moving a real robot to a hard-coded posture.  A
runner must validate the captured pose, add these offsets, and apply the normal
joint PD gains.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA = "e_skin.suspended_calibration_plan.v1"
DT = 0.02

MOTOR_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

PRESENT_23 = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26],
    dtype=np.int64,
)
LEG_MOTORS = np.arange(12, dtype=np.int64)

# Absolute PR-mode limits from the 23-DoF MJCF.  Missing motors remain zero.
LOWER_23 = np.array(
    [
        -2.5307,
        -0.5236,
        -2.7576,
        -0.0873,
        -0.8727,
        -0.2618,
        -2.5307,
        -2.9671,
        -2.7576,
        -0.0873,
        -0.8727,
        -0.2618,
        -2.6180,
        -3.0892,
        -1.5882,
        -2.6180,
        -1.0472,
        -1.9722,
        -3.0892,
        -2.2515,
        -2.6180,
        -1.0472,
        -1.9722,
    ],
    dtype=np.float64,
)
UPPER_23 = np.array(
    [
        2.8798,
        2.9671,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.8798,
        0.5236,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.6180,
        2.6704,
        2.2515,
        2.6180,
        2.0944,
        1.9722,
        2.6704,
        1.5882,
        2.6180,
        2.0944,
        1.9722,
    ],
    dtype=np.float64,
)

# Screening is the only physical default.  The full range is a second-stage
# protocol that must be selected explicitly after screening data is reviewed.
SCREENING_AMPLITUDE = {
    0: 0.08,
    1: 0.06,
    2: 0.06,
    3: 0.10,
    4: 0.06,
    5: 0.035,
    6: 0.08,
    7: 0.06,
    8: 0.06,
    9: 0.10,
    10: 0.06,
    11: 0.035,
    12: 0.06,
    15: 0.10,
    16: 0.08,
    17: 0.08,
    18: 0.10,
    19: 0.08,
    22: 0.10,
    23: 0.08,
    24: 0.08,
    25: 0.10,
    26: 0.08,
}
FULL_AMPLITUDE = {
    0: 0.25,
    1: 0.18,
    2: 0.20,
    3: 0.35,
    4: 0.20,
    5: 0.10,
    6: 0.25,
    7: 0.18,
    8: 0.20,
    9: 0.35,
    10: 0.20,
    11: 0.10,
    12: 0.18,
    15: 0.30,
    16: 0.25,
    17: 0.25,
    18: 0.35,
    19: 0.25,
    22: 0.30,
    23: 0.25,
    24: 0.25,
    25: 0.35,
    26: 0.25,
}


@dataclass(frozen=True)
class CalibrationPlan:
    time_s: np.ndarray
    phase: np.ndarray
    active_motor: np.ndarray
    direction: np.ndarray
    offsets: np.ndarray
    protocol: str
    scope: str

    def validate(self) -> None:
        n = len(self.time_s)
        if n < 2 or self.offsets.shape != (n, 29):
            raise ValueError("calibration plan must be N x 29 with N >= 2")
        if any(len(x) != n for x in (self.phase, self.active_motor, self.direction)):
            raise ValueError("calibration plan columns have unequal lengths")
        if not np.isfinite(self.time_s).all() or not np.isfinite(self.offsets).all():
            raise ValueError("calibration plan contains NaN or infinity")
        if np.max(np.abs(np.diff(self.time_s) - DT)) > 1e-8:
            raise ValueError(f"calibration plan must use a fixed {DT:.3f}s step")
        if np.max(np.abs(self.offsets[[0, -1]])) > 1e-10:
            raise ValueError("calibration plan must begin and end at zero offset")
        missing = np.setdiff1d(np.arange(29), PRESENT_23)
        if np.max(np.abs(self.offsets[:, missing])) > 1e-10:
            raise ValueError("23-DoF plan commands an absent motor")
        nonzero = np.sum(np.abs(self.offsets) > 1e-9, axis=1)
        if np.any(nonzero > 1):
            raise ValueError("suspended plan must excite at most one joint per sample")
        if np.any(~np.isin(self.active_motor, np.r_[-1, PRESENT_23])):
            raise ValueError("plan contains an invalid active motor")
        if np.any(~np.isin(self.direction, (-1, 0, 1))):
            raise ValueError("plan contains an invalid direction")
        for row, motor in zip(self.offsets, self.active_motor):
            nonzero_motor = np.flatnonzero(np.abs(row) > 1e-9)
            if len(nonzero_motor) == 1 and motor != nonzero_motor[0]:
                raise ValueError("active motor does not match the nonzero offset")
        rate = np.max(np.abs(np.diff(self.offsets, axis=0)), axis=0) / DT
        if np.max(rate) > 0.55:
            raise ValueError(f"plan target rate {np.max(rate):.3f}rad/s exceeds 0.55rad/s")
        amp = SCREENING_AMPLITUDE if self.protocol == "screening" else FULL_AMPLITUDE
        for motor, bound in amp.items():
            if np.max(np.abs(self.offsets[:, motor])) > bound + 1e-9:
                raise ValueError(f"motor {motor} exceeds {self.protocol} amplitude")

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1])


def _minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    return alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def build_plan(
    *,
    protocol: str = "screening",
    scope: str = "all",
    dt: float = DT,
) -> CalibrationPlan:
    """Build the safe staged plan shared by MuJoCo and the C++ robot runner."""
    if abs(dt - DT) > 1e-12:
        raise ValueError(f"the physical contract is fixed at {DT:.3f}s")
    if protocol not in {"screening", "full"}:
        raise ValueError("protocol must be screening or full")
    if scope not in {"legs", "all"}:
        raise ValueError("scope must be legs or all")
    motors = LEG_MOTORS if scope == "legs" else PRESENT_23
    amplitude = SCREENING_AMPLITUDE if protocol == "screening" else FULL_AMPLITUDE
    transition_s = 1.0 if protocol == "screening" else 1.5
    dwell_s = 0.6 if protocol == "screening" else 0.8
    fractions = (
        [0.0, 1.0, 0.0, -1.0, 0.0]
        if protocol == "screening"
        else [0.0, 0.25, 0.5, 1.0, 0.5, 0.0, -0.25, -0.5, -1.0, -0.5, 0.0, 0.25, 0.0]
    )

    phases: list[str] = []
    active: list[int] = []
    directions: list[int] = []
    offsets: list[np.ndarray] = []
    current = np.zeros(29, dtype=np.float64)

    def append_constant(seconds: float, phase: str, motor: int = -1, direction: int = 0):
        count = max(1, int(round(seconds / dt)))
        for _ in range(count):
            phases.append(phase)
            active.append(motor)
            directions.append(direction)
            offsets.append(current.copy())

    def append_transition(motor: int, value: float):
        nonlocal current
        start = float(current[motor])
        count = max(2, int(round(transition_s / dt)))
        alpha = np.arange(1, count + 1, dtype=np.float64) / count
        values = start + (value - start) * _minimum_jerk(alpha)
        direction = int(np.sign(value - start))
        for value_i in values:
            row = current.copy()
            row[motor] = value_i
            phases.append("transition")
            active.append(int(motor))
            directions.append(direction)
            offsets.append(row)
        current = offsets[-1].copy()

    append_constant(2.0, "settle")
    for motor in motors:
        a = float(amplitude[int(motor)])
        for fraction in fractions[1:]:
            append_transition(int(motor), a * fraction)
            append_constant(dwell_s, "dwell", int(motor), directions[-1])
        # Small dynamic probes estimate lag without the high accelerations of a
        # chirp.  Static inverse fitting deliberately ignores these samples.
        for frequency, scale, seconds in (
            ((0.20, 0.50, 5.0), (0.55, 0.25, 4.0)) if protocol == "full" else ((0.20, 0.40, 4.0),)
        ):
            count = int(round(seconds / dt))
            prior = float(current[motor])
            for k in range(count):
                value = scale * a * np.sin(2.0 * np.pi * frequency * (k + 1) * dt)
                row = current.copy()
                row[motor] = value
                phases.append(f"sine_{frequency:g}hz")
                active.append(int(motor))
                directions.append(int(np.sign(value - prior)))
                offsets.append(row)
                prior = value
            current = offsets[-1].copy()
        append_transition(int(motor), 0.0)
        append_constant(dwell_s, "neutral_dwell", int(motor), directions[-1])
    append_constant(2.0, "settle")

    # Samples are timestamped at the end of each 20-ms control interval.
    time_s = np.arange(1, len(offsets) + 1, dtype=np.float64) * dt
    plan = CalibrationPlan(
        time_s=time_s,
        phase=np.asarray(phases, dtype="U24"),
        active_motor=np.asarray(active, dtype=np.int16),
        direction=np.asarray(directions, dtype=np.int8),
        offsets=np.asarray(offsets, dtype=np.float64),
        protocol=protocol,
        scope=scope,
    )
    plan.validate()
    return plan


def save_plan(plan: CalibrationPlan, path: Path | str) -> Path:
    plan.validate()
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "schema",
        "protocol",
        "scope",
        "sample",
        "time_s",
        "phase",
        "active_motor",
        "direction",
    ] + [f"offset_{i}" for i in range(29)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for k in range(len(plan.time_s)):
            writer.writerow(
                [
                    SCHEMA,
                    plan.protocol,
                    plan.scope,
                    k,
                    f"{plan.time_s[k]:.9f}",
                    plan.phase[k],
                    int(plan.active_motor[k]),
                    int(plan.direction[k]),
                    *[f"{value:.9f}" for value in plan.offsets[k]],
                ]
            )
    return path


def load_plan(path: Path | str) -> CalibrationPlan:
    path = Path(path).expanduser().absolute()
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["schema"] != SCHEMA:
                raise ValueError(f"unsupported calibration plan schema {row['schema']!r}")
            rows.append(row)
    if not rows:
        raise ValueError("empty calibration plan")
    protocol = rows[0]["protocol"]
    scope = rows[0]["scope"]
    if any(row["protocol"] != protocol or row["scope"] != scope for row in rows):
        raise ValueError("plan metadata changes within the CSV")
    plan = CalibrationPlan(
        time_s=np.asarray([float(row["time_s"]) for row in rows]),
        phase=np.asarray([row["phase"] for row in rows]),
        active_motor=np.asarray([int(row["active_motor"]) for row in rows]),
        direction=np.asarray([int(row["direction"]) for row in rows]),
        offsets=np.asarray([[float(row[f"offset_{i}"]) for i in range(29)] for row in rows]),
        protocol=protocol,
        scope=scope,
    )
    plan.validate()
    return plan


def selected_motors(scope: str) -> Iterable[int]:
    return LEG_MOTORS.copy() if scope == "legs" else PRESENT_23.copy()
