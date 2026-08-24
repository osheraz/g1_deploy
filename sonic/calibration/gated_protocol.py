"""Review-gated suspended calibration protocol for the physical 23-DoF G1.

Every block starts and ends at zero offset from a captured hanging pose.  A
runner must hold that pose between blocks and obtain an explicit operator
approval before applying the next block.  The protocol contains no SONIC
inference: its motions are deterministic plant-identification probes.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from .protocol import MOTOR_NAMES, PRESENT_23, SCREENING_AMPLITUDE


SCHEMA = "e_skin.review_gated_calibration.v1"
DT = 0.02
MAX_TARGET_RATE_RAD_S = 0.35
DYNAMIC_MAX_TARGET_RATE_RAD_S = 2.0

# The fast pilot stays inside half of the original small screening envelope.
# The full screening uses the already-reviewed suspended full-range envelope
# from the legacy calibration design.  It is still a diagnostic excursion,
# not the mechanical joint limit and not a walking command.
FULL_SCREENING_AMPLITUDE: dict[int, float] = {
    # The physical pilot was 5--12x smaller than these values and produced no
    # faults, thermal/voltage stress, or material roll/pitch excursion.  In
    # particular, 0.0175--0.03 rad ankle probes barely overcame deadband.
    # Full screening therefore uses a genuinely identification-scale leg
    # envelope while retaining 0.04 rad modeled-limit margin around the
    # measured post-startup pose.
    0: .45, 1: .30, 2: .30, 3: .60, 4: .35, 5: .18,
    6: .45, 7: .30, 8: .30, 9: .60, 10: .35, 11: .18,
    12: .18,
    # Upper-body excursions stay at the collision-free screening envelope.
    # The requested expansion targets the leg/waist dynamics that affect
    # balance and slow walking; larger isolated arm motion caused torso/arm
    # self-contact in the 23-DoF MuJoCo model.
    15: .10, 16: .08, 17: .08, 18: .10, 19: .08,
    22: .10, 23: .08, 24: .08, 25: .10, 26: .08,
}


@dataclass(frozen=True)
class ReviewBlock:
    block_id: int
    name: str
    kind: str
    description: str
    phase: np.ndarray
    offsets: np.ndarray
    local_time_s: np.ndarray

    def validate(self) -> None:
        n = len(self.local_time_s)
        if n < 2 or self.offsets.shape != (n, 29) or len(self.phase) != n:
            raise ValueError(f"{self.name}: invalid sample dimensions")
        if not np.isfinite(self.offsets).all() or not np.isfinite(self.local_time_s).all():
            raise ValueError(f"{self.name}: non-finite plan value")
        if np.max(np.abs(self.offsets[[0, -1]])) > 1e-10:
            raise ValueError(f"{self.name}: block must start and end at captured neutral")
        if np.max(np.abs(np.diff(self.local_time_s) - DT)) > 1e-8:
            raise ValueError(f"{self.name}: samples must be exactly 50 Hz")
        absent = np.setdiff1d(np.arange(29), PRESENT_23)
        if np.max(np.abs(self.offsets[:, absent])) > 1e-12:
            raise ValueError(f"{self.name}: block commands an absent 23-DoF axis")
        for motor in PRESENT_23:
            limit = FULL_SCREENING_AMPLITUDE[int(motor)]
            if np.max(np.abs(self.offsets[:, motor])) > limit + 1e-9:
                raise ValueError(f"{self.name}: motor {motor} exceeds screening range")
        rate_limit = (
            DYNAMIC_MAX_TARGET_RATE_RAD_S
            if self.kind.startswith("dynamic_")
            else MAX_TARGET_RATE_RAD_S
        )
        rate = np.max(np.abs(np.diff(self.offsets, axis=0))) / DT
        if rate > rate_limit + 1e-8:
            raise ValueError(
                f"{self.name}: target rate {rate:.4f} exceeds "
                f"{rate_limit:.2f} rad/s"
            )

    @property
    def duration_s(self) -> float:
        return float(self.local_time_s[-1])

    @property
    def active_motors(self) -> tuple[int, ...]:
        active = np.flatnonzero(np.max(np.abs(self.offsets), axis=0) > 1e-10)
        return tuple(map(int, active))

    @property
    def peak_offsets(self) -> dict[int, float]:
        return {
            motor: float(np.max(np.abs(self.offsets[:, motor])))
            for motor in self.active_motors
        }


@dataclass(frozen=True)
class ReviewProtocol:
    name: str
    blocks: tuple[ReviewBlock, ...]

    def validate(self) -> None:
        if self.name not in {"pilot", "screening"}:
            raise ValueError("review protocol must be pilot or screening")
        if not self.blocks:
            raise ValueError("review protocol is empty")
        if [block.block_id for block in self.blocks] != list(range(len(self.blocks))):
            raise ValueError("block IDs must be contiguous from zero")
        if len({block.name for block in self.blocks}) != len(self.blocks):
            raise ValueError("block names must be unique")
        for block in self.blocks:
            block.validate()
            if self.name == "pilot":
                for motor in PRESENT_23:
                    pilot_limit = 0.50 * SCREENING_AMPLITUDE[int(motor)]
                    if np.max(np.abs(block.offsets[:, motor])) > pilot_limit + 1e-9:
                        raise ValueError(
                            f"{block.name}: motor {motor} exceeds the fast-pilot envelope"
                        )

    @property
    def motion_duration_s(self) -> float:
        return float(sum(block.duration_s for block in self.blocks))


class _BlockBuilder:
    def __init__(self):
        self.current = np.zeros(29, dtype=np.float64)
        self.rows: list[np.ndarray] = [self.current.copy()]
        self.phases: list[str] = ["neutral_start"]

    def hold(self, seconds: float, phase: str) -> None:
        count = max(1, int(round(seconds / DT)))
        for _ in range(count):
            self.rows.append(self.current.copy())
            self.phases.append(phase)

    def transition(self, goal: np.ndarray, seconds: float, phase: str) -> None:
        goal = np.asarray(goal, dtype=np.float64)
        count = max(2, int(round(seconds / DT)))
        start = self.current.copy()
        for sample in range(1, count + 1):
            alpha = sample / count
            blend = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
            self.rows.append(start + blend * (goal - start))
            self.phases.append(phase)
        self.current = goal.copy()

    def sine(self, amplitude: np.ndarray, frequency_hz: float, cycles: float, phase: str) -> None:
        if np.max(np.abs(self.current)) > 1e-10:
            raise ValueError("sine probe must begin at neutral")
        count = max(2, int(round(cycles / (frequency_hz * DT))))
        for sample in range(1, count + 1):
            angle = 2.0 * np.pi * cycles * sample / count
            self.rows.append(np.sin(angle) * amplitude)
            self.phases.append(phase)
        self.current[:] = 0.0
        self.rows[-1][:] = 0.0

    def gait_signature(
        self,
        scale: float,
        protocol: str,
        cycles: int = 1,
        frequency_hz: float | None = None,
        phase: str | None = None,
    ) -> None:
        if np.max(np.abs(self.current)) > 1e-10:
            raise ValueError("gait signature must begin at neutral")
        envelope = _envelope(protocol)
        if frequency_hz is None:
            frequency_hz = 0.12 if protocol == "pilot" else 0.08
        count = int(round(cycles / (frequency_hz * DT)))
        for sample in range(1, count + 1):
            angle = 2.0 * np.pi * cycles * sample / count
            wave = np.sin(angle)
            left_swing = max(0.0, wave) ** 2
            right_swing = max(0.0, -wave) ** 2
            row = np.zeros(29, dtype=np.float64)
            row[0] = scale * envelope[0] * 0.55 * wave
            row[6] = -scale * envelope[6] * 0.55 * wave
            row[3] = scale * envelope[3] * 0.80 * left_swing
            row[9] = scale * envelope[9] * 0.80 * right_swing
            row[4] = scale * envelope[4] * (
                -0.35 * wave + 0.30 * left_swing
            )
            row[10] = scale * envelope[10] * (
                0.35 * wave + 0.30 * right_swing
            )
            self.rows.append(row)
            self.phases.append(
                phase or f"gait_signature_scale_{scale:.2f}_hz_{frequency_hz:.2f}"
            )
        self.current[:] = 0.0
        self.rows[-1][:] = 0.0

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.max(np.abs(self.current)) > 1e-10:
            raise ValueError("block builder did not finish at neutral")
        offsets = np.asarray(self.rows, dtype=np.float64)
        local_time = np.arange(1, len(offsets) + 1, dtype=np.float64) * DT
        return offsets, np.asarray(self.phases), local_time


def _envelope(protocol: str) -> dict[int, float]:
    return SCREENING_AMPLITUDE if protocol == "pilot" else FULL_SCREENING_AMPLITUDE


def _amplitude(
    motors: dict[int, float], scale: float, protocol: str,
) -> np.ndarray:
    result = np.zeros(29, dtype=np.float64)
    envelope = _envelope(protocol)
    for motor, weight in motors.items():
        if motor not in PRESENT_23:
            raise ValueError(f"group references absent motor {motor}")
        result[motor] = envelope[motor] * weight * scale
    return result


def _single_block(block_id: int, motor: int, protocol: str) -> ReviewBlock:
    builder = _BlockBuilder()
    builder.hold(0.4 if protocol == "pilot" else 0.8, "neutral_observe")
    scales = (0.50,) if protocol == "pilot" else (0.25, 0.50, 1.00)
    # Minimum-jerk peak derivative is 1.875*displacement/duration.  Scale the
    # full-screening transition so its largest endpoint remains under the
    # static 0.35-rad/s target-rate contract.
    transition_s = (
        0.75 if protocol == "pilot" else
        max(2.0, 1.90 * FULL_SCREENING_AMPLITUDE[motor] / MAX_TARGET_RATE_RAD_S)
    )
    dwell_s = 0.30 if protocol == "pilot" else 1.20
    neutral_dwell_s = 0.20 if protocol == "pilot" else 0.30
    for scale in scales:
        goal = _amplitude({motor: 1.0}, scale, protocol)
        builder.transition(goal, transition_s, f"static_pos_scale_{scale:.2f}")
        builder.hold(dwell_s, f"dwell_pos_scale_{scale:.2f}")
        builder.transition(np.zeros(29), transition_s, "return_neutral")
        builder.hold(neutral_dwell_s, "neutral_dwell")
        builder.transition(-goal, transition_s, f"static_neg_scale_{scale:.2f}")
        builder.hold(dwell_s, f"dwell_neg_scale_{scale:.2f}")
        builder.transition(np.zeros(29), transition_s, "return_neutral")
        builder.hold(neutral_dwell_s, "neutral_dwell")
    sine_scale = 0.35 if protocol == "pilot" else 0.50
    sine_frequency = 0.20 if protocol == "pilot" else 0.10
    builder.sine(
        _amplitude({motor: 1.0}, sine_scale, protocol),
        sine_frequency,
        1.0,
        "sine_probe",
    )
    builder.hold(0.5 if protocol == "pilot" else 1.0, "neutral_end")
    offsets, phase, time_s = builder.finish()
    block = ReviewBlock(
        block_id=block_id,
        name=f"single_{MOTOR_NAMES[motor]}",
        kind="single_joint",
        description=(
            f"{MOTOR_NAMES[motor]} only: positive/negative static levels "
            f"and a {sine_frequency:.2f} Hz local response probe"
        ),
        phase=phase,
        offsets=offsets,
        local_time_s=time_s,
    )
    block.validate()
    return block


DYNAMIC_RATE_LIMIT: dict[int, float] = {
    0: 1.60, 1: 1.25, 2: 1.25, 3: 2.00, 4: 1.60, 5: 0.80,
    6: 1.60, 7: 1.25, 8: 1.25, 9: 2.00, 10: 1.60, 11: 0.80,
}


def _dynamic_single_block(block_id: int, motor: int) -> ReviewBlock:
    """Walking-band through OOD-frequency excitation for one leg joint."""
    builder = _BlockBuilder()
    builder.hold(0.5, "dynamic_neutral_observe")
    frequencies = ((0.35, 2.0), (0.75, 3.0), (1.25, 3.0), (2.20, 4.0))
    base_amplitude = 0.45 * FULL_SCREENING_AMPLITUDE[motor]
    for frequency_hz, cycles in frequencies:
        amplitude_value = min(
            base_amplitude,
            DYNAMIC_RATE_LIMIT[motor] / (2.0 * np.pi * frequency_hz),
        )
        amplitude = np.zeros(29, dtype=np.float64)
        amplitude[motor] = amplitude_value
        builder.sine(
            amplitude,
            frequency_hz,
            cycles,
            f"dynamic_single_hz_{frequency_hz:.2f}",
        )
        builder.hold(0.25, "dynamic_neutral_between_frequencies")
    builder.hold(0.5, "dynamic_neutral_end")
    offsets, phase, time_s = builder.finish()
    block = ReviewBlock(
        block_id=block_id,
        name=f"dynamic_{MOTOR_NAMES[motor]}",
        kind="dynamic_single_joint",
        description=(
            f"{MOTOR_NAMES[motor]} dynamic response at 0.35/0.75 Hz walking-band "
            f"and 1.25/2.20 Hz out-of-distribution frequencies"
        ),
        phase=phase,
        offsets=offsets,
        local_time_s=time_s,
    )
    block.validate()
    return block


GROUPS: tuple[tuple[str, str, dict[int, float]], ...] = (
    ("bilateral_hip_pitch", "bilateral hip pitch in phase", {0: 0.7, 6: 0.7}),
    ("bilateral_hip_roll", "symmetric bilateral hip roll", {1: 0.7, 7: -0.7}),
    ("bilateral_knee", "bilateral knee flexion", {3: 0.7, 9: 0.7}),
    ("bilateral_ankle_pitch", "bilateral ankle pitch", {4: 0.7, 10: 0.7}),
    (
        "bilateral_leg_flexion", "hip-knee-ankle flexion resembling a shallow squat",
        {0: -0.35, 3: 0.70, 4: 0.35, 6: -0.35, 9: 0.70, 10: 0.35},
    ),
    (
        "alternating_leg_clearance", "opposed sagittal leg articulation",
        {0: -0.35, 3: 0.70, 4: 0.35, 6: 0.35, 9: -0.70, 10: -0.35},
    ),
)


def _group_block(
    block_id: int, name: str, description: str, weights: dict[int, float], protocol: str,
) -> ReviewBlock:
    # The small pilot retains its original schedule and validated hash.  With
    # the enlarged full envelope, however, ±0.176 rad inward bilateral hip roll
    # caused the two ankle-roll links to touch.  Cap only that screening block
    # at ±0.144 rad; isolated hip-roll sweeps still reach ±0.30 rad.
    if protocol == "screening" and name == "bilateral_hip_roll":
        weights = {1: 0.48, 7: -0.48}
    builder = _BlockBuilder()
    builder.hold(0.5 if protocol == "pilot" else 1.0, "neutral_observe")
    scales = (0.35,) if protocol == "pilot" else (0.33, 0.67, 1.00)
    maximum_goal = max(abs(value) for value in _amplitude(weights, 1.0, protocol))
    transition_s = (
        1.0 if protocol == "pilot" else
        max(2.0, 1.90 * maximum_goal / MAX_TARGET_RATE_RAD_S)
    )
    dwell_s = 0.4 if protocol == "pilot" else 1.20
    neutral_dwell_s = 0.25 if protocol == "pilot" else 0.5
    for scale in scales:
        goal = _amplitude(weights, scale, protocol)
        builder.transition(goal, transition_s, f"combined_pos_scale_{scale:.2f}")
        builder.hold(dwell_s, f"combined_dwell_pos_scale_{scale:.2f}")
        builder.transition(np.zeros(29), transition_s, "return_neutral")
        builder.hold(neutral_dwell_s, "neutral_dwell")
        builder.transition(-goal, transition_s, f"combined_neg_scale_{scale:.2f}")
        builder.hold(dwell_s, f"combined_dwell_neg_scale_{scale:.2f}")
        builder.transition(np.zeros(29), transition_s, "return_neutral")
        builder.hold(neutral_dwell_s, "neutral_dwell")
    sine_scale = 0.25 if protocol == "pilot" else 0.50
    sine_frequency = 0.16 if protocol == "pilot" else 0.08
    builder.sine(
        _amplitude(weights, sine_scale, protocol),
        sine_frequency,
        1.0,
        "combined_sine_probe",
    )
    builder.hold(0.5 if protocol == "pilot" else 1.0, "neutral_end")
    offsets, phase, time_s = builder.finish()
    block = ReviewBlock(
        block_id, f"group_{name}", "combined_joint", description,
        phase, offsets, time_s,
    )
    block.validate()
    return block


def _gait_block(block_id: int, protocol: str) -> ReviewBlock:
    builder = _BlockBuilder()
    builder.hold(0.5 if protocol == "pilot" else 1.0, "neutral_observe")
    if protocol == "pilot":
        schedule = ((0.50, 0.30, 2), (1.00, 0.25, 2), (1.60, 0.15, 2))
        for frequency_hz, scale, cycles in schedule:
            builder.gait_signature(
                scale,
                protocol=protocol,
                cycles=cycles,
                frequency_hz=frequency_hz,
                phase=f"pilot_dynamic_gait_hz_{frequency_hz:.2f}",
            )
            builder.hold(0.35, "neutral_between_frequencies")
    else:
        scales = (0.25, 0.50, 0.75, 1.00)
        for scale in scales:
            builder.gait_signature(scale, protocol=protocol)
            builder.hold(1.0, "neutral_between_scales")
    builder.hold(0.5 if protocol == "pilot" else 1.0, "neutral_end")
    offsets, phase, time_s = builder.finish()
    block = ReviewBlock(
        block_id, "group_walking_signature", "walking_signature",
        "staged-frequency alternating hip/knee/ankle signature; suspended, not locomotion",
        phase, offsets, time_s,
    )
    block.validate()
    return block


def _dynamic_gait_block(block_id: int) -> ReviewBlock:
    """Coupled alternating response from normal cadence through OOD cadence."""
    builder = _BlockBuilder()
    builder.hold(0.5, "dynamic_neutral_observe")
    schedule = (
        (0.45, 0.65, 2),
        (0.85, 0.55, 3),
        (1.25, 0.42, 3),
        (1.80, 0.30, 4),
        (2.20, 0.22, 4),
    )
    for frequency_hz, scale, cycles in schedule:
        builder.gait_signature(
            scale,
            protocol="screening",
            cycles=cycles,
            frequency_hz=frequency_hz,
            phase=f"dynamic_coupled_gait_hz_{frequency_hz:.2f}",
        )
        builder.hold(0.30, "dynamic_neutral_between_frequencies")
    builder.hold(0.5, "dynamic_neutral_end")
    offsets, phase, time_s = builder.finish()
    block = ReviewBlock(
        block_id,
        "dynamic_coupled_walking_band_and_ood",
        "dynamic_coupled_gait",
        (
            "alternating hip/knee/ankle response from 0.45/0.85 Hz walking "
            "cadence through 1.25/1.80/2.20 Hz OOD cadence"
        ),
        phase,
        offsets,
        time_s,
    )
    block.validate()
    return block


def _dynamic_reversal_block(block_id: int) -> ReviewBlock:
    """Fast bounded direction reversals absent from periodic nominal walking."""
    builder = _BlockBuilder()
    builder.hold(0.5, "dynamic_neutral_observe")
    pose = _amplitude(
        {
            0: -0.35, 3: 0.70, 4: 0.35,
            6: 0.35, 9: -0.35, 10: -0.25,
        },
        0.50,
        "screening",
    )
    builder.transition(pose, 0.50, "dynamic_reversal_entry")
    builder.hold(0.10, "dynamic_reversal_dwell")
    reversal_duration = max(
        0.32,
        1.90 * 2.0 * float(np.max(np.abs(pose))) / DYNAMIC_MAX_TARGET_RATE_RAD_S,
    )
    for reversal in range(8):
        goal = -pose if reversal % 2 == 0 else pose
        builder.transition(goal, reversal_duration, f"dynamic_reversal_{reversal + 1}")
        builder.hold(0.10, "dynamic_reversal_dwell")
    builder.transition(np.zeros(29), 0.60, "dynamic_reversal_return")
    builder.hold(0.5, "dynamic_neutral_end")
    offsets, phase, time_s = builder.finish()
    block = ReviewBlock(
        block_id,
        "dynamic_asymmetric_direction_reversals",
        "dynamic_transient_reversal",
        (
            "eight rapid bounded left/right leg command reversals to probe "
            "transient lag, backlash, hysteresis, and nonperiodic OOD response"
        ),
        phase,
        offsets,
        time_s,
    )
    block.validate()
    return block


def _neutral_block(block_id: int, protocol: str) -> ReviewBlock:
    builder = _BlockBuilder()
    builder.hold(2.0 if protocol == "pilot" else 5.0, "pd_neutral_observe")
    offsets, phase, time_s = builder.finish()
    block = ReviewBlock(
        block_id, "neutral_pd_hold", "neutral_hold",
        "hold the captured pose to measure gravity/preload and suspension settling",
        phase, offsets, time_s,
    )
    block.validate()
    return block


def build_review_protocol(name: str = "pilot") -> ReviewProtocol:
    if name not in {"pilot", "screening"}:
        raise ValueError("protocol must be pilot or screening")
    blocks: list[ReviewBlock] = [_neutral_block(0, name)]
    # Mirrored joints are adjacent, with the nominally healthier right side
    # first to provide an immediate bilateral reference for the repaired left.
    legs = (6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5)
    arms_and_waist = (12, 25, 18, 26, 19, 22, 15, 23, 16, 24, 17)
    motors = legs if name == "pilot" else legs + arms_and_waist
    for motor in motors:
        blocks.append(_single_block(len(blocks), motor, name))
        if name == "screening" and motor < 12:
            blocks.append(_dynamic_single_block(len(blocks), motor))
    for group_name, description, weights in GROUPS:
        blocks.append(_group_block(len(blocks), group_name, description, weights, name))
    blocks.append(_gait_block(len(blocks), name))
    if name == "screening":
        blocks.append(_dynamic_gait_block(len(blocks)))
        blocks.append(_dynamic_reversal_block(len(blocks)))
    protocol = ReviewProtocol(name, tuple(blocks))
    protocol.validate()
    return protocol


def save_review_protocol(protocol: ReviewProtocol, directory: Path | str) -> Path:
    protocol.validate()
    directory = Path(directory).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    samples_path = directory / "protocol.csv"
    header = [
        "schema", "protocol", "block_id", "block_name", "block_kind",
        "sample", "local_time_s", "phase",
    ] + [f"offset_{i}" for i in range(29)]
    with samples_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for block in protocol.blocks:
            for sample in range(len(block.local_time_s)):
                writer.writerow([
                    SCHEMA, protocol.name, block.block_id, block.name, block.kind,
                    sample, f"{block.local_time_s[sample]:.9f}", block.phase[sample],
                    *[f"{value:.12g}" for value in block.offsets[sample]],
                ])
    manifest = {
        "schema": SCHEMA,
        "embodiment": "g1_23dof",
        "protocol": protocol.name,
        "control_dt_s": DT,
        "requires_operator_approval_each_block": True,
        "maximum_static_target_rate_rad_s": MAX_TARGET_RATE_RAD_S,
        "maximum_dynamic_target_rate_rad_s": DYNAMIC_MAX_TARGET_RATE_RAD_S,
        "dynamic_excitation": (
            None if protocol.name == "pilot" else {
                "single_leg_joint_frequencies_hz": [0.35, 0.75, 1.25, 2.20],
                "coupled_gait_frequencies_hz": [0.45, 0.85, 1.25, 1.80, 2.20],
                "includes_nonperiodic_direction_reversal": True,
                "intent": (
                    "cover observed walking band and bounded OOD frequency/transient response"
                ),
            }
        ),
        "full_range_enabled": protocol.name == "screening",
        "range_semantics": (
            "fast small-range workflow pilot" if protocol.name == "pilot"
            else "expanded suspended diagnostic envelope; not mechanical joint limits"
        ),
        "amplitude_envelope_rad": {
            str(motor): float(
                (0.50 * SCREENING_AMPLITUDE[motor])
                if protocol.name == "pilot"
                else FULL_SCREENING_AMPLITUDE[motor]
            )
            for motor in PRESENT_23
        },
        "physical_sensor_log": {
            "timing_and_transport": [
                "actual_time", "LowState age", "LowState tick", "mode_pr",
                "mode_machine", "remote buttons and analog axes",
            ],
            "imu": [
                "quaternion", "gyroscope", "accelerometer", "roll/pitch/yaw",
                "temperature", "captured steady quaternion and roll/pitch/yaw",
                "captured gyroscope, accelerometer, and temperature medians",
            ],
            "each_of_23_physical_motors": [
                "mode", "q", "dq", "ddq", "tau_est", "winding temperature",
                "driver temperature", "voltage", "raw sensor words", "fault code",
                "captured steady q/dq/ddq/torque/temperatures/voltage medians",
            ],
            "not_available_in_hg_lowstate": [
                "foot contact force", "external link/foot pose", "strap tension",
            ],
        },
        "motion_duration_s_excluding_approval_wait": protocol.motion_duration_s,
        "blocks": [
            {
                "block_id": block.block_id,
                "name": block.name,
                "kind": block.kind,
                "description": block.description,
                "duration_s": block.duration_s,
                "active_motors": list(block.active_motors),
                "active_joint_names": [MOTOR_NAMES[i] for i in block.active_motors],
                "peak_offsets_rad": {
                    str(i): value for i, value in block.peak_offsets.items()
                },
            }
            for block in protocol.blocks
        ],
    }
    (directory / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return samples_path


def load_review_protocol(path: Path | str) -> ReviewProtocol:
    path = Path(path).expanduser().absolute()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or rows[0].get("schema") != SCHEMA:
        raise ValueError("not a review-gated calibration protocol")
    protocol_name = rows[0]["protocol"]
    blocks: list[ReviewBlock] = []
    for block_id in sorted({int(row["block_id"]) for row in rows}):
        selected = [row for row in rows if int(row["block_id"]) == block_id]
        if [int(row["sample"]) for row in selected] != list(range(len(selected))):
            raise ValueError(f"block {block_id} sample indices are not contiguous")
        first = selected[0]
        if any(
            row["block_name"] != first["block_name"]
            or row["block_kind"] != first["block_kind"]
            or row["protocol"] != protocol_name
            for row in selected
        ):
            raise ValueError(f"block {block_id} metadata changes within the block")
        offsets = np.asarray([
            [float(row[f"offset_{i}"]) for i in range(29)] for row in selected
        ])
        block = ReviewBlock(
            block_id, first["block_name"], first["block_kind"],
            first["block_name"].replace("_", " "),
            np.asarray([row["phase"] for row in selected]), offsets,
            np.asarray([float(row["local_time_s"]) for row in selected]),
        )
        block.validate()
        blocks.append(block)
    protocol = ReviewProtocol(protocol_name, tuple(blocks))
    protocol.validate()
    return protocol
