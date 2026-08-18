"""Shared operator state machine for suspended interactive calibration.

The state machine is independent of MuJoCo, DDS, and input devices. A viewer
key, Unitree remote button, or scripted validation event becomes the same
``InteractiveCommand`` before it can affect a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .protocol import (
    FULL_AMPLITUDE,
    MOTOR_NAMES,
    PRESENT_23,
    SCREENING_AMPLITUDE,
)


class InteractiveCommand(str, Enum):
    CAPTURE_NEUTRAL = "capture_neutral"
    EXECUTE = "execute"
    NEUTRAL = "neutral"
    NEXT = "next"
    PREVIOUS = "previous"
    SINGLE_BANK = "single_bank"
    GROUP_BANK = "group_bank"
    STATIC_MODE = "static_mode"
    LOOP_MODE = "loop_mode"
    RANGE_UP = "range_up"
    RANGE_DOWN = "range_down"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    TEMPO_UP = "tempo_up"
    TEMPO_DOWN = "tempo_down"
    STATUS = "status"
    ABORT = "abort"


@dataclass(frozen=True)
class CalibrationSelection:
    name: str
    motors: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.motors or len(self.motors) != len(self.weights):
            raise ValueError("selection needs equal nonempty motors and weights")
        if len(set(self.motors)) != len(self.motors):
            raise ValueError("selection contains a duplicate motor")
        if any(motor not in PRESENT_23 for motor in self.motors):
            raise ValueError("selection contains an absent 23-DoF motor")
        if not np.isfinite(self.weights).all() or max(map(abs, self.weights)) > 1.0:
            raise ValueError("selection weights must be finite and within [-1, 1]")


SINGLE_SELECTIONS = tuple(
    CalibrationSelection(MOTOR_NAMES[int(motor)], (int(motor),), (1.0,))
    for motor in PRESENT_23
)

# These groups are identification motions, not SONIC references. Mirrored
# roll/yaw weights use opposite signs so the body motion is bilaterally
# symmetric. Composite leg motions stay within each joint's own envelope.
GROUP_SELECTIONS = (
    CalibrationSelection("bilateral_hip_pitch", (0, 6), (1.0, 1.0)),
    CalibrationSelection("bilateral_hip_roll_symmetric", (1, 7), (1.0, -1.0)),
    CalibrationSelection("bilateral_hip_yaw_symmetric", (2, 8), (1.0, -1.0)),
    CalibrationSelection("bilateral_knee", (3, 9), (1.0, 1.0)),
    CalibrationSelection("bilateral_ankle_pitch", (4, 10), (1.0, 1.0)),
    CalibrationSelection("bilateral_ankle_roll_symmetric", (5, 11), (1.0, -1.0)),
    CalibrationSelection(
        "bilateral_leg_flexion", (0, 3, 4, 6, 9, 10),
        (-0.5, 1.0, 0.5, -0.5, 1.0, 0.5),
    ),
    CalibrationSelection(
        "alternating_leg_clearance", (0, 3, 4, 6, 9, 10),
        (-0.5, 1.0, 0.5, 0.5, -1.0, -0.5),
    ),
    CalibrationSelection("bilateral_shoulder_pitch", (15, 22), (1.0, 1.0)),
    CalibrationSelection("bilateral_shoulder_roll", (16, 23), (1.0, -1.0)),
    CalibrationSelection("bilateral_shoulder_yaw", (17, 24), (1.0, -1.0)),
    CalibrationSelection("bilateral_elbow", (18, 25), (1.0, 1.0)),
    CalibrationSelection("bilateral_wrist_roll", (19, 26), (1.0, -1.0)),
)


class InteractiveCalibrationController:
    """Produce smooth 29-slot offsets under explicit operator commands."""

    STATIC_PHASES = (0.0, 1.0, 0.0, -1.0, 0.0)
    RANGE_NAMES = ("micro", "screening", "extended")

    def __init__(
        self, *, max_envelope: str = "screening", dt: float = 0.02,
        require_capture: bool = True,
    ):
        if max_envelope not in {"screening", "full"}:
            raise ValueError("max_envelope must be screening or full")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.max_envelope = max_envelope
        self.dt = float(dt)
        self.bank = "single"
        self.indices = {"single": 0, "group": 0}
        self.mode = "static"
        self.range_index = 1
        self.scale = 0.50
        self.tempo = 0.50
        self.running = False
        self.aborted = False
        self.captured = not require_capture
        self.static_phase_index = 0
        self.loop_phase = 0.0
        self.current_offset = np.zeros(29, dtype=np.float64)
        self._transition_start = self.current_offset.copy()
        self._transition_goal = self.current_offset.copy()
        self._transition_elapsed = 0.0
        self._transition_duration = 1.0
        self._pending_selection: tuple[str, int] | None = None
        self.last_message = (
            "ready at captured neutral" if self.captured
            else "waiting for neutral capture"
        )

    @property
    def selections(self) -> tuple[CalibrationSelection, ...]:
        return SINGLE_SELECTIONS if self.bank == "single" else GROUP_SELECTIONS

    @property
    def selection(self) -> CalibrationSelection:
        return self.selections[self.indices[self.bank]]

    @property
    def range_name(self) -> str:
        return self.RANGE_NAMES[self.range_index]

    def _base_amplitude(self, motor: int) -> float:
        if self.range_name == "micro":
            return 0.25 * SCREENING_AMPLITUDE[motor]
        if self.range_name == "screening":
            return SCREENING_AMPLITUDE[motor]
        if self.max_envelope != "full":
            raise RuntimeError("extended range is not enabled")
        return FULL_AMPLITUDE[motor]

    def selection_amplitude(self) -> np.ndarray:
        out = np.zeros(29, dtype=np.float64)
        for motor, weight in zip(self.selection.motors, self.selection.weights):
            out[motor] = self.scale * self._base_amplitude(motor) * weight
        return out

    def _start_transition(self, goal: np.ndarray) -> None:
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (29,) or not np.isfinite(goal).all():
            raise ValueError("interactive target must be a finite 29-vector")
        self._transition_start = self.current_offset.copy()
        self._transition_goal = goal.copy()
        self._transition_elapsed = 0.0
        self._transition_duration = max(0.75, 1.5 / self.tempo)

    def _request_selection(self, bank: str, index: int) -> None:
        selections = SINGLE_SELECTIONS if bank == "single" else GROUP_SELECTIONS
        index %= len(selections)
        if np.max(np.abs(self.current_offset)) > 1e-5 or self.running:
            self.running = False
            self.static_phase_index = 0
            self._pending_selection = (bank, index)
            self._start_transition(np.zeros(29))
            self.last_message = "returning neutral before changing selection"
            return
        self.bank = bank
        self.indices[bank] = index
        self.static_phase_index = 0
        self.loop_phase = 0.0
        self._pending_selection = None
        self.last_message = f"selected {self.selection.name}"

    def neutral(self, message: str = "neutral requested") -> None:
        self.running = False
        self.static_phase_index = 0
        self._start_transition(np.zeros(29))
        self.last_message = message

    def can_capture(self) -> bool:
        return (
            not self.aborted
            and not self.running
            and self._pending_selection is None
            and np.max(np.abs(self.current_offset)) < 1e-5
        )

    def apply(self, command: InteractiveCommand | str) -> str:
        command = InteractiveCommand(command)
        if self.aborted:
            return self.status()
        if command == InteractiveCommand.ABORT:
            self.aborted = True
            self.neutral("abort requested")
            return self.status()
        if command == InteractiveCommand.STATUS:
            self.last_message = "status requested"
            return self.status()
        if command == InteractiveCommand.CAPTURE_NEUTRAL:
            if not self.can_capture():
                self.last_message = "capture rejected: return fully to neutral first"
            else:
                self.captured = True
                self.static_phase_index = 0
                self.loop_phase = 0.0
                self.last_message = "neutral captured; ready"
            return self.status()
        if not self.captured:
            self.last_message = "capture neutral before changing or executing a motion"
            return self.status()
        if command == InteractiveCommand.NEUTRAL:
            self.neutral()
            return self.status()
        if command == InteractiveCommand.STATIC_MODE:
            self.mode = "static"
            self.neutral("static mode; neutral")
            return self.status()
        if command == InteractiveCommand.LOOP_MODE:
            self.mode = "loop"
            self.neutral("trajectory-loop mode; neutral")
            self.loop_phase = 0.0
            return self.status()
        if command in (InteractiveCommand.SINGLE_BANK, InteractiveCommand.GROUP_BANK):
            bank = "single" if command == InteractiveCommand.SINGLE_BANK else "group"
            self._request_selection(bank, self.indices[bank])
            return self.status()
        if command in (InteractiveCommand.NEXT, InteractiveCommand.PREVIOUS):
            delta = 1 if command == InteractiveCommand.NEXT else -1
            self._request_selection(self.bank, self.indices[self.bank] + delta)
            return self.status()
        if command in (InteractiveCommand.RANGE_UP, InteractiveCommand.RANGE_DOWN):
            delta = 1 if command == InteractiveCommand.RANGE_UP else -1
            upper = 2 if self.max_envelope == "full" else 1
            self.range_index = int(np.clip(self.range_index + delta, 0, upper))
            if not self.running:
                phase = self.STATIC_PHASES[self.static_phase_index]
                self._start_transition(phase * self.selection_amplitude())
            self.last_message = f"range preset {self.range_name}"
            return self.status()
        if command in (InteractiveCommand.SCALE_UP, InteractiveCommand.SCALE_DOWN):
            delta = 0.10 if command == InteractiveCommand.SCALE_UP else -0.10
            self.scale = float(np.clip(round(self.scale + delta, 2), 0.10, 1.00))
            if not self.running:
                phase = self.STATIC_PHASES[self.static_phase_index]
                self._start_transition(phase * self.selection_amplitude())
            self.last_message = f"amplitude scale {self.scale:.2f}"
            return self.status()
        if command in (InteractiveCommand.TEMPO_UP, InteractiveCommand.TEMPO_DOWN):
            delta = 0.25 if command == InteractiveCommand.TEMPO_UP else -0.25
            self.tempo = float(np.clip(round(self.tempo + delta, 2), 0.25, 1.00))
            self.last_message = f"tempo scale {self.tempo:.2f}"
            return self.status()
        if command == InteractiveCommand.EXECUTE:
            if self._pending_selection is not None:
                self.last_message = "wait for neutral before executing the queued selection"
                return self.status()
            if self.mode == "static":
                self.static_phase_index = (
                    self.static_phase_index + 1
                ) % len(self.STATIC_PHASES)
                phase = self.STATIC_PHASES[self.static_phase_index]
                self._start_transition(phase * self.selection_amplitude())
                self.last_message = f"static phase {phase:+.0f}"
            else:
                self.running = not self.running
                if self.running:
                    self.loop_phase = 0.0
                    self.last_message = "trajectory loop started"
                else:
                    self.neutral("trajectory loop stopped; returning neutral")
            return self.status()
        raise AssertionError(f"unhandled command {command}")

    def step(self) -> np.ndarray:
        if self.mode == "loop" and self.running and not self.aborted:
            self.loop_phase += 2.0 * np.pi * (0.10 * self.tempo) * self.dt
            raw = np.sin(self.loop_phase) * self.selection_amplitude()
            # The sine starts at zero. A per-motor rate bound also protects a
            # live range/scale adjustment from becoming a command jump.
            max_delta = 0.35 * self.dt
            delta = np.clip(raw - self.current_offset, -max_delta, max_delta)
            self.current_offset += delta
        else:
            self._transition_elapsed = min(
                self._transition_duration,
                self._transition_elapsed + self.dt,
            )
            alpha = self._transition_elapsed / self._transition_duration
            blend = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
            self.current_offset = (
                self._transition_start
                + blend * (self._transition_goal - self._transition_start)
            )
            if alpha >= 1.0 and self._pending_selection is not None:
                bank, index = self._pending_selection
                self.bank = bank
                self.indices[bank] = index
                self._pending_selection = None
                self.last_message = f"selected {self.selection.name}"
        if not np.isfinite(self.current_offset).all():
            raise RuntimeError("interactive calibration generated a non-finite target")
        return self.current_offset.copy()

    def status_dict(self) -> dict[str, object]:
        return {
            "bank": self.bank,
            "selection": self.selection.name,
            "motors": list(self.selection.motors),
            "mode": self.mode,
            "range": self.range_name,
            "scale": self.scale,
            "tempo": self.tempo,
            "running": self.running,
            "static_phase": self.STATIC_PHASES[self.static_phase_index],
            "pending_selection": self._pending_selection is not None,
            "aborted": self.aborted,
            "captured": self.captured,
            "message": self.last_message,
        }

    def status(self) -> str:
        state = self.status_dict()
        return (
            f"bank={state['bank']} selection={state['selection']} "
            f"mode={state['mode']} range={state['range']} "
            f"scale={state['scale']:.2f} tempo={state['tempo']:.2f} "
            f"running={int(state['running'])} :: {state['message']}"
        )
