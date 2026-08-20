"""Planner navigation from keyboard axes or the Unitree wireless remote.

``JoystickTeleop`` is the lightweight keyboard/scripted stand-in.
``GamepadManagerTeleop`` mirrors the native C++ ``gamepad_manager`` button
protocol used on the real robot.  Both emit no VR fields, so SONIC selects G1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.pico_client import LocomotionMode
from common.remote_controller import KeyMap
from common.scripted_teleop import TeleopBase

DEADZONE = 0.15
YAW_GAIN = 1.5


MOTION_SETS = (
    (
        LocomotionMode.SLOW_WALK,
        LocomotionMode.WALK,
        LocomotionMode.RUN,
        LocomotionMode.FORWARD_JUMP,
        LocomotionMode.STEALTH_WALK,
        LocomotionMode.INJURED_WALK,
    ),
    (
        LocomotionMode.IDLE_SQUAT,
        LocomotionMode.IDLE_KNEEL_TWO_LEGS,
        LocomotionMode.IDLE_KNEEL,
        LocomotionMode.CRAWLING,
        LocomotionMode.ELBOW_CRAWLING,
    ),
    (
        LocomotionMode.IDLE_BOXING,
        LocomotionMode.WALK_BOXING,
        LocomotionMode.LEFT_PUNCH,
        LocomotionMode.RIGHT_PUNCH,
        LocomotionMode.RANDOM_PUNCH,
        LocomotionMode.LEFT_HOOK,
        LocomotionMode.RIGHT_HOOK,
    ),
    (
        LocomotionMode.LEDGE_WALKING,
        LocomotionMode.OBJECT_CARRYING,
        LocomotionMode.STEALTH_WALK_2,
        LocomotionMode.HAPPY_DANCE_WALK,
        LocomotionMode.ZOMBIE_WALK,
        LocomotionMode.GUN_WALK,
        LocomotionMode.SCARE_WALK,
    ),
)

MOTION_SET_NAMES = ("STANDING", "SQUAT_CRAWL", "BOXING", "STYLED_WALKING")
STATIC_LOWER_MODES = frozenset(
    (
        LocomotionMode.IDLE_SQUAT,
        LocomotionMode.IDLE_KNEEL_TWO_LEGS,
        LocomotionMode.IDLE_KNEEL,
    )
)
CRAWLING_MODES = frozenset(
    (LocomotionMode.CRAWLING, LocomotionMode.ELBOW_CRAWLING)
)
PUNCH_MODES = frozenset(
    (
        LocomotionMode.LEFT_PUNCH,
        LocomotionMode.RIGHT_PUNCH,
        LocomotionMode.LEFT_HOOK,
        LocomotionMode.RIGHT_HOOK,
    )
)


@dataclass
class _Transition:
    target: LocomotionMode
    remaining_s: float
    final: LocomotionMode | None = None


class JoystickTeleop(TeleopBase):
    def __init__(
        self,
        publish_hz: float = 10.0,
        locomotion_mode: LocomotionMode = LocomotionMode.WALK,
        yaw_command_scale: float = 1.0,
    ):
        super().__init__(publish_hz)
        self.navigation_frame = "world"
        self.lx = 0.0     # right positive
        self.ly = 0.0     # forward positive
        self.rx = 0.0     # turn, right positive
        self.locomotion_mode = LocomotionMode(locomotion_mode)
        self.yaw_rad = 0.0
        self.yaw_command_scale = yaw_command_scale
        self._yaw_inplace_scale = None
        self._previous_buttons = np.zeros(16, dtype=np.int8)
        self._idle_capture_requested = False
        self._idle_capture_chord_down = False

    @property
    def yaw_command_scale(self) -> float:
        return self._yaw_command_scale

    @yaw_command_scale.setter
    def yaw_command_scale(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value) or not 0.0 < value <= 2.0:
            raise ValueError("yaw_command_scale must be finite and in (0, 2]")
        self._yaw_command_scale = value

    @property
    def yaw_inplace_scale(self) -> float:
        """Heading-rate scale used when the stick is turn-only.

        Walk-plus-turn keeps ``yaw_command_scale``. In-place turning can use a
        larger floor so a short stride still produces stepping, not a stand-spin.
        """
        if self._yaw_inplace_scale is None:
            return self._yaw_command_scale
        return self._yaw_inplace_scale

    @yaw_inplace_scale.setter
    def yaw_inplace_scale(self, value: float | None) -> None:
        if value is None:
            self._yaw_inplace_scale = None
            return
        value = float(value)
        if not np.isfinite(value) or not 0.0 < value <= 2.0:
            raise ValueError("yaw_inplace_scale must be finite and in (0, 2]")
        self._yaw_inplace_scale = value

    def _yaw_scale_for_stick(self, *, deadzone: float) -> float:
        if max(abs(self.lx), abs(self.ly)) < deadzone:
            return self.yaw_inplace_scale
        return self.yaw_command_scale

    def _update_idle_capture_chord(self, buttons: np.ndarray) -> bool:
        """Reserve an F1+F3 rising-edge chord for deliberate stand capture."""
        chord = bool(buttons[KeyMap.F1] and buttons[KeyMap.F3])
        rising = chord and not self._idle_capture_chord_down
        self._idle_capture_chord_down = chord
        if rising:
            self._idle_capture_requested = True
            print(
                "[IDLE CAPTURE] F1+F3: waiting for one second of neutral, "
                "steady standing data"
            )
        return rising

    def consume_idle_capture_request(self) -> bool:
        requested = self._idle_capture_requested
        self._idle_capture_requested = False
        return requested

    def idle_capture_stationary(self) -> bool:
        """Whether operator input is neutral enough to collect stand samples."""
        return (
            int(self.mode()) == int(LocomotionMode.IDLE)
            and max(abs(self.lx), abs(self.ly), abs(self.rx)) < DEADZONE
        )

    def tick(self, dt: float) -> None:
        super().tick(dt)
        if abs(self.rx) >= DEADZONE:
            self.yaw_rad += (
                YAW_GAIN
                * self._yaw_scale_for_stick(deadzone=DEADZONE)
                * (-self.rx)
                * float(dt)
            )

    def set_axes(self, lx: float, ly: float, rx: float = 0.0) -> None:
        self.lx, self.ly, self.rx = float(lx), float(ly), float(rx)

    def update_remote(self, remote) -> None:
        """Update sticks and apply rising-edge D-pad mode selection.

        D-pad avoids X (episode logging), A (controller stop), and Select
        (shutdown). Neutral left stick always emits canonical IDLE regardless
        of the selected walking mode.
        """
        self.set_axes(remote.lx, remote.ly, remote.rx)
        buttons = np.asarray(remote.button, dtype=np.int8).reshape(-1)
        if buttons.size < self._previous_buttons.size:
            raise ValueError(
                f"Unitree remote exposes {buttons.size} buttons; expected at least 16"
            )
        self._update_idle_capture_chord(buttons)
        for index, mode in (
            (KeyMap.up, LocomotionMode.SLOW_WALK),
            (KeyMap.right, LocomotionMode.WALK),
            (KeyMap.down, LocomotionMode.RUN),
        ):
            if buttons[index] and not self._previous_buttons[index]:
                self.set_locomotion_mode(mode)
                print(f"[JOYSTICK] locomotion mode: {mode.name}")
        self._previous_buttons[:] = buttons[: self._previous_buttons.size]

    def set_locomotion_mode(self, mode: LocomotionMode | int) -> None:
        self.locomotion_mode = LocomotionMode(mode)

    def _command(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        facing = np.array(
            [np.cos(self.yaw_rad), np.sin(self.yaw_rad), 0.0], np.float32
        )

        raw_mag = float(np.clip(np.hypot(self.lx, self.ly), 0.0, 1.0))
        if raw_mag < DEADZONE:
            return np.zeros(3, np.float32), facing, -1.0, int(LocomotionMode.IDLE)

        mag = min((raw_mag - DEADZONE) / (1.0 - DEADZONE), 1.0)
        movement_local = np.array([-self.lx, self.ly], np.float64) * (mag / raw_mag)
        rotation_facing = np.array(
            [[-facing[1], facing[0]], [facing[0], facing[1]]], np.float64
        )
        movement_xy = rotation_facing @ movement_local
        movement = np.array([movement_xy[0], movement_xy[1], 0.0], np.float64)

        if self.locomotion_mode == LocomotionMode.SLOW_WALK:
            speed = 0.1 + 0.5 * mag
        elif self.locomotion_mode == LocomotionMode.WALK:
            speed = -1.0
        elif self.locomotion_mode == LocomotionMode.RUN:
            speed = 1.5 + 3.0 * mag
        else:
            speed = mag
        return movement.astype(np.float32), facing, float(speed), int(self.locomotion_mode)

    def mode(self) -> int:
        return self._command()[3]

    def nav_command(self) -> dict:
        movement, facing, speed, _ = self._command()
        return {
            "movement": movement,
            "facing": facing,
            "speed": speed,
            "height": -1.0,
        }

    def vr_3point(self):
        return None


class GamepadManagerTeleop(JoystickTeleop):
    """Python mirror of the native C++ planner ``GamepadManager``.

    The state table, set membership, fixed speed/height values, stick
    smoothing, direction binning, and rising-edge controls intentionally track
    ``sonic_cpp/.../input_interface/gamepad_manager.hpp``.  System-level
    actions are exposed as ``control_started`` and ``stop_requested`` so the
    MuJoCo host can reproduce WAIT_FOR_CONTROL and clean shutdown.
    """

    SMOOTH = 0.3
    MANAGER_DEADZONE = 0.05
    MANAGER_YAW_GAIN = 2.0  # native: 0.02 rad at its 100-Hz input rate
    CRAWL_TRANSITION_S = 2.0
    BOXING_REVERT_S = 0.5   # actual C++ threshold (comment there says 1 s)

    def __init__(self, publish_hz: float = 10.0):
        super().__init__(
            publish_hz=publish_hz,
            locomotion_mode=LocomotionMode.SLOW_WALK,
        )
        self.control_started = False
        self.stop_requested = False
        self.motion_set_index = 0
        self.mode_index_in_set = 0
        self.movement_speed = 0.4
        self.height = -1.0
        self.ry = 0.0
        self._emergency_idle = False
        self._transition: _Transition | None = None
        self._boxing_revert_s: float | None = None

    @property
    def motion_set_name(self) -> str:
        return MOTION_SET_NAMES[self.motion_set_index]

    def _pressed(self, buttons: np.ndarray, index: int) -> bool:
        return bool(buttons[index] and not self._previous_buttons[index])

    def _apply_speed_height(self, mode: LocomotionMode) -> None:
        self.locomotion_mode = mode
        if mode == LocomotionMode.SLOW_WALK:
            self.movement_speed, self.height = 0.4, -1.0
        elif mode == LocomotionMode.RUN:
            self.movement_speed, self.height = 1.5, -1.0
        elif mode == LocomotionMode.CRAWLING:
            self.movement_speed, self.height = 0.7, 0.4
        elif mode == LocomotionMode.ELBOW_CRAWLING:
            self.movement_speed, self.height = 0.7, 0.3
        elif mode in STATIC_LOWER_MODES:
            self.movement_speed, self.height = -1.0, 0.4
        elif mode in MOTION_SETS[2]:
            self.movement_speed, self.height = 0.7, -1.0
        else:
            self.movement_speed, self.height = -1.0, -1.0

    def set_locomotion_mode(self, mode: LocomotionMode | int) -> None:
        """Deterministic test hook that preserves manager speed/height rules."""
        self._apply_speed_height(LocomotionMode(mode))

    def _apply_mode_index(self) -> None:
        self._apply_speed_height(MOTION_SETS[self.motion_set_index][self.mode_index_in_set])

    def _begin_crawl_transition(self, target: LocomotionMode) -> None:
        current = self.locomotion_mode
        if target == LocomotionMode.CRAWLING:
            if current == target:
                return
            if current not in (
                LocomotionMode.IDLE_KNEEL_TWO_LEGS,
                LocomotionMode.IDLE_KNEEL,
            ):
                self._apply_speed_height(LocomotionMode.IDLE_KNEEL_TWO_LEGS)
            self._transition = _Transition(target, self.CRAWL_TRANSITION_S)
        elif target == LocomotionMode.ELBOW_CRAWLING:
            if current == target:
                return
            if current == LocomotionMode.CRAWLING:
                self._transition = _Transition(target, self.CRAWL_TRANSITION_S)
            elif current in (
                LocomotionMode.IDLE_KNEEL_TWO_LEGS,
                LocomotionMode.IDLE_KNEEL,
            ):
                self._transition = _Transition(
                    LocomotionMode.CRAWLING,
                    self.CRAWL_TRANSITION_S,
                    final=target,
                )
            else:
                self._apply_speed_height(LocomotionMode.IDLE_KNEEL_TWO_LEGS)
                self._transition = _Transition(
                    LocomotionMode.CRAWLING,
                    self.CRAWL_TRANSITION_S,
                    final=target,
                )

    def _begin_exit_crawl(self, target: LocomotionMode) -> None:
        if self.locomotion_mode == LocomotionMode.ELBOW_CRAWLING:
            self._apply_speed_height(LocomotionMode.CRAWLING)
            self._transition = _Transition(
                LocomotionMode.IDLE_KNEEL_TWO_LEGS,
                self.CRAWL_TRANSITION_S,
                final=target,
            )
        elif self.locomotion_mode == LocomotionMode.CRAWLING:
            self._apply_speed_height(LocomotionMode.IDLE_KNEEL_TWO_LEGS)
            self._transition = _Transition(target, self.CRAWL_TRANSITION_S)

    def _select_target(self, target: LocomotionMode) -> None:
        was_crawling = self.locomotion_mode in CRAWLING_MODES
        if target in CRAWLING_MODES:
            self._begin_crawl_transition(target)
        elif was_crawling:
            self._begin_exit_crawl(target)
        else:
            self._transition = None
            self._apply_speed_height(target)

    def _select_motion_set(self, index: int) -> None:
        self.motion_set_index = index
        self.mode_index_in_set = 0
        self._boxing_revert_s = None
        self._select_target(MOTION_SETS[index][0])
        if index == 2:
            self._boxing_revert_s = self.BOXING_REVERT_S
        print(
            f"[GAMEPAD MANAGER] set={self.motion_set_name} "
            f"mode={self.locomotion_mode.name}"
        )

    def _cycle(self, delta: int) -> None:
        modes = MOTION_SETS[self.motion_set_index]
        self.mode_index_in_set = (self.mode_index_in_set + delta) % len(modes)
        self._select_target(modes[self.mode_index_in_set])
        print(
            f"[GAMEPAD MANAGER] set={self.motion_set_name} "
            f"mode={self.locomotion_mode.name}"
        )

    def _boxing_mode(self, mode: LocomotionMode, *, revert: bool) -> None:
        self._transition = None
        self._apply_speed_height(mode)
        self._boxing_revert_s = self.BOXING_REVERT_S if revert else None
        print(f"[GAMEPAD MANAGER] set=BOXING mode={mode.name}")

    def update_remote(self, remote) -> None:
        raw = (remote.lx, remote.ly, remote.rx, remote.ry)
        filtered = [0.0 if abs(float(value)) < self.MANAGER_DEADZONE else float(value)
                    for value in raw]
        self.lx = self.lx * (1.0 - self.SMOOTH) + filtered[0] * self.SMOOTH
        self.ly = self.ly * (1.0 - self.SMOOTH) + filtered[1] * self.SMOOTH
        self.rx = self.rx * (1.0 - self.SMOOTH) + filtered[2] * self.SMOOTH
        self.ry = self.ry * (1.0 - self.SMOOTH) + filtered[3] * self.SMOOTH

        buttons = np.asarray(remote.button, dtype=np.int8).reshape(-1)
        if buttons.size < self._previous_buttons.size:
            raise ValueError(
                f"Unitree remote exposes {buttons.size} buttons; expected at least 16"
            )
        capture_chord_rising = self._update_idle_capture_chord(buttons)
        pressed = lambda key: self._pressed(buttons, key)
        self._emergency_idle = pressed(KeyMap.A)

        if pressed(KeyMap.start):
            if not self.control_started:
                self.control_started = True
                print("[GAMEPAD MANAGER] Start: policy control enabled")
        if pressed(KeyMap.select):
            self.stop_requested = True
            print("[GAMEPAD MANAGER] Select: stop requested")

        for key, index in (
            (KeyMap.up, 0),
            (KeyMap.down, 1),
            (KeyMap.left, 2),
            (KeyMap.right, 3),
        ):
            if pressed(key):
                self._select_motion_set(index)

        if self.motion_set_index == 2:
            if pressed(KeyMap.X):
                self._boxing_mode(LocomotionMode.WALK_BOXING, revert=False)
            if pressed(KeyMap.Y):
                self._boxing_mode(LocomotionMode.RANDOM_PUNCH, revert=False)
            if pressed(KeyMap.B):
                self._boxing_mode(LocomotionMode.IDLE_BOXING, revert=True)
            if pressed(KeyMap.L1):
                self._boxing_mode(LocomotionMode.LEFT_PUNCH, revert=True)
            if pressed(KeyMap.R1):
                self._boxing_mode(LocomotionMode.RIGHT_PUNCH, revert=True)
            if pressed(KeyMap.L2) and not capture_chord_rising:
                self._boxing_mode(LocomotionMode.LEFT_HOOK, revert=True)
            if pressed(KeyMap.R2) and not capture_chord_rising:
                self._boxing_mode(LocomotionMode.RIGHT_HOOK, revert=True)
        else:
            if pressed(KeyMap.X):
                self._cycle(+1)
            if pressed(KeyMap.Y):
                self._cycle(-1)
            if pressed(KeyMap.B):
                self.mode_index_in_set = 0
                self._select_target(MOTION_SETS[self.motion_set_index][0])
                print(
                    f"[GAMEPAD MANAGER] B: reset set={self.motion_set_name} "
                    f"mode={self.locomotion_mode.name}"
                )
            if pressed(KeyMap.L1):
                self.yaw_rad += np.pi / 4.0
            if pressed(KeyMap.R1):
                self.yaw_rad -= np.pi / 4.0
            if self.motion_set_index == 1 and not capture_chord_rising:
                if pressed(KeyMap.L2):
                    self.height = max(self.height - 0.1, 0.2)
                if pressed(KeyMap.R2):
                    self.height = min(self.height + 0.1, 0.8)

        self._previous_buttons[:] = buttons[: self._previous_buttons.size]

    def tick(self, dt: float) -> None:
        # Before Start, native GamepadManager updates its local selection state
        # but does not enable the planner.  Keep planner sequence 0 while still
        # advancing deterministic transition timers.
        if self.control_started:
            TeleopBase.tick(self, dt)
        else:
            self.t += float(dt)
        if abs(self.rx) >= self.MANAGER_DEADZONE:
            self.yaw_rad -= (
                self.MANAGER_YAW_GAIN
                * self._yaw_scale_for_stick(deadzone=self.MANAGER_DEADZONE)
                * self.rx
                * float(dt)
            )

        if self._transition is not None:
            self._transition.remaining_s -= float(dt)
            if self._transition.remaining_s <= 0.0:
                transition = self._transition
                self._apply_speed_height(transition.target)
                if transition.final is None or transition.final == transition.target:
                    self._transition = None
                else:
                    self._transition = _Transition(
                        transition.final,
                        self.CRAWL_TRANSITION_S,
                    )
                print(f"[GAMEPAD MANAGER] transition mode={self.locomotion_mode.name}")

        if self._boxing_revert_s is not None:
            self._boxing_revert_s -= float(dt)
            if self._boxing_revert_s <= 0.0:
                self._boxing_revert_s = None
                self._apply_speed_height(LocomotionMode.WALK_BOXING)
                print("[GAMEPAD MANAGER] boxing auto-revert mode=WALK_BOXING")

    def _manager_command(self) -> tuple[np.ndarray, np.ndarray, float, float, int]:
        facing = np.array(
            [np.cos(self.yaw_rad), np.sin(self.yaw_rad), 0.0], np.float32
        )
        if not self.control_started or self._emergency_idle:
            return np.zeros(3, np.float32), facing, -1.0, -1.0, int(LocomotionMode.IDLE)

        stick_active = (
            abs(self.lx) >= self.MANAGER_DEADZONE
            or abs(self.ly) >= self.MANAGER_DEADZONE
        )
        if stick_active:
            raw_angle = float(np.arctan2(self.ly, self.lx))
            binned = round(raw_angle / (np.pi / 4.0)) * (np.pi / 4.0)
            movement_angle = binned - np.pi / 2.0 + self.yaw_rad
            movement = np.array(
                [np.cos(movement_angle), np.sin(movement_angle), 0.0], np.float32
            )
        else:
            movement = np.zeros(3, np.float32)

        mode = self.locomotion_mode
        speed = self.movement_speed
        height = self.height
        if not stick_active:
            if self.motion_set_index == 0:
                mode, speed, height = LocomotionMode.IDLE, -1.0, -1.0
            elif self.motion_set_index == 1:
                speed = 0.0 if mode in CRAWLING_MODES else -1.0
            elif self.motion_set_index in (2, 3):
                speed, height = 0.0, -1.0
                if self.motion_set_index == 2 and mode in PUNCH_MODES:
                    movement = facing.copy()

        if mode in STATIC_LOWER_MODES:
            movement = np.zeros(3, np.float32)
            speed = 0.0
        return movement, facing, float(speed), float(height), int(mode)

    def mode(self) -> int:
        return self._manager_command()[4]

    def idle_capture_stationary(self) -> bool:
        return (
            int(self.mode()) == int(LocomotionMode.IDLE)
            and max(abs(self.lx), abs(self.ly), abs(self.rx))
            < self.MANAGER_DEADZONE
        )

    def nav_command(self) -> dict:
        movement, facing, speed, height, _ = self._manager_command()
        return {
            "movement": movement,
            "facing": facing,
            "speed": speed,
            "height": height,
        }
