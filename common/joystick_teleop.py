"""Robot-frame navigation from the Unitree remote sticks.

Headset-free stand-in for the PICO planner publisher; emits no VR three-point
fields, so SONIC selects its G1 encoder. Commands are declared `local`
(current pelvis frame) and rotated once by the live reference heading in
PicoReferenceWindow -- forward on the stick is forward for the robot, wherever
it faces. The real PICO app publishes world-frame and passes through untouched.
"""
from __future__ import annotations

import numpy as np

from common.pico_client import LocomotionMode
from common.scripted_teleop import TeleopBase

DEADZONE = 0.15
MAX_TURN = 0.6      # rad of heading offset per replan at full right stick


class JoystickTeleop(TeleopBase):
    def __init__(
        self,
        publish_hz: float = 10.0,
        locomotion_mode: LocomotionMode = LocomotionMode.WALK,
    ):
        super().__init__(publish_hz)
        self.navigation_frame = "local"
        self.lx = 0.0     # right positive
        self.ly = 0.0     # forward positive
        self.rx = 0.0     # turn, right positive
        self.locomotion_mode = LocomotionMode(locomotion_mode)

    def set_axes(self, lx: float, ly: float, rx: float = 0.0) -> None:
        self.lx, self.ly, self.rx = float(lx), float(ly), float(rx)

    def set_locomotion_mode(self, mode: LocomotionMode | int) -> None:
        self.locomotion_mode = LocomotionMode(mode)

    def _command(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        turn = -self.rx * MAX_TURN if abs(self.rx) >= DEADZONE else 0.0
        facing = np.array([np.cos(turn), np.sin(turn), 0.0], np.float32)

        raw_mag = float(np.clip(np.hypot(self.lx, self.ly), 0.0, 1.0))
        if raw_mag < DEADZONE:
            return np.zeros(3, np.float32), facing, -1.0, int(LocomotionMode.IDLE)

        mag = min((raw_mag - DEADZONE) / (1.0 - DEADZONE), 1.0)
        movement = np.array(
            [self.ly, -self.lx, 0.0], np.float64) * (mag / raw_mag)

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
