"""Official-PLANNER-style navigation from the Unitree remote sticks.

This is a headset-free stand-in for the PICO planner publisher.  It deliberately
does not emit VR three-point fields: the planner supplies the complete upper
body and SONIC must select its G1 encoder.
"""
from __future__ import annotations

import numpy as np

from common.pico_client import LocomotionMode
from common.scripted_teleop import TeleopBase

DEADZONE = 0.15
YAW_GAIN = 1.5


class JoystickTeleop(TeleopBase):
    """Mirror ``PlannerStreamer`` joystick math from the official PICO manager."""

    def __init__(
        self,
        publish_hz: float = 10.0,
        locomotion_mode: LocomotionMode = LocomotionMode.WALK,
    ):
        super().__init__(publish_hz)
        # The accumulated vectors below are already in the planner-session
        # world frame, exactly like PicoClient/YawAccumulator output.
        self.navigation_frame = "world"
        self.lx = 0.0     # right positive
        self.ly = 0.0     # forward positive
        self.rx = 0.0     # turn, right positive
        self.locomotion_mode = LocomotionMode(locomotion_mode)
        self.yaw_angle = 0.0

    def set_axes(self, lx: float, ly: float, rx: float = 0.0) -> None:
        self.lx, self.ly, self.rx = float(lx), float(ly), float(rx)

    def set_locomotion_mode(self, mode: LocomotionMode | int) -> None:
        self.locomotion_mode = LocomotionMode(mode)

    def reset_yaw(self) -> None:
        self.yaw_angle = 0.0

    def tick(self, dt: float) -> None:
        super().tick(dt)
        if abs(self.rx) >= DEADZONE:
            self.yaw_angle += YAW_GAIN * (-self.rx) * float(dt)

    def _command(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        facing = np.array(
            [np.cos(self.yaw_angle), np.sin(self.yaw_angle), 0.0],
            dtype=np.float32,
        )
        raw_mag = float(np.clip(np.hypot(self.lx, self.ly), 0.0, 1.0))
        if raw_mag < DEADZONE:
            return np.zeros(3, np.float32), facing, -1.0, int(LocomotionMode.IDLE)

        mag = min((raw_mag - DEADZONE) / (1.0 - DEADZONE), 1.0)
        movement_local = np.array([-self.lx, self.ly], np.float64) * (mag / raw_mag)
        # Official PICO rotation: local lateral/forward -> persistent planner world.
        perp_x, perp_y = -facing[1], facing[0]
        rotation_facing = np.array(
            [[perp_x, perp_y], [facing[0], facing[1]]], dtype=np.float64
        )
        movement_xy = rotation_facing @ movement_local
        movement = np.array([movement_xy[0], movement_xy[1], 0.0], np.float32)

        if self.locomotion_mode == LocomotionMode.SLOW_WALK:
            speed = 0.1 + 0.5 * mag
        elif self.locomotion_mode == LocomotionMode.WALK:
            speed = -1.0
        elif self.locomotion_mode == LocomotionMode.RUN:
            speed = 1.5 + 3.0 * mag
        else:
            speed = mag
        return movement, facing, float(speed), int(self.locomotion_mode)

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
