"""Nav command from stick axes: PicoClient stand-in, like ScriptedTeleop.

Left stick = move (pelvis frame), right stick x = turn, arms held raised.
Feed it axes each step: robot -> remote_controller, sim -> keyboard.
"""
from __future__ import annotations

import numpy as np

from common.scripted_teleop import IDLE, OBJECT_CARRYING, IDENT_QUATS, TeleopBase

DEADZONE = 0.15
MAX_TURN = 0.6      # rad of heading offset per replan at full stick
MAX_SPEED = 0.3     # planner default speed: what target_vel=-1 walks at


class JoystickTeleop(TeleopBase):
    def __init__(self, publish_hz: float = 5.0):
        super().__init__(publish_hz)
        self.lx = 0.0     # right positive
        self.ly = 0.0     # forward positive
        self.rx = 0.0     # turn, right positive

    def set_axes(self, lx: float, ly: float, rx: float = 0.0) -> None:
        self.lx, self.ly, self.rx = float(lx), float(ly), float(rx)

    def _move(self) -> np.ndarray:
        v = np.array([self.ly, -self.lx], np.float32)    # pelvis: x fwd, y left
        return v if np.linalg.norm(v) > DEADZONE else np.zeros(2, np.float32)

    def mode(self) -> int:
        walking = np.linalg.norm(self._move()) > 0 or abs(self.rx) > DEADZONE
        return OBJECT_CARRYING if walking else IDLE

    def nav_command(self) -> dict:
        v = self._move()
        mag = min(float(np.linalg.norm(v)), 1.0)
        turn = -self.rx * MAX_TURN if abs(self.rx) > DEADZONE else 0.0
        movement = np.zeros(3, np.float32)
        if mag > 0:
            movement[:2] = v / np.linalg.norm(v)
        return {
            "movement": movement,
            "facing": np.array([np.cos(turn), np.sin(turn), 0.0], np.float32),
            "speed": mag * MAX_SPEED if mag > 0 else -1.0,
            "height": -1.0,
        }

    def vr_3point(self):
        return self.raised_wrists(0.18, 0.25), IDENT_QUATS.copy()
