"""Scripted teleop: stands in for PicoClient. No headset, no ZMQ.

    nav   walk forward, arms at rest
    vr3   arms raised, base idle
    both  both
    carry planner OBJECT_CARRYING gait, arms raised
"""
from __future__ import annotations

import numpy as np

IDLE, SLOW_WALK, OBJECT_CARRYING = 0, 1, 21   # sonic/planner/navigation.py

# obs.VR_ENDPOINT_NAMES order, pelvis frame, standing. Values are the official
# neutral (COMFORT_POS/QUAT_LOCAL in sonic/push_env/clip_controller.py) -- the
# quats are real hand orientations, not identity.
WRIST_L, WRIST_R, HEAD = 0, 1, 2
REST_VR = np.array([[0.0903, 0.1615, -0.2411],
                    [0.1280, -0.1522, -0.2461],
                    [0.0241, -0.0081, 0.4028]], np.float32)
REST_QUATS = np.array([[0.7295, 0.3145, 0.5533, -0.2506],
                       [0.7320, -0.2639, 0.5395, 0.3217],
                       [0.9991, 0.0110, 0.0402, -0.0002]], np.float32)

TESTS = ("nav", "vr3", "both", "idle", "carry")


class TeleopBase:
    """seq bumps at publish_hz: each bump is one planner replan (~60 ms).
    10 Hz matches the official deploy's planner thread."""

    def __init__(self, publish_hz: float = 10.0):
        self.publish_hz = publish_hz
        self.latest_seq = 0
        self.t = 0.0
        self._next_pub = 0.0

    def tick(self, dt: float) -> None:
        self.t += dt
        if self.t >= self._next_pub:
            self._next_pub = self.t + 1.0 / self.publish_hz
            self.latest_seq += 1

    def raised_wrists(self, fwd: float, up: float, osc: float = 0.0) -> np.ndarray:
        """REST_VR with both wrists ramped (2s) forward/up from rest."""
        pos = REST_VR.copy()
        ramp = min(self.t / 2.0, 1.0)
        for w in (WRIST_L, WRIST_R):
            pos[w, 0] += fwd * ramp
            pos[w, 2] += (up + osc) * ramp
        return pos

    def set_axes(self, lx, ly, rx=0.0) -> None:
        pass

    def hands(self):
        return None

    def stop(self) -> None:
        pass


class ScriptedTeleop(TeleopBase):
    def __init__(self, test: str = "idle", speed: float = 0.2, wrist_amp: float = 0.25,
                 wrist_hz: float = 0.25, publish_hz: float = 5.0):
        if test not in TESTS:
            raise SystemExit(f"unknown teleop test {test!r}; expected one of {TESTS}")
        super().__init__(publish_hz)
        self.test = test
        self.speed = speed
        self.wrist_amp = wrist_amp
        self.wrist_hz = wrist_hz

    def mode(self) -> int:
        if self.test == "carry":
            return OBJECT_CARRYING
        return SLOW_WALK if self.test in ("nav", "both") else IDLE

    def nav_command(self) -> dict:
        walking = self.test in ("nav", "both", "carry")
        return {
            "movement": np.array([1.0, 0.0, 0.0] if walking else [0.0, 0.0, 0.0], np.float32),
            "facing": np.array([1.0, 0.0, 0.0], np.float32),
            "speed": self.speed if walking else -1.0,
            "height": -1.0,
        }

    def vr_3point(self):
        if self.test in ("vr3", "both", "carry"):
            osc = 0.08 * np.sin(2 * np.pi * self.wrist_hz * self.t)
            pos = self.raised_wrists(0.18, self.wrist_amp, osc)
        else:
            pos = REST_VR.copy()
        return pos, REST_QUATS.copy()
