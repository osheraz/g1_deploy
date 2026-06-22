from __future__ import annotations

import threading
import numpy as np

# Unity Y-up -> Z-up, matching GMR XRobotStreamer.coordinate_transform_unity_data
_R_U2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def _rot_to_quat_wxyz(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


_Q_U2Z = _rot_to_quat_wxyz(_R_U2Z)


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _xform_pose(pose7):
    # raw xrt pose = [x, y, z, qx, qy, qz, qw] (Unity frame, scalar-last)
    p = np.asarray(pose7[:3], np.float64)
    qw_last = np.asarray(pose7[3:7], np.float64)
    q_wxyz = np.array([qw_last[3], qw_last[0], qw_last[1], qw_last[2]])
    pos = p @ _R_U2Z.T
    ori = _quat_mul_wxyz(_Q_U2Z, q_wxyz)
    return pos.astype(np.float32), ori.astype(np.float32)


class PicoReader:
    """XRoboToolkit 3-point teleop reader (head + 2 controllers), Z-up frame, quats wxyz.
    Inert if the SDK is absent (latest() -> None) so the deploy loop never blocks."""

    def __init__(self, rate_hz=100.0):
        self.dt = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._latest = None
        self._run = False
        self._thread = None
        try:
            import xrobotoolkit_sdk as xrt
            xrt.init()
            self._xrt = xrt
        except Exception as e:
            print(f"[PicoReader] SDK unavailable ({e}); reader inert")
            self._xrt = None

    def start(self):
        if self._xrt is None or self._run:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False

    def _loop(self):
        import time
        xrt = self._xrt
        while self._run:
            head_p, head_q = _xform_pose(xrt.get_headset_pose())
            lp, lq = _xform_pose(xrt.get_left_controller_pose())
            rp, rq = _xform_pose(xrt.get_right_controller_pose())
            frame = {
                "head": (head_p, head_q),
                "left_wrist": (lp, lq),
                "right_wrist": (rp, rq),
                "left_grip": float(xrt.get_left_grip()),
                "right_grip": float(xrt.get_right_grip()),
                "left_trig": float(xrt.get_left_trigger()),
                "right_trig": float(xrt.get_right_trigger()),
            }
            with self._lock:
                self._latest = frame
            time.sleep(self.dt)

    def latest(self):
        with self._lock:
            return self._latest
