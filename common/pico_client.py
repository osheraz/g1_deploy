"""ZMQ SUB for the SONIC teleop `planner` topic.

Mirrors ``SimpleDepthClient`` and ``TactileClient``: a daemon thread holding the
latest message. Same split the official implementation uses -- a separate
process owns the PICO/XR SDK and publishes; the deploy only subscribes.

Wire format, copied from NVlabs/GR00T-WholeBodyControl
``gear_sonic/utils/teleop/zmq/zmq_planner_sender.py``:

    [b"planner"][1280-byte JSON header][packed little-endian payload]

The header is JSON describing every field, so the payload is self-describing:

    {"v":1,"endian":"le","count":1,
     "fields":[{"name":"mode","dtype":"i32","shape":[1]}, ...]}

Always present, in this order:

    mode        i32[1]   LocomotionMode
    movement    f32[3]   base nav command, x/y/z
    facing      f32[3]   base facing
    speed       f32[1]   -1 = default
    height      f32[1]   -1 = default

Optional, appended in this order when the sender includes them:

    upper_body_position  f32[N]
    upper_body_velocity  f32[N]
    left_hand_joints     f32[N]   Dex3 hand control
    right_hand_joints    f32[N]
    vr_position          f32[9]   3 endpoints, xyz
    vr_orientation       f32[12]  3 endpoints, wxyz
    vr_compliance        f32[3]   ignored by the official deploy
"""
from __future__ import annotations

import json
import threading
import time
from enum import IntEnum
from typing import Optional

import numpy as np
import zmq

HEADER_SIZE = 1280
DEFAULT_PORT = 5560
TOPIC = b"planner"

_DTYPE = {"i32": np.int32, "f32": np.float32, "f64": np.float64,
          "u8": np.uint8, "i64": np.int64}
DEFAULT_VR_ORIENTATION = np.array(
    [[0.7295, 0.3145, 0.5533, -0.2506],
     [0.7320, -0.2639, 0.5395, 0.3217],
     [0.9991, 0.0110, 0.0402, -0.0002]],
    dtype=np.float64,
)


class LocomotionMode(IntEnum):
    """Copied from gear_sonic/scripts/pico_manager_thread_server.py."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


def decode_planner_message(msg: bytes) -> dict:
    """Split one `planner` message into ``{field_name: ndarray}``."""
    if not msg.startswith(TOPIC):
        raise ValueError(f"not a planner message: {msg[:16]!r}")
    body = msg[len(TOPIC):]
    if len(body) < HEADER_SIZE:
        raise ValueError(f"planner message is shorter than its {HEADER_SIZE}-byte header")
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\x00").decode("utf-8"))
    if not isinstance(header.get("fields"), list):
        raise ValueError("planner header has no fields list")
    payload = body[HEADER_SIZE:]

    out, off = {}, 0
    for f in header["fields"]:
        dt = np.dtype(_DTYPE[f["dtype"]]).newbyteorder(
            "<" if header.get("endian", "le") == "le" else ">"
        )
        n = int(np.prod(f["shape"]))
        nbytes = n * dt.itemsize
        if off + nbytes > len(payload):
            raise ValueError(
                f"planner payload ends inside field {f['name']!r}: "
                f"need {off + nbytes}, have {len(payload)} bytes"
            )
        out[f["name"]] = np.frombuffer(payload[off:off + nbytes], dtype=dt).reshape(
            f["shape"]
        ).copy()
        off += nbytes
    if off != len(payload):
        raise ValueError(f"planner payload has {len(payload) - off} trailing bytes")
    required = {"mode", "movement", "facing", "speed", "height"}
    missing = sorted(required - out.keys())
    if missing:
        raise ValueError(f"planner message is missing required fields: {missing}")
    return out


class PicoClient(threading.Thread):
    """Latest teleop command from the PICO publisher."""

    # Official PICO's YawAccumulator sends planner-world vectors.  Reference
    # providers must not rotate these by the current pelvis yaw a second time.
    navigation_frame = "world"

    def __init__(self, server_address: str = "127.0.0.1", port: int = DEFAULT_PORT):
        super().__init__(daemon=True)
        self.server_address = server_address
        self.port = port

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{server_address}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self.socket.setsockopt(zmq.RCVTIMEO, 500)

        self._lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.latest: Optional[dict] = None
        self.latest_seq = 0
        self._latest_monotonic: Optional[float] = None

    def run(self) -> None:
        while self.running:
            try:
                msg = self.socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            try:
                fields = decode_planner_message(msg)
            except Exception:
                continue
            with self._lock:
                self.latest = fields
                self.frame_count += 1
                self.latest_seq += 1
                self._latest_monotonic = time.monotonic()

    def stop(self) -> None:
        if self.socket is None:
            return
        self.running = False
        if self.is_alive():
            self.join(timeout=2.0)
        self.socket.close(linger=0)
        self.context.term()
        self.socket = None
        self.context = None

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            return None if self.latest is None else dict(self.latest)

    def age_s(self) -> float:
        """Monotonic age of the latest complete planner message."""
        with self._lock:
            timestamp = self._latest_monotonic
        return float("inf") if timestamp is None else time.monotonic() - timestamp

    # -- convenience accessors, all None when that field was not sent --

    def mode(self) -> Optional[int]:
        s = self.snapshot()
        return None if s is None else int(s["mode"][0])

    def nav_command(self) -> Optional[dict]:
        """Base navigation: movement, facing, speed, height."""
        s = self.snapshot()
        if s is None:
            return None
        return {
            "movement": s["movement"],
            "facing": s["facing"],
            "speed": float(s["speed"][0]),
            "height": float(s["height"][0]),
        }

    def hands(self) -> Optional[dict]:
        s = self.snapshot()
        if s is None or "left_hand_joints" not in s:
            return None
        return {"left": s["left_hand_joints"], "right": s["right_hand_joints"]}

    def vr_3point(self) -> Optional[tuple]:
        """``(pos (3,3), quat_wxyz (3,4))`` or None if not streamed."""
        s = self.snapshot()
        # Official ZMQManager enables encoder mode 1 when vr_position is
        # present. Orientation is optional and falls back to its calibrated
        # comfort quaternions.
        if s is None or "vr_position" not in s:
            return None
        try:
            position = s["vr_position"].reshape(3, 3).astype(np.float64)
        except ValueError:
            return None
        orientation = DEFAULT_VR_ORIENTATION.copy()
        if "vr_orientation" in s:
            try:
                orientation = s["vr_orientation"].reshape(3, 4).astype(np.float64)
            except ValueError:
                pass
        return position, orientation


def main():
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Print the teleop planner stream.")
    ap.add_argument("--server", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    c = PicoClient(args.server, args.port)
    c.start()
    print(f"subscribed to tcp://{args.server}:{args.port} topic={TOPIC.decode()}")
    last = 0
    try:
        while True:
            time.sleep(1.0)
            hz = c.frame_count - last
            last = c.frame_count
            s = c.snapshot()
            if s is None:
                print(f"[pico] {hz} Hz | no data")
                continue
            nav = c.nav_command()
            vr = c.vr_3point()
            print(f"[pico] {hz} Hz | mode={LocomotionMode(int(s['mode'][0])).name} "
                  f"move={np.round(nav['movement'], 3)} facing={np.round(nav['facing'], 3)} "
                  f"speed={nav['speed']:.2f} height={nav['height']:.2f} "
                  f"vr={'yes' if vr is not None else 'no'} "
                  f"hands={'yes' if c.hands() is not None else 'no'} "
                  f"fields={sorted(s)}")
    except KeyboardInterrupt:
        pass
    finally:
        c.stop()


if __name__ == "__main__":
    main()
