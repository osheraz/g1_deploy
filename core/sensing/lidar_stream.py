"""Sensor-only LiDAR preview transport shared by live ROS, replay and viewers.

The preview protocol is not a qualified control observation. Capture clocks and
receipt clocks remain distinct. Each single-part message is one complete scan,
so ZMQ conflation drops whole old previews without corrupting multipart payloads.
"""

from __future__ import annotations

import json
from collections import deque
import struct
import threading
import time

import numpy as np

SCHEMA = "hv2.lidar_preview.v1"
MAX_POINTS = 250_000


def encode_scan(metadata, points, point_stamp_ns):
    points = np.asarray(points, dtype="<f4")
    stamps = np.asarray(point_stamp_ns, dtype="<i8")
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) > MAX_POINTS
        or stamps.shape != (len(points),)
    ):
        raise ValueError("Invalid point or timestamp shape")
    header = dict(
        metadata, schema=SCHEMA, point_count=len(points), xyz_dtype="<f4", stamp_dtype="<i8"
    )
    raw = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    return struct.pack("<I", len(raw)) + raw + points.tobytes() + stamps.tobytes()


def decode_scan(payload):
    if len(payload) < 4:
        raise ValueError("Truncated LiDAR packet")
    length = struct.unpack_from("<I", payload)[0]
    if not 0 < length <= 256_000 or 4 + length > len(payload):
        raise ValueError("Invalid header length")
    header = json.loads(payload[4 : 4 + length])
    if not isinstance(header, dict):
        raise ValueError("Invalid LiDAR header")
    n = header.get("point_count")
    if header.get("schema") != SCHEMA or type(n) is not int or not 0 <= n <= MAX_POINTS:
        raise ValueError("Invalid LiDAR schema or count")
    if header.get("xyz_dtype") != "<f4" or header.get("stamp_dtype") != "<i8":
        raise ValueError("Unsupported layout")
    if len(payload) != 4 + length + n * 20:
        raise ValueError("LiDAR payload length mismatch")
    offset = 4 + length
    xyz = np.frombuffer(payload, dtype="<f4", count=n * 3, offset=offset).reshape(n, 3)
    stamps = np.frombuffer(payload, dtype="<i8", count=n, offset=offset + n * 12)
    return header, xyz, stamps


class LidarClient(threading.Thread):
    """Nonblocking latest-complete-scan receiver; network runs outside the UI."""

    def __init__(self, endpoint="tcp://127.0.0.1:15560"):
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self.received = 0
        self.skipped = 0
        self.restarts = 0
        self.bad_packets = 0
        self.error = None
        self._session = None
        self._sequence = None
        self._last_receive = None
        self.hz = 0.0
        self._receive_times = deque(maxlen=90)
        self._recent = deque(maxlen=32)

    def run(self):
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(self.endpoint)
            while not self._stop_event.is_set():
                try:
                    payload = socket.recv()
                except zmq.Again:
                    continue
                now = time.monotonic()
                try:
                    header, points, stamps = decode_scan(payload)
                    session = str(header["session"])
                    seq = int(header["sequence"])
                    if self._session != session:
                        if self._session is not None:
                            self.restarts += 1
                        self._sequence = None
                        self._session = session
                        self._receive_times.clear()
                        self.hz = 0.0
                    if self._sequence is not None:
                        if seq <= self._sequence:
                            raise ValueError("Out-of-order sequence")
                        self.skipped += max(0, seq - self._sequence - 1)
                    self._sequence = seq
                    self._receive_times.append(now)
                    while len(self._receive_times) > 2 and now - self._receive_times[0] > 1.0:
                        self._receive_times.popleft()
                    if len(self._receive_times) > 1:
                        self.hz = (len(self._receive_times) - 1) / max(
                            now - self._receive_times[0], 1e-6
                        )
                    self._last_receive = now
                    self.received += 1
                    with self._lock:
                        if self._latest is not None and self._latest[0]["session"] != session:
                            self._recent.clear()
                        self._latest = (header, points, stamps, now)
                        self._recent.append(self._latest)
                    self.error = None
                except (ValueError, KeyError, TypeError, OverflowError, UnicodeError) as exc:
                    self.bad_packets += 1
                    self.error = str(exc)
        except zmq.ZMQError as exc:
            self.error = str(exc)
        finally:
            socket.close()
            context.term()

    def snapshot(self):
        with self._lock:
            return self._latest

    def recent(self):
        """Bounded immutable snapshots for display persistence, never policy history."""
        with self._lock:
            return tuple(self._recent)

    def summary(self):
        sample = self.snapshot()
        if sample is None:
            return f'lidar: {self.error or "waiting"} ({self.endpoint})'
        h, p, _, arrival = sample
        age = time.monotonic() - arrival
        valid = np.isfinite(p).all(axis=1) & (np.linalg.norm(p, axis=1) > 0.1)
        return (
            f"lidar: {'STALE' if age>.5 else 'ON'} source={h.get('source','unknown')} "
            f"receive={self.hz:.1f}Hz age={age*1000:.0f}ms points={len(p)} valid={valid.sum()} "
            f"frame={h.get('frame','unknown')} units={h.get('units','unknown')} "
            f"skipped_previews={self.skipped} bad={self.bad_packets} restarts={self.restarts}"
        )

    def close(self):
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=2)
