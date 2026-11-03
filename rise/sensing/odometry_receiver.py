"""Receive timestamped pelvis estimates from the local odometry bridge."""

import json
import socket
import select
import time
import threading
import numpy as np


class PelvisOdometryReceiver:
    """Local bridge from an external qualified estimator; no inertial invention.

    UDP JSON packets must carry actual mapped capture timestamps and pelvis
    position in the same gravity-aligned inertial axes as the pelvis quaternion.
    The estimator may run separately; LiDAR is not a planner/tracker requirement.
    """

    def __init__(self, port, *, background=False):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", int(port)))
        self.socket.setblocking(False)
        self.latest = None
        self.packet = None
        self.received_s = None
        self.error = None
        self.stream_id = None
        self.last_stamp = None
        self.stream_changed = False
        self._snapshot = (None, None, None, None)
        self._stop = threading.Event()
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._run, daemon=True, name="pelvis-odometry")
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            # Wake on packets instead of polling 500 times/s and adding up to
            # 2 ms of avoidable delivery delay. Timeout only bounds shutdown.
            ready, _, _ = select.select([self.socket], [], [], 0.05)
            if ready:
                self._receive()

    def _receive(self):
        # Bound work on the control thread even if a producer floods the socket.
        for _ in range(32):
            try:
                data, _ = self.socket.recvfrom(8192)
            except BlockingIOError:
                break
            self.received_s = time.monotonic()
            try:
                packet = json.loads(data)
                if (
                    packet.get("schema") != "hv2.pelvis_odometry.v1"
                    or packet.get("frame") != "pelvis_imu_gravity_frame"
                    or not packet.get("estimator")
                ):
                    raise ValueError(
                        "Odometry requires a declared estimator and calibrated pelvis/IMU common frame"
                    )
                stream = packet.get("stream_id")
                if self.stream_id is not None and stream != self.stream_id:
                    self.stream_changed = True
                self.stream_id = stream
                if self.stream_changed:
                    raise ValueError(
                        "Odometry estimator restarted; episode origin is no longer valid"
                    )
                stamp = packet.get("capture_monotonic_s")
                if stamp is not None and np.isfinite(float(stamp)):
                    if (
                        self.last_stamp is not None
                        and float(stamp) <= self.last_stamp
                        and (
                            packet.get("tracking_valid") is True
                            or packet.get("geometry_valid") is True
                        )
                    ):
                        raise ValueError("Odometry clock reset/out-of-order")
                    self.last_stamp = max(
                        float(stamp),
                        self.last_stamp if self.last_stamp is not None else float(stamp),
                    )
                self.packet = packet
                self.error = None
            except (ValueError, TypeError, AttributeError) as exc:
                self.packet = None
                self.error = str(exc)
            self._snapshot = (self.packet, self.received_s, self.error, self.stream_id)

    def sample(self, now_s=None, *, required=False, max_age_s=0.05, check_age=True):
        """Return diagnostics without using unqualified measurements as feedback.

        Required mode retains the existing clock/validity failure behavior. A
        monitoring-only deployment can inspect finite estimates and their status.
        """
        if self._thread is None:
            self._receive()
        raw, received, error, stream_id = self._snapshot
        # Live callers read the clock AFTER selecting the background snapshot.
        # A packet can arrive after a caller's earlier timestamp; comparing it
        # to that timestamp incorrectly labels its local receipt as future.
        # Explicit times remain available for deterministic/offline checks.
        if now_s is None:
            now_s = time.monotonic()
        packet = raw or {}
        self.latest = None
        try:
            position = np.asarray(packet.get("position_m"), dtype=float)
        except (ValueError, TypeError):
            position = np.asarray([])
        stamp = packet.get("capture_monotonic_s")
        stamp = float(stamp) if isinstance(stamp, (int, float)) and np.isfinite(stamp) else None
        finite = position.shape == (3,) and np.isfinite(position).all() and stamp is not None
        uncertainty = packet.get("clock_uncertainty_s")
        uncertainty = (
            float(uncertainty)
            if isinstance(uncertainty, (int, float)) and np.isfinite(uncertainty)
            else None
        )
        qualified = (
            packet.get("tracking_valid") is True
            and uncertainty is not None
            and 0 <= uncertainty <= 0.005
        )
        age = None if stamp is None else float(now_s - stamp)
        delay = packet.get("position_filter_delay_s", 0.0)
        valid_delay = isinstance(delay, (int, float)) and np.isfinite(delay) and delay >= 0
        if not valid_delay:
            error = error or "invalid_position_filter_delay"
        effective_stamp = None if stamp is None or not valid_delay else stamp - float(delay)
        effective_age = None if effective_stamp is None else float(now_s - effective_stamp)
        geometry = packet.get("geometry_valid", packet.get("tracking_valid", False)) is True
        fresh = (
            finite
            and valid_delay
            and (not check_age or (0 <= age and 0 <= effective_age <= max_age_s))
        )
        reason = error or packet.get("invalid_reason") or ""
        if not raw:
            reason = reason or "waiting_for_odometry"
        elif not finite:
            reason = "invalid_pose"
        elif not fresh:
            reason = "stale_or_future_odometry"
        elif not geometry:
            reason = reason or "geometry_invalid"
        elif not qualified:
            reason = reason or "clock_offset_unqualified"
        result = dict(
            position_m=position.tolist() if finite else None,
            capture_s=effective_stamp,
            age_s=effective_age,
            measurement_capture_s=stamp,
            measurement_age_s=age,
            position_filter_delay_s=delay if valid_delay else None,
            received_monotonic_s=received,
            arrival_age_s=None if received is None else float(now_s - received),
            clock_uncertainty_s=uncertainty,
            age_is_estimated=uncertainty is None,
            geometry_valid=geometry,
            tracking_valid=bool(qualified and fresh and geometry and not reason),
            available=bool(fresh and geometry and not error),
            invalid_reason=reason or None,
            estimator=packet.get("estimator"),
            stream_id=stream_id,
            used_for_steering=False,
        )
        for key in (
            "tilt_disagreement_rad",
            "yaw_disagreement_rad",
            "state_pair_error_s",
            "processing_s",
            "arrival_monotonic_s",
            "lidar_stamp_s",
            "clock_source",
            "orientation_disagreement_rad",
            "position_raw_m",
            "position_filter",
            "position_filter_tau_s",
        ):
            result[key] = packet.get(key)
        if required:
            if error:
                raise ValueError(error)
            if not raw:
                raise RuntimeError("Waiting for qualified pelvis odometry")
            if raw and not qualified:
                raise RuntimeError("Pelvis estimator tracking/clock qualification lost")
            if raw and not finite:
                raise ValueError("Invalid pelvis odometry")
            if raw and (not fresh or not geometry):
                raise RuntimeError("Pelvis odometry is stale or geometrically invalid")
            if reason:
                raise RuntimeError("Pelvis odometry invalid: " + str(reason))
        if result["tracking_valid"]:
            self.latest = (position.astype(np.float32), effective_stamp)
            result["used_for_steering"] = bool(required)
        return result

    def poll(self):
        self.sample(required=True, check_age=False)
        return self.latest

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.socket.close()
