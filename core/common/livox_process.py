"""ROS 2 Livox CustomMsg receiver isolated from the 50 Hz control process."""

from __future__ import annotations

from collections import deque
import multiprocessing as mp
import signal
import time
import traceback

import numpy as np
import torch


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _rotate_by_rotvec(points: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Batch Rodrigues rotation without a scipy dependency in the ROS process."""
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = rotvec / np.maximum(angle, 1.0e-9)
    c, s = np.cos(angle), np.sin(angle)
    cross = np.cross(axis, points)
    dot = np.sum(axis * points, axis=-1, keepdims=True)
    return points * c + cross * s + axis * dot * (1.0 - c)


def rotational_deskew(
    xyz: np.ndarray,
    point_times_ns: np.ndarray,
    scan_end_ns: int,
    imu_samples,
) -> np.ndarray:
    """Rotate each return into the scan-end LiDAR frame.

    Translation is intentionally not integrated from accelerometer data: without
    a trustworthy initial linear velocity that would manufacture a drifting pose.
    """
    if len(imu_samples) < 1 or xyz.shape[0] == 0:
        return xyz
    times = np.asarray([sample[0] for sample in imu_samples], dtype=np.float64)
    omega = np.asarray([sample[1] for sample in imu_samples], dtype=np.float64)
    order = np.argsort(times)
    times, omega = times[order], omega[order]
    query = np.minimum(point_times_ns.astype(np.float64), float(scan_end_ns))
    grid = np.unique(
        np.concatenate(
            (query, times[(times >= query.min()) & (times <= scan_end_ns)],
             np.asarray([float(scan_end_ns)]))
        )
    )
    omega_grid = np.stack(
        [np.interp(grid, times, omega[:, axis]) for axis in range(3)], axis=-1
    )
    increments = 0.5 * (omega_grid[:-1] + omega_grid[1:]) * np.diff(grid)[:, None] * 1.0e-9
    integrated = np.zeros_like(omega_grid)
    if increments.shape[0]:
        integrated[:-1] = np.cumsum(increments[::-1], axis=0)[::-1]
    query_indices = np.searchsorted(grid, query)
    # R_end^T R_point ~= Exp(-integral(omega dt)), with trapezoidal IMU
    # integration over the whole remaining firing interval.
    return _rotate_by_rotvec(xyz, -integrated[query_indices]).astype(np.float32)


def rolling_window_points(point_history, end_ns: int, duration_s: float, point_count: int):
    """Select a fixed-size chronological sample from a rolling firing window."""
    duration_ns = max(1, int(float(duration_s) * 1.0e9))
    start_ns = int(end_ns) - duration_ns
    while point_history and int(point_history[0][0][-1]) < start_ns:
        point_history.popleft()
    point_ns = np.concatenate([item[0] for item in point_history])
    xyz = np.concatenate([item[1] for item in point_history])
    reflectivity = np.concatenate([item[2] for item in point_history])
    tags = np.concatenate([item[3] for item in point_history])
    in_window = (point_ns >= start_ns) & (point_ns <= end_ns)
    point_ns, xyz = point_ns[in_window], xyz[in_window]
    reflectivity, tags = reflectivity[in_window], tags[in_window]
    if point_ns.size == 0:
        return None
    # Do not publish a short startup fragment as a nominal full-duration scan.
    if int(point_ns.min()) > start_ns + duration_ns // 5:
        return None
    if point_ns.size > point_count:
        keep = np.linspace(0, point_ns.size - 1, point_count).round().astype(np.int64)
        point_ns, xyz = point_ns[keep], xyz[keep]
        reflectivity, tags = reflectivity[keep], tags[keep]
    return point_ns, xyz, reflectivity, tags, start_ns


def _child_main(
    lidar_topic, imu_topic, point_count, scan_duration_s, frames, seq,
    receipt_stamp, sensor_stamp, rate_hz, stop, err_text,
):
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu
        from livox_ros_driver2.msg import CustomMsg

        bufs = frames.numpy()
        imu_history = deque(maxlen=4096)
        point_history = deque()
        counter = 0
        last_receipt = 0.0
        rate_ema = 0.0

        class LivoxNode(Node):
            def __init__(self):
                super().__init__("hv2_livox_receiver")
                self.create_subscription(
                    Imu, imu_topic, self.on_imu, qos_profile_sensor_data
                )
                self.create_subscription(
                    CustomMsg, lidar_topic, self.on_lidar, qos_profile_sensor_data
                )

            def on_imu(self, msg):
                imu_history.append(
                    (
                        _stamp_ns(msg.header.stamp),
                        np.array(
                            [msg.angular_velocity.x, msg.angular_velocity.y,
                             msg.angular_velocity.z],
                            dtype=np.float32,
                        ),
                    )
                )

            def on_lidar(self, msg):
                nonlocal counter, last_receipt, rate_ema
                total = min(int(msg.point_num), len(msg.points))
                if total <= 0:
                    return
                packet = msg.points[:total]
                timebase = int(msg.timebase)
                offsets = np.asarray([p.offset_time for p in packet], dtype=np.uint64)
                packet_ns = timebase + offsets
                packet_xyz = np.asarray(
                    [[p.x, p.y, p.z] for p in packet], dtype=np.float32
                )
                packet_reflectivity = np.asarray(
                    [p.reflectivity for p in packet], dtype=np.float32
                )
                packet_tags = np.asarray([p.tag for p in packet], dtype=np.float32)
                end_ns = int(packet_ns.max())
                point_history.append(
                    (packet_ns, packet_xyz, packet_reflectivity, packet_tags)
                )

                window = rolling_window_points(
                    point_history, end_ns, scan_duration_s, point_count
                )
                if window is None:
                    return
                point_ns, xyz, reflectivity, tags, start_ns = window

                margin_ns = 20_000_000
                imu_window = [
                    sample for sample in imu_history
                    if start_ns - margin_ns <= sample[0] <= end_ns + margin_ns
                ]
                # During a fall, publishing an un-deskewed scan is worse than
                # dropping it. Startup/freshness watchdogs make this fail closed.
                if (
                    len(imu_window) < 2
                    or imu_window[0][0] > start_ns + margin_ns
                    or imu_window[-1][0] < end_ns - margin_ns
                ):
                    return
                xyz = rotational_deskew(xyz, point_ns, end_ns, imu_window)
                ranges = np.linalg.norm(xyz, axis=-1)
                valid = np.isfinite(xyz).all(axis=-1) & (ranges >= 0.10) & (ranges <= 3.0)

                receipt = time.monotonic()
                if last_receipt > 0.0 and receipt > last_receipt:
                    instant_rate = 1.0 / (receipt - last_receipt)
                    rate_ema = (
                        instant_rate
                        if rate_ema <= 0.0
                        else 0.9 * rate_ema + 0.1 * instant_rate
                    )
                last_receipt = receipt
                counter += 1
                out = bufs[counter % 2]
                out.fill(0.0)
                n = min(point_count, xyz.shape[0])
                out[:n, :3] = xyz[:n]
                out[:n, 3] = valid[:n]
                out[:n, 4] = (point_ns[:n].astype(np.float64) - float(end_ns)) * 1.0e-9
                out[:n, 5] = reflectivity[:n] / 255.0
                out[:n, 6] = tags[:n] / 255.0
                sensor_stamp[0] = end_ns
                receipt_stamp[0] = receipt
                rate_hz[0] = rate_ema
                seq[0] = counter

        rclpy.init(args=None)
        node = LivoxNode()
        try:
            while rclpy.ok() and int(stop[0]) == 0:
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    except Exception:
        err_text.value = traceback.format_exc()[-8000:].encode()


class LivoxProcess:
    """Double-buffered fixed-size scans: xyz, valid, dt, reflectivity, tag."""

    point_dim = 7

    def __init__(
        self, lidar_topic: str, imu_topic: str, point_count: int = 1024,
        scan_duration_s: float = 0.10,
    ):
        ctx = mp.get_context("spawn")
        self.point_count = int(point_count)
        self.frames = torch.zeros(
            2, self.point_count, self.point_dim, dtype=torch.float32
        ).share_memory_()
        self.seq = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.receipt_stamp = torch.zeros(1, dtype=torch.float64).share_memory_()
        self.sensor_stamp = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.rate = torch.zeros(1, dtype=torch.float64).share_memory_()
        self._stop = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.err_text = ctx.Array("c", 8192)
        self._frames_np = self.frames.numpy()
        self.proc = ctx.Process(
            target=_child_main,
            args=(
                str(lidar_topic), str(imu_topic), self.point_count,
                float(scan_duration_s), self.frames, self.seq, self.receipt_stamp,
                self.sensor_stamp, self.rate, self._stop, self.err_text,
            ),
            daemon=True,
        )

    def start(self) -> None:
        self.proc.start()

    @property
    def latest_seq(self) -> int:
        return int(self.seq[0])

    @property
    def latest_scan(self):
        n = self.latest_seq
        return None if n == 0 else self._frames_np[n % 2]

    def snapshot(self):
        """Copy one coherent scan and its receipt timestamps from shared memory."""
        for _ in range(4):
            seq_before = self.latest_seq
            if seq_before == 0:
                return None, 0, float("inf"), 0.0, 0
            receipt = float(self.receipt_stamp[0])
            sensor = int(self.sensor_stamp[0])
            scan = self._frames_np[seq_before % 2].copy()
            if seq_before == self.latest_seq:
                age = time.monotonic() - receipt
                return scan, seq_before, age, receipt, sensor
        raise RuntimeError("Livox shared scan changed repeatedly while being copied")

    @property
    def age_s(self) -> float:
        stamp = float(self.receipt_stamp[0])
        return float("inf") if stamp <= 0.0 else time.monotonic() - stamp

    @property
    def receipt_time(self) -> float:
        return float(self.receipt_stamp[0])

    @property
    def rate_hz(self) -> float:
        return float(self.rate[0])

    @property
    def error(self) -> str:
        return self.err_text.value.decode(errors="replace")

    @property
    def running(self) -> bool:
        return self.proc.is_alive()

    def stop(self) -> None:
        self._stop[0] = 1
        self.proc.join(timeout=2.0)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=1.0)
