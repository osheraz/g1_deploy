"""Publish live ROS or saved LiDAR scans to the asynchronous Viser preview.

No motor publishers. Live mode subscribes to an already-running Livox driver;
replay mode never imports ROS or connects to the robot. The protocol explicitly
marks this as diagnostic data rather than a qualified planner observation.
"""

if __name__ == "__main__" and not __package__:
    import _bootstrap  # System Python workers use this checkout, not the policy virtualenv.
from rise.paths import ROOT
import argparse
from pathlib import Path
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, str(ROOT / "g1_deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sensing.lidar_stream import encode_scan


def replay(args, publish):
    with np.load(args.capture / "sensors.npz") as source:
        data = {k: source[k] for k in source.files}
    times = data["scan_receipt_monotonic_s"]
    started = time.monotonic()
    seq = 0
    while True:
        session = str(uuid.uuid4())
        begin = time.monotonic()
        for k, t in enumerate(times):
            target = begin + (t - times[0]) / args.rate
            while time.monotonic() < target:
                if args.duration_s and time.monotonic() - started >= args.duration_s:
                    return
                time.sleep(min(0.02, target - time.monotonic()))
            if args.duration_s and time.monotonic() - started >= args.duration_s:
                return
            state_index = int(np.argmin(abs(data["state_receipt_monotonic_s"] - t)))
            state = dict(
                q=data["q"][state_index].tolist(),
                quat_wxyz=data["pelvis_quat_wxyz"][state_index].tolist(),
                tick=int(data["state_tick_ms"][state_index]),
                mode_machine=int(data["mode_machine"][state_index]),
                pair_error_s=float(data["state_receipt_monotonic_s"][state_index] - t),
            )
            if "torso_quat_wxyz" in data:
                torso_index = int(np.argmin(abs(data["torso_receipt_monotonic_s"] - t)))
                state["torso_quat_wxyz"] = data["torso_quat_wxyz"][torso_index].tolist()
            lo, hi = data["scan_offsets"][k : k + 2]
            seq += 1
            now = time.monotonic()
            publish(
                dict(
                    session=session,
                    sequence=seq,
                    source="replay",
                    frame="mid360_link",
                    units="m",
                    sensor_stamp_ns=int(data["scan_stamp_ns"][k]),
                    source_receipt_monotonic_s=float(t),
                    publish_monotonic_s=now,
                    capture_monotonic_s=None,
                    clock_qualified=False,
                    recorded_time_s=float(t - times[0]),
                    state=state,
                ),
                data["points"][lo:hi],
                np.nan_to_num(data["point_timestamp_ns"][lo:hi], nan=0.0).astype(np.int64),
            )
        if not args.loop:
            return


def live(args, publish):
    import os

    if int(os.environ.get("ROS_DOMAIN_ID", "0")) == 0:
        raise ValueError("Set a nonzero ROS_DOMAIN_ID; Unitree uses domain 0")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2, Imu
    from rise.sensing.pelvis_odometry import LowStateReader, LidarClock
    from rise.tools.sensors.record_lidar_live import cloud_array

    reader = LowStateReader(args.interface)
    clock = LidarClock()
    session = str(uuid.uuid4())
    seq = 0
    rclpy.init()
    node = rclpy.create_node("hv2_lidar_preview_publisher")
    stamp = lambda msg: int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    node.create_subscription(
        Imu,
        "/livox/imu",
        lambda msg: clock.observe(stamp(msg) * 1e-9, time.monotonic()),
        qos_profile_sensor_data,
    )

    def callback(msg):
        nonlocal seq
        arrival = time.monotonic()
        data = cloud_array(msg)
        seq += 1
        points = np.stack([data[c] for c in "xyz"], axis=-1)
        stamps = (
            data["timestamp"].astype(np.int64)
            if "timestamp" in data.dtype.names
            else np.full(len(points), stamp(msg), dtype=np.int64)
        )
        end = int(stamps.max()) if len(stamps) else stamp(msg)
        capture = clock.to_monotonic(end * 1e-9) if clock.ready else None
        sample = reader.get(end * 1e-9, monotonic_s=capture) if capture is not None else None
        state = None
        if sample is not None:
            q = sample["quat_xyzw"]
            state = dict(
                q=sample["q"].tolist(),
                quat_wxyz=q[[3, 0, 1, 2]].tolist(),
                tick=sample["tick"],
                mode_machine=sample["mode_machine"],
                pair_error_s=sample["monotonic_s"] - capture,
            )
        publish(
            dict(
                session=session,
                sequence=seq,
                source="live",
                frame=msg.header.frame_id,
                units="m",
                sensor_stamp_ns=stamp(msg),
                source_receipt_monotonic_s=arrival,
                publish_monotonic_s=time.monotonic(),
                capture_monotonic_s=capture,
                clock_qualified=False,
                state=state,
                rejected_state_crc=reader.rejected_crc,
            ),
            points,
            stamps,
        )

    node.create_subscription(PointCloud2, args.topic, callback, qos_profile_sensor_data)
    started = time.monotonic()
    try:
        while not args.duration_s or time.monotonic() - started < args.duration_s:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        reader.subscriber.Close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, help="Saved real capture directory; omit for live ROS")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--rate", type=float, default=1.0)
    p.add_argument("--endpoint", default="tcp://127.0.0.1:15560")
    p.add_argument("--interface", default="enp132s0")
    p.add_argument("--topic", default="/livox/lidar")
    p.add_argument("--duration-s", type=float, default=0.0)
    a = p.parse_args()
    if not np.isfinite(a.rate) or a.rate <= 0 or not np.isfinite(a.duration_s) or a.duration_s < 0:
        p.error("Invalid rate/duration")
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(a.endpoint)
    count = 0
    last = time.monotonic()

    def publish(header, points, stamps):
        nonlocal count, last
        socket.send(encode_scan(header, points, stamps))
        count += 1
        if time.monotonic() - last > 3:
            print(
                f'LiDAR preview: {count} scans, state paired={header["state"] is not None}',
                flush=True,
            )
            last = time.monotonic()

    print(
        f'LiDAR {"replay" if a.capture else "live"} preview: {a.endpoint}; no motor commands',
        flush=True,
    )
    try:
        if a.capture:
            replay(a, publish)
        else:
            live(a, publish)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
