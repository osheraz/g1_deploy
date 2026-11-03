"""Combine packed Livox clouds into longer DLIO sweeps, preserving point times.

Three 30 Hz packets give approximately 0.1 s sweeps at 10 Hz. This is the
estimator input, independent of the planner's rolling-window/FOV contract.
DLIO must deskew the resulting sweep using the original per-point timestamps.
"""

import argparse
import copy
from array import array


def join_cloud_packets(messages):
    if not messages:
        raise ValueError("No clouds")
    first = messages[0]
    signature = lambda m: (
        m.header.frame_id,
        m.is_bigendian,
        m.point_step,
        [(f.name, f.offset, f.datatype, f.count) for f in m.fields],
    )
    for msg in messages:
        if (
            msg.height != 1
            or msg.row_step != msg.point_step * msg.width
            or len(msg.data) != msg.row_step
        ):
            raise ValueError("Expected packed one-row Livox cloud")
        if signature(msg) != signature(first):
            raise ValueError("Cloud layouts or frames changed within sweep")
    out = copy.deepcopy(first)
    out.width = sum(m.width for m in messages)
    out.row_step = out.width * out.point_step
    out.is_dense = all(m.is_dense for m in messages)
    # ROS's bytes setter validates each byte in Python. Its typed-array path
    # validates the element type once; the sensor bytes themselves are identical.
    out.data = array("B", b"".join(bytes(m.data) for m in messages))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-topic", default="/livox/lidar_filtered")
    p.add_argument("--output-topic", default="/livox/lidar_dlio")
    p.add_argument("--packets", type=int, default=3)
    a = p.parse_args()
    if a.input_topic == a.output_topic or not 1 <= a.packets <= 10:
        p.error("Different topics and 1..10 packets required")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2

    rclpy.init()
    node = rclpy.create_node("hv2_livox_dlio_window")
    # DLIO requests reliable clouds, unlike its sensor-data IMU subscription.
    pub = node.create_publisher(PointCloud2, a.output_topic, 10)
    pending = []

    def callback(msg):
        pending.append(msg)
        if len(pending) >= a.packets:
            pub.publish(join_cloud_packets(pending))
            pending.clear()

    node.create_subscription(PointCloud2, a.input_topic, callback, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
