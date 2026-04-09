from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray


SENSOR_TOPIC_CONFIG = {
    "LF": "left_foot",
    "RF": "right_foot",
    "LH": "left_hand",
    "RH": "right_hand",
    "WB": "upper_body",
}


class TactileBridgeNode(Node):
    def __init__(self, sensor_names: list[str]):
        super().__init__("tactile_bridge")
        self.sensor_names = [name for name in sensor_names if name in SENSOR_TOPIC_CONFIG]
        self.frame_count = 0
        self.samples = {
            sensor: {
                "raw_len": 0,
                "gyro_len": 0,
                "last_rx_time": 0.0,
                "raw_values": [],
            }
            for sensor in self.sensor_names
        }

        for sensor in self.sensor_names:
            role = SENSOR_TOPIC_CONFIG[sensor]
            prefix = f"/jq_tactile/{role}/{sensor}"
            self.create_subscription(
                UInt8MultiArray,
                f"{prefix}/raw_sensor_bytes",
                self._make_raw_cb(sensor),
                10,
            )
            self.create_subscription(
                UInt8MultiArray,
                f"{prefix}/raw_gyro_bytes",
                self._make_gyro_cb(sensor),
                10,
            )

    def _make_raw_cb(self, sensor: str):
        def cb(msg: UInt8MultiArray):
            self.samples[sensor]["raw_len"] = len(msg.data)
            self.samples[sensor]["last_rx_time"] = time.time()
            self.samples[sensor]["raw_values"] = list(msg.data[:256])
            self.frame_count += 1

        return cb

    def _make_gyro_cb(self, sensor: str):
        def cb(msg: UInt8MultiArray):
            self.samples[sensor]["gyro_len"] = len(msg.data)
            self.samples[sensor]["last_rx_time"] = time.time()

        return cb

    def snapshot(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "samples": self.samples,
            "timestamp": time.time(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", action="append", dest="sensors", default=[])
    parser.add_argument("--period", type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = TactileBridgeNode(args.sensors)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    next_emit = time.time()

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            now = time.time()
            if now >= next_emit:
                print(json.dumps(node.snapshot()), flush=True)
                next_emit = now + args.period
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
