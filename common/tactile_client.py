from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray


SENSOR_TOPIC_CONFIG = {
    "LF": {
        "role": "left_foot",
    },
    "RF": {
        "role": "right_foot",
    },
    "LH": {
        "role": "left_hand",
    },
    "RH": {
        "role": "right_hand",
    },
    "WB": {
        "role": "upper_body",
    },
}


@dataclass
class TactileSample:
    sensor: str
    raw_sensor_bytes: bytes = b""
    raw_gyro_bytes: bytes = b""


class TactileClient:
    def __init__(
        self,
        sensor_names: list[str],
        node_name: str = "deploy_tactile_client",
    ):
        self.sensor_names = [name for name in sensor_names if name in SENSOR_TOPIC_CONFIG]
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self.node = Node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.samples: Dict[str, TactileSample] = {
            sensor: TactileSample(sensor=sensor)
            for sensor in self.sensor_names
        }

        for sensor in self.sensor_names:
            role = SENSOR_TOPIC_CONFIG[sensor]["role"]
            prefix = f"/jq_tactile/{role}/{sensor}"

            self.node.create_subscription(
                UInt8MultiArray,
                f"{prefix}/raw_sensor_bytes",
                self._make_raw_cb(sensor),
                10,
            )
            self.node.create_subscription(
                UInt8MultiArray,
                f"{prefix}/raw_gyro_bytes",
                self._make_gyro_cb(sensor),
                10,
            )

        self.thread = threading.Thread(target=self._spin, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _spin(self) -> None:
        while self.running and rclpy.ok():
            self.executor.spin_once(timeout_sec=0.1)

    def _make_raw_cb(self, sensor: str):
        def cb(msg: UInt8MultiArray) -> None:
            with self._lock:
                sample = self.samples[sensor]
                sample.raw_sensor_bytes = bytes(msg.data)
                self.frame_count += 1

        return cb

    def _make_gyro_cb(self, sensor: str):
        def cb(msg: UInt8MultiArray) -> None:
            with self._lock:
                self.samples[sensor].raw_gyro_bytes = bytes(msg.data)

        return cb

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=2.0)
        self.executor.shutdown()
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def snapshot(self) -> Dict[str, dict]:
        snapshot: Dict[str, dict] = {}

        with self._lock:
            samples = {
                sensor: TactileSample(
                    sensor=sample.sensor,
                    raw_sensor_bytes=sample.raw_sensor_bytes,
                    raw_gyro_bytes=sample.raw_gyro_bytes,
                )
                for sensor, sample in self.samples.items()
            }

        for sensor in self.sensor_names:
            sample = samples[sensor]
            snapshot[sensor] = {
                "raw_sensor_bytes": sample.raw_sensor_bytes,
                "raw_gyro_bytes": sample.raw_gyro_bytes,
            }

        return snapshot
