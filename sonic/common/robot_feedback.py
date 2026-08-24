"""Official-compatible ``g1_debug`` feedback for the PICO manager.

The official PICO process polls ``body_q_measured`` when entering VR3PT and
uses it for wrist forward-kinematics calibration.  A 23-DoF G1 is expanded to
the release model's 29-DoF MuJoCo ordering with its six absent joints fixed at
zero.
"""
from __future__ import annotations

import time

import msgpack
import numpy as np
import zmq

from sonic.model import robot_g1
from sonic.model.robot_profile import get_robot_profile


class RobotFeedbackPublisher:
    """Non-blocking PUB using NVIDIA's ``[topic][msgpack map]`` wire format."""

    def __init__(self, port: int = 5558, topic: str = "g1_debug", robot_profile="g1_23dof"):
        self.robot_profile = get_robot_profile(robot_profile)
        self.port = int(port)
        self.topic = topic.encode("utf-8")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 10)
        self.socket.setsockopt(zmq.SNDBUF, 32768)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{self.port}")
        self.index = 0
        self.dropped = 0
        print(f"PICO feedback: tcp://*:{self.port} topic={self.topic.decode()}")

    def expand_to_mujoco(self, values) -> np.ndarray:
        return self.robot_profile.physical_to_mujoco(
            np.asarray(values, np.float64), fill=0.0
        )

    def publish(
        self,
        *,
        q_physical,
        dq_physical,
        command_q_physical,
        base_quat_wxyz,
        base_ang_vel,
    ) -> None:
        measured_q = self.expand_to_mujoco(q_physical)
        measured_dq = self.expand_to_mujoco(dq_physical)
        command_q = self.expand_to_mujoco(command_q_physical)
        zeros7 = [0.0] * 7
        data = {
            "control_loop_type": f"python_{self.robot_profile.name}",
            "index": self.index,
            "monotonic_timestamp": time.monotonic(),
            "base_quat": np.asarray(base_quat_wxyz, np.float64).reshape(4).tolist(),
            "base_ang_vel": np.asarray(base_ang_vel, np.float64).reshape(3).tolist(),
            "body_q": measured_q.tolist(),
            "body_dq": measured_dq.tolist(),
            "last_action": command_q.tolist(),
            "body_q_measured": measured_q.tolist(),
            "left_hand_q_measured": zeros7,
            "right_hand_q_measured": zeros7,
        }
        self.index += 1
        message = self.topic + msgpack.packb(data, use_bin_type=True)
        try:
            self.socket.send(message, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.dropped += 1

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()
