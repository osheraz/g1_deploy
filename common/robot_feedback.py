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

from policy.sonic import robot_g1


class RobotFeedbackPublisher:
    """Non-blocking PUB using NVIDIA's ``[topic][msgpack map]`` wire format."""

    def __init__(self, port: int = 5558, topic: str = "g1_debug"):
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

    @staticmethod
    def expand_23_to_mujoco(values_23) -> np.ndarray:
        values_il = np.zeros(robot_g1.NUM_DOF, dtype=np.float64)
        values_il[robot_g1.IL_FROM_OUR23] = np.asarray(values_23, np.float64).reshape(23)
        return robot_g1.il_to_mj(values_il)

    def publish(
        self,
        *,
        q_23,
        dq_23,
        command_q_23,
        base_quat_wxyz,
        base_ang_vel,
    ) -> None:
        measured_q = self.expand_23_to_mujoco(q_23)
        measured_dq = self.expand_23_to_mujoco(dq_23)
        command_q = self.expand_23_to_mujoco(command_q_23)
        zeros7 = [0.0] * 7
        data = {
            "control_loop_type": "python23",
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
