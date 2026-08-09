from __future__ import annotations

from typing import Union
import time
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
    create_zero_cmd,
    init_cmd_hg,
    init_cmd_go,
    MotorMode,
)
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class RobotInterface:
    def __init__(self, config: Config):
        self.config = config
        self.remote_controller = RemoteController()

        self.low_cmd = None
        self.low_state = None
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)

        if config.msg_type == "hg":
            self._init_hg()
        elif config.msg_type == "go":
            self._init_go()
        else:
            raise ValueError("Invalid msg_type")

        self.wait_for_low_state()
        self._init_low_cmd()

    # -------- init / I/O --------

    def _init_hg(self):
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher(self.config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(
            self.config.lowstate_topic, LowStateHG
        )
        self.lowstate_subscriber.Init(self._low_state_hg_handler, 10)

    def _init_go(self):
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(self.config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(
            self.config.lowstate_topic, LowStateGo
        )
        self.lowstate_subscriber.Init(self._low_state_go_handler, 10)

    def _init_low_cmd(self):
        if self.config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else:
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def _low_state_hg_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def _low_state_go_handler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # -------- modes (same as before) --------

    def zero_torque_state(self):
        print("Enter zero torque state.")
        create_zero_cmd(self.low_cmd)
        self.send_cmd()
        time.sleep(self.config.control_dt)

    def move_to_default_pos(self):

        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.action_joint2motor_idx + self.config.fixed_joint2motor_idx
        kps = np.array(self.config.kps)
        kds = np.array(self.config.kds)
        default_pos = np.array(self.config.default_angles, dtype=np.float32)

        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[i].q

        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[j].q = (
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                )
                self.low_cmd.motor_cmd[j].qd = 0.0
                self.low_cmd.motor_cmd[j].kp = kps[j]
                self.low_cmd.motor_cmd[j].kd = kds[j]
                self.low_cmd.motor_cmd[j].tau = 0.0
            self.send_cmd()
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.default_angles)):
                cmd = self.low_cmd.motor_cmd[i]
                cmd.q = self.config.default_angles[i]
                cmd.qd = 0.0
                cmd.kp = self.config.kps[i]
                cmd.kd = self.config.kds[i]
                cmd.tau = 0.0

            # create_zero_cmd(self.low_cmd)
            self.send_cmd()
            time.sleep(self.config.control_dt)

    # -------- state / features --------

    def update_joint_state(self):
        for i, motor_idx in enumerate(self.config.action_joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

    def get_imu(self):
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_idx = self.config.fixed_joint2motor_idx[0]
            waist_yaw = self.low_state.motor_state[waist_idx].q
            waist_yaw_omega = self.low_state.motor_state[waist_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel,
            )
        return quat, ang_vel

    def get_base_features_scaled(self):

        qj = self.qj.copy()
        dqj = self.dqj.copy()
        quat, ang_vel = self.get_imu()

        projected_gravity = get_gravity_orientation(quat)
        dof_pos = (qj - self.config.default_angles_23) * self.config.dof_pos_scale
        dof_vel = dqj * self.config.dof_vel_scale
        base_ang_vel = ang_vel * self.config.ang_vel_scale

        return base_ang_vel, projected_gravity, dof_pos, dof_vel

    def apply_target_dof_pos(self, target_dof_pos: np.ndarray):
        # hardware joint limits: last line of defense for every mode
        target_dof_pos = np.clip(
            target_dof_pos, self.config.dof_lower, self.config.dof_upper
        )
        for i, motor_idx in enumerate(self.config.action_joint2motor_idx):
            cmd = self.low_cmd.motor_cmd[motor_idx]
            cmd.q = target_dof_pos[i]
            cmd.qd = 0.0
            cmd.kp = self.config.kps[motor_idx]
            cmd.kd = self.config.kds[motor_idx]
            cmd.tau = 0.0

        for i, motor_idx in enumerate(self.config.fixed_joint2motor_idx):
            cmd = self.low_cmd.motor_cmd[motor_idx]
            cmd.q = self.config.fixed_target[i]
            cmd.qd = 0.0
            cmd.kp = self.config.fixed_kps[i]
            cmd.kd = self.config.fixed_kds[i]
            cmd.tau = 0.0

        self.send_cmd()
