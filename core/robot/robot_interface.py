from __future__ import annotations

import threading
from collections import deque
from typing import Union
import time
import copy
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__IMUState_,
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_ as IMUStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from core.common.command_helper import (
    create_zero_cmd,
    init_cmd_hg,
    init_cmd_go,
    MotorMode,
)
from core.common.rotation_helper import get_gravity_orientation, transform_imu_data
from core.common.remote_controller import RemoteController, KeyMap
from core.config import Config
from sonic.model.robot_profile import get_robot_profile


class RobotInterface:
    def __init__(self, config: Config):
        self.config = config
        self.robot_profile = get_robot_profile(config.sonic_robot_profile)
        self.remote_controller = RemoteController()

        self.low_cmd = None
        self.low_state = None
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.last_target_unclipped = config.physical_default_angles.copy()
        self.last_target_dof_pos = config.physical_default_angles.copy()
        self.last_target_saturated = np.zeros(config.num_actions, dtype=bool)
        self._low_state_monotonic = 0.0
        self._low_state_history = deque(maxlen=256)
        self._torso_imu = None
        self._torso_imu_monotonic = 0.0
        self._torso_imu_history = deque(maxlen=512)
        self._state_lock = threading.Lock()
        self.rejected_crc_count = 0

        if config.msg_type == "hg":
            self._init_hg()
        elif config.msg_type == "go":
            self._init_go()
        else:
            raise ValueError("Invalid msg_type")

        self.wait_for_low_state()
        if self.config.require_torso_imu:
            self.wait_for_torso_imu()
        if config.msg_type == "hg" and config.policy_type == "sonic":
            # Refuse a selected 23/29 command map before _init_low_cmd() or any
            # moving/default-pose target can be published to the wrong plant.
            self.robot_profile.require_mode_machine(self.mode_machine_)
            print(
                f"Validated {self.robot_profile.name} against "
                f"LowState mode_machine={self.mode_machine_}."
            )
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
        if self.config.require_torso_imu or self.config.imu_type == "torso":
            self._torso_imu = unitree_hg_msg_dds__IMUState_()
            self.torso_imu_subscriber = ChannelSubscriber(
                self.config.torso_imu_topic, IMUStateHG
            )
            self.torso_imu_subscriber.Init(self._torso_imu_handler, 50)

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
        if self.config.validate_low_state_crc:
            received = int(msg.crc)
            calculated = int(CRC().Crc(copy.deepcopy(msg)))
            if received != calculated:
                self.rejected_crc_count += 1
                return
        with self._state_lock:
            self.low_state = msg
            self._low_state_monotonic = time.monotonic()
            self._low_state_history.append((self._low_state_monotonic, msg))
            self.mode_machine_ = msg.mode_machine
        self.remote_controller.set(msg.wireless_remote)

    def _low_state_go_handler(self, msg: LowStateGo):
        with self._state_lock:
            self.low_state = msg
            self._low_state_monotonic = time.monotonic()
            self._low_state_history.append((self._low_state_monotonic, msg))
        self.remote_controller.set(msg.wireless_remote)

    def _torso_imu_handler(self, msg: IMUStateHG):
        now = time.monotonic()
        with self._state_lock:
            self._torso_imu = msg
            self._torso_imu_monotonic = now
            self._torso_imu_history.append((now, msg))

    def send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def wait_for_low_state(self):
        while self.low_state_snapshot().tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def wait_for_torso_imu(self):
        deadline = time.monotonic() + self.config.torso_imu_startup_timeout_s
        while True:
            _, age = self.torso_imu_snapshot_with_age()
            if np.isfinite(age):
                print(f"Torso IMU live on {self.config.torso_imu_topic}.")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"No torso IMU on {self.config.torso_imu_topic!r} after "
                    f"{self.config.torso_imu_startup_timeout_s:.1f} s. "
                    "The 29-DoF LiDAR deployment forbids pelvis fallback."
                )
            time.sleep(0.01)

    def low_state_snapshot(self):
        """Return one coherent SDK message reference for a control-step read."""
        with self._state_lock:
            return self.low_state

    def low_state_snapshot_with_age(self):
        """Return one message and its monotonic age from the same lock scope."""
        with self._state_lock:
            state = self.low_state
            timestamp = self._low_state_monotonic
        age = float("inf") if timestamp <= 0.0 else time.monotonic() - timestamp
        return state, age

    def torso_imu_snapshot_with_age(self):
        with self._state_lock:
            imu = self._torso_imu
            timestamp = self._torso_imu_monotonic
        age = float("inf") if timestamp <= 0.0 else time.monotonic() - timestamp
        return imu, age

    def sensor_snapshot_with_age(self):
        """One lock acquisition for the controller's pelvis and torso samples."""
        now = time.monotonic()
        with self._state_lock:
            state, state_t = self.low_state, self._low_state_monotonic
            torso, torso_t = self._torso_imu, self._torso_imu_monotonic
        state_age = float("inf") if state_t <= 0.0 else now - state_t
        torso_age = float("inf") if torso_t <= 0.0 else now - torso_t
        return state, state_age, torso, torso_age

    def _nearest_history_sample(self, history, target_monotonic: float):
        with self._state_lock:
            samples = tuple(history)
        if not samples:
            return None, float("inf")
        stamp, sample = min(samples, key=lambda item: abs(item[0] - target_monotonic))
        return sample, abs(float(stamp) - float(target_monotonic))

    def low_state_nearest(self, target_monotonic: float):
        return self._nearest_history_sample(self._low_state_history, target_monotonic)

    def torso_imu_nearest(self, target_monotonic: float):
        return self._nearest_history_sample(self._torso_imu_history, target_monotonic)

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

        state = self.low_state_snapshot()
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i, motor_idx in enumerate(dof_idx):
            init_dof_pos[i] = state.motor_state[motor_idx].q

        for i in range(num_step):
            state, age = self.low_state_snapshot_with_age()
            if age > self.config.max_low_state_age_s:
                raise RuntimeError(
                    f"LowState became stale while moving to default ({age:.3f}s)"
                )
            max_dq = max(
                abs(float(state.motor_state[motor_idx].dq)) for motor_idx in dof_idx
            )
            if max_dq > self.config.max_joint_velocity:
                raise RuntimeError(
                    f"measured joint velocity {max_dq:.2f}rad/s while moving to "
                    f"default exceeds {self.config.max_joint_velocity:.2f}rad/s"
                )
            alpha = i / num_step
            for j, motor_idx in enumerate(dof_idx):
                target_pos = default_pos[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                )
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd()
            time.sleep(self.config.control_dt)

    def default_pos_state(self, start_button=KeyMap.A, button_name="A"):
        print("Enter default pos state.")
        print(f"Waiting for the Button {button_name} signal...")
        soft_age_limit = float(self.config.max_low_state_age_s)
        hold_age_limit = float(
            getattr(
                self.config,
                "max_default_hold_low_state_age_s",
                max(soft_age_limit, 0.5),
            )
        )
        stale_warning_active = False
        while self.remote_controller.button[start_button] != 1:
            _, age = self.low_state_snapshot_with_age()
            if age > hold_age_limit:
                raise RuntimeError(
                    "LowState was continuously unavailable in default-pose "
                    f"hold for {age:.3f}s (limit {hold_age_limit:.3f}s)"
                )
            if age > soft_age_limit:
                if not stale_warning_active:
                    print(
                        "[LOWSTATE] Brief startup gap "
                        f"({age * 1e3:.0f}ms > {soft_age_limit * 1e3:.0f}ms); "
                        "continuing the unchanged default-pose hold"
                    )
                    stale_warning_active = True
            elif stale_warning_active:
                print("[LOWSTATE] Startup state stream recovered")
                stale_warning_active = False
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

    def update_joint_state(self, state=None):
        state = self.low_state_snapshot() if state is None else state
        for i, motor_idx in enumerate(self.config.action_joint2motor_idx):
            self.qj[i] = state.motor_state[motor_idx].q
            self.dqj[i] = state.motor_state[motor_idx].dq

    def active_joint_state(self, state=None):
        state = self.low_state_snapshot() if state is None else state
        q = np.asarray(
            [state.motor_state[i].q for i in self.config.action_joint2motor_idx],
            dtype=np.float32,
        )
        dq = np.asarray(
            [state.motor_state[i].dq for i in self.config.action_joint2motor_idx],
            dtype=np.float32,
        )
        return q, dq

    @property
    def low_state_age_s(self) -> float:
        with self._state_lock:
            timestamp = self._low_state_monotonic
        if timestamp <= 0.0:
            return float("inf")
        return time.monotonic() - timestamp

    @property
    def low_state_tick(self) -> int:
        return int(getattr(self.low_state_snapshot(), "tick", -1))

    def motor_state_arrays(self, count: int = 29, state=None) -> dict[str, np.ndarray]:
        """Raw motor-order state for diagnostics; absent fields use sentinels."""
        state = self.low_state_snapshot() if state is None else state
        states = state.motor_state
        q = np.array([states[i].q for i in range(count)], np.float32)
        dq = np.array([states[i].dq for i in range(count)], np.float32)
        temperature = np.full((count, 2), np.nan, np.float32)
        motor_error = np.full(count, -1, np.int32)
        tau_est = np.full(count, np.nan, np.float32)
        for i in range(count):
            motor = states[i]
            raw_temp = np.asarray(
                getattr(motor, "temperature", np.nan), dtype=np.float32
            ).reshape(-1)
            temperature[i, :min(2, raw_temp.size)] = raw_temp[:2]
            motor_error[i] = int(getattr(
                motor, "motorstate", getattr(motor, "lost", -1)
            ))
            tau_est[i] = float(getattr(motor, "tau_est", np.nan))
        return {
            "q": q,
            "dq": dq,
            "temperature": temperature,
            "motor_error": motor_error,
            "tau_est": tau_est,
        }

    def commanded_q_29(self) -> np.ndarray:
        return np.array([self.low_cmd.motor_cmd[i].q for i in range(29)], np.float32)

    def get_imu(self, state=None):
        state = self.low_state_snapshot() if state is None else state
        quat = state.imu_state.quaternion
        ang_vel = np.array(state.imu_state.gyroscope, dtype=np.float32)

        if self.config.imu_type == "torso":
            torso, age = self.torso_imu_snapshot_with_age()
            if torso is not None and age <= self.config.max_torso_imu_age_s:
                return (
                    np.asarray(torso.quaternion, dtype=np.float32),
                    np.asarray(torso.gyroscope, dtype=np.float32),
                )
            if self.config.require_torso_imu:
                raise RuntimeError(f"Torso IMU stale for {age:.3f} s")
            # Legacy 23-DoF approximation retained only for old configs.
            waist_idx = 12  # waist yaw; fixed[0] is waist roll on the 23-DoF profile
            waist_yaw = state.motor_state[waist_idx].q
            waist_yaw_omega = state.motor_state[waist_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel,
            )
        return quat, ang_vel

    def get_torso_imu(self, sample=None):
        """Return the physical secondary IMU. There is deliberately no fallback."""
        if sample is None:
            sample, age = self.torso_imu_snapshot_with_age()
        else:
            age = 0.0
        if sample is None:
            raise RuntimeError("Torso IMU has not produced a sample")
        if age > self.config.max_torso_imu_age_s:
            raise RuntimeError(f"Torso IMU stale for {age:.3f} s")
        return (
            np.asarray(sample.quaternion, dtype=np.float32),
            np.asarray(sample.gyroscope, dtype=np.float32),
            np.asarray(sample.accelerometer, dtype=np.float32),
        )

    def get_base_features_scaled(self, imu=None):

        qj = self.qj.copy()
        dqj = self.dqj.copy()
        quat, ang_vel = self.get_imu() if imu is None else imu

        projected_gravity = get_gravity_orientation(quat)
        dof_pos = (qj - self.config.physical_default_angles) * self.config.dof_pos_scale
        dof_vel = dqj * self.config.dof_vel_scale
        base_ang_vel = ang_vel * self.config.ang_vel_scale

        return base_ang_vel, projected_gravity, dof_pos, dof_vel

    def apply_target_dof_pos(self, target_dof_pos: np.ndarray):
        # Match the NVIDIA C++ reference: pass the policy's scaled position
        # targets directly to the active motors. Joint limits and target-rate
        # shaping here changed the learned closed loop and made the Python and
        # C++ deployments behave differently.
        target_dof_pos = np.asarray(target_dof_pos, dtype=np.float32).copy()
        if target_dof_pos.shape != self.config.physical_default_angles.shape:
            raise ValueError(
                f"expected {self.config.physical_default_angles.shape} target, "
                f"got {target_dof_pos.shape}"
            )
        if not np.all(np.isfinite(target_dof_pos)):
            raise RuntimeError("policy produced a non-finite joint target")
        self.last_target_unclipped = target_dof_pos.copy()
        self.last_target_dof_pos = target_dof_pos.copy()
        self.last_target_saturated = np.zeros_like(target_dof_pos, dtype=bool)
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
        return target_dof_pos.copy()
