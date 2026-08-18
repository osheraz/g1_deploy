"""Reusable suspended G1 23-DoF MuJoCo plants.

The legacy/default fixture holds the free root rigidly.  The review-gated
calibration uses two compliant points on the torso near the shoulder straps,
which preserves gravity-driven tilt and sway while keeping the feet clear.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .joint_mapping import JointCalibrationProfile
from .protocol import MOTOR_NAMES, PRESENT_23
from policy.sonic.robot_profile import MUJOCO_JOINT_NAMES


def load_faults(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text())
    if payload.get("schema") != "e_skin.synthetic_joint_faults.v1":
        raise ValueError("unsupported fault profile")
    faults = {}
    for key, value in payload.get("joints", {}).items():
        motor = int(key)
        if motor not in PRESENT_23:
            raise ValueError(f"fault targets absent motor {motor}")
        faults[motor] = {
            "gain": float(value.get("gain", 1.0)),
            "bias_rad": float(value.get("bias_rad", 0.0)),
            "direction_bias_rad": float(value.get("direction_bias_rad", 0.0)),
            "lag_s": float(value.get("lag_s", 0.0)),
        }
        if faults[motor]["gain"] <= 0 or faults[motor]["lag_s"] < 0:
            raise ValueError(f"invalid fault parameters for motor {motor}")
    return faults


def _set_pd(model, actuator: int, kp: float, kd: float, effort: float) -> None:
    model.actuator_gaintype[actuator] = 0
    model.actuator_gainprm[actuator, 0] = kp
    model.actuator_biastype[actuator] = 1
    model.actuator_biasprm[actuator, 0] = 0.0
    model.actuator_biasprm[actuator, 1] = -kp
    model.actuator_biasprm[actuator, 2] = -kd
    model.actuator_ctrllimited[actuator] = 0
    if effort > 0:
        model.actuator_forcerange[actuator] = [-effort, effort]
        model.actuator_forcelimited[actuator] = 1


class SuspendedMujocoPlant:
    def __init__(
        self,
        *,
        mjcf: Path,
        config: Path,
        fault_profile: Path | None = None,
        calibration_profile: Path | None = None,
        control_dt: float = 0.02,
        suspension: str = "fixed_pelvis",
        root_quat_wxyz: np.ndarray | None = None,
        shoulder_stiffness_n_m: float = 3200.0,
        shoulder_damping_ns_m: float = 180.0,
        shoulder_angular_damping_nms: float = 60.0,
    ):
        if suspension not in {"fixed_pelvis", "two_shoulder"}:
            raise ValueError("suspension must be fixed_pelvis or two_shoulder")
        if (
            shoulder_stiffness_n_m <= 0
            or shoulder_damping_ns_m < 0
            or shoulder_angular_damping_nms < 0
        ):
            raise ValueError("shoulder suspension stiffness/damping is invalid")
        self.control_dt = float(control_dt)
        self.suspension = suspension
        self.shoulder_stiffness_n_m = float(shoulder_stiffness_n_m)
        self.shoulder_damping_ns_m = float(shoulder_damping_ns_m)
        self.shoulder_angular_damping_nms = float(shoulder_angular_damping_nms)
        self.model = mujoco.MjModel.from_xml_path(str(mjcf.expanduser()))
        self.data = mujoco.MjData(self.model)
        cfg = yaml.safe_load(config.expanduser().read_text())
        self.neutral = np.asarray(cfg["default_angles"], dtype=np.float64)
        self.kp = np.asarray(cfg["kps"], dtype=np.float64)
        self.kd = np.asarray(cfg["kds"], dtype=np.float64)
        self.profile = (
            JointCalibrationProfile.load(calibration_profile)
            if calibration_profile else None
        )
        self.faults = load_faults(fault_profile)
        self.present = np.asarray(PRESENT_23, dtype=np.int64)
        self.joint_ids = np.asarray([
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT,
                MUJOCO_JOINT_NAMES[motor],
            )
            for motor in range(29)
        ])
        if np.any(self.joint_ids < 0):
            missing = [MOTOR_NAMES[i] for i in np.flatnonzero(self.joint_ids < 0)]
            raise RuntimeError(f"23-DoF compatibility MJCF is missing joints: {missing}")
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        actuator_by_joint = {
            int(self.model.actuator_trnid[a, 0]): a for a in range(self.model.nu)
        }
        self.actuators = np.asarray([
            actuator_by_joint[int(joint)] for joint in self.joint_ids
        ])
        for motor, joint, actuator in zip(range(29), self.joint_ids, self.actuators):
            effort = (
                float(self.model.jnt_actfrcrange[joint, 1])
                if self.model.jnt_actfrclimited[joint] else 0.0
            )
            _set_pd(self.model, int(actuator), self.kp[motor], self.kd[motor], effort)
        self.root_position = np.asarray([0.0, 0.0, 1.20], dtype=np.float64)
        self.root_quat = np.asarray(
            [1.0, 0.0, 0.0, 0.0] if root_quat_wxyz is None else root_quat_wxyz,
            dtype=np.float64,
        )
        if self.root_quat.shape != (4,) or not np.isfinite(self.root_quat).all():
            raise ValueError("root quaternion must be a finite wxyz vector")
        quat_norm = np.linalg.norm(self.root_quat)
        if quat_norm < 1e-8:
            raise ValueError("root quaternion has zero norm")
        self.root_quat /= quat_norm
        self.torso_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.foot_bodies = np.asarray([
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
            ),
        ])
        if self.torso_body < 0 or np.any(self.foot_bodies < 0):
            raise RuntimeError("suspension model is missing torso or foot bodies")
        self.shoulder_local_points = np.asarray([
            [0.004, 0.145, 0.220],
            [0.004, -0.145, 0.220],
        ], dtype=np.float64)
        self.shoulder_anchors = np.zeros((2, 3), dtype=np.float64)
        self.shoulder_points = np.zeros((2, 3), dtype=np.float64)
        self.last_suspension_forces = np.zeros((2, 3), dtype=np.float64)
        self.decimation = max(1, round(self.control_dt / self.model.opt.timestep))
        self.filtered_effective = self.neutral.copy()
        self.previous_desired = self.neutral.copy()
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.root_position
        self.data.qpos[3:7] = self.root_quat
        self.data.qpos[self.qadr] = self.neutral
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.actuators] = self.neutral
        self.filtered_effective = self.neutral.copy()
        self.previous_desired = self.neutral.copy()
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.torso_body].reshape(3, 3)
        self.shoulder_points = (
            self.data.xpos[self.torso_body]
            + (rotation @ self.shoulder_local_points.T).T
        )
        equilibrium_extension = (
            np.sum(self.model.body_mass) * abs(float(self.model.opt.gravity[2]))
            / (2.0 * self.shoulder_stiffness_n_m)
        )
        self.shoulder_anchors = self.shoulder_points.copy()
        self.shoulder_anchors[:, 2] += equilibrium_extension

    def _apply_shoulder_fixture(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        rotation = self.data.xmat[self.torso_body].reshape(3, 3)
        origin = np.asarray(self.data.xpos[self.torso_body], dtype=np.float64)
        points = origin + (rotation @ self.shoulder_local_points.T).T
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.torso_body, velocity, 0,
        )
        omega, linear = velocity[:3], velocity[3:]
        forces = np.zeros((2, 3), dtype=np.float64)
        for index, point in enumerate(points):
            point_velocity = linear + np.cross(omega, point - origin)
            force = (
                self.shoulder_stiffness_n_m * (self.shoulder_anchors[index] - point)
                - self.shoulder_damping_ns_m * point_velocity
            )
            # A finite cap prevents a numerical launch if an experimental
            # initial attitude is grossly inconsistent with the fixture.
            magnitude = np.linalg.norm(force)
            if magnitude > 600.0:
                force *= 600.0 / magnitude
            mujoco.mj_applyFT(
                self.model, self.data, force, np.zeros(3), point,
                self.torso_body, self.data.qfrc_applied,
            )
            forces[index] = force
        angular_drag = -self.shoulder_angular_damping_nms * omega
        drag_norm = np.linalg.norm(angular_drag)
        if drag_norm > 120.0:
            angular_drag *= 120.0 / drag_norm
        mujoco.mj_applyFT(
            self.model, self.data, np.zeros(3), angular_drag, origin,
            self.torso_body, self.data.qfrc_applied,
        )
        self.shoulder_points = points
        self.last_suspension_forces = forces

    def step(self, target_model: np.ndarray) -> dict[str, np.ndarray]:
        desired = np.asarray(target_model, dtype=np.float64)
        if desired.shape != (29,) or not np.isfinite(desired).all():
            raise ValueError("target_model must be a finite 29-vector")
        direction = np.sign(desired - self.previous_desired)
        command = desired.copy() if self.profile is None else self.profile.apply_29(
            desired, direction
        )
        effective = command.copy()
        for motor, fault in self.faults.items():
            delta = command[motor] - self.neutral[motor]
            raw = (
                self.neutral[motor] + fault["gain"] * delta
                + fault["bias_rad"]
                + np.sign(direction[motor]) * fault["direction_bias_rad"]
            )
            tau = fault["lag_s"]
            alpha = 1.0 if tau <= 0 else self.control_dt / (tau + self.control_dt)
            self.filtered_effective[motor] += alpha * (
                raw - self.filtered_effective[motor]
            )
            effective[motor] = self.filtered_effective[motor]
        self.data.ctrl[self.actuators] = effective
        for _ in range(self.decimation):
            if self.suspension == "fixed_pelvis":
                self.data.qpos[:3] = self.root_position
                self.data.qpos[3:7] = self.root_quat
                self.data.qvel[:6] = 0.0
                self.data.qfrc_applied[:] = 0.0
            else:
                self._apply_shoulder_fixture()
            mujoco.mj_step(self.model, self.data)
            if self.suspension == "fixed_pelvis":
                self.data.qpos[:3] = self.root_position
                self.data.qpos[3:7] = self.root_quat
                self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.previous_desired = desired.copy()
        q = np.asarray(self.data.qpos[self.qadr], dtype=np.float64).copy()
        dq = np.asarray(self.data.qvel[self.vadr], dtype=np.float64).copy()
        tau = np.asarray(self.data.actuator_force[self.actuators], dtype=np.float64).copy()
        contact_pairs = []
        for contact in self.data.contact[:self.data.ncon]:
            geom1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            ) or f"geom_{int(contact.geom1)}"
            geom2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            ) or f"geom_{int(contact.geom2)}"
            contact_pairs.append(f"{geom1}|{geom2}")
        return {
            "target_model": desired.copy(),
            "command_hardware": command.copy(),
            "effective_model_target": effective.copy(),
            "q_measured": q,
            "dq_measured": dq,
            "tau_est": tau,
            "direction": direction,
            "root_position": np.asarray(self.data.qpos[:3], dtype=np.float64).copy(),
            "root_quaternion": np.asarray(self.data.qpos[3:7], dtype=np.float64).copy(),
            "root_linear_velocity": np.asarray(self.data.qvel[:3], dtype=np.float64).copy(),
            "root_angular_velocity": np.asarray(self.data.qvel[3:6], dtype=np.float64).copy(),
            "imu_quaternion": np.asarray(self.data.qpos[3:7], dtype=np.float64).copy(),
            "imu_gyroscope": np.asarray(self.data.qvel[3:6], dtype=np.float64).copy(),
            "imu_accelerometer": self._sensor("imu_acc", 3),
            "shoulder_points": self.shoulder_points.copy(),
            "shoulder_anchors": self.shoulder_anchors.copy(),
            "suspension_forces": self.last_suspension_forces.copy(),
            "foot_link_height": np.asarray(
                self.data.xpos[self.foot_bodies, 2], dtype=np.float64
            ).copy(),
            "contact_count": np.asarray(self.data.ncon, dtype=np.int64),
            "contact_pairs": np.asarray(";".join(contact_pairs)),
        }

    def _sensor(self, name: str, expected_dim: int) -> np.ndarray:
        sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor < 0:
            return np.full(expected_dim, np.nan, dtype=np.float64)
        start = int(self.model.sensor_adr[sensor])
        dim = int(self.model.sensor_dim[sensor])
        if dim != expected_dim:
            raise RuntimeError(f"sensor {name} has dimension {dim}, expected {expected_dim}")
        return np.asarray(self.data.sensordata[start:start + dim], dtype=np.float64).copy()

    def joint_anchor(self, motor: int) -> np.ndarray:
        return np.asarray(self.data.xanchor[self.joint_ids[motor]], np.float64).copy()
