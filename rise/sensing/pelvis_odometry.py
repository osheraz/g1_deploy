"""Pelvis odometry and local terrain grid for the recovery runtime, from the ETH sensing stack on bluefin.

Live motion: Livox -> SI IMU / grouped scans -> DLIO -> this bridge -> runtime UDP.
Body filtering and Elevation Mapping CuPy are optional, separate map paths.

Runs under the ROS 2 Humble system Python. Frames: DLIO `base_link` is the Mid-360 point-cloud frame
(`mid360_link` in the G1 URDF); the pelvis is reached through the waist joints from the Unitree low state.
Output positions are in the pelvis-IMU gravity frame (z up, yaw of the IMU's world frame), origin at DLIO start.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rise.paths import ROOT

import argparse
import json
import os
import socket
import struct
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np

REPO = ROOT
URDF = REPO / "humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf"
LIDAR_LINK = "mid360_link"
PELVIS = "pelvis"
G1_JOINTS = (
    "left_hip_pitch_joint left_hip_roll_joint left_hip_yaw_joint left_knee_joint left_ankle_pitch_joint left_ankle_roll_joint "
    "right_hip_pitch_joint right_hip_roll_joint right_hip_yaw_joint right_knee_joint right_ankle_pitch_joint right_ankle_roll_joint "
    "waist_yaw_joint waist_roll_joint waist_pitch_joint "
    "left_shoulder_pitch_joint left_shoulder_roll_joint left_shoulder_yaw_joint left_elbow_joint left_wrist_roll_joint "
    "left_wrist_pitch_joint left_wrist_yaw_joint right_shoulder_pitch_joint right_shoulder_roll_joint right_shoulder_yaw_joint "
    "right_elbow_joint right_wrist_roll_joint right_wrist_pitch_joint right_wrist_yaw_joint"
).split()
ODOMETRY_SCHEMA = "hv2.pelvis_odometry.v1"
ODOMETRY_FRAME = "pelvis_imu_gravity_frame"
TERRAIN_SCHEMA = b"hv2.terrain_grid.v1"
_PROCESS_HANDLES = []


def load_clock_calibration(path, now_s=None):
    """Read an independently measured, host/session-limited timing calibration."""
    data = json.loads(Path(path).read_text())
    now = time.monotonic() if now_s is None else now_s
    fields = (
        "lidar_offset_s",
        "lidar_error_bound_s",
        "robot_tick_offset_s",
        "robot_error_bound_s",
        "valid_from_monotonic_s",
        "valid_until_monotonic_s",
    )
    if data.get("schema") != "hv2.motion_clock_calibration.v1" or not data.get("source"):
        raise ValueError("Clock calibration requires its schema and measurement source")
    if any(not isinstance(data.get(k), (int, float)) or not np.isfinite(data[k]) for k in fields):
        raise ValueError("Clock calibration requires finite offsets, bounds and validity times")
    if not all(0 <= data[k] <= 0.005 for k in ("lidar_error_bound_s", "robot_error_bound_s")):
        raise ValueError("Clock error bounds exceed the existing 5 ms contract")
    if not data["valid_from_monotonic_s"] <= now <= data["valid_until_monotonic_s"]:
        raise ValueError("Clock calibration is outside its measured validity interval")
    if data.get("host_boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise ValueError("Clock calibration belongs to another host boot")
    return data


ESTIMATOR = "dlio_mid360+g1_waist_fk"
SCAN_X = np.arange(-0.45, 1.2 + 1e-6, 0.15)
SCAN_Y = np.arange(-0.75, 0.75 + 1e-6, 0.15)


def quat_wxyz_to_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_of(rotation):
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tilt_between(rotation_a, rotation_b):
    """Angle between the gravity directions the two body-to-world rotations imply in the body frame (yaw-invariant)."""
    return float(np.arccos(np.clip(rotation_a[2, :] @ rotation_b[2, :], -1.0, 1.0)))


class G1Body:
    """URDF kinematics for the pelvis-LiDAR chain plus per-link boxes for self-filtering scans."""

    def __init__(self, urdf_path=URDF, margin_m=0.06, *, load_body_boxes=True):
        import trimesh
        import yourdfpy

        self.robot = yourdfpy.URDF.load(
            str(urdf_path),
            load_meshes=False,
            load_collision_meshes=False,
            build_scene_graph=True,
            build_collision_scene_graph=False,
        )
        if list(self.robot.actuated_joint_names) != G1_JOINTS:
            raise ValueError("URDF joint order differs from the Unitree motor order")
        self.q = np.zeros(len(G1_JOINTS))
        self.robot.update_cfg(self.q)
        base = Path(urdf_path).parent
        self.boxes = []
        for name, link in (self.robot.link_map.items() if load_body_boxes else []):
            for visual in link.visuals:
                if visual.geometry.mesh is None:
                    continue
                mesh = trimesh.load(str(base / visual.geometry.mesh.filename), force="mesh")
                origin = np.eye(4) if visual.origin is None else visual.origin
                points = trimesh.transform_points(mesh.vertices, origin)
                self.boxes.append((name, points.min(0) - margin_m, points.max(0) + margin_m))

    def set_joints(self, q):
        self.q = np.asarray(q, dtype=np.float64)
        self.robot.update_cfg(self.q)

    def set_waist_joints(self, q):
        """Update only the chain needed for pelvis/torso/fixed-LiDAR geometry."""
        self.q = np.asarray(q, dtype=np.float64)
        self.robot.update_cfg({G1_JOINTS[i]: self.q[i] for i in range(12, 15)})

    def transform(self, child, parent):
        return self.robot.get_transform(child, parent)

    def pelvis_from_lidar(self):
        """T such that p_pelvis = T @ p_lidar for the current joints (LiDAR frame pose in the pelvis frame)."""
        return self.transform(LIDAR_LINK, PELVIS)

    def body_mask(self, points_lidar):
        """True for points inside any inflated link box, given the current joints."""
        lidar_to_pelvis = self.pelvis_from_lidar()
        homogeneous = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1))], axis=1)
        mask = np.zeros(points_lidar.shape[0], dtype=bool)
        for name, low, high in self.boxes:
            link_from_pelvis = np.linalg.inv(self.transform(name, PELVIS))
            local = homogeneous @ (link_from_pelvis @ lidar_to_pelvis).T
            mask |= np.all((local[:, :3] >= low) & (local[:, :3] <= high), axis=1)
        return mask


class LidarClock:
    """Maps LiDAR stamps to bluefin CLOCK_MONOTONIC from message arrivals (minimum-latency estimate)."""

    def __init__(self, window=400, offset_error_bound_s=None, calibrated_offset_s=None):
        self.samples = deque(maxlen=window)
        if offset_error_bound_s is not None and (
            not np.isfinite(offset_error_bound_s) or offset_error_bound_s < 0
        ):
            raise ValueError("Clock offset error bound must be finite and nonnegative")
        self.offset_error_bound_s = offset_error_bound_s
        if calibrated_offset_s is not None and (
            not np.isfinite(calibrated_offset_s) or offset_error_bound_s is None
        ):
            raise ValueError(
                "A calibrated clock offset requires a finite offset and its error bound"
            )
        self.calibrated_offset_s = calibrated_offset_s
        self.last_stamp = None
        self.frozen_offset_s = None

    def observe(self, stamp_s, arrival_monotonic_s):
        if not np.isfinite(stamp_s) or stamp_s <= 0 or not np.isfinite(arrival_monotonic_s):
            return
        if self.last_stamp is not None and stamp_s <= self.last_stamp:
            return
        self.last_stamp = stamp_s
        self.samples.append(arrival_monotonic_s - stamp_s)

    @property
    def ready(self):
        return len(self.samples) >= 20

    def offset_s(self):
        if self.calibrated_offset_s is not None:
            return float(self.calibrated_offset_s)
        if self.frozen_offset_s is None and self.ready:
            self.frozen_offset_s = float(min(self.samples))
        if self.frozen_offset_s is not None:
            return self.frozen_offset_s
        return float(min(self.samples))

    def arrival_jitter_s(self):
        if not self.samples:
            return float("inf")
        offsets = np.fromiter(self.samples, dtype=np.float64)
        return float(np.percentile(offsets, 50.0) - offsets.min())

    def uncertainty_s(self):
        # Equal packet delays produce zero jitter even with a large clock bias.
        # An external bound on the mapped timestamp error is required for use
        # by the motor runtime. Diagnostic recording remains available without it.
        return (
            float("inf") if self.offset_error_bound_s is None else float(self.offset_error_bound_s)
        )

    def to_monotonic(self, stamp_s):
        return stamp_s + self.offset_s()


class StateHistory:
    """Bounded nearest-state lookup in workstation receipt time, not sensor epoch.

    Receipt matching fixes use of the latest pose for an old scan. It does not
    by itself calibrate either device clock or prove motion-time accuracy.
    """

    def __init__(self, capacity=4096, max_pair_error_s=0.02):
        self.samples = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.max_pair_error_s = max_pair_error_s
        self.ordered = True

    def append(self, sample):
        with self.lock:
            if self.samples and sample["tick"] == self.samples[-1]["tick"]:
                return
            if self.samples and sample["monotonic_s"] <= self.samples[-1]["monotonic_s"]:
                self.ordered = False
            self.samples.append(sample)

    def get(self, monotonic_s=None):
        with self.lock:
            if not self.samples:
                return None
            if monotonic_s is None:
                return self.samples[-1]
            if not self.ordered:
                sample = min(self.samples, key=lambda s: abs(s["monotonic_s"] - monotonic_s))
            else:
                # Live odometry asks for a recent sample. Search backward only
                # to its timestamp bracket, preserving the older-on-tie rule.
                sample = self.samples[-1]
                error = abs(sample["monotonic_s"] - monotonic_s)
                for candidate in reversed(self.samples):
                    distance = abs(candidate["monotonic_s"] - monotonic_s)
                    if distance <= error:
                        sample, error = candidate, distance
                    if candidate["monotonic_s"] < monotonic_s:
                        break
            if abs(sample["monotonic_s"] - monotonic_s) > self.max_pair_error_s:
                return None
            return sample


class LowStateReader:
    """CRC-checked historical Unitree states; explicitly mapped timestamp lookup."""

    def __init__(self, interface, domain_id=0, clock_calibration=None):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, IMUState_

        from core.robot.unitree_crc import CRC

        ChannelFactoryInitialize(int(domain_id), str(interface))
        self.history = StateHistory()
        self.torso_history = StateHistory()
        self.crc = CRC()
        self.rejected_crc = 0
        self.clock_calibration = clock_calibration
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._receive, 10)
        self.torso_subscriber = ChannelSubscriber("rt/secondary_imu", IMUState_)
        self.torso_subscriber.Init(self._receive_torso, 10)

    def _receive_torso(self, message):
        now = time.monotonic()
        w, x, y, z = message.quaternion
        self.torso_history.append(dict(tick=now, monotonic_s=now, quat_xyzw=np.array([x, y, z, w])))

    def close(self):
        self.subscriber.Close()
        self.torso_subscriber.Close()

    def _receive(self, message):
        arrival = time.monotonic()
        if int(message.crc) != int(self.crc.Crc(message)):
            self.rejected_crc += 1
            return
        q = np.array([message.motor_state[i].q for i in range(len(G1_JOINTS))], dtype=np.float64)
        w, x, y, z = message.imu_state.quaternion
        calibration = self.clock_calibration
        capture = (
            arrival
            if calibration is None
            else int(message.tick) * 0.001 + calibration["robot_tick_offset_s"]
        )
        sample = dict(
            q=q,
            quat_xyzw=np.array([x, y, z, w], dtype=np.float64),
            gyro=np.asarray(message.imu_state.gyroscope, dtype=np.float64),
            tick=int(message.tick),
            mode_machine=int(message.mode_machine),
            monotonic_s=capture,
            arrival_monotonic_s=arrival,
            clock_uncertainty_s=None if calibration is None else calibration["robot_error_bound_s"],
        )
        self.history.append(sample)

    def get(self, stamp_s=None, *, monotonic_s=None):
        if stamp_s is not None and monotonic_s is None:
            return None
        state = self.history.get(monotonic_s)
        if state is None:
            return None
        state = dict(state)
        torso = self.torso_history.get(state.get("arrival_monotonic_s", state["monotonic_s"]))
        if torso is not None:
            state["torso_quat_xyzw"] = torso["quat_xyzw"]
        return state


class NpzStateReader:
    """Replays joints and pelvis quaternion from record_lidar_sim.py at a bag stamp (sim replay only)."""

    def __init__(self, path, env_index, epoch_s):
        rec = np.load(path, allow_pickle=False)
        self.t = rec["t"].astype(np.float64)
        self.q = rec["dof_pos"][:, env_index].astype(np.float64)
        self.quat_xyzw = rec["pelvis_quat_xyzw"][:, env_index].astype(np.float64)
        self.gyro = rec["pelvis_ang_vel_world"][:, env_index].astype(np.float64)
        self.epoch_s = float(epoch_s)

    def get(self, stamp_s=None, *, monotonic_s=None):
        if stamp_s is None:
            return None
        t = stamp_s - self.epoch_s
        if t < self.t[0] - 1e-6 or t > self.t[-1] + 1e-6:
            return None
        k = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))
        return dict(
            q=self.q[k],
            quat_xyzw=self.quat_xyzw[k],
            gyro=self.gyro[k],
            tick=k,
            monotonic_s=time.monotonic(),
        )


def pointcloud_dtype(msg):
    codes = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    endian = ">" if msg.is_bigendian else "<"
    return np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [
                (
                    np.dtype(endian + codes[f.datatype])
                    if f.count == 1
                    else (np.dtype(endian + codes[f.datatype]), (f.count,))
                )
                for f in msg.fields
            ],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )


class BodyFilterNode:
    """Republishes /livox/lidar without the robot's own body, using the low-state joints."""

    def __init__(
        self,
        node,
        body,
        low_state,
        in_topic="/livox/lidar",
        out_topic="/livox/lidar_filtered",
        max_state_age_s=0.05,
    ):
        from sensor_msgs.msg import PointCloud2, Imu
        from rclpy.qos import qos_profile_sensor_data

        self.node, self.body, self.low_state = node, body, low_state
        self.max_state_age_s = max_state_age_s
        self.publisher = node.create_publisher(PointCloud2, out_topic, 5)
        self.subscription = node.create_subscription(
            PointCloud2, in_topic, self.on_cloud, qos_profile_sensor_data
        )
        self.clock = LidarClock()
        self.imu_subscription = node.create_subscription(
            Imu,
            "/livox/imu",
            lambda msg: self.clock.observe(PelvisBridge.stamp_s(msg.header), time.monotonic()),
            qos_profile_sensor_data,
        )
        self.removed = deque(maxlen=100)

    def on_cloud(self, msg):
        if not self.clock.ready:
            return
        stamp = PelvisBridge.stamp_s(msg.header)
        state = self.low_state.get(stamp, monotonic_s=self.clock.to_monotonic(stamp))
        if state is None or time.monotonic() - state["monotonic_s"] > self.max_state_age_s:
            return
        raw = (
            np.ndarray(
                (msg.height, msg.width, msg.point_step),
                dtype=np.uint8,
                buffer=bytes(msg.data),
                strides=(msg.row_step, msg.point_step, 1),
            )
            .reshape(-1, msg.point_step)
            .copy()
        )
        fields = np.frombuffer(raw.tobytes(), dtype=pointcloud_dtype(msg))
        xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
        self.body.set_joints(state["q"])
        keep = ~self.body.body_mask(xyz)
        self.removed.append(int((~keep).sum()))
        out = msg.__class__()
        out.header, out.fields, out.is_bigendian, out.point_step, out.is_dense = (
            msg.header,
            msg.fields,
            msg.is_bigendian,
            msg.point_step,
            msg.is_dense,
        )
        out.height, out.width = 1, int(keep.sum())
        out.row_step = out.width * msg.point_step
        out.data = raw[keep].tobytes()
        self.publisher.publish(out)


class PelvisBridge:
    """DLIO pose + low state -> pelvis position packet; elevation map -> local terrain grid packet."""

    def __init__(
        self,
        node,
        body,
        low_state,
        odometry_port,
        terrain_port,
        grid_window_m=3.0,
        record=None,
        max_odom_gap_s=0.1,
        max_state_age_s=0.05,
        max_tilt_disagreement_rad=np.radians(5.0),
        max_jump_m=0.5,
        clock_offset_error_bound_s=None,
        mount_rotation=None,
        calibrated_clock_offset_s=None,
        odometry_topic="/dlio/odom_node/odom",
        publish_diagnostics=False,
        clock_calibration=None,
        recording_enabled=None,
    ):
        from rise.sensing.measured_pose import PelvisPoseEstimator, CausalPositionFilter

        self.estimator = PelvisPoseEstimator(body, mount_rotation=mount_rotation)
        self.position_filter = CausalPositionFilter()
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu

        self.node, self.body, self.low_state = node, body, low_state
        self.publish_diagnostics = publish_diagnostics
        self.clock_calibration = clock_calibration
        self.stream_id = uuid.uuid4().hex
        self.last_message_stamp = None
        self.clock = LidarClock(
            offset_error_bound_s=clock_offset_error_bound_s,
            calibrated_offset_s=calibrated_clock_offset_s,
        )
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.odometry_address = ("127.0.0.1", int(odometry_port)) if odometry_port else None
        self.terrain_address = ("127.0.0.1", int(terrain_port)) if terrain_port else None
        self.grid_window_m = float(grid_window_m)
        self.limits = dict(
            odom_gap=max_odom_gap_s,
            state_age=max_state_age_s,
            tilt=max_tilt_disagreement_rad,
            jump=max_jump_m,
        )
        self.yaw_align = None
        self.last = None
        self.pelvis_xy_aligned = None
        self.pelvis_z_aligned = None
        self.sent = 0
        self.invalid = 0
        self.unreachable = 0
        self.record = None if record is None else open(record, "ab")
        self.recording_enabled = recording_enabled
        self.pending_record = None
        self.status_path = None if record is None else Path(record).with_suffix(".status.json")
        self.last_status_s = 0.0
        self.terrain_record = (
            None if record is None or terrain_port is None else open(str(record) + ".terrain", "ab")
        )
        from rclpy.qos import qos_profile_sensor_data

        node.create_subscription(Imu, "/livox/imu", self.on_imu, qos_profile_sensor_data)
        node.create_subscription(Odometry, odometry_topic, self.on_odom, 20)
        if terrain_port is not None:
            from grid_map_msgs.msg import GridMap

            node.create_subscription(
                GridMap, "/elevation_mapping/elevation_map_raw", self.on_grid, 2
            )

    @staticmethod
    def stamp_s(header):
        return float(header.stamp.sec) + 1e-9 * float(header.stamp.nanosec)

    def on_imu(self, msg):
        self.clock.observe(self.stamp_s(msg.header), time.monotonic())

    def on_odom(self, msg):
        now = time.monotonic()
        stamp = self.stamp_s(msg.header)
        if stamp == self.last_message_stamp:
            return
        if self.last_message_stamp is not None and stamp < self.last_message_stamp:
            self.stream_id = uuid.uuid4().hex  # force receiver to invalidate episode origin
        self.last_message_stamp = stamp
        state = (
            self.low_state.get(stamp, monotonic_s=self.clock.to_monotonic(stamp))
            if self.clock.ready
            else None
        )
        pose = msg.pose.pose
        lidar_in_odom = np.eye(4)
        lidar_in_odom[:3, :3] = quat_wxyz_to_matrix(
            (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
        )
        lidar_in_odom[:3, 3] = (pose.position.x, pose.position.y, pose.position.z)
        reason = ""
        if stamp <= 0.0:
            reason = "stamp"
        elif state is None or now - state["monotonic_s"] > self.limits["state_age"]:
            reason = "low_state"
        elif not self.clock.ready:
            reason = "clock"
        info = {}
        position = np.full(3, np.nan)
        tilt = float("nan")
        if not reason:
            try:
                info = self.estimator.update(
                    stamp,
                    lidar_in_odom[:3, 3],
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ],
                    state,
                )
                position = np.asarray(info["position_m"])
                tilt = info["tilt_disagreement_rad"]
                self.yaw_align = info["yaw_align_rad"]
                if tilt > self.limits["tilt"]:
                    reason = "tilt"
                elif self.last is not None:
                    gap = stamp - self.last[0]
                    if not 0 < gap <= self.limits["odom_gap"]:
                        reason = "gap"
                    elif np.linalg.norm(position - self.last[1]) > self.limits["jump"]:
                        reason = "jump"
                self.last = (stamp, position.copy())
                self.pelvis_xy_aligned, self.pelvis_z_aligned = position[:2].copy(), float(
                    position[2]
                )
            except ValueError as exc:
                reason = str(exc)
        valid = not reason
        geometry_valid = valid
        state_uncertainty = None if state is None else state.get("clock_uncertainty_s")
        clock_valid = (
            self.clock.uncertainty_s() <= 0.005
            and state_uncertainty is not None
            and state_uncertainty <= 0.005
        )
        if self.clock_calibration is not None:
            clock_valid &= (
                self.clock_calibration["valid_from_monotonic_s"]
                <= now
                <= self.clock_calibration["valid_until_monotonic_s"]
            )
        if valid and not clock_valid:
            reason = "clock_offset_unqualified"
            valid = False
        if not valid:
            self.pelvis_xy_aligned = None
            self.pelvis_z_aligned = None
        packet = dict(
            schema=ODOMETRY_SCHEMA,
            frame=ODOMETRY_FRAME,
            estimator=ESTIMATOR,
            tracking_valid=bool(valid),
            clock_uncertainty_s=self.clock.uncertainty_s() if self.clock.ready else float("inf"),
            arrival_jitter_s=self.clock.arrival_jitter_s(),
            position_m=[float(v) for v in position],
            capture_monotonic_s=(
                self.clock.to_monotonic(stamp) if self.clock.ready else float("nan")
            ),
            lidar_stamp_s=stamp,
            tilt_disagreement_rad=tilt,
            yaw_align_rad=self.yaw_align,
            invalid_reason=reason,
            lidar_pose_odom=[
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ],
        )
        packet.update(info, geometry_valid=geometry_valid, arrival_monotonic_s=now)
        # Filter only real, finite incoming estimates. Preserve raw XYZ and all
        # validity decisions; a smoother trajectory is not evidence of accuracy.
        if geometry_valid or reason == "tilt":
            filtered, delay = self.position_filter.update(position, packet["capture_monotonic_s"])
            packet.update(
                position_raw_m=position.tolist(),
                position_m=filtered.tolist(),
                position_filter="causal_exponential_v1",
                position_filter_tau_s=self.position_filter.tau_s,
                position_filter_delay_s=delay,
            )
        else:
            self.position_filter.reset()
        packet.update(
            stream_id=self.stream_id,
            invalid_reason=reason,
            state_pair_error_s=(
                None
                if state is None or not self.clock.ready
                else abs(state["monotonic_s"] - self.clock.to_monotonic(stamp))
            ),
            state_clock_uncertainty_s=state_uncertainty,
            clock_source=(
                "declared calibration"
                if self.clock_calibration
                else "frozen minimum-arrival estimate; unqualified"
            ),
            processing_s=time.monotonic() - now,
        )
        packet = json.loads(json.dumps(packet), parse_constant=lambda _: None)
        self.record_pose(packet)
        if self.odometry_address is not None and (valid or self.publish_diagnostics):
            self._send(json.dumps(packet).encode(), self.odometry_address)
            if valid:
                self.sent += 1
        if self.status_path is not None and now - self.last_status_s >= 0.5:
            status = dict(
                arrival_monotonic_s=now,
                geometry_valid=geometry_valid,
                tracking_valid=bool(valid),
                invalid_reason=reason,
                capture_monotonic_s=packet["capture_monotonic_s"],
            )
            temporary = self.status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status))
            temporary.replace(self.status_path)
            self.last_status_s = now
        if not valid:
            self.invalid += 1

    def record_pose(self, packet):
        if self.record is None:
            return
        if self.recording_enabled is not None and not self.recording_enabled.is_set():
            self.pending_record = packet  # one origin sample, not an idle trajectory
            return
        if self.pending_record is not None:
            self.record.write((json.dumps(self.pending_record) + "\n").encode())
            self.pending_record = None
        self.record.write((json.dumps(packet) + "\n").encode())
        self.record.flush()

    def _send(self, payload, address):
        try:
            self.socket.sendto(payload, address)
        except OSError:
            self.unreachable += 1

    def on_grid(self, msg):
        if (
            (self.terrain_address is None and self.terrain_record is None)
            or self.pelvis_xy_aligned is None
            or self.yaw_align is None
        ):
            return
        layer = decode_layer(msg, "elevation")
        resolution = float(msg.info.resolution)
        center = np.array([msg.info.pose.position.x, msg.info.pose.position.y])
        rows, cols = layer.shape
        xs = center[0] + (np.arange(cols) + 0.5 - 0.5 * cols) * resolution
        ys = center[1] + (np.arange(rows) + 0.5 - 0.5 * rows) * resolution
        pelvis_odom_xy = rot_z(-self.yaw_align)[:2, :2] @ self.pelvis_xy_aligned
        half = int(round(0.5 * self.grid_window_m / resolution))
        ix = int(np.argmin(np.abs(xs - pelvis_odom_xy[0])))
        iy = int(np.argmin(np.abs(ys - pelvis_odom_xy[1])))
        window = layer[max(iy - half, 0) : iy + half, max(ix - half, 0) : ix + half]
        wx = xs[max(ix - half, 0) : ix + half]
        wy = ys[max(iy - half, 0) : iy + half]
        heights = window.astype(np.float32)
        observed = np.isfinite(heights)
        heights = np.where(observed, heights, 0.0).astype(np.float32)
        header = json.dumps(
            dict(
                schema=TERRAIN_SCHEMA.decode(),
                frame=ODOMETRY_FRAME,
                estimator=ESTIMATOR,
                capture_monotonic_s=self.clock.to_monotonic(self.stamp_s(msg.header)),
                lidar_stamp_s=self.stamp_s(msg.header),
                resolution_m=resolution,
                rows=int(heights.shape[0]),
                cols=int(heights.shape[1]),
                yaw_align_rad=self.yaw_align,
                origin_odom_xy=[float(wx[0]), float(wy[0])],
                pelvis_position_m=[
                    float(self.pelvis_xy_aligned[0]),
                    float(self.pelvis_xy_aligned[1]),
                    self.pelvis_z_aligned,
                ],
            )
        ).encode()
        payload = (
            struct.pack("<I", len(header))
            + header
            + heights.tobytes()
            + observed.astype(np.uint8).tobytes()
        )
        if self.terrain_record is not None and (
            self.recording_enabled is None or self.recording_enabled.is_set()
        ):
            self.terrain_record.write(struct.pack("<I", len(payload)) + payload)
        if self.terrain_address is not None:
            self._send(payload, self.terrain_address)


def decode_layer(msg, name):
    """GridMap layer as a (rows=y, cols=x) array; NaN where the map has no estimate."""
    index = list(msg.layers).index(name)
    array = msg.data[index]
    dims = array.layout.dim
    data = np.asarray(array.data, dtype=np.float32)
    if dims[0].label == "column_index":
        return data.reshape((dims[0].size, dims[1].size)).T
    return data.reshape((dims[0].size, dims[1].size))


def decode_terrain_packet(payload):
    """Inverse of PelvisBridge.on_grid: (header dict, heights [rows, cols], observed mask)."""
    (length,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4 : 4 + length].decode())
    rows, cols = header["rows"], header["cols"]
    offset = 4 + length
    heights = np.frombuffer(payload[offset : offset + 4 * rows * cols], dtype=np.float32).reshape(
        rows, cols
    )
    observed = (
        np.frombuffer(payload[offset + 4 * rows * cols :], dtype=np.uint8)
        .reshape(rows, cols)
        .astype(bool)
    )
    return header, heights, observed


def sample_height_scan(
    header, heights, observed, pelvis_xy_aligned, heading_rad, scan_x=SCAN_X, scan_y=SCAN_Y
):
    """Teacher-layout height scan around the pelvis in the given heading frame (nearest cell), plus its observed mask."""
    resolution = header["resolution_m"]
    origin = np.asarray(header["origin_odom_xy"])
    yaw_align = header["yaw_align_rad"]
    gx, gy = np.meshgrid(scan_x, scan_y, indexing="ij")
    local = np.stack([gx.ravel(), gy.ravel()], axis=1)
    aligned = local @ rot_z(heading_rad)[:2, :2].T + np.asarray(pelvis_xy_aligned)[None]
    odom = aligned @ rot_z(-yaw_align)[:2, :2].T
    ix = np.round((odom[:, 0] - origin[0]) / resolution).astype(int)
    iy = np.round((odom[:, 1] - origin[1]) / resolution).astype(int)
    in_bounds = (ix >= 0) & (ix < heights.shape[1]) & (iy >= 0) & (iy < heights.shape[0])
    ix = np.clip(ix, 0, heights.shape[1] - 1)
    iy = np.clip(iy, 0, heights.shape[0] - 1)
    valid = in_bounds & observed[iy, ix]
    return np.where(valid, heights[iy, ix], 0.0).reshape(gx.shape), valid.reshape(gx.shape)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("role", choices=["body-filter", "bridge"])
    parser.add_argument(
        "--interface", default="enp132s0", help="network interface of the Unitree low state"
    )
    parser.add_argument("--urdf", default=str(URDF))
    parser.add_argument(
        "--odometry-port", type=int, default=None, help="runtime pelvis_odometry_port"
    )
    parser.add_argument("--terrain-port", type=int, default=None)
    parser.add_argument("--grid-window-m", type=float, default=3.0)
    parser.add_argument("--report-s", type=float, default=5.0)
    parser.add_argument(
        "--clock-offset-error-bound-s",
        type=float,
        default=None,
        help="Externally measured timestamp-error bound; omitted means diagnostic-only odometry",
    )
    parser.add_argument(
        "--calibrated-clock-offset-s",
        type=float,
        help="Measured monotonic minus LiDAR clock offset; requires its error bound",
    )
    parser.add_argument(
        "--mount-calibration",
        type=Path,
        help="Optional lidar_mount_candidate JSON; nominal mount remains default",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="append every bridge packet here (JSON lines; terrain grids in <record>.terrain)",
    )
    parser.add_argument(
        "--controlled-record",
        action="store_true",
        help="Publish live estimates immediately; record only after start on stdin",
    )
    parser.add_argument(
        "--publish-diagnostics",
        action="store_true",
        help="Publish unqualified/invalid packets with explicit status for monitoring",
    )
    parser.add_argument(
        "--clock-calibration",
        type=Path,
        help="Measured LiDAR and robot clock offsets, bounds and validity interval",
    )
    parser.add_argument(
        "--sim-state",
        default=None,
        help="record_lidar_sim.py npz: replay joints/pelvis IMU instead of the robot",
    )
    parser.add_argument("--sim-env", type=int, default=0)
    parser.add_argument(
        "--sim-epoch-s", type=float, default=1.7e9, help="bag epoch used by lidar_sim_to_bag.py"
    )
    args = parser.parse_args()
    import rclpy

    rclpy.init()
    node = rclpy.create_node(args.role.replace("-", "_"))
    body = G1Body(args.urdf, load_body_boxes=args.role == "body-filter")
    calibration = (
        None if args.clock_calibration is None else load_clock_calibration(args.clock_calibration)
    )
    low_state = (
        NpzStateReader(args.sim_state, args.sim_env, args.sim_epoch_s)
        if args.sim_state
        else LowStateReader(args.interface, clock_calibration=calibration)
    )
    _PROCESS_HANDLES.append(low_state)
    if args.role == "body-filter":
        worker = BodyFilterNode(node, body, low_state)
        report = lambda: node.get_logger().info(
            f"removed body points per scan: {np.mean(worker.removed) if worker.removed else float('nan'):.0f}"
        )
    else:
        if args.odometry_port is None and args.record is None:
            parser.error("Provide --record for diagnostics or --odometry-port for runtime")
        mount = (
            None
            if args.mount_calibration is None
            else json.loads(args.mount_calibration.read_text())["rotation_lidar_to_torso"]
        )
        import threading

        recording_enabled = threading.Event() if args.controlled_record else None
        worker = PelvisBridge(
            node,
            body,
            low_state,
            args.odometry_port,
            args.terrain_port,
            args.grid_window_m,
            record=args.record,
            clock_offset_error_bound_s=args.clock_offset_error_bound_s,
            mount_rotation=mount,
            calibrated_clock_offset_s=args.calibrated_clock_offset_s,
            publish_diagnostics=args.publish_diagnostics,
            clock_calibration=calibration,
            recording_enabled=recording_enabled,
        )
        if recording_enabled is not None:

            def start_recording():
                import sys

                for line in sys.stdin:
                    if json.loads(line).get("command") == "start":
                        recording_enabled.set()

            threading.Thread(target=start_recording, daemon=True).start()
        if calibration is not None:
            worker.clock = LidarClock(
                calibrated_offset_s=calibration["lidar_offset_s"],
                offset_error_bound_s=calibration["lidar_error_bound_s"],
            )
        report = lambda: node.get_logger().info(
            f"sent {worker.sent} valid, {worker.invalid} invalid, {worker.unreachable} unreachable; clock uncertainty {worker.clock.uncertainty_s() * 1e3 if worker.clock.ready else float('nan'):.2f} ms; yaw align {worker.yaw_align}"
        )
    node.create_timer(args.report_s, report)
    try:
        rclpy.spin(node)
    finally:
        # DDS listener destruction can hang. Flush files first and let the
        # dedicated process exit own the listener lifetime, like raw capture.
        if hasattr(worker, "record") and worker.record:
            worker.record.close()
        if hasattr(worker, "terrain_record") and worker.terrain_record:
            worker.terrain_record.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    import sys

    code = 0
    try:
        main()
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
