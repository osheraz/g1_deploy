"""Pose-free Mid-360 preprocessing for fall-recovery deployment."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from core.common.rotation_helper import get_gravity_orientation
from motion_diffusion.models.lidar_encoder import LidarVoxelSpec
from humanoidverse2.envs.env_utils.embodied_lidar import (
    EMBODIED_LIDAR_CONTRACT,
    LIVOX_MID360_CONTRACT,
)


def _relative_fk_poses(pos, quat_xyzw, root_index: int):
    root_rot = Rotation.from_quat(quat_xyzw[root_index])
    inv = root_rot.inv()
    rel_pos = inv.apply(pos - pos[root_index])
    rel_rot = (inv * Rotation.from_quat(quat_xyzw)).as_quat()
    return rel_pos.astype(np.float32), rel_rot.astype(np.float32)


def gravity_aligned_scan(
    raw_scan: np.ndarray,
    torso_quat_wxyz: np.ndarray,
    torso_pos_pelvis: np.ndarray,
    torso_quat_pelvis_xyzw: np.ndarray,
    translation_torso_lidar,
    rpy_torso_lidar,
    *,
    previous_heading_pelvis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform LiDAR points into a pelvis-centred gravity/heading frame.

    Only torso gravity and current waist FK are used; no world translation or
    persistent pose estimate appears in this transform.
    """
    scan = np.asarray(raw_scan, dtype=np.float32)
    if scan.ndim != 2 or scan.shape[1] < 4:
        raise ValueError("raw LiDAR scan must be [points, >=4]")
    r_pt = Rotation.from_quat(torso_quat_pelvis_xyzw).as_matrix()
    r_tl = Rotation.from_euler("xyz", rpy_torso_lidar).as_matrix()
    p_t = scan[:, :3] @ r_tl.T + np.asarray(translation_torso_lidar, dtype=np.float64)
    p_p = p_t @ r_pt.T + np.asarray(torso_pos_pelvis, dtype=np.float64)

    gravity_t = np.asarray(get_gravity_orientation(torso_quat_wxyz), dtype=np.float64)
    gravity_p = r_pt @ gravity_t
    up = -gravity_p / max(np.linalg.norm(gravity_p), 1.0e-8)
    # Match humanoidverse2.character_yaw_matrix: project root X + root Z.
    # The +Z term supplies a heading through the pitch=90-degree fall pose.
    forward = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    heading = forward - up * float(np.dot(forward, up))
    if np.linalg.norm(heading) < 0.08:
        candidate = (
            np.asarray(previous_heading_pelvis, dtype=np.float64)
            if previous_heading_pelvis is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        heading = candidate - up * float(np.dot(candidate, up))
    if np.linalg.norm(heading) < 0.08:
        heading = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        heading -= up * float(np.dot(heading, up))
    heading /= max(np.linalg.norm(heading), 1.0e-8)
    lateral = np.cross(up, heading)
    lateral /= max(np.linalg.norm(lateral), 1.0e-8)
    heading = np.cross(lateral, up)
    basis_pelvis = np.stack((heading, lateral, up), axis=-1)
    xyz = p_p @ basis_pelvis
    valid = scan[:, 3:4]
    return np.concatenate((xyz.astype(np.float32), valid), axis=-1), heading.astype(np.float32)


def _body_radius(name: str) -> float:
    low = name.lower()
    if "torso" in low or "pelvis" in low or "waist" in low:
        return 0.20
    if "shoulder" in low or "hip" in low:
        return 0.14
    if "wrist" in low or "ankle" in low or "foot" in low:
        return 0.09
    return 0.11


def filter_robot_returns(
    points: np.ndarray,
    body_centers: np.ndarray,
    body_names: list[str],
) -> np.ndarray:
    """Invalidate returns inside conservative link-centred self-filter spheres."""
    out = points.copy()
    if len(body_names) == 0 or body_centers.size == 0:
        return out
    radii2 = np.square(np.asarray([_body_radius(n) for n in body_names], dtype=np.float32))
    dist2 = np.square(out[:, None, :3] - body_centers[None]).sum(axis=-1)
    self_hit = (dist2 < radii2[None]).any(axis=1)
    out[self_hit, 3] = 0.0
    return out


class DeployLidarHistory:
    """Transforms each new scan once and stores only independent scan tensors."""

    def __init__(self, config: dict, body_names: list[str], device):
        self.config = dict(config)
        point_contract = self.config.get("point_contract")
        sensor_contract = self.config.get("sensor_contract")
        if point_contract not in (None, EMBODIED_LIDAR_CONTRACT):
            raise ValueError(
                f"LiDAR point contract {point_contract!r} != {EMBODIED_LIDAR_CONTRACT!r}"
            )
        if sensor_contract not in (None, LIVOX_MID360_CONTRACT):
            raise ValueError(
                f"LiDAR sensor contract {sensor_contract!r} != {LIVOX_MID360_CONTRACT!r}"
            )
        self.spec = LidarVoxelSpec.from_mapping(config)
        self.body_names = list(body_names)
        self.device = torch.device(device)
        self.translation = tuple(
            float(v) for v in config.get("translation_m", (0.0002835, 0.00003, 0.428434))
        )
        self.rpy = tuple(
            float(v) for v in config.get("rpy_rad", (math.pi, 0.05112069379091391, 0.0))
        )
        self.self_filter = bool(config.get("self_filter", True))
        self._scans: list[np.ndarray] = []
        self._omega: list[np.ndarray] = []
        self._capture_time: list[float] = []
        self._last_seq = -1
        self._heading = None

    def reset(self) -> None:
        self._scans.clear()
        self._omega.clear()
        self._capture_time.clear()
        self._last_seq = -1
        self._heading = None

    def update(
        self,
        *,
        seq: int,
        raw_scan: np.ndarray,
        scan_age_s: float,
        torso_quat_wxyz: np.ndarray,
        torso_gyro: np.ndarray,
        fk_pos: np.ndarray,
        fk_quat_xyzw: np.ndarray,
        torso_index: int,
        root_index: int,
        now: float,
    ) -> None:
        if int(seq) == self._last_seq:
            return
        pos_p, rot_p = _relative_fk_poses(fk_pos, fk_quat_xyzw, root_index)
        points, self._heading = gravity_aligned_scan(
            raw_scan,
            torso_quat_wxyz,
            pos_p[torso_index],
            rot_p[torso_index],
            self.translation,
            self.rpy,
            previous_heading_pelvis=self._heading,
        )
        # Transform all FK link centres through the same gravity basis by using
        # zero-offset pseudo points. Reconstructing the basis avoids a second API.
        gravity_t = np.asarray(get_gravity_orientation(torso_quat_wxyz), dtype=np.float64)
        r_pt = Rotation.from_quat(rot_p[torso_index]).as_matrix()
        up = -(r_pt @ gravity_t)
        up /= max(np.linalg.norm(up), 1.0e-8)
        heading = self._heading.astype(np.float64)
        lateral = np.cross(up, heading)
        lateral /= max(np.linalg.norm(lateral), 1.0e-8)
        basis = np.stack((heading, lateral, up), axis=-1)
        if self.self_filter:
            points = filter_robot_returns(points, pos_p @ basis, self.body_names)

        self._last_seq = int(seq)
        self._scans.append(points)
        omega_torso = np.asarray(torso_gyro, dtype=np.float64).reshape(3)
        omega_pelvis = omega_torso @ r_pt.T
        omega_character = omega_pelvis @ basis
        self._omega.append(omega_character.astype(np.float32))
        self._capture_time.append(float(now) - float(scan_age_s))
        keep = self.spec.history_steps
        self._scans = self._scans[-keep:]
        self._omega = self._omega[-keep:]
        self._capture_time = self._capture_time[-keep:]
        while len(self._scans) < keep:
            self._scans.insert(0, self._scans[0].copy())
            self._omega.insert(0, self._omega[0].copy())
            period = 1.0 / float(self.config.get("scan_rate_hz", 30.0))
            self._capture_time.insert(0, self._capture_time[0] - period)

    def tensors(self, now: float) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._scans:
            raise RuntimeError("No transformed LiDAR scan is available")
        points = torch.as_tensor(
            np.stack(self._scans), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        metadata = np.zeros((self.spec.history_steps, 4), dtype=np.float32)
        metadata[:, 0] = np.maximum(0.0, float(now) - np.asarray(self._capture_time))
        metadata[:, 1:] = np.stack(self._omega)
        return points, torch.as_tensor(metadata, device=self.device).unsqueeze(0)

    def flatten(self, now: float) -> torch.Tensor:
        points, metadata = self.tensors(now)
        return torch.cat((points.flatten(1), metadata.flatten(1)), dim=-1)
