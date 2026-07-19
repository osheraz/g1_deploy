from __future__ import annotations

import numpy as np


def _quat_conj_xyzw(q):
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _quat_to_rot6d_xyzw(q):
    # matches humanoidverse2 reference_tracking_obs._quat_xyzw_to_rot6d (first 2 cols)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    col0 = np.stack([1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)], axis=-1)
    col1 = np.stack([2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)], axis=-1)
    return np.concatenate([col0, col1], axis=-1)


class ReferenceProvider:
    """Replays a retargeted reference clip and builds `future_ref_motion_proto`."""

    def __init__(self, clip_path, default_dof_pos, future_steps, loop_mode="clamp"):
        d = np.load(clip_path)
        self.dof_pos = d["dof_pos"].astype(np.float32)
        self.dof_vel = d["dof_vel"].astype(np.float32)
        self.torso_quat = d["torso_quat_xyzw"].astype(np.float32)
        self.root_pos = d["root_pos"].astype(np.float32) if "root_pos" in d.files else None
        self.T = len(self.dof_pos)
        self.default = np.asarray(default_dof_pos, np.float32)
        self.future_steps = [int(s) for s in future_steps]
        self.loop_mode = loop_mode
        self.i = 0

    def reset(self):
        self.i = 0

    def step(self):
        self.i += 1
        if self.loop_mode == "clamp":
            self.i = min(self.i, self.T - 1)
        else:
            self.i %= self.T

    def _frame(self, k):
        j = self.i + k
        if self.loop_mode == "clamp":
            j = min(j, self.T - 1)
        else:
            j %= self.T
        return j

    def future_proto(self, robot_quat_wxyz):
        q = np.asarray(robot_quat_wxyz, np.float32)
        anchor_inv = _quat_conj_xyzw(q[[1, 2, 3, 0]])
        blocks = []
        for s in self.future_steps:
            j = self._frame(s)
            rel = _quat_mul_xyzw(anchor_inv, self.torso_quat[j])
            ori6 = _quat_to_rot6d_xyzw(rel)
            blocks.append(np.concatenate(
                [ori6, self.dof_vel[j], self.dof_pos[j] - self.default]
            ))
        return np.concatenate(blocks).astype(np.float32)
