"""Vendored from the SONIC research repo so this deploy is standalone.
Source: humanoidverse2/utils/rotation_np.py
Pure numpy; do not edit here -- re-copy if upstream changes.
"""
"""Pure-numpy quaternion/rotation helpers (wxyz, w-first) — the numpy mirror of the torch
functions in ``isaac_utils.rotations``, for the torch-free SONIC deploy path. All broadcasting."""

import numpy as np


def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_inv(q):
    """Unit-quat inverse == conjugate."""
    return quat_conjugate(q)


def quat_mul(a, b):
    """Hamilton product, wxyz."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_apply(q, v):
    """Rotate vec v by quat q, wxyz."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    xyz = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


quat_rotate = quat_apply


def matrix_from_quat(q):
    """3x3 rotation matrix from wxyz quat."""
    q = np.asarray(q, dtype=np.float64)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / np.sum(q * q, axis=-1)
    o = np.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], axis=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m):
    """wxyz quat from 3x3 rotation matrix (trace method)."""
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    s = np.sqrt(np.maximum(t + 1.0, 1e-12)) * 2.0
    q = np.zeros(m.shape[:-2] + (4,))
    q[..., 0] = 0.25 * s
    q[..., 1] = (m[..., 2, 1] - m[..., 1, 2]) / s
    q[..., 2] = (m[..., 0, 2] - m[..., 2, 0]) / s
    q[..., 3] = (m[..., 1, 0] - m[..., 0, 1]) / s
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rot6d_to_quat(r6):
    """6D (first two columns of R, block order) -> quat wxyz, via Gram-Schmidt.
    Inverse of motion_diffusion's block-order ``quat_to_rot6d`` (the diffusion feature codec)."""
    c0, c1 = r6[..., 0:3], r6[..., 3:6]
    b0 = c0 / np.linalg.norm(c0, axis=-1, keepdims=True)
    c1 = c1 - np.sum(b0 * c1, axis=-1, keepdims=True) * b0
    b1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    b2 = np.cross(b0, b1)
    return matrix_to_quat(np.stack([b0, b1, b2], axis=-1))


def heading_quat(q):
    """Yaw-only (heading) quaternion, wxyz."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    o = np.zeros(q.shape[:-1] + (4,))
    o[..., 0] = np.cos(0.5 * yaw)
    o[..., 3] = np.sin(0.5 * yaw)
    return o


def subtract_frame_transforms(t01, q01, t02, q02):
    """Express frame 2 in frame 1."""
    q01i = quat_inv(q01)
    return quat_apply(q01i, np.asarray(t02) - np.asarray(t01)), quat_mul(q01i, q02)
