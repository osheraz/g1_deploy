from __future__ import annotations

import numpy as np

from policy.reference_provider import (
    _quat_conj_xyzw,
    _quat_mul_xyzw,
    _quat_to_rot6d_xyzw,
)


def _quat_apply_xyzw(q, v):
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], np.float32)


class PicoReferenceProvider:
    """Sparse-goal obs blocks (motion_anchor_* + ref_body_pos_b/ori_b) from live 3-point
    VR poses, matching reference_tracking_obs getters (anchor=pelvis, w_last). Returns a
    dict {obs_name: block} so PolicyRunner sorted-concat picks up the trained keys.
    tracked_order MUST match the trained tracked_body_names; emit MUST match its goal keys."""

    def __init__(self, tracked_order=("head", "left_wrist", "right_wrist"),
                 emit=("motion_anchor_pos_b", "motion_anchor_ori_b",
                       "ref_body_pos_b", "ref_body_ori_b")):
        self.tracked_order = list(tracked_order)
        self.emit = set(emit)

    def reset(self):
        pass

    def step(self):
        pass

    def blocks(self, pico_frame, ref_anchor_pos, ref_anchor_quat_wxyz,
              robot_anchor_pos, robot_anchor_quat_wxyz,
              ref_anchor_vel=None, ref_anchor_angvel=None):
        rq = _wxyz_to_xyzw(np.asarray(robot_anchor_quat_wxyz, np.float32))
        anchor_inv = _quat_conj_xyzw(rq)
        rp = np.asarray(robot_anchor_pos, np.float32)
        ra_p = np.asarray(ref_anchor_pos, np.float32)
        ra_q = _wxyz_to_xyzw(np.asarray(ref_anchor_quat_wxyz, np.float32))
        z3 = np.zeros(3, np.float32)
        rv = z3 if ref_anchor_vel is None else np.asarray(ref_anchor_vel, np.float32)
        rw = z3 if ref_anchor_angvel is None else np.asarray(ref_anchor_angvel, np.float32)

        out = {}
        if "motion_anchor_pos_b" in self.emit:
            out["motion_anchor_pos_b"] = _quat_apply_xyzw(anchor_inv, ra_p - rp)
        if "motion_anchor_ori_b" in self.emit:
            out["motion_anchor_ori_b"] = _quat_to_rot6d_xyzw(
                _quat_mul_xyzw(anchor_inv, ra_q))
        if "motion_anchor_vel_b" in self.emit:
            out["motion_anchor_vel_b"] = _quat_apply_xyzw(anchor_inv, rv)
        if "motion_anchor_ang_vel_b" in self.emit:
            out["motion_anchor_ang_vel_b"] = _quat_apply_xyzw(anchor_inv, rw)

        pos_blocks, ori_blocks = [], []
        for name in self.tracked_order:
            bp, bq = pico_frame[name]
            bp = np.asarray(bp, np.float32)
            bq = _wxyz_to_xyzw(np.asarray(bq, np.float32))
            pos_blocks.append(_quat_apply_xyzw(anchor_inv, bp - rp))
            ori_blocks.append(_quat_to_rot6d_xyzw(_quat_mul_xyzw(anchor_inv, bq)))
        if "ref_body_pos_b" in self.emit:
            out["ref_body_pos_b"] = np.concatenate(pos_blocks).astype(np.float32)
        if "ref_body_ori_b" in self.emit:
            out["ref_body_ori_b"] = np.concatenate(ori_blocks).astype(np.float32)
        return out
