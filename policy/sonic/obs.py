"""Vendored from the SONIC research repo so this deploy is standalone.
Source: sonic/policy/obs.py
Pure numpy; do not edit here -- re-copy if upstream changes.
"""
"""Pure-numpy SONIC encoder/decoder obs builders. IsaacLab joint/body order; all quats wxyz (w-first)."""

import numpy as np

from . import robot_g1

# fixed index sets (IsaacLab order)
LOWER_JOINT_IL_IDX = robot_g1.ISAACLAB_TO_MUJOCO_DOF[:12]  # legs: mujoco 0..11 -> isaaclab
WRIST_IL_IDX = np.array([23, 24, 25, 26, 27, 28])         # wrists for smpl term
# Official G1 carrier bodies and the endpoint names exposed by SONIC's VR3
# interface.  Slot 3 is a virtual head/neck endpoint carried by torso_link; it
# is not the torso-link origin.
VR_CARRIER_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
)
VR_ENDPOINT_NAMES = (
    "left_wrist_endpoint",
    "right_wrist_endpoint",
    "virtual_head_endpoint",
)
VR_BODY_IDX = np.array(
    [robot_g1.BODY_NAMES.index(name) for name in VR_CARRIER_BODY_NAMES]
)
VR_ENDPOINT_OFFSETS = np.array(
    [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]],
    dtype=np.float64,
)
# Backward-compatible name used by historical Stage 2 runners and logs.
VR_OFFSETS = VR_ENDPOINT_OFFSETS
ANCHOR_BODY_IDX = robot_g1.BODY_NAMES.index(robot_g1.ANCHOR_BODY)  # pelvis == 0

ENCODER_MODE_IDX = {"g1": 0, "teleop": 1, "smpl": 2, "soma": 3}
DOWN_DIR = np.array([0.0, 0.0, -1.0])

# future windows: g1/teleop 10 @ step 5; smpl 10 @ step 1
G1_N, G1_STEP = 10, 5
SMPL_N, SMPL_STEP = 10, 1

# per-term dims (sum to 1762), INTEGRATION.md order
ENC_TERM_DIMS = [4, 580, 10, 60, 240, 9, 12, 6, 1, 720, 60, 60]
DEC_TERM_DIMS = [64, 30, 290, 290, 290, 30]


# numpy quat/rotation primitives live in the shared rotation lib (wxyz, torch-free deploy mirror)
from .rotation_np import (  # noqa: E402
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_inv,
    quat_mul,
    quat_rotate,
    subtract_frame_transforms,
)


def quat_to_rot6d(q):
    """6D orientation in SONIC's on-wire order: first two COLUMNS of R, interleaved
    (mat[...,:2].reshape) to match the released encoder. NOT motion_diffusion's block-order rot6d."""
    mat = matrix_from_quat(q)                      # (...,3,3)
    return mat[..., :2].reshape(q.shape[:-1] + (6,))


def _validated_vr_targets(vr_targets, batch_size):
    """Return strict VR3 arrays in official endpoint order and WXYZ format."""
    pos = np.asarray(vr_targets[0], dtype=np.float64).reshape(batch_size, 3, 3)
    quat = np.asarray(vr_targets[1], dtype=np.float64).reshape(batch_size, 3, 4)
    if not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("VR3 targets must contain only finite values")
    norms = np.linalg.norm(quat, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3):
        raise ValueError(
            "VR3 target quaternions must be normalized WXYZ values in "
            f"{VR_ENDPOINT_NAMES} order; norms={norms.tolist()}"
        )
    return pos, quat


# encoder obs (1762), BATCHED over leading dim N
def build_encoder_obs(mode, dof_fut, dvel_fut, rootq_fut, rootz_fut,
                      body_pos_cur, body_quat_cur, rootz_cur, robot_anchor_quat_wxyz,
                      wrist_dof_smpl, vr_targets=None, hold_idx=None,
                      vr_targets_local=False):
    """Full (N,1762) encoder buffer in INTEGRATION.md term order. mode: g1|teleop|smpl. Concat order must be exact.
    Pre-gathered inputs (N leading): dof_fut/dvel_fut (N,10,29), rootq_fut (N,10,4), rootz_fut (N,10),
    body_pos_cur (N,14,3), body_quat_cur (N,14,4), rootz_cur (N,), robot_anchor_quat_wxyz (N,4),
    wrist_dof_smpl (N,10,6) = smpl-window wrist dof.
    vr_targets: optional ((N,3,3) world pos, (N,3,4) wxyz quat); else 3 points come from motion.
    When ``vr_targets_local=True``, both arrays bypass localization.  The caller
    declares which desired-pose root they use.  NVIDIA's live PICO path supplies
    calibrated operator-current-pelvis-local values directly; offline training
    constructs the analogous terms from the desired motion pelvis.  Neither is
    a request to rotate external targets by the separate kinematic-planner root.
    hold_idx: IsaacLab joint idxs held at default 0 (phantom: joints absent on the robot -> don't command them)."""
    rq = np.asarray(robot_anchor_quat_wxyz, dtype=np.float64).reshape(-1, 4)  # (N,4)
    N = rq.shape[0]
    dof = np.asarray(dof_fut, dtype=np.float64).copy()      # (N,10,29)
    dvel = np.asarray(dvel_fut, dtype=np.float64).copy()
    if hold_idx is not None:                                # locked absent joints -> default 0, not the ref motion
        dof[:, :, hold_idx] = 0.0; dvel[:, :, hold_idx] = 0.0

    # 1. encoder_index: mode-id in slot 0 (g1=0/teleop=1/smpl=2), rest zeros (NOT one-hot)
    t1 = np.zeros((N, 4))
    t1[:, 0] = ENCODER_MODE_IDX[mode]

    # 2. command_multi_future_nonflat (580): [pos_f0..f9 (290) | vel_f0..f9 (290)], frame-major 29/frame
    t2 = np.concatenate([dof.reshape(N, -1), dvel.reshape(N, -1)], axis=1)  # 290 + 290

    # 3. command_z_multi_future (10): ref root z per future frame
    t3 = np.asarray(rootz_fut, dtype=np.float64).reshape(N, 10).copy()

    # 4. motion_anchor_ori_b_mf (60): quat_mul(quat_inv(robot_anchor), ref_root_quat_f) -> 6D per frame
    ref_q = np.asarray(rootq_fut, dtype=np.float64).reshape(N, 10, 4)
    dif = quat_mul(quat_inv(rq[:, None, :]), ref_q)          # broadcast (N,1,4) x (N,10,4)
    t4 = quat_to_rot6d(dif).reshape(N, -1)                   # 10*6

    # 5. command_multi_future_lower_body (240): [lb_pos_f0..f9 (120) | lb_vel_f0..f9 (120)]
    lb_pos = dof[:, :, LOWER_JOINT_IL_IDX].reshape(N, -1)    # 10*12
    lb_vel = dvel[:, :, LOWER_JOINT_IL_IDX].reshape(N, -1)
    t5 = np.concatenate([lb_pos, lb_vel], axis=1)

    # 6. vr_3point_local_target (9): 3 endpoints in desired-pose pelvis frame.
    bpc = np.asarray(body_pos_cur, dtype=np.float64).reshape(N, -1, 3)
    bqc = np.asarray(body_quat_cur, dtype=np.float64).reshape(N, -1, 4)
    a_pos = bpc[:, ANCHOR_BODY_IDX]                          # (N,3) ref pelvis pos
    a_q = bqc[:, ANCHOR_BODY_IDX]                            # (N,4) ref pelvis quat
    if vr_targets is None:                                  # from motion: body pos + endpoint offset
        vr_q = bqc[:, VR_BODY_IDX]                           # (N,3,4)
        vr_pos_w = bpc[:, VR_BODY_IDX] + quat_apply(vr_q, VR_ENDPOINT_OFFSETS[None])
    elif vr_targets_local:                                  # explicit local pass-through
        vr_pos_local, vr_q_local = _validated_vr_targets(vr_targets, N)
        t6 = vr_pos_local.reshape(N, -1)
        t7 = vr_q_local.reshape(N, -1)
    else:                                                   # external: already the head/hand targets
        vr_pos_w, vr_q = _validated_vr_targets(vr_targets, N)
    if vr_targets is None or not vr_targets_local:
        vr_diff = vr_pos_w - a_pos[:, None, :]
        t6 = quat_apply(quat_inv(a_q[:, None, :]), vr_diff).reshape(N, -1)

    # 7. local orientations, or offline quat_inv(ref_anchor) * vr_body_quat
    if vr_targets is None or not vr_targets_local:
        t7 = quat_mul(quat_inv(a_q[:, None, :]), vr_q).reshape(N, -1)

    # 8. motion_anchor_ori_b (6): ref-vs-robot root ori, 6D (current)
    _, ori8 = subtract_frame_transforms(np.zeros((N, 3)), rq, np.zeros((N, 3)), a_q)
    t8 = quat_to_rot6d(ori8)                                 # (N,6)

    # 9. command_z (1): ref root z at current frame
    t9 = np.asarray(rootz_cur, dtype=np.float64).reshape(N, 1).copy()

    # 10/11. smpl terms (720+60): zero-filled; smpl mode needs smpl_root_quat in npz (TODO).
    t10 = np.zeros((N, 720))
    t11 = np.zeros((N, 60))

    # 12. joint_pos_multi_future_wrist_for_smpl (60): 10 smpl frames x 6 wrist DOF pos
    t12 = np.asarray(wrist_dof_smpl, dtype=np.float64).reshape(N, -1)

    # deploy encoder order: z + current-frame anchor BEFORE multi-future anchor
    obs = np.concatenate([t1, t2, t3, t9, t8, t4, t5, t6, t7, t10, t11, t12], axis=1).astype(np.float32)
    assert obs.shape[1] == 1762, obs.shape
    return obs


# decoder obs (994), BATCHED over leading dim N
def build_decoder_obs(token64, history):
    """(N,994) buffer: token(64) | angvel(30) | jpos_rel(290) | jvel(290) | last_act(290) | grav(30). History RAW, frame-major.
    token64 (N,64); history arrays leading N: ang_vel (N,10,3), joint_pos_rel/joint_vel/last_action (N,10,29), gravity (N,10,3)."""
    tok = np.asarray(token64, dtype=np.float64).reshape(-1, 64)
    N = tok.shape[0]
    obs = np.concatenate([
        tok,
        np.asarray(history["ang_vel"]).reshape(N, -1),         # 10*3
        np.asarray(history["joint_pos_rel"]).reshape(N, -1),   # 10*29
        np.asarray(history["joint_vel"]).reshape(N, -1),       # 10*29
        np.asarray(history["last_action"]).reshape(N, -1),     # 10*29
        np.asarray(history["gravity"]).reshape(N, -1),         # 10*3
    ], axis=1).astype(np.float32)
    assert obs.shape[1] == 994, obs.shape
    return obs


def gravity_dir(base_quat_wxyz):
    """Gravity unit-vec in base frame."""
    return quat_apply(quat_inv(np.asarray(base_quat_wxyz, dtype=np.float64)), DOWN_DIR)
