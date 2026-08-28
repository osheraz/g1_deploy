"""Pure-numpy mirror of official SONIC observation gatherers.

The observation math remains in the official 29-DoF IsaacLab convention.  The
only embodiment-specific operation is the optional missing-joint reference
mask supplied by :mod:`robot_profile`.
"""

import numpy as np

from . import robot_g1
from .model_profile import get_model_profile

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
    heading_quat,
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
def build_encoder_obs(
    mode,
    dof_fut,
    dvel_fut,
    rootq_fut,
    rootz_fut,
    body_pos_cur,
    body_quat_cur,
    rootz_cur,
    robot_anchor_quat_wxyz,
    wrist_dof_smpl,
    vr_targets=None,
    hold_idx=None,
    vr_targets_local=False,
    *,
    model_profile="default",
    smpl_joints_fut=None,
    smpl_rootq_fut=None,
):
    """Build the exact released encoder flat buffer.

    ``mode`` selects one encoder and every term not consumed by that encoder is
    zero-filled, as in the official C++ gatherer.  Model families are not
    interchangeable: Default is 1762-D, low-latency 1247-D, and v1.1 1751-D.
    SMPL positions are the publisher's already pelvis-local ``(N,F,24,3)``
    representation; root quaternions are world WXYZ values.
    """
    profile = (
        get_model_profile(model_profile)
        if isinstance(model_profile, str)
        else model_profile
    )
    gf = profile.g1_future_frames
    sf = profile.smpl_future_frames
    rq = np.asarray(robot_anchor_quat_wxyz, dtype=np.float64).reshape(-1, 4)  # (N,4)
    N = rq.shape[0]
    dof = np.asarray(dof_fut, dtype=np.float64).reshape(N, gf, 29).copy()
    dvel = np.asarray(dvel_fut, dtype=np.float64).reshape(N, gf, 29).copy()
    if hold_idx is not None:                                # locked absent joints -> default 0, not the ref motion
        dof[:, :, hold_idx] = 0.0; dvel[:, :, hold_idx] = 0.0

    # 1. encoder_index: mode-id in slot 0 (g1=0/teleop=1/smpl=2), rest zeros (NOT one-hot)
    t1 = np.zeros((N, 4))
    t1[:, 0] = ENCODER_MODE_IDX[mode]

    # 2. command_multi_future_nonflat (580): [pos_f0..f9 (290) | vel_f0..f9 (290)], frame-major 29/frame
    t2 = np.concatenate([dof.reshape(N, -1), dvel.reshape(N, -1)], axis=1)  # 290 + 290

    # 3. command_z_multi_future (10): ref root z per future frame
    t3 = np.asarray(rootz_fut, dtype=np.float64).reshape(N, gf).copy()

    # 4. motion_anchor_ori_b_mf (60): quat_mul(quat_inv(robot_anchor), ref_root_quat_f) -> 6D per frame
    ref_q = np.asarray(rootq_fut, dtype=np.float64).reshape(N, gf, 4)
    robot_orientation = heading_quat(rq) if profile.heading_normalized else rq
    dif = quat_mul(quat_inv(robot_orientation[:, None, :]), ref_q)
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
    _, ori8 = subtract_frame_transforms(
        np.zeros((N, 3)), robot_orientation, np.zeros((N, 3)), a_q
    )
    t8 = quat_to_rot6d(ori8)                                 # (N,6)

    # 9. command_z (1): ref root z at current frame
    t9 = np.asarray(rootz_cur, dtype=np.float64).reshape(N, 1).copy()

    # 10/11. SMPL local joints and root orientation.
    if smpl_joints_fut is None or smpl_rootq_fut is None:
        if mode == "smpl":
            raise ValueError(
                "SMPL encoder mode requires smpl_joints_fut and smpl_rootq_fut"
            )
        t10 = np.zeros((N, sf * 24 * 3))
        t11 = np.zeros((N, sf * 6))
    else:
        t10 = np.asarray(smpl_joints_fut, dtype=np.float64).reshape(N, sf * 24 * 3)
        smpl_q = np.asarray(smpl_rootq_fut, dtype=np.float64).reshape(N, sf, 4)
        smpl_dif = quat_mul(quat_inv(robot_orientation[:, None, :]), smpl_q)
        t11 = quat_to_rot6d(smpl_dif).reshape(N, sf * 6)

    # 12. joint_pos_multi_future_wrist_for_smpl (60): 10 smpl frames x 6 wrist DOF pos
    wrist = np.asarray(wrist_dof_smpl, dtype=np.float64).reshape(N, sf, 6).copy()
    if hold_idx is not None:
        wrist[:, :, np.isin(WRIST_IL_IDX, np.asarray(hold_idx, dtype=np.int64))] = 0.0
    t12 = wrist.reshape(N, -1)

    # The YAML listing order is artifact identity, not the exported graph's
    # flat-input order.  In particular, Low Latency is single-anchor-first
    # while v1.1 is multi-anchor-first even though neither has root-height
    # terms.  Keep that as an explicit model-profile contract.
    values = {
        "mode": t1,
        "joint_future": t2,
        "root_height_future": t3,
        "root_height_current": t9,
        "anchor_single": t8,
        "anchor_multi": t4,
        "lower_body_future": t5,
        "vr_position": t6,
        "vr_orientation": t7,
        "smpl_joints": t10,
        "smpl_anchor": t11,
        "smpl_wrists": t12,
    }
    anchors = {
        "single_then_multi": ("anchor_single", "anchor_multi"),
        "multi_then_single": ("anchor_multi", "anchor_single"),
    }[profile.encoder_anchor_order]
    if profile.include_root_height_terms:
        layout = (
            "mode", "joint_future", "root_height_future",
            "root_height_current", *anchors,
            "lower_body_future", "vr_position", "vr_orientation",
            "smpl_joints", "smpl_anchor", "smpl_wrists",
        )
    else:
        layout = (
            "mode", "joint_future", *anchors, "lower_body_future",
            "vr_position", "vr_orientation", "smpl_joints",
            "smpl_anchor", "smpl_wrists",
        )
    required_parts = {
        "g1": {"mode", "joint_future", "anchor_multi"},
        "teleop": {
            "mode", "anchor_single", "lower_body_future",
            "vr_position", "vr_orientation",
        },
        "smpl": {"mode", "smpl_joints", "smpl_anchor", "smpl_wrists"},
    }
    if mode not in required_parts:
        raise ValueError(
            f"unsupported SONIC encoder mode {mode!r}; expected one of "
            f"{sorted(required_parts)}"
        )
    keep = required_parts[mode]
    masked_parts = [
        values[name] if name in keep else np.zeros_like(values[name])
        for name in layout
    ]
    obs = np.concatenate(masked_parts, axis=1).astype(np.float32)
    if obs.shape != (N, profile.encoder_input_dim):
        raise AssertionError(
            f"{profile.name} encoder observation shape {obs.shape}; expected "
            f"({N}, {profile.encoder_input_dim})"
        )
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
