"""Closed-loop sim2sim test: exported tracker ONNX + ReferenceProvider in MuJoCo.

Reuses the deploy obs path (policy/reference_provider.py + the sorted actor_obs
concat) so a pass here exercises the actual deploy code in a different engine.
Reports per-joint tracking error vs the reference clip.

  python tools/test_tracker_mujoco.py \
    --onnx policy/model_last.onnx --clip policy/reference_clip.npz \
    --mjcf <repo>/others/TWIST/assets/g1/g1_sim2sim_with_wrist_roll.xml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy.reference_provider import ReferenceProvider

DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
]
DEFAULT = np.array([-0.2, 0, 0, 0.42, -0.23, 0, -0.2, 0, 0, 0.42, -0.23, 0, 0.0,
                    0.35, 0.16, 0, 0.87, 0, 0.35, -0.16, 0, 0.87, 0], np.float32)
# training gains (humanoidverse2 g1_23dof.yaml stiffness/damping), NOT deploy/real gains
KP = np.array([100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20, 400,
               90, 60, 20, 60, 40, 90, 60, 20, 60, 40], np.float32)
KD = np.array([2.5, 2.5, 2.5, 5, 0.2, 0.1, 2.5, 2.5, 2.5, 5, 0.2, 0.1, 5.0,
               2, 1, 0.4, 1, 1.0, 2, 1, 0.4, 1, 1.0], np.float32)
HOLD = {"waist_roll_joint": 0.0, "waist_pitch_joint": 0.0}  # not in 23dof


def gravity_orientation(q_wxyz):
    qw, qx, qy, qz = q_wxyz
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--control-dt", type=float, default=0.02)
    ap.add_argument("--future-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--action-scale", type=float, default=0.25)
    ap.add_argument("--init-height", type=float, default=0.78)
    ap.add_argument("--zero-action", action="store_true",
                    help="diagnostic: hold default pose, ignore policy")
    ap.add_argument("--action-ema", type=float, default=1.0,
                    help="EMA on action: 1.0 off, lower = smoother")
    ap.add_argument("--foot-radius", type=float, default=0.022,
                    help="foot sphere radius (TWIST 0.005 is near-point; 0 keeps MJCF)")
    ap.add_argument("--trace", action="store_true", help="print per-step height/tilt/dof-err")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.mjcf)
    m.opt.timestep = 0.001
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.jnt_stiffness[:] = 0.0          # PD provides all stiffness/damping; keep armature
    m.dof_damping[:] = 0.0
    m.dof_frictionloss[:] = 0.0
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fid >= 0:
        m.geom_priority[fid] = 1
        m.geom_friction[fid, 0] = 0.8
        m.geom_condim[fid] = 3
    d = mujoco.MjData(m)
    decim = max(1, round(args.control_dt / m.opt.timestep))

    if args.foot_radius > 0:  # fatten foot point-contacts (training URDF foot is denser)
        for bn in ("left_ankle_roll_link", "right_ankle_roll_link"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
            for gi in range(m.ngeom):
                if m.geom_bodyid[gi] == bid and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    m.geom_size[gi, 0] = args.foot_radius

    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in DOF_NAMES}
    qadr = np.array([m.jnt_qposadr[jid[n]] for n in DOF_NAMES])
    vadr = np.array([m.jnt_dofadr[jid[n]] for n in DOF_NAMES])
    torso_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    # affine position-actuator PD (ProtoMotions): force = kp*(ctrl-q) - kd*qd, integrated implicitly
    act_for_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}

    def _set_pd(a, kp, kd, eff):
        m.actuator_gaintype[a] = 0
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biastype[a] = 1
        m.actuator_biasprm[a, 0] = 0.0
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kd
        m.actuator_ctrllimited[a] = 0
        if eff > 0:
            m.actuator_forcerange[a] = [-eff, eff]
            m.actuator_forcelimited[a] = 1

    pd_act = np.array([act_for_joint[jid[n]] for n in DOF_NAMES])
    for k, n in enumerate(DOF_NAMES):
        j = jid[n]
        eff = float(m.jnt_actfrcrange[j, 1]) if m.jnt_actfrclimited[j] else 0.0
        _set_pd(pd_act[k], float(KP[k]), float(KD[k]), eff)
    hold_act, hold_tgt = [], []
    for n, v in HOLD.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0 and j in act_for_joint:
            _set_pd(act_for_joint[j], 200.0, 5.0, 0.0)
            hold_act.append(act_for_joint[j]); hold_tgt.append(v)
    hold_act = np.array(hold_act, dtype=int)
    hold_tgt = np.array(hold_tgt, np.float32)

    sess = ort.InferenceSession(args.onnx, sess_options=_so())
    ref = ReferenceProvider(args.clip, DEFAULT, args.future_steps, "clamp")

    # init on the clip's frame 0 (the Isaac eval resets the robot onto the reference)
    if ref.root_pos is not None:
        d.qpos[:3] = ref.root_pos[0]
    else:
        d.qpos[:3] = [0, 0, args.init_height]
    d.qpos[3:7] = ref.torso_quat[0][[3, 0, 1, 2]]   # xyzw -> wxyz
    d.qpos[qadr] = ref.dof_pos[0]
    d.qvel[vadr] = ref.dof_vel[0]
    mujoco.mj_forward(m, d)
    action = np.zeros(23, np.float32)

    errs, fell = [], False
    for step in range(ref.T):
        q = d.qpos[qadr].astype(np.float32)
        qd = d.qvel[vadr].astype(np.float32)
        base_q = d.qpos[3:7].astype(np.float32)              # pelvis wxyz (gravity/ang_vel)
        torso_q = d.xquat[torso_bid].astype(np.float32)      # torso wxyz (proto anchor)
        ang_vel = d.qvel[3:6].astype(np.float32)             # base-local
        proj_g = gravity_orientation(base_q)

        vals = {
            "actions": action,
            "base_ang_vel": ang_vel * 0.25,
            "dof_pos": (q - DEFAULT) * 1.0,
            "dof_vel": qd * 0.1,
            "future_ref_motion_proto": ref.future_proto(torso_q),
            "projected_gravity": proj_g,
        }
        obs = np.concatenate([vals[k] for k in sorted(vals)])[None].astype(np.float32)
        new_action = sess.run(None, {sess.get_inputs()[0].name: obs})[0].squeeze()
        new_action = np.clip(new_action, -100, 100).astype(np.float32)
        # EMA smoothing (ProtoMotions deploy): alpha=1 off, lower = smoother
        action = args.action_ema * new_action + (1.0 - args.action_ema) * action

        if args.trace and step < 40:
            de = float(np.abs(q - ref.dof_pos[ref.i]).mean())
            print(f"  t{step:3d} h={d.qpos[2]:.3f} gz={proj_g[2]:+.2f} "
                  f"gx={proj_g[0]:+.2f} gy={proj_g[1]:+.2f} dof_err={de:.3f}", flush=True)

        target = DEFAULT if args.zero_action else action * args.action_scale + DEFAULT
        d.ctrl[pd_act] = target
        if len(hold_act):
            d.ctrl[hold_act] = hold_tgt
        for _ in range(decim):
            mujoco.mj_step(m, d)

        errs.append(np.abs(d.qpos[qadr] - ref.dof_pos[ref.i]).mean())
        if d.qpos[2] < 0.4:
            fell = True
            print(f"FELL at step {step}/{ref.T} (height {d.qpos[2]:.2f})")
            break
        ref.step()

    errs = np.array(errs)
    print(f"steps {len(errs)}/{ref.T}  fell={fell}")
    print(f"dof tracking err: mean {errs.mean():.4f} rad  final {errs[-1]:.4f}  max {errs.max():.4f}")


def _so():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return so


if __name__ == "__main__":
    main()
