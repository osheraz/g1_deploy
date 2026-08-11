"""Closed-loop sim2sim gate for the SONIC tracker in MuJoCo.

Runs the real deploy objects -- SonicRunner and MotionReferenceWindow, with the
gains, default pose and per-joint action_scale from the config -- against a
different physics engine than the one they were trained in. A pass here is the
prerequisite for touching hardware; a fall here means do not.

  python scripts/sim2sim_sonic.py \
      --mjcf ~/GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_23dof.xml

Reports height, tilt and per-joint tracking error against the clip.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from policy.sonic_reference import MotionReferenceWindow
from policy.sonic.robot_profile import MUJOCO_JOINT_NAMES, get_robot_profile
from common.scripted_teleop import ScriptedTeleop
from policy.sonic_runner import SonicRunner

def gravity_orientation(q_wxyz):
    qw, qx, qy, qz = q_wxyz
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)], np.float32)


def draw_command(viewer, d, teleop):
    """Spheres at the VR targets, a line along the movement direction."""
    scn = viewer.user_scn
    scn.ngeom = 0
    root, quat = d.qpos[:3], d.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)

    vr = teleop.vr_3point()
    if vr is not None:
        colors = [[0.2, 1, 0.4, 1], [1, 0.4, 0.2, 1], [0.2, 0.6, 1, 1]]  # L, R, torso
        for p, c in zip(vr[0], colors):
            w = root + R @ p
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.04, 0, 0]), w, np.eye(3).flatten(),
                                np.array(c, np.float32))
            scn.ngeom += 1

    nav = teleop.nav_command()
    if nav["speed"] > 0:
        tip = root + R @ (np.asarray(nav["movement"]) * 0.6)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array([1, 1, 0.2, 1], np.float32))
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 5,
                             root.astype(np.float64), tip.astype(np.float64))
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="g1_23dof_sonic.yaml")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--clip", default=None, help="default: sonic_motion_clip from the config")
    ap.add_argument("--pico", choices=["nav", "vr3", "both", "idle", "carry", "joystick"],
                    default=None,
                    help="drive the planner with a scripted teleop command")
    ap.add_argument("--planner", action="store_true",
                    help="force the G1 encoder, exactly deploy --mode planner")
    ap.add_argument("--idle-hold", action="store_true",
                    help="IDLE holds the default pose instead of the planner's "
                         "stepping idle plan, as deploy --idle-hold")
    ap.add_argument("--slow", action="store_true",
                    help="joystick SLOW_WALK instead of WALK, as deploy --slow")
    ap.add_argument("--speed", type=float, default=0.2, help="--pico nav: m/s")
    ap.add_argument("--axes", default=None,
                    help="--pico joystick: fixed 'lx,ly,rx' instead of the keyboard")
    ap.add_argument("--axes-until", type=int, default=None,
                    help="release the stick (axes to 0) after N steps")
    ap.add_argument("--remote", action="store_true",
                    help="--pico joystick: real Unitree remote via DDS "
                         "(robot on and reachable; nothing publishes lowcmd)")
    ap.add_argument("--blend", action=argparse.BooleanOptionalAction, default=True,
                    help="same one-second raised-cosine handover as deployment "
                         "(use --no-blend only for official parity diagnostics)")
    ap.add_argument("--replan-ungated", action="store_true",
                    help="replan on every command instead of official_gated, as deploy")
    ap.add_argument("--steps", type=int, default=None, help="default: whole clip")
    ap.add_argument("--control-dt", type=float, default=None)
    ap.add_argument("--init-height", type=float, default=0.78)
    ap.add_argument("--zero-action", action="store_true",
                    help="diagnostic: hold default pose, ignore policy")
    ap.add_argument("--trace", action="store_true", help="print per-step height/tilt/dof-err")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--realtime", action="store_true",
                    help="with --view, pace the sim to wall clock")
    args = ap.parse_args()

    cfg = Config(args.config)
    profile = get_robot_profile(cfg.sonic_robot_profile)
    dof_names = list(profile.physical_joint_names)
    default = np.asarray(cfg.physical_default_angles, np.float32)
    idx = np.asarray(cfg.action_joint2motor_idx)
    kp = np.asarray(cfg.kps, np.float32)[idx]
    kd = np.asarray(cfg.kds, np.float32)[idx]
    action_scale = np.asarray(cfg.action_scale, np.float32)
    if args.control_dt is None:
        args.control_dt = float(cfg.control_dt)

    # MJCF as authored: its own timestep, damping, friction, foot geoms
    m = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).expanduser()))
    d = mujoco.MjData(m)
    decim = max(1, round(args.control_dt / m.opt.timestep))

    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in dof_names}
    missing_model_joints = [name for name, joint_id in jid.items() if joint_id < 0]
    if missing_model_joints:
        raise SystemExit(
            f"{args.mjcf} does not contain the {profile.name} physical joints: "
            f"{missing_model_joints}"
        )
    qadr = np.array([m.jnt_qposadr[jid[n]] for n in dof_names])
    vadr = np.array([m.jnt_dofadr[jid[n]] for n in dof_names])

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

    try:
        pd_act = np.array([act_for_joint[jid[n]] for n in dof_names])
    except KeyError as exc:
        raise SystemExit(f"physical joint has no MuJoCo actuator: joint id {exc.args[0]}") from exc
    for k, n in enumerate(dof_names):
        j = jid[n]
        eff = float(m.jnt_actfrcrange[j, 1]) if m.jnt_actfrclimited[j] else 0.0
        _set_pd(pd_act[k], float(kp[k]), float(kd[k]), eff)
    # Undriven joints get the config's fixed_kps/fixed_kds, as on the robot.
    fkp = np.asarray(cfg.fixed_kps, np.float32)
    fkd = np.asarray(cfg.fixed_kds, np.float32)
    hold_act, hold_tgt = [], []
    fixed_names = [MUJOCO_JOINT_NAMES[i] for i in cfg.fixed_joint2motor_idx]
    for k, n in enumerate(fixed_names):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0 and j in act_for_joint:
            _set_pd(act_for_joint[j], float(fkp[k]), float(fkd[k]), 0.0)
            hold_act.append(act_for_joint[j])
            hold_tgt.append(float(cfg.fixed_target[k]))
    hold_act = np.array(hold_act, dtype=int)
    hold_tgt = np.array(hold_tgt, np.float32)

    teleop = None
    if args.pico:
        from policy.pico_reference import PicoReferenceWindow

        if args.pico == "joystick":
            from common.joystick_teleop import JoystickTeleop

            from common.pico_client import LocomotionMode
            teleop = JoystickTeleop(
                locomotion_mode=LocomotionMode.SLOW_WALK if args.slow
                else LocomotionMode.WALK)
            if args.axes:
                teleop.set_axes(*map(float, args.axes.split(",")))
            if args.remote:
                from unitree_sdk2py.core.channel import (
                    ChannelFactoryInitialize, ChannelSubscriber)
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
                from common.remote_controller import RemoteController

                remote = RemoteController()
                ChannelFactoryInitialize(0)
                sub = ChannelSubscriber(cfg.lowstate_topic, LowState_)
                sub.Init(lambda msg: remote.set(msg.wireless_remote), 10)
                print("remote: reading Unitree sticks from rt/lowstate")
                last_remote = (0.0, 0.0, 0.0)
        else:
            teleop = ScriptedTeleop(args.pico, speed=args.speed)
        # wall-paced viewing uses the deploy's planner thread; headless sim
        # outruns the wall clock, so it replans synchronously instead
        ref = PicoReferenceWindow(cfg, teleop, threaded=args.realtime,
                                  gated=not args.replan_ungated,
                                  force_g1=args.planner,
                                  idle_hold=args.idle_hold)
        runner = SonicRunner(cfg, ref)
        ref_dof = None
        d.qpos[:3] = [0.0, 0.0, 0.793]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[qadr] = default
        mujoco.mj_forward(m, d)
        ref.initialize_from_robot(d.qpos[3:7].copy(), d.qpos[qadr].copy())
        # pico: run until the viewer closes; headless still needs a bound
        n_steps = args.steps or (10**9 if args.view else 500)
    else:
        clip_path = args.clip or cfg.sonic_motion_clip
        ref = MotionReferenceWindow(
            clip_path,
            loop=False,
            model_profile=cfg.sonic_model_variant,
            robot_profile=cfg.sonic_robot_profile,
        )
        runner = SonicRunner(cfg, ref)
        clip = np.load(Path(clip_path).expanduser())
        ref_dof = profile.isaaclab_to_physical(clip["dof_il"])
        ref_vel = profile.isaaclab_to_physical(clip["dof_vel_il"])
        root_pos = clip["body_pos_il"][:, 0]
        root_quat = clip["body_quat_wxyz_il"][:, 0]

        # start on the clip's frame 0, as the Isaac eval does
        d.qpos[:3] = [0.0, 0.0, float(root_pos[0, 2])]
        d.qpos[3:7] = root_quat[0]
        d.qpos[qadr] = ref_dof[0]
        d.qvel[vadr] = ref_vel[0]
        mujoco.mj_forward(m, d)
        ref.initialize_from_robot(d.qpos[3:7].copy(), d.qpos[qadr].copy())
        n_steps = args.steps or ref.T

    viewer = None
    key_cb = None
    if args.pico == "joystick":
        # tap to set, X stops: W/S fwd/back, A/D strafe, Q/E turn
        # GLFW arrows: they don't collide with the viewer's letter shortcuts
        UP, DOWN, RIGHT, LEFT = 265, 264, 262, 263
        KEYS = {UP: (0, -1, 0), DOWN: (0, 1, 0),
                LEFT: (-1, 0, 0), RIGHT: (1, 0, 0),
                ord("["): (0, 0, -1), ord("]"): (0, 0, 1)}

        def key_cb(key):
            if key == ord("0"):
                teleop.set_axes(0, 0, 0)
            elif key in KEYS:
                dlx, dly, drx = KEYS[key]
                teleop.set_axes(np.clip(teleop.lx + 0.5 * dlx, -1, 1),
                                np.clip(teleop.ly - 0.5 * dly, -1, 1),
                                np.clip(teleop.rx + 0.5 * drx, -1, 1))
            print(f"\raxes lx={teleop.lx:+.1f} ly={teleop.ly:+.1f} "
                  f"rx={teleop.rx:+.1f}   ", end="", flush=True)

    if args.view:
        # not `import mujoco.viewer`: that shadows the module-level import
        from mujoco import viewer as _mj_viewer

        viewer = _mj_viewer.launch_passive(m, d, key_callback=key_cb)
        if key_cb is not None:
            print("keys: arrows move  [ ] turn  0 stop")
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        print("viewer open -- close the window to stop")

    import time as _time
    t_wall = _time.perf_counter()

    qlog = []
    errs, heights, tilts, fell = [], [], [], False
    handover_origin = d.qpos[qadr].astype(np.float32).copy()
    for step in range(n_steps):
        if viewer is not None and not viewer.is_running():
            print("viewer closed")
            break
        q = d.qpos[qadr].astype(np.float32)
        qd = d.qvel[vadr].astype(np.float32)
        base_q = d.qpos[3:7].astype(np.float32)              # pelvis wxyz
        ang_vel = d.qvel[3:6].astype(np.float32)             # base-local
        proj_g = gravity_orientation(base_q)

        # SONIC builds its own observation from RAW state; this is exactly the
        # call deploy_eskin.py makes.
        action = runner.compute_action(
            dof_pos=q.astype(np.float64),
            dof_vel=qd.astype(np.float64),
            base_quat_wxyz=base_q.astype(np.float64),
            base_ang_vel_body=ang_vel.astype(np.float64),  # qvel[3:6] is body frame
        )

        if args.trace and step < 40 and ref_dof is not None:
            de = float(np.abs(q - ref_dof[min(ref.frame, len(ref_dof) - 1)]).mean())
            print(f"  t{step:3d} h={d.qpos[2]:.3f} gz={proj_g[2]:+.2f} "
                  f"gx={proj_g[0]:+.2f} gy={proj_g[1]:+.2f} dof_err={de:.3f}", flush=True)

        target = default if args.zero_action else action * action_scale + default
        if args.blend and step < 50:
            phase = step / 49.0
            alpha = 0.5 - 0.5 * np.cos(np.pi * phase)
            target = alpha * target + (1.0 - alpha) * handover_origin
        d.ctrl[pd_act] = np.clip(target, cfg.dof_lower, cfg.dof_upper)
        if len(hold_act):
            d.ctrl[hold_act] = hold_tgt
        for _ in range(decim):
            mujoco.mj_step(m, d)

        if teleop is not None:
            if args.remote:
                cur = (remote.lx, remote.ly, remote.rx)
                if cur != last_remote:      # dead/idle remote must not clobber keys
                    last_remote = cur
                    teleop.set_axes(*cur)
            if args.axes_until is not None and step == args.axes_until:
                teleop.set_axes(0, 0, 0)
            teleop.tick(args.control_dt)

        if viewer is not None:
            if teleop is not None:
                draw_command(viewer, d, teleop)
            viewer.sync()
            if args.realtime:
                t_wall += args.control_dt
                lag = t_wall - _time.perf_counter()
                if lag > 0:
                    _time.sleep(lag)

        if ref_dof is not None:
            f = min(ref.frame, len(ref_dof) - 1)
            errs.append(np.abs(d.qpos[qadr] - ref_dof[f]).mean())
        heights.append(float(d.qpos[2]))
        qlog.append(d.qpos[qadr].copy())
        tilts.append(float(np.degrees(np.arccos(np.clip(-proj_g[2], -1, 1)))))
        if d.qpos[2] < 0.4:
            fell = True
            print(f"FELL at step {step}/{n_steps} (height {d.qpos[2]:.2f})")
            break
        ref.step()

    if viewer is not None:
        viewer.close()
    errs, heights, tilts = np.array(errs), np.array(heights), np.array(tilts)
    Q = np.degrees(np.ptp(np.array(qlog), axis=0))
    print(f"\nsteps {len(heights)}/{n_steps}" + ("  FELL" if fell else ""))
    if errs.size:
        print(f"  track err  {np.degrees(errs.mean()):.1f} deg")
    print(f"  travel     {np.linalg.norm(d.qpos[:2]):.2f} m")
    print(f"  arm/leg    {Q[13:].max():.0f} / {Q[:12].max():.0f} deg")
    print(f"  height     {heights.min():.3f} .. {heights.max():.3f} m")
    print(f"  tilt       {tilts.max():.1f} deg")


if __name__ == "__main__":
    main()
