#!/usr/bin/env python3
"""Replay a recorded episode in viser: robot pose, camera, tactile.

Counterpart to replay_rollout.py, which reads deploy_vigor's flat .npz. This one
reads the episode directories deploy_eskin.py writes:

    episodes/episode_001/
        state.npz     one row per control step
        frames.bin    colour+depth, still encoded
        meta.json

    python scripts/replay_episode.py --episode ~/episodes/episode_001

Opens http://localhost:8080 -- scrub the Index slider or tick Play.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))
sys.path.insert(0,str(DEPLOY_ROOT.parent))

from sonic.common.episode import load_episode, load_native_recovery, load_raw_motion, attach_measured_translation  # noqa: E402
from core.common.viser_display import DeployViser  # noqa: E402
from core.config import DEFAULT_URDF_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", help="episode directory")
    source.add_argument("--telemetry", help="Native diffusion telemetry.jsonl.gz or its session directory")
    source.add_argument('--capture',type=Path,help='Raw sensor directory for sensor-only replay')
    ap.add_argument('--validate-only',action='store_true',help='Validate replay data and URDF without starting Viser')
    ap.add_argument('--validation-output',type=Path)
    ap.add_argument("--port",type=int,default=8080)
    ap.add_argument("--measured-pose",type=Path,help="Offline measured_pose.jsonl for relative XYZ; requires --telemetry")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--urdf", default=DEFAULT_URDF_PATH)
    ap.add_argument("--deadzone", type=int, default=12,
                    help="tactile: suppress values within this of the resting baseline")
    args = ap.parse_args()

    ep = Path(args.telemetry or args.capture or args.episode).expanduser()
    if args.telemetry or args.capture:
        state,get_frame,meta=load_native_recovery(ep) if args.telemetry else load_raw_motion(ep)
        if args.measured_pose:attach_measured_translation(state,meta,args.measured_pose)
    else:
        if args.measured_pose:ap.error('--measured-pose requires --telemetry or --capture')
        state, get_frame = load_episode(ep)
        meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}

    n = len(state["t"])
    sensors = list(meta.get("sensors", []))
    if sensors:
        from sonic.common.tactile_render import render_sensor_panels
    has_cam = "frame_offset" in state

    print(f"{ep.name}: {n} steps, {meta.get('duration_s', 0):.1f}s, "
          f"tactile={sensors or 'none'}, camera={'yes' if has_cam else 'no'}")
    print("fields:", ", ".join(sorted(k for k in state if k != "frame_offset")))

    qj = state["qj"].astype(np.float32) if "qj" in state else None
    imu = (state.get('display_quat_wxyz',state["imu_quat_wxyz"]).astype(np.float32)
           if "imu_quat_wxyz" in state else None)
    cmd_key = "command_q_physical" if "command_q_physical" in state else "command_q_23"
    cmd = state[cmd_key].astype(np.float32) if cmd_key in state else None
    temp = state.get("motor_temperature_29x2")
    err_key = "tracking_error_q_physical" if "tracking_error_q_physical" in state else "tracking_error_q_23"
    err = state[err_key].astype(np.float32) if err_key in state else None
    joint_names = list(meta.get("physical_joint_names", []))
    if not joint_names and qj is not None:
        from core.common.viser_display import JOINT_NAMES_BY_MOTOR_INDEX
        if qj.shape[1]!=len(JOINT_NAMES_BY_MOTOR_INDEX):
            raise ValueError('Episode needs physical_joint_names for this joint count')
        joint_names=list(JOINT_NAMES_BY_MOTOR_INDEX)

    if args.validate_only:
        import yourdfpy
        model=yourdfpy.URDF.load(args.urdf,load_meshes=False)
        missing=set(joint_names)-set(model.actuated_joint_names)
        if missing:raise ValueError(f'Recorded joints missing from URDF: {sorted(missing)}')
        if not np.isfinite(qj).all() or not np.isfinite(imu).all():raise ValueError('Nonfinite robot replay state')
        valid=state.get('position_valid',np.zeros(n,bool))
        positions=state.get('pelvis_position_relative_m')
        result=dict(frames=n,duration_s=float(state['t'][-1]),joints=len(joint_names),viewer_started=False,
            translation_present=positions is not None,translation_coverage=float(valid.mean()),
            position_max_displacement_m=None if not valid.any() else float(np.linalg.norm(positions[valid],axis=1).max()),
            heading_max_change_deg=float(np.degrees(np.abs(state.get('yaw_relative_rad',np.zeros(n))).max())))
        if args.validation_output:
            args.validation_output.parent.mkdir(parents=True,exist_ok=True)
            args.validation_output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return

    # Resting baseline per sensor, so light contact is visible rather than
    # swamped by each taxel's offset. Same idea as the live viewer.
    baselines = {}
    for s in sensors:
        key = f"tactile_{s.lower()}_raw_u8"
        if key in state and len(state[key]) >= 30:
            baselines[s] = np.round(state[key][:30].mean(axis=0)).astype(int).tolist()

    dv = DeployViser(
        urdf_path=Path(args.urdf).expanduser(),
        with_depth=has_cam,
        depth_min=0.1,
        depth_max=3.0,
        depth_scale=1.0,
        tactile_sensors=sensors,
        joint_names=joint_names or None,
        port=args.port,
    )
    if args.telemetry or args.capture:dv.server.gui.add_markdown(meta['explanation'])
    if args.measured_pose:
        report_path=args.measured_pose.parent/'evaluation.json'
        if report_path.is_file():
            report=json.loads(report_path.read_text())
            dv.server.gui.add_markdown(
                f"**Tracking diagnostics**\n\n"
                f"Heading disagreement RMS / max: {report['yaw_disagreement_rms_deg']:.2f} / "
                f"{report['yaw_disagreement_max_deg']:.2f} degrees.\n\n"
                f"Geometry checks passed: {100*report['geometry_valid_fraction']:.1f}%.\n\n"
                f"Largest odometry gap: {report['odometry_gap_max_ms']:.1f} ms.\n\n"
                "Live timing remains unqualified. Contacts: not implemented.")
    positions=state.get('pelvis_position_relative_m')
    if positions is not None:
        valid=state['position_valid']
        segments=np.stack([positions[:-1],positions[1:]],axis=1)[valid[:-1]&valid[1:]]
        if len(segments):
            dv.server.scene.add_line_segments('/pelvis_path',points=segments,
                colors=(30,180,240),line_width=3.)
        dv.server.scene.add_frame('/episode_origin',axes_length=.2,axes_radius=.006)

    # commanded pose as a second robot, 0.9 m to the side: divergence between
    # the two IS the tracking error (a dragged leg reads as measured lagging)
    cmd_vis = None
    if cmd is not None:
        from viser.extras import ViserUrdf

        cmd_frame = dv.server.scene.add_frame("/cmd", position=(0.0, 0.9, 0.0),
                                              show_axes=False)
        cmd_vis = ViserUrdf(dv.server, urdf_or_path=Path(args.urdf).expanduser(),
                            root_node_name="/cmd", load_meshes=True,
                            load_collision_meshes=False)

    folder = dv.server.gui.add_folder("Replay")
    with folder:
        idx = dv.server.gui.add_slider("Index", min=0, max=max(n - 1, 1), step=1,
                                       initial_value=0)
        play = dv.server.gui.add_checkbox("Play", False)
        restart = dv.server.gui.add_button("Restart")
        @restart.on_click
        def on_restart(_):
            idx.value=0
            play.value=False
        fps = dv.server.gui.add_number("FPS", float(args.fps))
        label = dv.server.gui.add_text("t", "0.000 s")
        position_label=dv.server.gui.add_text('Relative XYZ (m)', 'Unavailable: pose-only replay')
        heading_label=dv.server.gui.add_text('Steering heading change (deg)', '-')
        orientation_label=dv.server.gui.add_text('Orientation from start (xyzw)', '-')
        err_label = dv.server.gui.add_text("worst err", "-")
        temp_label = dv.server.gui.add_text("ankle temps L/R", "-")

    def render(k: int):
        ls = type("LS", (), {})()
        ls.tick = int(k)
        ls.mode_pr = 0
        ls.mode_machine = 0
        q = type("IMU", (), {})()
        q.quaternion = tuple(float(x) for x in (imu[k] if imu is not None
                                                else (1.0, 0.0, 0.0, 0.0)))
        ls.imu_state = q

        color = depth = None
        if has_cam:
            color, depth_mm = get_frame(k)
            if depth_mm is not None:
                depth = depth_mm.astype(np.float32) * 0.001   # mm -> m

        dv.update(ls, qj[k] if qj is not None else np.zeros(23, np.float32),
                  depth_raw=depth, rgb_bgr=color)
        if 'yaw_relative_rad' in state:
            heading_label.value=f'{np.degrees(state["yaw_relative_rad"][k]):+.2f}'
            orientation_label.value=', '.join(f'{v:+.4f}' for v in state['orientation_relative_xyzw'][k])
        if positions is not None:
            dv.base_frame.visible=bool(state['position_valid'][k])
            if state['position_valid'][k]:
                dv.base_frame.position=tuple(positions[k])
                position_label.value=', '.join(f'{v:+.3f}' for v in positions[k])
                if not state['position_geometry_valid'][k]:position_label.value+=' [tilt disagreement: inspect estimate]'
            else:position_label.value='Unavailable: missing or invalid position estimate'

        if cmd_vis is not None:
            cmd_vis.update_cfg({name: float(cmd[k][i])
                                for i, name in enumerate(joint_names)})
        if err is not None:
            e = np.degrees(np.abs(err[k]))
            j = int(e.argmax())
            err_label.value = f"{joint_names[j]} {e[j]:.1f} deg"
        if temp is not None:
            t = np.asarray(temp[k])
            temp_label.value = (f"p {t[4].max():.0f}/{t[10].max():.0f}  "
                                f"r {t[5].max():.0f}/{t[11].max():.0f} C")

        panels = {}
        for s in sensors:
            raw = state.get(f"tactile_{s.lower()}_raw_u8")
            if raw is not None:
                panels.update(render_sensor_panels(
                    s, raw[k].tolist(), baseline_values=baselines.get(s),
                    deadzone=args.deadzone))
        if panels:
            dv.update_tactile_images(panels)

        label.value = f"{float(state['t'][k]):.3f} s   step {k}/{n - 1}"

    render(0)
    print(f"\nviser: http://localhost:{args.port}",flush=True)

    k, last = 0, time.perf_counter()
    while True:
        kk = int(idx.value)
        if kk != k:
            k = kk
            render(k)
        if play.value:
            now = time.perf_counter()
            interval=(float(state['t'][k+1]-state['t'][k]) if args.telemetry and k<n-1 else 1.0/args.fps)
            if (now - last) >= interval*args.fps/max(1e-6,float(fps.value)):
                last = now
                k = min(k + 1, n - 1)
                idx.value = k
                render(k)
                if k == n - 1:
                    play.value = False
        time.sleep(0.002)


if __name__ == "__main__":
    main()
