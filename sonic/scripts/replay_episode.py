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


def depth_display_limits(source, *, native):
    """Use the recorded planner view, not the camera's wider capture range."""
    if not native:
        return .1, 3.
    session = source.parent if source.is_file() else source
    runtime = session/'model_trace/planner/runtime.json'
    if runtime.is_file():
        view = json.loads(runtime.read_text()).get('sampling_contract', {}).get('depth_condition', {})
        if view.get('enabled'):
            return float(view['min_depth_m']), float(view['max_depth_m'])
    return .1, 2.  # Native recovery without a model trace: current 2 m view.


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
    ap.add_argument('--live-estimates',action='store_true',help='Display only XYZ recorded by the live runtime')
    ap.add_argument('--external-camera',type=Path,help='Directory containing frames.mjpg and its timestamp index')
    ap.add_argument('--depth-recording',type=Path,help='Recorded metric depth directory')
    ap.add_argument('--full-validation',action='store_true',help='Decode every saved camera frame during validation')
    ap.add_argument('--allow-late-origin',action='store_true',help='Show a clearly labelled trajectory from the first available estimate if XYZ at policy start is missing')
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--urdf", default=DEFAULT_URDF_PATH)
    ap.add_argument("--deadzone", type=int, default=12,
                    help="tactile: suppress values within this of the resting baseline")
    args = ap.parse_args()
    if args.live_estimates and (not args.telemetry or args.measured_pose):ap.error('--live-estimates requires telemetry and no offline measured-pose')

    ep = Path(args.telemetry or args.capture or args.episode).expanduser()
    if args.telemetry or args.capture:
        state,get_frame,meta=load_native_recovery(ep,live_estimates=args.live_estimates) if args.telemetry else load_raw_motion(ep)
        if args.measured_pose:attach_measured_translation(state,meta,args.measured_pose,allow_late_origin=args.allow_late_origin)
    else:
        if args.measured_pose:ap.error('--measured-pose requires --telemetry or --capture')
        state, get_frame = load_episode(ep)
        meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}

    n = len(state["t"])
    sensors = list(meta.get("sensors", []))
    if sensors:
        from sonic.common.tactile_render import render_sensor_panels
    has_cam = "frame_offset" in state
    external_camera=None
    camera_indices=None
    if args.external_camera:
        from sonic.common.recorded_camera import RecordedCamera
        external_camera=RecordedCamera(args.external_camera)
        camera_query=state.get('capture_monotonic_s',state.get('receipt_monotonic_s'))
        if camera_query is None:ap.error('External camera replay requires robot monotonic timestamps')
        camera_indices=external_camera.indices_at(camera_query)
    recorded_depth=None;depth_indices=None;depth_error=None;depth_summary={}
    if args.depth_recording:
        import torch
        from motion_diffusion.behavioral.hardware.depth_recording import RecordedDepth
        torch.set_num_threads(1)
        try:
            recorded_depth=RecordedDepth(args.depth_recording)
            depth_query=state.get('depth_capture_monotonic_s',state.get('receipt_monotonic_s'))
            if depth_query is None:raise ValueError('Depth replay requires recorded monotonic timestamps')
            depth_indices=recorded_depth.indices_at(depth_query)
            summary=args.depth_recording/'summary.json'
            if summary.is_file():depth_summary=json.loads(summary.read_text())
        except (OSError,ValueError,KeyError) as exc:
            recorded_depth=None;depth_error=str(exc)
            print(f'Depth unavailable: {exc}. Continuing video and robot replay.',flush=True)

    print(f"{ep.name}: {n} steps, {meta.get('duration_s', 0):.1f}s, "
          f"tactile={sensors or 'none'}, camera={'yes' if has_cam or external_camera else 'no'}")
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
            translation_geometry_valid_fraction=float(state.get('position_geometry_valid',np.zeros(n,bool)).mean()),
            position_origin_delay_s=meta.get('position_origin_delay_s'),
            position_max_displacement_m=None if not valid.any() else float(np.linalg.norm(positions[valid],axis=1).max()),
            heading_max_change_deg=float(np.degrees(np.abs(state.get('yaw_relative_rad',np.zeros(n))).max())),
            external_camera=None if external_camera is None else external_camera.validate(camera_query,full=args.full_validation),
            depth_error=depth_error,
            depth=None if recorded_depth is None else dict(frames=len(recorded_depth.rows),
                timeline_coverage=float((depth_indices>=0).mean()),recording_summary=depth_summary))
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

    depth_min, depth_max = depth_display_limits(ep, native=bool(args.telemetry or args.capture))
    dv = DeployViser(
        urdf_path=Path(args.urdf).expanduser(),
        with_depth=has_cam or recorded_depth is not None,
        depth_min=depth_min,
        depth_max=depth_max,
        depth_scale=1.0,
        tactile_sensors=sensors,
        joint_names=joint_names or None,
        port=args.port,
        replay_layout=True,
    )
    with dv.server.gui.add_folder('Recording details',expand_by_default=False):
        if args.telemetry or args.capture:dv.server.gui.add_markdown(meta['explanation'])
        dv.server.gui.add_markdown('Video follows the recorded exposure timestamps. Depth shows the latest calibrated '
            'camera capture logged at each tick, before model encoding; it is not a traced planner dependency.'
            if recorded_depth else ('Depth unavailable: '+depth_error if depth_error else
                'Depth was not recorded in this run. Future recorded deployments save it automatically.'))
        if depth_summary.get('error') or depth_summary.get('dropped'):
            dv.server.gui.add_markdown(f'Depth recorder: {depth_summary}')
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
        connected=valid[:-1]&valid[1:]
        segments=np.stack([positions[:-1],positions[1:]],axis=1)[connected]
        if len(segments):
            geometry=state['position_geometry_valid']
            colors=np.tile(np.array([30,180,240],np.uint8),(len(segments),2,1))
            colors[~(geometry[:-1]&geometry[1:])[connected]]=(240,145,30)
            dv.server.scene.add_line_segments('/pelvis_path',points=segments,
                colors=colors,line_width=3.)
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

    folder = dv.server.gui.add_folder("Replay",order=-30)
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
        if has_cam or recorded_depth is not None:
            show_depth=dv.server.gui.add_checkbox('Show depth',True)
            @show_depth.on_update
            def on_depth_visibility(_):dv.depth_folder.visible=show_depth.value
        else:dv.server.gui.add_markdown('Depth: unavailable; see Recording details.' if depth_error else 'Depth: not recorded in this run.')
        label = dv.server.gui.add_text("t", "0.000 s")
        xyz_status=dv.server.gui.add_text('XYZ status','Unavailable',disabled=True)
    with dv.server.gui.add_folder('Motion values',expand_by_default=False):
        position_label=dv.server.gui.add_text('Relative XYZ (m)', 'Unavailable: pose-only replay')
        heading_label=dv.server.gui.add_text('Steering heading change (deg)', '-')
        orientation_label=dv.server.gui.add_text('Orientation from start (xyzw)', '-')
        camera_label=dv.server.gui.add_text('External camera', 'Unavailable' if external_camera is None else '-')
        err_label = dv.server.gui.add_text("worst err", "-")
        temp_label = dv.server.gui.add_text("ankle temps L/R", "-")

    last_camera_index=None;last_depth_index=None
    def render(k: int):
        nonlocal last_camera_index,last_depth_index
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
        if external_camera:
            frame_index=int(camera_indices[k])
            if frame_index != last_camera_index:
                color=external_camera.frame(frame_index)
                if color is None:color=np.zeros((270,480,3),np.uint8)
                last_camera_index=frame_index
            camera_label.value=('Unavailable: outside recording or camera gap' if frame_index<0 else
                f'frame {frame_index}; exposure {(camera_query[k]-external_camera.times[frame_index])*1000:.1f} ms before robot sample')
        if recorded_depth is not None and int(depth_indices[k])!=last_depth_index:
            last_depth_index=int(depth_indices[k])
            depth,valid=recorded_depth.frame(last_depth_index)
            if depth is None:depth=np.full(recorded_depth.view_shape,np.nan,dtype=np.float32)
            else:depth=np.where(valid,depth,np.nan)

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
                if meta.get('position_origin_delay_s',0)>0:position_label.value+=f' [from +{meta["position_origin_delay_s"]:.2f}s, not A]'
                if not state['position_geometry_valid'][k]:position_label.value+=' [tilt disagreement: inspect estimate]'
            else:position_label.value='Unavailable: missing or invalid position estimate'
            xyz_status.value=('Missing estimate' if not state['position_valid'][k] else
                ('Offline estimate' if args.measured_pose else 'Live estimate')+
                ('; tilt disagreement (orange trace)' if not state['position_geometry_valid'][k] else '; accuracy unqualified'))

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
