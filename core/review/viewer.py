#!/usr/bin/env python3
"""Replay a recorded episode in viser: robot pose, camera, tactile.

Counterpart to vigor/replay.py, which reads VIGOR's flat .npz logs. This one
reads the episode directories sonic/deploy.py writes:

    episodes/episode_001/
        state.npz     one row per control step
        frames.bin    colour+depth, still encoded
        meta.json

    python g1_deploy/core/review/viewer.py --episode ~/episodes/episode_001

Opens http://localhost:8080 -- scrub the Index slider or tick Play.
``--sessions FILE`` replays several recordings in one viewer with a dropdown and
Previous/Next buttons; ``--preload`` loads all of them before opening.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))
sys.path.insert(0, str(DEPLOY_ROOT.parent))

from core.recording.episode import (
    load_episode,
    load_native_recovery,
    load_raw_motion,
    attach_measured_translation,
    NoControlObservations,
)
from core.recording.annotations import RecoveryInterval, validate_interval, session_directory
from core.review.display import DeployViser  # noqa: E402
from core.config import DEFAULT_URDF_PATH  # noqa: E402


def depth_display_limits(source, *, native):
    """Use the recorded planner view, not the camera's wider capture range."""
    if not native:
        return 0.1, 3.0
    session = source.parent if source.is_file() else source
    if session.name == "review" and (session.parent / "telemetry.jsonl.gz").is_file():
        session = session.parent  # Derived telemetry retains its session's model view.
    runtime = session / "model_trace/planner/runtime.json"
    if runtime.is_file():
        view = (
            json.loads(runtime.read_text()).get("sampling_contract", {}).get("depth_condition", {})
        )
        if view.get("enabled"):
            return float(view["min_depth_m"]), float(view["max_depth_m"])
    return 0.1, 2.0  # Native recovery without a model trace: current 2 m view.


def session_spec(args):
    text = lambda value: None if value is None else str(value)
    return dict(
        label=Path(args.telemetry or args.capture or args.episode).expanduser().name,
        telemetry=args.telemetry,
        capture=text(args.capture),
        episode=args.episode,
        measured_pose=text(args.measured_pose),
        allow_late_origin=bool(args.allow_late_origin),
        live_estimates=bool(args.live_estimates),
        external_camera=text(args.external_camera),
        depth_recording=text(args.depth_recording),
    )


def load_session(spec):
    """Everything one recording contributes to the viewer; frames stay on disk."""
    ep = Path(spec.get("telemetry") or spec.get("capture") or spec["episode"]).expanduser()
    external_camera = camera_query = camera_indices = None
    if spec.get("external_camera"):
        from core.recording.camera import RecordedCamera

        external_camera = RecordedCamera(Path(spec["external_camera"]))
    if spec.get("telemetry") or spec.get("capture"):
        try:
            state, get_frame, meta = (
                load_native_recovery(ep, live_estimates=bool(spec.get("live_estimates")))
                if spec.get("telemetry")
                else load_raw_motion(ep)
            )
        except NoControlObservations:
            if external_camera is None:
                raise
            times = external_camera.times
            state = dict(t=times - times[0], playback_monotonic_s=times, receipt_monotonic_s=times)
            get_frame = lambda _: (None, None)
            meta = dict(
                camera_only=True,
                timeline_source="external_camera_exposure",
                duration_s=float(times[-1] - times[0]),
                physical_joint_names=[],
                explanation="Camera-only recording: no policy observations or XYZ trajectory. "
                "Playback starts at the first recorded camera exposure; robot pose is unavailable.",
            )
        if spec.get("measured_pose") and not meta.get("camera_only"):
            attach_measured_translation(
                state,
                meta,
                Path(spec["measured_pose"]),
                allow_late_origin=bool(spec.get("allow_late_origin")),
            )
    else:
        state, get_frame = load_episode(ep)
        meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}
    n = len(state["t"])
    if external_camera is not None:
        camera_query = state.get(
            "playback_monotonic_s",
            state.get("capture_monotonic_s", state.get("receipt_monotonic_s")),
        )
        if camera_query is None:
            raise ValueError("External camera replay requires robot monotonic timestamps")
        camera_indices = external_camera.indices_at(camera_query)
    recorded_depth = depth_indices = depth_error = None
    depth_summary = {}
    if spec.get("depth_recording"):
        import torch
        from rise.recording.depth import RecordedDepth

        torch.set_num_threads(1)
        directory = Path(spec["depth_recording"])
        try:
            recorded_depth = RecordedDepth(directory)
            depth_query = state.get("depth_capture_monotonic_s", state.get("receipt_monotonic_s"))
            if depth_query is None:
                raise ValueError("Depth replay requires recorded monotonic timestamps")
            depth_indices = recorded_depth.indices_at(depth_query)
            summary = directory / "summary.json"
            if summary.is_file():
                depth_summary = json.loads(summary.read_text())
        except (OSError, ValueError, KeyError) as exc:
            recorded_depth = None
            depth_error = str(exc)
            print(f"Depth unavailable: {exc}. Continuing video and robot replay.", flush=True)
    qj = state["qj"].astype(np.float32) if "qj" in state else None
    imu = (
        state.get("display_quat_wxyz", state["imu_quat_wxyz"]).astype(np.float32)
        if "imu_quat_wxyz" in state
        else None
    )
    cmd_key = "command_q_physical" if "command_q_physical" in state else "command_q_23"
    cmd = state[cmd_key].astype(np.float32) if cmd_key in state else None
    err_key = (
        "tracking_error_q_physical"
        if "tracking_error_q_physical" in state
        else "tracking_error_q_23"
    )
    err = state[err_key].astype(np.float32) if err_key in state else None
    joint_names = list(meta.get("physical_joint_names", []))
    if not joint_names and qj is not None:
        from core.review.display import JOINT_NAMES_BY_MOTOR_INDEX

        if qj.shape[1] != len(JOINT_NAMES_BY_MOTOR_INDEX):
            raise ValueError("Episode needs physical_joint_names for this joint count")
        joint_names = list(JOINT_NAMES_BY_MOTOR_INDEX)
    sensors = list(meta.get("sensors", []))
    # Resting baseline per sensor, so light contact is visible rather than
    # swamped by each taxel's offset. Same idea as the live viewer.
    baselines = {}
    for s in sensors:
        key = f"tactile_{s.lower()}_raw_u8"
        if key in state and len(state[key]) >= 30:
            baselines[s] = np.round(state[key][:30].mean(axis=0)).astype(int).tolist()
    steering_goal = None
    for folder in (ep if ep.is_dir() else ep.parent, ep.parent, ep.parent.parent):
        path = folder / "steering_goal.json"
        if path.is_file():
            steering_goal = json.loads(path.read_text())
            break
    return dict(
        spec=spec,
        label=spec.get("label") or ep.name,
        ep=ep,
        state=state,
        get_frame=get_frame,
        meta=meta,
        n=n,
        steering_goal=steering_goal,
        has_cam="frame_offset" in state,
        external_camera=external_camera,
        camera_query=camera_query,
        camera_indices=camera_indices,
        recorded_depth=recorded_depth,
        depth_indices=depth_indices,
        depth_error=depth_error,
        depth_summary=depth_summary,
        qj=qj,
        imu=imu,
        cmd=cmd,
        err=err,
        temp=state.get("motor_temperature_29x2"),
        joint_names=joint_names,
        sensors=sensors,
        baselines=baselines,
        positions=state.get("pelvis_position_relative_m"),
        depth_limits=depth_display_limits(
            ep, native=bool(spec.get("telemetry") or spec.get("capture"))
        ),
    )


def name_markdown(label):
    """Category and run name on their own lines, large enough to read at a glance."""
    parts = label.split("/")
    return "\n\n".join(["## " + parts[-1]] + [f"**{p}**" for p in parts[:-1]])


def details_text(s):
    parts = []
    if s["spec"].get("telemetry") or s["spec"].get("capture"):
        parts.append(s["meta"]["explanation"])
    parts.append(
        "Video follows the recorded exposure timestamps. Depth shows the latest calibrated "
        "camera capture logged at each tick, before model encoding; it is not a traced planner dependency."
        if s["recorded_depth"]
        else (
            "Depth unavailable: " + s["depth_error"]
            if s["depth_error"]
            else "Depth was not recorded in this run. Future recorded deployments save it automatically."
        )
    )
    if s["depth_summary"].get("error") or s["depth_summary"].get("dropped"):
        parts.append(f'Depth recorder: {s["depth_summary"]}')
    if s["spec"].get("measured_pose"):
        report_path = Path(s["spec"]["measured_pose"]).parent / "evaluation.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            parts.append(
                f"**Tracking diagnostics**\n\n"
                f"Heading disagreement RMS / max: {report['yaw_disagreement_rms_deg']:.2f} / "
                f"{report['yaw_disagreement_max_deg']:.2f} degrees.\n\n"
                f"Geometry checks passed: {100*report['geometry_valid_fraction']:.1f}%.\n\n"
                f"Largest odometry gap: {report['odometry_gap_max_ms']:.1f} ms.\n\n"
                "Live timing remains unqualified. Contacts: not implemented."
            )
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--episode", help="episode directory")
    source.add_argument(
        "--telemetry", help="Native diffusion telemetry.jsonl.gz or its session directory"
    )
    source.add_argument("--capture", type=Path, help="Raw sensor directory for sensor-only replay")
    source.add_argument(
        "--sessions",
        type=Path,
        help="JSON list of recordings (telemetry, external_camera, depth_recording, ...) to browse in one viewer",
    )
    ap.add_argument(
        "--preload", action="store_true", help="Load every listed session before opening the viewer"
    )
    ap.add_argument(
        "--browse-argv",
        help="JSON argv of replay_motion.py; enables rescanning the browsed folder while viewing",
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate replay data and URDF without starting Viser",
    )
    ap.add_argument("--validation-output", type=Path)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--measured-pose",
        type=Path,
        help="Offline measured_pose.jsonl for relative XYZ; requires --telemetry",
    )
    ap.add_argument(
        "--live-estimates",
        action="store_true",
        help="Display only XYZ recorded by the live runtime",
    )
    ap.add_argument(
        "--external-camera",
        type=Path,
        help="Directory containing frames.mjpg and its timestamp index",
    )
    ap.add_argument("--depth-recording", type=Path, help="Recorded metric depth directory")
    ap.add_argument(
        "--full-validation",
        action="store_true",
        help="Decode every saved camera frame during validation",
    )
    ap.add_argument(
        "--allow-late-origin",
        action="store_true",
        help="Show a clearly labelled trajectory from the first available estimate if XYZ at policy start is missing",
    )
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback multiplier, 0.05–4 (0.25 = quarter speed)",
    )
    ap.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat the selected recording (default: enabled)",
    )
    ap.add_argument("--urdf", default=DEFAULT_URDF_PATH)
    ap.add_argument(
        "--deadzone",
        type=int,
        default=12,
        help="tactile: suppress values within this of the resting baseline",
    )
    args = ap.parse_args()
    if not 0.05 <= args.speed <= 4.0:
        ap.error("--speed must be between 0.05 and 4")
    if args.sessions:
        specs = json.loads(args.sessions.expanduser().read_text())
        if not specs:
            ap.error("--sessions lists no recordings")
        if (
            args.measured_pose
            or args.live_estimates
            or args.external_camera
            or args.depth_recording
        ):
            ap.error(
                "--sessions carries each recording's options; do not pass them on the command line"
            )
    else:
        if not (args.telemetry or args.capture or args.episode):
            ap.error("one of --episode, --telemetry, --capture or --sessions is required")
        if args.live_estimates and (not args.telemetry or args.measured_pose):
            ap.error("--live-estimates requires telemetry and no offline measured-pose")
        if args.measured_pose and not (args.telemetry or args.capture):
            ap.error("--measured-pose requires --telemetry or --capture")
        specs = [session_spec(args)]
    for s in specs:
        s["label"] = (
            s.get("label") or Path(s.get("telemetry") or s.get("capture") or s["episode"]).name
        )
    labels = [s["label"] for s in specs]
    if len(set(labels)) != len(labels):
        ap.error("Session labels must be unique")

    loaded = {}

    def session(i):
        spec = specs[i]
        if spec["label"] not in loaded:
            started = time.perf_counter()
            print(f'Loading recording: {spec["label"]}', flush=True)
            loaded[spec["label"]] = load_session(spec)
            print(f"Recording loaded in {time.perf_counter()-started:.2f}s.", flush=True)
        return loaded[spec["label"]]

    def drop(i, exc):
        """Forget a recording that vanished or cannot be read; the viewer keeps running."""
        print(f'{specs[i]["label"]}: {exc}; skipping', flush=True)
        loaded.pop(specs[i]["label"], None)
        del specs[i]
        del labels[i]

    if args.preload:
        i = 0
        while i < len(specs):
            try:
                session(i)
                i += 1
            except (OSError, ValueError) as exc:
                drop(i, exc)
    while specs:
        try:
            cur = session(0)
            break
        except (OSError, ValueError) as exc:
            drop(0, exc)
    if not specs:
        sys.exit("No replayable recordings")
    if len(specs) == 1 and not args.validate_only:
        print(
            f"{cur['label']}: {cur['n']} steps, {cur['meta'].get('duration_s', 0):.1f}s, "
            f"camera={'yes' if cur['has_cam'] or cur['external_camera'] else 'no'}, depth={'yes' if cur['recorded_depth'] else 'no'}",
            flush=True,
        )

    if args.validate_only:
        if cur["qj"] is not None:
            import yourdfpy

            model = yourdfpy.URDF.load(args.urdf, load_meshes=False)
            missing = set(cur["joint_names"]) - set(model.actuated_joint_names)
            if missing:
                raise ValueError(f"Recorded joints missing from URDF: {sorted(missing)}")
        if any(
            value is not None and not np.isfinite(value).all() for value in (cur["qj"], cur["imu"])
        ):
            raise ValueError("Nonfinite robot replay state")
        state, meta, n = cur["state"], cur["meta"], cur["n"]
        valid = state.get("position_valid", np.zeros(n, bool))
        positions = cur["positions"]
        result = dict(
            frames=n,
            duration_s=float(state["t"][-1]),
            joints=len(cur["joint_names"]),
            viewer_started=False,
            camera_only=bool(meta.get("camera_only")),
            robot_observations_present=cur["qj"] is not None,
            timeline_source=meta.get("timeline_source"),
            nonincreasing_capture_steps=meta.get("nonincreasing_capture_steps", 0),
            translation_present=positions is not None,
            translation_coverage=float(valid.mean()),
            translation_geometry_valid_fraction=float(
                state.get("position_geometry_valid", np.zeros(n, bool)).mean()
            ),
            position_origin_delay_s=meta.get("position_origin_delay_s"),
            position_max_displacement_m=(
                None if not valid.any() else float(np.linalg.norm(positions[valid], axis=1).max())
            ),
            heading_max_change_deg=(
                None
                if meta.get("camera_only")
                else float(np.degrees(np.abs(state.get("yaw_relative_rad", np.zeros(n))).max()))
            ),
            external_camera=(
                None
                if cur["external_camera"] is None
                else cur["external_camera"].validate(cur["camera_query"], full=args.full_validation)
            ),
            depth_error=cur["depth_error"],
            depth=(
                None
                if cur["recorded_depth"] is None
                else dict(
                    frames=len(cur["recorded_depth"].rows),
                    timeline_coverage=float((cur["depth_indices"] >= 0).mean()),
                    recording_summary=cur["depth_summary"],
                )
            ),
        )
        if args.validation_output:
            args.validation_output.parent.mkdir(parents=True, exist_ok=True)
            args.validation_output.write_text(json.dumps(result, indent=2) + "\n")
        else:
            print(json.dumps(result, indent=2))
        return

    if cur["sensors"]:
        from sonic.sensing.tactile_render import render_sensor_panels
    depth_min, depth_max = cur["depth_limits"]
    dv = DeployViser(
        urdf_path=Path(args.urdf).expanduser(),
        with_depth=cur["has_cam"]
        or cur["recorded_depth"] is not None
        or any(s.get("depth_recording") for s in specs),
        depth_min=depth_min,
        depth_max=depth_max,
        depth_scale=1.0,
        tactile_sensors=cur["sensors"],
        joint_names=cur["joint_names"] or None,
        port=args.port,
        replay_layout=True,
        robot_visible=cur["qj"] is not None,
    )
    with dv.server.gui.add_folder("Recording details", expand_by_default=False):
        details = dv.server.gui.add_markdown(details_text(cur))
    playback_notice = dv.server.gui.add_markdown("", order=-25)
    trace = {}
    path_stretches = []
    with dv.server.gui.add_folder("Steering", order=-32):
        fit_route = dv.server.gui.add_checkbox("Fit route to motion", True)
        route_note = dv.server.gui.add_markdown("")

    def draw_trace(s):
        for node in trace.values():
            node.remove()
        trace.clear()
        goal = s.get("steering_goal")
        if goal and not s["meta"].get("position_origin_delay_s", 0):
            from motion_diffusion.inference.steering import (
                goal_display_keypoints,
                goal_display_segments,
                fit_route_for_display,
            )

            fitted = None
            if fit_route.value and s["positions"] is not None:
                marks = session_directory(s["ep"]) / "review/recovery_interval.json"
                if marks.is_file():
                    stand = float(json.loads(marks.read_text())["end"]["replay_time_s"])
                    fitted, _ = fit_route_for_display(
                        goal,
                        np.asarray(s["state"]["t"]),
                        s["positions"],
                        s["state"]["position_valid"],
                        stand,
                    )
            route = goal_display_keypoints(goal, route=fitted)
            segments = goal_display_segments(goal) if route is None else []
            route_note.content = (
                "**Magenta: display fit of the command** (cut where the robot stood, time-warped to the "
                "measured pelvis, blended toward it). Untick *Fit route to motion* for the recorded command."
                if fitted is not None
                else "**Magenta: commanded route as recorded.** Blue/orange: measured estimate."
            )
        else:
            route = None
            segments = []
        magenta = (230, 40, 210)
        if route is not None and len(route[0]) > 1:
            knots, heading = route
            # markers scale with the route so a short get-up path is not buried under them
            span = float(np.linalg.norm(np.ptp(knots, axis=0)))
            radius = float(np.clip(0.025 * span, 0.008, 0.025))
            trace["goal"] = dv.server.scene.add_spline_catmull_rom(
                "/steering_goal",
                points=knots,
                curve_type="centripetal",
                line_width=5.0,
                color=magenta,
                segments=40 * len(knots),
            )
            for i, knot in enumerate(knots):
                trace[f"goal_knot_{i}"] = dv.server.scene.add_icosphere(
                    f"/steering_goal_knot_{i}",
                    radius=radius,
                    color=magenta,
                    opacity=0.55,
                    position=tuple(float(v) for v in knot),
                    subdivisions=2,
                )
            # commanded heading at the start and at the end, where the turn is defined
            arrows = np.stack(
                [knots[[0, -1]], knots[[0, -1]] + max(0.2, 0.4 * span) * heading[[0, -1]]], axis=1
            )
            trace["goal_heading"] = dv.server.scene.add_line_segments(
                "/steering_goal_heading",
                points=arrows,
                colors=np.array(magenta, np.uint8),
                line_width=4.0,
            )
        elif len(segments):
            trace["goal"] = dv.server.scene.add_line_segments(
                "/steering_goal",
                points=segments,
                colors=np.array(magenta, np.uint8),
                line_width=4.0,
            )
        positions = s["positions"]
        if positions is None:
            return
        state = s["state"]
        valid = state["position_valid"]
        connected = valid[:-1] & valid[1:]
        segments = np.stack([positions[:-1], positions[1:]], axis=1)[connected]
        if len(segments):
            geometry = state["position_geometry_valid"]
            colors = np.tile(np.array([30, 180, 240], np.uint8), (len(segments), 2, 1))
            colors[~(geometry[:-1] & geometry[1:])[connected]] = (240, 145, 30)
            # drawn as the robot walks it: one hidden stretch per 10 samples, revealed by render(k)
            starts = np.flatnonzero(connected)
            path_stretches.clear()
            for first in range(0, len(segments), 10):
                node = dv.server.scene.add_line_segments(
                    f"/pelvis_path/{first}",
                    points=segments[first : first + 10],
                    colors=colors[first : first + 10],
                    line_width=3.0,
                )
                node.visible = False
                trace[f"path_{first}"] = node
                path_stretches.append((int(starts[first]), node))
        trace["origin"] = dv.server.scene.add_frame(
            "/episode_origin", axes_length=0.2, axes_radius=0.006
        )

    def reveal_path(k):
        for start, node in path_stretches:
            shown = start <= k
            if node.visible != shown:
                node.visible = shown

    draw_trace(cur)
    fit_route.on_update(lambda _: draw_trace(cur))

    # commanded pose as a second robot, 0.9 m to the side: divergence between
    # the two IS the tracking error (a dragged leg reads as measured lagging)
    cmd_vis = None
    if cur["cmd"] is not None:
        from viser.extras import ViserUrdf

        cmd_frame = dv.server.scene.add_frame("/cmd", position=(0.0, 0.9, 0.0), show_axes=False)
        cmd_vis = ViserUrdf(
            dv.server,
            urdf_or_path=Path(args.urdf).expanduser(),
            root_node_name="/cmd",
            load_meshes=True,
            load_collision_meshes=False,
        )

    lock = threading.RLock()
    k = 0
    switching = False
    annotation = None
    annotation_cache = {}
    if args.sessions:
        with dv.server.gui.add_folder("Session", order=-40):
            run_name = dv.server.gui.add_markdown(name_markdown(cur["label"]))
            dropdown = dv.server.gui.add_dropdown(
                "Recording", options=labels, initial_value=cur["label"]
            )
            session_status = dv.server.gui.add_text("Folder", "-", disabled=True)
            previous_button = dv.server.gui.add_button("Previous")
            next_button = dv.server.gui.add_button("Next", color="blue")
            dropdown.on_update(lambda _: None if switching else move(to=dropdown.value))
            previous_button.on_click(lambda _: move(-1))
            next_button.on_click(lambda _: move(1))
    scene_name = dv.server.scene.add_label(
        "/run_name", text=cur["label"], position=(0.0, 0.0, 1.25), anchor="bottom-center"
    )
    folder = dv.server.gui.add_folder("Replay", order=-30)
    with folder:
        idx = dv.server.gui.add_slider(
            "Index", min=0, max=max(cur["n"] - 1, 1), step=1, initial_value=0
        )
        play = dv.server.gui.add_checkbox("Play", True)
        speed = dv.server.gui.add_number("Speed (×)", args.speed, min=0.05, max=4.0, step=0.05)
        slow_motion = dv.server.gui.add_button_group("Slow motion", ("¼×", "½×", "1×"))
        slow_motion.on_click(
            lambda _: setattr(speed, "value", {"¼×": 0.25, "½×": 0.5, "1×": 1.0}[slow_motion.value])
        )
        loop = dv.server.gui.add_checkbox("Loop", args.loop)
        if dv.depth_image_handle is not None:
            show_depth = dv.server.gui.add_checkbox("Show depth", True)

            @show_depth.on_update
            def on_depth_visibility(_):
                dv.depth_folder.visible = show_depth.value

        else:
            dv.server.gui.add_markdown(
                "Depth: unavailable; see Recording details."
                if cur["depth_error"]
                else "Depth: not recorded in this run."
            )
        label = dv.server.gui.add_text("t", "0.000 s")
        xyz_status = dv.server.gui.add_text("XYZ status", "Unavailable", disabled=True)
    with dv.server.gui.add_folder("Mark clip", order=-35):
        interval_status = dv.server.gui.add_markdown("**NOT MARKED**")
        mark_start = dv.server.gui.add_button("Mark start")
        mark_next = dv.server.gui.add_button("Mark next")
        start_label = dv.server.gui.add_text("Start", "—", disabled=True)
        end_label = dv.server.gui.add_text("End", "—", disabled=True)
        save_all = dv.server.gui.add_button("Save all")
        save_status = dv.server.gui.add_markdown("No unsaved marks.")
    with dv.server.gui.add_folder("Motion values", expand_by_default=False):
        position_label = dv.server.gui.add_text("Relative XYZ (m)", "Unavailable: pose-only replay")
        heading_label = dv.server.gui.add_text("Steering heading change (deg)", "-")
        orientation_label = dv.server.gui.add_text("Orientation from start (xyzw)", "-")
        camera_label = dv.server.gui.add_text(
            "External camera", "Unavailable" if cur["external_camera"] is None else "-"
        )
        err_label = dv.server.gui.add_text("worst err", "-")
        temp_label = dv.server.gui.add_text("ankle temps L/R", "-")

    last_camera_index = None
    last_depth_index = None

    def render(k: int):
        nonlocal last_camera_index, last_depth_index
        s = cur
        state, meta, n = s["state"], s["meta"], s["n"]
        ls = type("LS", (), {})()
        ls.tick = int(k)
        ls.mode_pr = 0
        ls.mode_machine = 0
        q = type("IMU", (), {})()
        q.quaternion = tuple(
            float(x) for x in (s["imu"][k] if s["imu"] is not None else (1.0, 0.0, 0.0, 0.0))
        )
        ls.imu_state = q

        color = depth = None
        if s["has_cam"]:
            color, depth_mm = s["get_frame"](k)
            if depth_mm is not None:
                depth = depth_mm.astype(np.float32) * 0.001  # mm -> m
        external_camera = s["external_camera"]
        if external_camera:
            frame_index = int(s["camera_indices"][k])
            if frame_index != last_camera_index:
                color = external_camera.frame(frame_index)
                if color is None:
                    color = np.zeros((270, 480, 3), np.uint8)
                last_camera_index = frame_index
            camera_label.value = (
                "Unavailable: outside recording or camera gap"
                if frame_index < 0
                else (
                    f"frame {frame_index}; camera exposure"
                    if meta.get("camera_only")
                    else f'frame {frame_index}; exposure {(s["camera_query"][k]-external_camera.times[frame_index])*1000:.1f} ms before robot sample'
                )
            )
        recorded_depth = s["recorded_depth"]
        if recorded_depth is not None and int(s["depth_indices"][k]) != last_depth_index:
            last_depth_index = int(s["depth_indices"][k])
            depth, valid = recorded_depth.frame(last_depth_index)
            if depth is None:
                depth = np.full(recorded_depth.view_shape, np.nan, dtype=np.float32)
            else:
                depth = np.where(valid, depth, np.nan)

        camera_only = bool(meta.get("camera_only"))
        playback_notice.content = (
            "**Camera-only recording. Robot motion and XYZ were not recorded.**"
            if camera_only
            else ""
        )
        playback_notice.visible = camera_only
        dv.state_folder.visible = s["qj"] is not None
        if s["qj"] is not None:
            dv.joint_names_by_motor_index = s["joint_names"]
            dv.update(ls, s["qj"][k], depth_raw=depth, rgb_bgr=color)
        else:
            if color is not None:
                dv.rgb_image_handle.image = color[..., ::-1]
            if depth is not None and dv.depth_image_handle is not None:
                dv._update_depth_image(depth)
        reveal_path(k)
        if "yaw_relative_rad" in state:
            heading_label.value = f'{np.degrees(state["yaw_relative_rad"][k]):+.2f}'
            orientation_label.value = ", ".join(
                f"{v:+.4f}" for v in state["orientation_relative_xyzw"][k]
            )
        else:
            heading_label.value = orientation_label.value = "Unavailable: no robot observations"
        positions = s["positions"]
        if positions is not None:
            dv.base_frame.visible = bool(state["position_valid"][k])
            if state["position_valid"][k]:
                dv.base_frame.position = tuple(positions[k])
                position_label.value = ", ".join(f"{v:+.3f}" for v in positions[k])
                if meta.get("position_origin_delay_s", 0) > 0:
                    position_label.value += (
                        f' [from +{meta["position_origin_delay_s"]:.2f}s, not A]'
                    )
                if not state["position_geometry_valid"][k]:
                    position_label.value += " [tilt disagreement: inspect estimate]"
            else:
                position_label.value = "Unavailable: missing or invalid position estimate"
            xyz_status.value = (
                "Missing estimate"
                if not state["position_valid"][k]
                else ("Offline estimate" if s["spec"].get("measured_pose") else "Live estimate")
                + (
                    "; tilt disagreement (orange trace)"
                    if not state["position_geometry_valid"][k]
                    else "; accuracy unqualified"
                )
            )
        else:
            dv.base_frame.visible = s["qj"] is not None
            position_label.value = (
                "Unavailable: camera-only replay"
                if camera_only
                else "Unavailable: pose-only replay"
            )
            xyz_status.value = "Unavailable"

        if cmd_vis is not None:
            cmd_frame.visible = s["cmd"] is not None
        if cmd_vis is not None and s["cmd"] is not None:
            cmd_vis.update_cfg(
                {name: float(s["cmd"][k][i]) for i, name in enumerate(s["joint_names"])}
            )
        if s["err"] is not None:
            e = np.degrees(np.abs(s["err"][k]))
            j = int(e.argmax())
            err_label.value = f"{s['joint_names'][j]} {e[j]:.1f} deg"
        if s["temp"] is not None:
            t = np.asarray(s["temp"][k])
            temp_label.value = (
                f"p {t[4].max():.0f}/{t[10].max():.0f}  " f"r {t[5].max():.0f}/{t[11].max():.0f} C"
            )

        panels = {}
        for sensor in s["sensors"]:
            raw = state.get(f"tactile_{sensor.lower()}_raw_u8")
            if raw is not None:
                panels.update(
                    render_sensor_panels(
                        sensor,
                        raw[k].tolist(),
                        baseline_values=s["baselines"].get(sensor),
                        deadzone=args.deadzone,
                    )
                )
        if panels:
            dv.update_tactile_images(panels)

        label.value = f"{float(state['t'][k]):.3f} s   step {k}/{n - 1}"

    known = {s["label"]: s for s in specs}

    def rescan():
        """Rediscover the browsed folder: moved-out sessions disappear, moved-in ones appear."""
        nonlocal specs, labels, switching
        if not args.browse_argv:
            return
        import rise.replay as replay_motion

        specs[:] = replay_motion.rescan(json.loads(args.browse_argv), known)
        labels[:] = [s["label"] for s in specs]
        for label in list(loaded):
            if label not in labels:
                loaded.pop(label)
        switching = True
        try:
            dropdown.options = labels
        finally:
            switching = False

    def switch(i):
        nonlocal cur, k, last, last_camera_index, last_depth_index, switching
        while specs:
            i = min(max(i, 0), len(specs) - 1)
            try:
                cur = session(i)
                break
            except (OSError, ValueError) as exc:
                drop(i, exc)
        if not specs:
            cur = None
            play.value = False
            session_status.value = "No recordings left"
            return
        k = 0
        last_camera_index = last_depth_index = None
        idx.max = max(cur["n"] - 1, 1)
        idx.value = 0
        last = time.perf_counter()
        play.value = True
        bind_annotations()
        details.content = details_text(cur)
        draw_trace(cur)
        if dv.depth_image_handle is not None:
            dv.depth_min_gui.value, dv.depth_max_gui.value = cur["depth_limits"]
        camera_label.value = "Unavailable" if cur["external_camera"] is None else "-"
        switching = True
        try:
            dropdown.value = cur["label"]
            run_name.content = name_markdown(cur["label"])
            scene_name.text = cur["label"]
            session_status.value = f"{i+1}/{len(specs)}"
        finally:
            switching = False
        render(0)

    def move(delta=0, *, to=None):
        """Rescan, then go to a neighbour (delta), a named recording (to) or refresh the list (0)."""
        nonlocal switching
        with lock:
            current = cur["label"] if cur else None
            rescan()
            if to is not None:
                i = labels.index(to) if to in labels else 0
            elif current in labels:
                i = labels.index(current) + delta
            else:
                i = 0 if delta >= 0 else len(specs) - 1
            if to is None and not delta and current in labels:
                switching = True
                try:
                    session_status.value = f"{i+1}/{len(specs)}"
                finally:
                    switching = False
                return
            switch(i)

    def seek(index):
        nonlocal k, last
        if cur is None:
            return
        play.value = False
        k = min(max(int(index), 0), cur["n"] - 1)
        idx.value = k
        last = time.perf_counter()
        render(k)

    def refresh_annotation_labels():
        def label_point(point):
            return "—" if point is None else f'{point["replay_time_s"]:.3f} s'

        start_label.value = label_point(annotation.start)
        end_label.value = label_point(annotation.end)
        complete = (
            annotation.start is not None
            and annotation.end is not None
            and annotation.end["sample_index"] > annotation.start["sample_index"]
        )
        if annotation.dirty and complete:
            interval_status.content = "**✅ MARKED — not saved yet**"
        elif annotation.dirty:
            interval_status.content = "**START MARKED — choose the end, then Mark next**"
        elif annotation.status == "Saved":
            interval_status.content = "**✅ MARKED — saved**"
        else:
            interval_status.content = "**NOT MARKED**"
            if annotation.status.startswith("Cannot load"):
                interval_status.content += "\n\n" + annotation.status
        pending = sum(a.dirty for a in annotation_cache.values())
        save_status.content = (
            f"**{pending} recording(s) unsaved — Save all when finished.**"
            if pending
            else "All marks saved."
        )

    def bind_annotations():
        nonlocal annotation
        key = str(cur["ep"].resolve())
        if key not in annotation_cache:
            annotation_cache[key] = RecoveryInterval(cur)
        annotation = annotation_cache[key]
        if annotation.start is None and annotation.status == "Not marked":
            # The operator uses the true beginning for every clip. Merely
            # opening a recording does not create an unsaved annotation.
            annotation.start = annotation.point(0)
        refresh_annotation_labels()

    def mark_boundary(name):
        with lock:
            if cur is None:
                return
            seek(idx.value)
            annotation.mark(name, k)
            refresh_annotation_labels()

    mark_start.on_click(lambda _: mark_boundary("start"))

    @mark_next.on_click
    def on_mark_next(_):
        with lock:
            if cur is None:
                return
            seek(idx.value)
            if annotation.start is None:
                interval_status.content = "**Mark start first.**"
                return
            if k <= annotation.start["sample_index"]:
                interval_status.content = "**Choose an end after the start.**"
                return
            annotation.mark("end", k)
            refresh_annotation_labels()
            if args.sessions and labels.index(cur["label"]) < len(labels) - 1:
                move(1)

    @save_all.on_click
    def on_save_all(_):
        with lock:
            play.value = False
            pending = [a for a in annotation_cache.values() if a.dirty]
            # Validate every draft before writing, so an unfinished clip cannot
            # silently disappear from a successful-looking batch save.
            try:
                for draft in pending:
                    validate_interval(draft.document(), draft.session["state"])
            except ValueError as exc:
                save_status.content = f'**NOT SAVED** — {draft.session["label"]}: {exc}'
                return
            try:
                for draft in pending:
                    path = draft.save()
                    print(
                        f'{draft.session["label"]}: saved recovery interval to {path}', flush=True
                    )
            except (OSError, ValueError) as exc:
                if annotation is not None:
                    refresh_annotation_labels()
                save_status.content = f'**Save incomplete** — {draft.session["label"]}: {exc}'
                return
            if annotation is not None:
                refresh_annotation_labels()
            save_status.content = f"**All marks saved ({len(pending)} updated).**"

    last = time.perf_counter()
    bind_annotations()
    render(0)
    last = time.perf_counter()
    play.value = True

    @play.on_update
    def on_play(_):
        nonlocal k, last
        with lock:
            last = time.perf_counter()
            if cur is not None and play.value and k == cur["n"] - 1 and k > 0:
                k = 0
                idx.value = 0
                render(0)

    @speed.on_update
    def on_speed(_):
        nonlocal last
        with lock:
            last = time.perf_counter()

    if args.sessions:
        session_status.value = f"1/{len(specs)}"
    print(f"\nviser: http://localhost:{args.port}", flush=True)

    while True:
        try:
            with lock:
                if cur is None:
                    time.sleep(0.1)
                    continue
                n = cur["n"]
                kk = min(int(idx.value), n - 1)
                if kk != k:
                    k = kk
                    last = time.perf_counter()
                    render(k)
                if play.value:
                    now = time.perf_counter()
                    interval = (
                        float(cur["state"]["t"][k + 1] - cur["state"]["t"][k])
                        if (cur["spec"].get("telemetry") or cur["spec"].get("capture"))
                        and k < n - 1
                        else 1.0 / args.fps
                    )
                    if (now - last) >= interval / float(speed.value):
                        last = now
                        stop = n - 1
                        k = 0 if k == stop and loop.value and n > 1 else min(k + 1, stop)
                        idx.value = k
                        render(k)
                        if k == stop and (not loop.value or n == 1):
                            play.value = False
        except (OSError, ValueError) as exc:
            if not args.sessions:
                raise
            with lock:
                broken = cur["label"]
                print(f"{broken}: {exc}; moving on", flush=True)
                rescan()
                if broken in labels:
                    i = labels.index(broken)
                    drop(i, "unreadable")
                    try:
                        dropdown.options = labels
                    except Exception:
                        pass
                else:
                    i = 0
                switch(min(i, len(specs) - 1) if specs else 0)
            continue
        time.sleep(0.002)


if __name__ == "__main__":
    main()
