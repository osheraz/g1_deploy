"""Review a stopped hardware run: video, sensor checks, XYZ and synchronized Viser.

Run after deployment stops. Never connects to the robot or starts a sensor driver.
"""

from steerable_recovery.paths import (
    RESULTS_ROOT,
    PAPER_RESULTS_ROOT,
    HIGHLIGHTS_ROOT,
    PAPER_HIGHLIGHTS_ROOT,
    ABORTED_ROOT,
    ROOT,
    CONFIG_ROOT,
    PACKAGE_ROOT,
)
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from steerable_recovery.recording.session import DEFAULT_WS, camera_report, capture_saved


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def resolve_session(path):
    path = path.expanduser()
    if not path.is_absolute() and len(path.parts) == 1 and path.name != ".." and not path.is_dir():
        path = RESULTS_ROOT / path
    path = path.resolve()
    # Accept the "Saved and verified raw sensors" path as well as "Session".
    if path.name == "sensors" and (path.parent / "telemetry.jsonl.gz").is_file():
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"Session directory not found: {path}")
    return path


def review_telemetry(source, review, *, quick=False):
    """Recover complete records from an interrupted gzip, keeping raw bytes intact.

    Only an absent gzip ending is recoverable here. CRC failures and malformed
    complete JSON records remain errors, rather than silently skipping data.
    quick reuses a previous recovery of the same unchanged raw file.
    """
    import gzip
    import hashlib

    pending = review / "telemetry_recovered.pending.gz"
    recovered = review / "telemetry_recovered.jsonl.gz"
    report_path = review / "telemetry_recovery.json"
    if quick and recovered.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("source_bytes") == source.stat().st_size:
            return recovered, report
    buffer = b""
    records = 0
    interrupted = None
    try:
        with gzip.open(source, "rb") as stream, gzip.open(pending, "wb", compresslevel=1) as output:
            while True:
                try:
                    chunk = stream.read1(65536)
                except EOFError as exc:
                    interrupted = str(exc)
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    json.loads(line)
                    output.write(line + b"\n")
                    records += 1
            if not interrupted and buffer:
                json.loads(buffer)
            if interrupted and not records:
                raise ValueError(f"No complete telemetry records recoverable from {source}")
        if not interrupted:
            return source, None
        pending.replace(recovered)
        with source.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        report = dict(
            source=str(source),
            source_sha256=sha,
            source_bytes=source.stat().st_size,
            recovered=str(recovered),
            complete_records=records,
            discarded_tail_bytes=len(buffer),
            error=interrupted,
            note="Recovered complete JSON records only. Raw telemetry is unchanged. "
            "The end of this run may be missing; no state, commands or stop event were invented.",
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Telemetry was interrupted: recovered {records} complete records; original preserved.",
            flush=True,
        )
        return recovered, report
    finally:
        pending.unlink(missing_ok=True)


def finalize_camera(recording, review, *, export_video=True):
    camera = recording / "camera"
    result = dict(available=False, mp4=None, error=None)
    if not (camera / "frames.mjpg").is_file():
        return result
    result["available"] = True
    config = read_json(recording / "session.json").get("camera") or {}
    saved = read_json(recording / "recording_result.json").get("camera") or {}
    if not (camera / "frames.npz").is_file():
        trigger = read_json(recording / "policy_start.json").get("monotonic_s")
        saved = camera_report(recording, config.get("fps", 30), policy_start_monotonic_s=trigger)
        (review / "camera_finalize.json").write_text(json.dumps(saved, indent=2) + "\n")
    result["error"] = saved.get("error") or saved.get("remux_error")
    if not (camera / "frames.npz").is_file():
        result.update(available=False, error=result["error"] or "Camera timestamp index is missing")
        return result
    # MP4 is for sharing. Viser uses indexed MJPEG exposure times instead.
    video = review / "external.mp4"
    raw = camera / "frames.mjpg"
    start = policy_start_frame(recording)
    if export_video and (
        not video.is_file()
        or video.stat().st_mtime_ns < raw.stat().st_mtime_ns
        or read_json(review / "export.json").get("start_frame") != start
    ):
        import numpy as np, shutil

        with np.load(camera / "frames.npz") as index:
            fps = float(index["fps"])
            offset = int(index["offset"][start])
            frames = len(index["seq"]) - start
        pending = review / "external.pending.mp4"
        print("Preparing external.mp4 after recording stopped...", flush=True)
        with (review / "video_export.log").open("w") as log, raw.open("rb") as source:
            try:
                source.seek(offset)
                encoder = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "mjpeg",
                        "-framerate",
                        str(fps),
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "2",
                        "-preset",
                        "fast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(pending),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    shutil.copyfileobj(source, encoder.stdin, 1 << 20)
                finally:
                    encoder.stdin.close()
                    encoder.wait(timeout=600)
                if encoder.returncode:
                    raise subprocess.CalledProcessError(encoder.returncode, "ffmpeg")
                pending.replace(video)
                (review / "export.json").write_text(
                    json.dumps(
                        dict(
                            start_frame=start,
                            frames=frames,
                            fps=fps,
                            note="external.mp4, depth_aligned.mp4 and frames/ begin at the first camera frame after A (policy start)",
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            except (OSError, subprocess.SubprocessError) as exc:
                result["error"] = f"Video export failed: {exc}; see {review}/video_export.log"
    if video.is_file():
        result["mp4"] = str(video)
    return result


def policy_start_frame(recording):
    """Index of the first camera frame at or after A; 0 when the policy never started."""
    import numpy as np

    start = read_json(recording / "policy_start.json").get("monotonic_s")
    if start is None:
        return 0
    with np.load(recording / "camera/frames.npz") as index:
        times = index["ts_monotonic_s"]
    return min(int(np.searchsorted(times, start)), len(times) - 1)


def export_depth_video(session, recording, review):
    """review/depth_aligned.mp4: one frame per external camera frame, showing the
    latest depth capture at that exposure, so it plays in step with external.mp4."""
    import numpy as np
    from steerable_recovery.recording.depth import RecordedDepth

    camera = recording / "camera"
    depth_dir = recording / "depth"
    video = review / "depth_aligned.mp4"
    if not (camera / "frames.npz").is_file() or not (depth_dir / "frames.jsonl").is_file():
        return None
    start = policy_start_frame(recording)
    marker = review / "depth_export.json"
    if (
        video.is_file()
        and video.stat().st_mtime_ns >= (depth_dir / "frames.bin").stat().st_mtime_ns
        and read_json(marker).get("start_frame") == start
    ):
        return str(video)
    import torch

    torch.set_num_threads(1)
    with np.load(camera / "frames.npz") as index:
        times = index["ts_monotonic_s"].astype(float)[start:]
        fps = float(index["fps"])
    depth = RecordedDepth(depth_dir)
    indices = depth.indices_at(times)
    runtime = session / "model_trace/planner/runtime.json"
    view = (
        json.loads(runtime.read_text()).get("sampling_contract", {}).get("depth_condition", {})
        if runtime.is_file()
        else {}
    )
    vmin, vmax = (
        (float(view["min_depth_m"]), float(view["max_depth_m"]))
        if view.get("enabled")
        else (0.1, 2.0)
    )
    height, width = depth.view_shape
    scale = max(1, 320 // width)
    pending = review / "depth_aligned.pending.mp4"
    with (review / "depth_export.log").open("w") as log:
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width*scale}x{height*scale}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(pending),
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        last = None
        image = np.full((height, width, 3), (100, 0, 130), np.uint8)
        try:
            for i in indices:
                if i != last:
                    last = int(i)
                    if last < 0:
                        image = np.full((height, width, 3), (100, 0, 130), np.uint8)
                    else:
                        d, valid = depth.frame(last)
                        g = (np.clip((d - vmin) / max(vmax - vmin, 1e-3), 0.0, 1.0) * 255).astype(
                            np.uint8
                        )
                        image = np.stack([g, g, g], axis=-1)
                        image[~valid] = (100, 0, 130)
                encoder.stdin.write(
                    np.repeat(np.repeat(image, scale, axis=0), scale, axis=1).tobytes()
                )
        finally:
            encoder.stdin.close()
            encoder.wait(timeout=600)
    if encoder.returncode:
        raise ValueError(f"Depth video export failed; see {review}/depth_export.log")
    pending.replace(video)
    marker.write_text(
        json.dumps(dict(start_frame=start, frames=len(times), fps=fps), indent=2) + "\n"
    )
    return str(video)


def export_frames(session, recording, review):
    """review/frames/external/NNNNNN.jpg (original camera JPEG bytes) and
    review/frames/depth/NNNNNN.png with the same numbering as depth_aligned.mp4."""
    import numpy as np

    camera = recording / "camera"
    depth_dir = recording / "depth"
    root = review / "frames"
    if not (camera / "frames.npz").is_file():
        return None
    import shutil

    start = policy_start_frame(recording)
    with np.load(camera / "frames.npz") as index:
        times = index["ts_monotonic_s"].astype(float)[start:]
        offsets = index["offset"][start:]
        sizes = index["bytes"][start:]
        fps = float(index["fps"])
    if read_json(root / "index.json").get("first_camera_frame") != start:
        shutil.rmtree(root, ignore_errors=True)
    external = root / "external"
    external.mkdir(parents=True, exist_ok=True)
    if len(list(external.glob("*.jpg"))) != len(times):
        with (camera / "frames.mjpg").open("rb") as raw:
            for i, (offset, size) in enumerate(zip(offsets, sizes)):
                raw.seek(int(offset))
                (external / f"{i:06d}.jpg").write_bytes(raw.read(int(size)))
    indices = []
    if (depth_dir / "frames.jsonl").is_file():
        import cv2, torch

        torch.set_num_threads(1)
        from steerable_recovery.recording.depth import RecordedDepth

        depth_out = root / "depth"
        depth_out.mkdir(exist_ok=True)
        depth = RecordedDepth(depth_dir)
        indices = depth.indices_at(times)
        if len(list(depth_out.glob("*.png"))) != len(times):
            runtime = session / "model_trace/planner/runtime.json"
            view = (
                json.loads(runtime.read_text())
                .get("sampling_contract", {})
                .get("depth_condition", {})
                if runtime.is_file()
                else {}
            )
            vmin, vmax = (
                (float(view["min_depth_m"]), float(view["max_depth_m"]))
                if view.get("enabled")
                else (0.1, 2.0)
            )
            height, width = depth.view_shape
            last = None
            image = np.full((height, width, 3), (100, 0, 130), np.uint8)
            for i, k in enumerate(indices):
                if int(k) != last:
                    last = int(k)
                    if last < 0:
                        image = np.full((height, width, 3), (100, 0, 130), np.uint8)
                    else:
                        d, valid = depth.frame(last)
                        g = (np.clip((d - vmin) / max(vmax - vmin, 1e-3), 0.0, 1.0) * 255).astype(
                            np.uint8
                        )
                        image = np.stack([g, g, g], axis=-1)
                        image[~valid] = (100, 0, 130)
                cv2.imwrite(str(depth_out / f"{i:06d}.png"), image)
    (root / "index.json").write_text(
        json.dumps(
            dict(
                first_camera_frame=start,
                frames=len(times),
                fps=fps,
                exposure_monotonic_s=times.tolist(),
                depth_frame_index=[int(k) for k in indices],
                note="Frame 000000 is the first camera frame after A (policy start). "
                "external/NNNNNN.jpg is the original camera JPEG; depth/NNNNNN.png is the latest depth capture at that exposure "
                "(purple = no valid depth), same numbering as depth_aligned.mp4",
            ),
            indent=None,
        )
        + "\n"
    )
    return str(root)


def has_live_odometry(telemetry):
    import gzip

    with gzip.open(telemetry, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "control" and row.get("pelvis_odometry") is not None:
                return True
    return False


def preserve_attempt(path):
    if path.exists():
        archived = path.with_name(path.name + f".previous_{time.time_ns()}")
        path.rename(archived)
        print(f"Previous processing attempt preserved: {archived}", flush=True)


def prepare_xyz(session, raw, telemetry, calibration, *, capture_only=False):
    replay = session / "motion_replay"
    odom = replay / "dlio"
    pose = replay / ("pose_calibrated" if calibration else "pose_nominal")
    if not (raw / "sensors.npz").is_file() or not (raw / "bag/metadata.yaml").is_file():
        raise ValueError("Complete raw LiDAR/IMU capture is missing; XYZ cannot be reconstructed")
    replay.mkdir(exist_ok=True)
    report = read_json(odom / "summary.json")
    if not report or report.get("error") or not (odom / "odometry.npz").is_file():
        workspace = Path(os.environ.get("LIDAR_WS", DEFAULT_WS))
        setup = workspace / "install/setup.bash"
        if not setup.is_file():
            raise ValueError(f"Missing {setup}; set LIDAR_WS for offline XYZ")
        preserve_attempt(odom)
        print("Replaying saved LiDAR offline; no live sensors or motor connection.", flush=True)
        shell = 'set -e; source /opt/ros/humble/setup.bash; source "$1"; shift; export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=91; exec "$@"'
        subprocess.run(
            [
                "bash",
                "-c",
                shell,
                "replay-env",
                str(setup),
                "/usr/bin/python3",
                str(PACKAGE_ROOT / "tools/sensors/replay_dlio.py"),
                str(raw / "bag"),
                "--config",
                str(CONFIG_ROOT / "lidar/dlio.yaml"),
                "--output",
                str(odom),
                "--imu-unit",
                "g",
                "--scan-packets",
                "3",
                "--timeout-s",
                "360",
            ],
            check=True,
        )
        report = read_json(odom / "summary.json")
    if report.get("error"):
        raise ValueError(f'Offline DLIO failed: {report["error"]}')
    calibration = None if calibration is None else str(calibration.resolve())
    signature = dict(
        telemetry=None if capture_only else str(telemetry),
        mount_calibration=calibration,
        calibration_contents=None if calibration is None else Path(calibration).read_text(),
        odometry_mtime_ns=(odom / "odometry.npz").stat().st_mtime_ns,
    )
    if not (pose / "measured_pose.jsonl").is_file() or read_json(pose / "source.json") != signature:
        preserve_attempt(pose)
        command = [
            sys.executable,
            str(PACKAGE_ROOT / "tools/sensors/evaluate_measured_motion.py"),
            str(odom / "odometry.npz"),
            "--capture",
            str(raw),
            "--output",
            str(pose),
        ]
        if calibration:
            command += ["--mount-calibration", calibration]
        with (replay / "pose_evaluation.log").open("w") as log:
            run = subprocess.run(
                command + ([] if capture_only else ["--deployment-log", str(telemetry)]),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if run.returncode:
            # Keep late estimates for display, never invent an A-relative origin.
            if (
                "Sensor recording does not cover deployment episode start"
                not in (replay / "pose_evaluation.log").read_text()
            ):
                raise RuntimeError(f"Pelvis evaluation failed; see {replay}/pose_evaluation.log")
            print(
                "XYZ at policy start is unavailable. Preparing later motion with an explicitly late origin.",
                flush=True,
            )
            preserve_attempt(pose)
            with (replay / "pose_evaluation_late.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        (pose / "source.json").write_text(json.dumps(signature, indent=2) + "\n")
    return pose / "measured_pose.jsonl"


def save_timing_report(telemetry, output):
    """Named measured endpoints; no assumption that asynchronous stages add up."""
    import gzip
    import numpy as np

    with gzip.open(telemetry, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    controls = [r for r in rows if r.get("event") == "control"]
    issued = [r for r in controls if r.get("command_publish_s") is not None]

    def stats(seconds):
        x = np.asarray(seconds, dtype=float) * 1000
        return dict(
            samples=len(x),
            p50_ms=None if not len(x) else float(np.median(x)),
            p95_ms=None if not len(x) else float(np.percentile(x, 95)),
            max_ms=None if not len(x) else float(x.max()),
        )

    result = dict(
        control_step=stats([r["control_wall_s"] for r in controls]),
        control_ticks=len(controls),
        commands_issued=len(issued),
        planner_pass=stats([r["sample_wall_s"] for r in rows if r.get("event") == "plan"]),
        robot_observation_to_command=stats(
            [r["command_publish_s"] - r["observation_capture_s"] for r in issued]
        ),
        latest_depth_age_at_command=stats(
            [
                r["command_publish_s"] - r["depth_timing"]["capture_s"]
                for r in issued
                if r.get("depth_timing")
            ]
        ),
        note="Milliseconds. Depth age describes the latest logged frame, not a traced planner-to-command dependency. "
        "Offline DLIO replay does not establish live odometry latency.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Deployment timing (p95): "
        + ", ".join(
            f'{name} {v["p95_ms"]:.2f} ms'
            for name, v in result.items()
            if isinstance(v, dict) and v["p95_ms"] is not None
        ),
        flush=True,
    )
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "session",
        type=Path,
        help="Run name under logs/real_world, or a session/sensors directory path",
    )
    p.add_argument("--port", type=int, default=8081)
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Viser playback multiplier, 0.05–4 (0.25 = quarter speed)",
    )
    p.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat the selected recording in Viser (default: enabled)",
    )
    p.add_argument(
        "--unmarked", action="store_true", help="Open only recordings without saved start/end marks"
    )
    p.add_argument(
        "--pose-only",
        action="store_true",
        help="Joints, orientation and video; skip XYZ processing",
    )
    p.add_argument(
        "--live-estimates",
        action="store_true",
        help="Use XYZ actually logged during deployment; no offline DLIO",
    )
    p.add_argument(
        "--offline-xyz",
        action="store_true",
        help="Explicitly reconstruct XYZ from raw LiDAR; default reuses live estimates or cached offline motion",
    )
    p.add_argument(
        "--export-video",
        action="store_true",
        help="Also encode a shareable MP4; Viser reads the original camera frames directly",
    )
    p.add_argument(
        "--export-depth",
        action="store_true",
        help="Also encode review/depth_aligned.mp4, frame-aligned with external.mp4",
    )
    p.add_argument(
        "--export-frames",
        action="store_true",
        help="Also write every camera frame (jpg) and aligned depth frame (png) under review/frames/",
    )
    p.add_argument(
        "--export-overlay",
        action="store_true",
        help="Also blend the picked highlight frames into review/overlay_*.jpg",
    )
    p.add_argument(
        "--export-strip",
        action="store_true",
        help="Also put the picked highlight frames side by side into review/strip_*.jpg",
    )
    p.add_argument(
        "--overlay-opacity",
        type=float,
        nargs=2,
        default=(0.4, 0.7),
        metavar=("OLDEST", "NEWEST"),
        help="Opacity of the earlier robot cut-outs in the overlay, from the oldest to the newest pick; the last pick is opaque",
    )
    p.add_argument(
        "--full-validation",
        action="store_true",
        help="Decode all video frames; default validates index and samples three frames",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save checks/requested exports without starting Viser",
    )
    p.add_argument(
        "--capture-only",
        action="store_true",
        help="Standalone raw capture without policy telemetry (also autodetected)",
    )
    p.add_argument(
        "--mount-calibration", type=Path, help="Optional explicit fixed-mount calibration candidate"
    )
    p.add_argument(
        "--preload",
        action="store_true",
        help="Browsing a folder: load every session before opening Viser so Next is instant",
    )
    return p


def browse_target(session):
    target = session.expanduser()
    if not target.is_absolute() and not target.exists() and (RESULTS_ROOT / target).exists():
        target = RESULTS_ROOT / target
    return target.resolve()


def browse_sessions(target):
    """Session directories under target, in path order; anything under a hidden folder is skipped."""
    return sorted(
        t.parent
        for t in target.rglob("telemetry.jsonl.gz")
        if "hidden" not in t.relative_to(target).parts
    )


def cached_viewer_spec(a, session, *, policy_trials_only=False):
    """Open an already reviewed run without rereading every telemetry stream.

    Frame decoding and robot/depth loading remain lazy in the shared viewer. Explicit
    exports, validation and reconstruction still go through the full review path.
    """
    if any(
        getattr(a, key, False)
        for key in (
            "prepare_only",
            "export_video",
            "export_depth",
            "export_frames",
            "export_overlay",
            "export_strip",
            "full_validation",
            "offline_xyz",
            "mount_calibration",
            "capture_only",
        )
    ):
        return None
    if a.live_estimates and a.pose_only:
        return None
    telemetry = session / "telemetry.jsonl.gz"
    review = session / "review"
    recording = session / "sensors"
    paths = [
        review / "review.json",
        session / "motion_replay/deployment_timing.json",
        session / "motion_replay/replay_validation.json",
    ]
    if not telemetry.is_file() or any(not path.is_file() for path in paths):
        return None
    if any(path.stat().st_mtime_ns < telemetry.stat().st_mtime_ns for path in paths):
        return None
    result = read_json(paths[0])
    timing = read_json(paths[1])
    if policy_trials_only and timing.get("commands_issued") == 0:
        raise ValueError(f"{session.name}: no policy commands; excluded from trial browser")
    recovered = review / "telemetry_recovered.jsonl.gz"
    if result.get("telemetry_recovery"):
        report = read_json(review / "telemetry_recovery.json")
        if (
            not recovered.is_file()
            or report.get("source_bytes") != telemetry.stat().st_size
            or recovered.stat().st_mtime_ns < telemetry.stat().st_mtime_ns
        ):
            return None
        telemetry = recovered
    spec = dict(
        label=session.name,
        telemetry=str(telemetry),
        capture=None,
        external_camera=None,
        depth_recording=None,
        live_estimates=False,
        measured_pose=None,
        allow_late_origin=False,
    )
    if result.get("camera", {}).get("available"):
        if not all(
            (recording / "camera" / name).is_file() for name in ("frames.mjpg", "frames.npz")
        ):
            return None
        spec["external_camera"] = str(recording / "camera")
    if (recording / "depth/frames.jsonl").is_file():
        spec["depth_recording"] = str(recording / "depth")
    if a.live_estimates and result.get("xyz_source") != "recorded_live":
        return None
    if not a.pose_only:
        if result.get("xyz_source") == "recorded_live":
            spec["live_estimates"] = True
        elif result.get("xyz_source") == "offline_dlio":
            selection = read_json(review / "xyz_selection.json").get("pose")
            if not selection or not (session / selection).is_file():
                return None
            spec.update(measured_pose=str(session / selection), allow_late_origin=True)
    return spec


def archive_empty_launch(session):
    """Move a launch where A was never pressed and nothing was recorded to logs/aborted_deployments.

    Returns the destination, or None when the session holds any episode data.
    """
    import collections, gzip, shutil

    sensors = session / "sensors"
    if (sensors / "policy_start.json").is_file() or read_json(
        sensors / "recording_result.json"
    ).get("started"):
        return None
    events = collections.Counter()
    recovered = session / "review/telemetry_recovered.jsonl.gz"
    with gzip.open(
        recovered if recovered.is_file() else session / "telemetry.jsonl.gz", "rt"
    ) as stream:
        try:
            for line in stream:
                events[json.loads(line).get("event")] += 1
        except EOFError:
            pass  # Interrupted launch: complete records were already counted.
    if events.get("control") or events.get("plan"):
        return None
    archive = ABORTED_ROOT
    archive.mkdir(exist_ok=True)
    destination = archive / session.name
    if destination.exists():
        destination = archive / f"{session.name}_{time.time_ns()}"
    shutil.move(str(session), str(destination))
    receipt = archive / f'cleanup_no_policy_{time.strftime("%Y%m%d")}_browse.json'
    report = read_json(receipt) or dict(
        reason="Browse cleanup: A never pressed, no policy commands, no recorded episode; all files preserved",
        moves=[],
    )
    report["moves"].append(
        dict(source=str(session), destination=str(destination), events=dict(events))
    )
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return destination


def browse_specs(p, a, target, known=None):
    """Viewer specs for the sessions currently under target.

    known maps labels to earlier specs (reused) or None (skipped once, silently
    thereafter); it is updated in place so repeated rescans stay quiet. Launches
    that never ran the policy are moved out of the browsed folder.
    """
    known = {} if known is None else known
    specs = []
    for s in browse_sessions(target):
        label = str(s.relative_to(target))
        # Keep this browser's queue stable after Save all, while a new launch
        # resumes only the recordings still missing marks.
        if a.unmarked and label not in known and (s / "review/recovery_interval.json").is_file():
            continue
        if label not in known:
            try:
                spec = cached_viewer_spec(a, s, policy_trials_only=True)
                if spec is None:
                    print(f"Preparing replay: {label}", flush=True)
                    spec = review(p, a, s, quick=True, policy_trials_only=True)
                known[label] = dict(spec, label=label)
            except ValueError as exc:
                known[label] = None
                moved = archive_empty_launch(s) if "no policy commands" in str(exc) else None
                print(
                    (
                        f"  moved empty launch {label} to {moved.relative_to(ROOT)}"
                        if moved
                        else f"  skipping {exc}"
                    ),
                    flush=True,
                )
        if known[label] is not None:
            specs.append(known[label])
    return specs


def rescan(argv, known):
    """Called by the viewer: rediscover the browsed folder with the original command-line options."""
    p = parser()
    a = p.parse_args(argv)
    return browse_specs(p, a, browse_target(a.session), known)


def main():
    p = parser()
    a = p.parse_args()
    if not 0.05 <= a.speed <= 4.0:
        p.error("--speed must be between 0.05 and 4")
    target = browse_target(a.session)
    collection = (
        target.is_dir()
        and target.name != "sensors"
        and not (target / "telemetry.jsonl.gz").is_file()
        and not (target / "raw/sensors.npz").is_file()
    )
    sessions = browse_sessions(target) if collection else []
    if a.unmarked:
        if collection:
            sessions = [s for s in sessions if not (s / "review/recovery_interval.json").is_file()]
            if not sessions:
                print("All recordings are marked. Nothing left to mark.", flush=True)
                return
        elif (target / "review/recovery_interval.json").is_file():
            print("This recording is already marked.", flush=True)
            return
    if not sessions:
        try:
            session = resolve_session(target)
            spec = cached_viewer_spec(a, session)
            if spec is None:
                spec = review(p, a, session)
        except ValueError as exc:
            sys.exit(str(exc))
        if not a.prepare_only:
            os.execv(sys.executable, viewer_command(spec, a.port, speed=a.speed, loop=a.loop))
        return
    print(
        f"{len(sessions)} sessions under {target}:",
        *[f"  {i+1:3d}. {s.relative_to(target)}" for i, s in enumerate(sessions)],
        sep="\n",
        flush=True,
    )
    specs = browse_specs(p, a, target)
    manifest = target / ("review_sessions_unmarked.json" if a.unmarked else "review_sessions.json")
    manifest.write_text(json.dumps(specs, indent=2) + "\n")
    if a.prepare_only or not specs:
        return
    print(
        "Opening viewer; recordings load when selected (no batch replay or video export).",
        flush=True,
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(ROOT / "g1_deploy/core/review/viewer.py"),
            "--sessions",
            str(manifest),
            "--browse-argv",
            json.dumps(sys.argv[1:]),
            *(["--preload"] if a.preload else []),
            "--urdf",
            str(ROOT / "humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf"),
            "--port",
            str(a.port),
            "--speed",
            str(a.speed),
            "--loop" if a.loop else "--no-loop",
        ],
    )


def viewer_command(spec, port, *, speed=1.0, loop=True):
    command = [
        sys.executable,
        str(ROOT / "g1_deploy/core/review/viewer.py"),
        "--urdf",
        str(ROOT / "humanoidverse2/data/robots/g1/g1_29dof_sonic.urdf"),
        "--port",
        str(port),
        "--speed",
        str(speed),
        "--loop" if loop else "--no-loop",
    ]
    for key in ("telemetry", "capture", "external_camera", "depth_recording", "measured_pose"):
        if spec.get(key):
            command += ["--" + key.replace("_", "-"), spec[key]]
    for key in ("live_estimates", "allow_late_origin"):
        if spec.get(key):
            command.append("--" + key.replace("_", "-"))
    return command


def review(p, a, session, *, quick=False, policy_trials_only=False):
    """Check one session, write its review files and return the viewer spec.

    quick reuses a previous validation instead of decoding camera samples again.
    """
    telemetry = session / "telemetry.jsonl.gz"
    if not telemetry.is_file() and (session / "raw/sensors.npz").is_file():
        a.capture_only = True
    if a.live_estimates and (a.capture_only or a.pose_only or a.mount_calibration or a.offline_xyz):
        p.error("--live-estimates requires a policy session and no offline pose options")
    if not a.capture_only and not telemetry.is_file():
        p.error(f"Missing {telemetry}")
    recording = session if a.capture_only else session / "sensors"
    depth_only = (recording / "depth/summary.json").is_file() and not (recording / "raw").exists()
    if (
        recording.is_dir()
        and not (recording / "recording_result.json").is_file()
        and not depth_only
    ):
        p.error(
            'Recorder has not finalized this run. Wait for "Saved and verified raw sensors" before reviewing.'
        )
    review = session / "review"
    review.mkdir(exist_ok=True)
    result = dict(
        session=str(session),
        recording=read_json(recording / "recording_result.json"),
        model=read_json(session / "session.json").get("model"),
        policy_end={
            name: read_json(session / name)
            for name in ("FAILED.json", "STOPPED.json", "COMPLETE.json")
            if (session / name).is_file()
        },
        capture_error=capture_saved(recording) if recording.is_dir() else "No raw recording",
        xyz_source="none",
        xyz_error=None,
    )
    result["camera"] = finalize_camera(recording, review, export_video=a.export_video)
    result["depth_video"] = (
        export_depth_video(session, recording, review)
        if a.export_depth and result["camera"]["available"]
        else None
    )
    result["frames"] = (
        export_frames(session, recording, review)
        if a.export_frames and result["camera"]["available"]
        else None
    )
    result["overlay"] = None
    if not a.capture_only:
        telemetry, result["telemetry_recovery"] = review_telemetry(telemetry, review, quick=quick)
        timing = session / "motion_replay/deployment_timing.json"
        result["timing"] = (
            read_json(timing)
            if quick and timing.is_file() and timing.stat().st_mtime >= telemetry.stat().st_mtime
            else save_timing_report(telemetry, timing)
        )
        if policy_trials_only and result["timing"].get("commands_issued") == 0:
            raise ValueError(f"{session.name}: no policy commands; excluded from trial browser")
        if (
            a.export_overlay
            and result["camera"]["available"]
            and result["timing"].get("commands_issued")
        ):
            from steerable_recovery.tools.media.figures import export_overlay

            result["overlay"] = export_overlay(
                session, recording, review, opacity=tuple(a.overlay_opacity)
            )
        if (
            a.export_strip
            and result["camera"]["available"]
            and result["timing"].get("commands_issued")
        ):
            from steerable_recovery.tools.media.figures import export_strip

            result["strip"] = export_strip(session, recording, review)
    spec = dict(
        label=session.name,
        capture=str(session / "raw") if a.capture_only else None,
        telemetry=None if a.capture_only else str(telemetry),
        external_camera=str(recording / "camera") if result["camera"]["available"] else None,
        depth_recording=(
            str(recording / "depth") if (recording / "depth/frames.jsonl").is_file() else None
        ),
        live_estimates=False,
        measured_pose=None,
        allow_late_origin=False,
    )
    camera_only = (
        not a.capture_only
        and result.get("timing", {}).get("control_ticks") == 0
        and result["camera"]["available"]
    )
    if camera_only:
        result["xyz_error"] = (
            "No policy observations or XYZ trajectory; showing the recorded camera footage."
        )
        print(
            "No policy observations: recorded camera footage is available for Viser playback.",
            flush=True,
        )
    use_live = not camera_only and (
        a.live_estimates
        or (
            not a.pose_only
            and not a.offline_xyz
            and not a.mount_calibration
            and not a.capture_only
            and has_live_odometry(telemetry)
        )
    )
    if use_live:
        from core.recording.episode import load_native_recovery

        try:
            load_native_recovery(telemetry, live_estimates=True)
            spec["live_estimates"] = True
            result["xyz_source"] = "recorded_live"
        except ValueError as exc:
            result["xyz_error"] = str(exc)
            print(
                f"Live XYZ unavailable: {exc}. Showing joints, orientation and video.", flush=True
            )
    if not a.pose_only and not use_live and not camera_only:
        try:
            calibration = a.mount_calibration or (
                session / "odometry_mount.json"
                if (session / "odometry_mount.json").is_file()
                else None
            )
            selection = read_json(review / "xyz_selection.json")
            pose = (
                session / selection["pose"]
                if selection.get("pose") and not a.mount_calibration
                else session
                / "motion_replay"
                / ("pose_calibrated" if calibration else "pose_nominal")
                / "measured_pose.jsonl"
            )
            if a.offline_xyz or a.mount_calibration:
                pose = prepare_xyz(
                    session, recording / "raw", telemetry, calibration, capture_only=a.capture_only
                )
            elif not pose.is_file():
                raise ValueError(
                    "No live XYZ or cached offline estimate. Use --offline-xyz to reconstruct; record with --live-odometry next time"
                )
            from core.recording.episode import (
                load_native_recovery,
                load_raw_motion,
                attach_measured_translation,
            )

            state, _, meta = (
                load_raw_motion(recording / "raw")
                if a.capture_only
                else load_native_recovery(telemetry)
            )
            attach_measured_translation(state, meta, pose, allow_late_origin=True)
            result["xyz_evaluation"] = read_json(pose.parent / "evaluation.json")
            spec.update(measured_pose=str(pose), allow_late_origin=True)
            result["xyz_source"] = "offline_dlio"
            (review / "xyz_selection.json").write_text(
                json.dumps(dict(pose=str(pose.relative_to(session))), indent=2) + "\n"
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            result["xyz_error"] = str(exc)
            print(
                f"XYZ unavailable: {exc}. Continuing with recorded joints, orientation and video.",
                flush=True,
            )
    validation = session / "motion_replay/replay_validation.json"
    if not (
        quick and validation.is_file() and validation.stat().st_mtime >= telemetry.stat().st_mtime
    ):
        run = subprocess.run(
            [
                *viewer_command(spec, a.port),
                "--validate-only",
                *(["--full-validation"] if a.full_validation else []),
                "--validation-output",
                str(validation),
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode:
            reason = (run.stderr.strip().splitlines() or [f"exit status {run.returncode}"])[-1]
            raise ValueError(f"{session.name} cannot be replayed: {reason}")
    result["replay"] = read_json(validation)
    (review / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    ending = next(iter(result["policy_end"]), "no policy end marker")
    outcome = result["policy_end"].get(ending, {})
    detail = outcome.get("error") or outcome.get("reason") or ""
    lines = [
        "# Run review",
        "",
        f"Session: `{session}`",
        "",
        (
            "- [External video](external.mp4)"
            if result["camera"]["mp4"]
            else (
                "- External video available directly in Viser; use --export-video for MP4."
                if result["camera"]["available"]
                else "- External video not recorded."
            )
        ),
        "- [Checks and timing](review.json)",
        f'- Model: {result["model"] or "not recorded"}.',
        f"- Policy outcome: {ending} {detail}.",
        (
            "- Playback: camera only; robot pose and XYZ are unavailable."
            if result["replay"].get("camera_only")
            else "- Playback: recorded robot observations."
        ),
        f'- XYZ source: {result["xyz_source"]}. Estimated motion, not ground truth.',
        f'- Position origin delay: {result["replay"].get("position_origin_delay_s")} s (0 means policy start).',
        f'- XYZ coverage: {100*result["replay"]["translation_coverage"]:.1f}%.',
        f'- Reused/regressed observation timestamps: {result["replay"].get("nonincreasing_capture_steps",0)}. '
        f'Playback clock: {result["replay"].get("timeline_source","capture")}.',
        f'- Recording error: {result["capture_error"] or result["recording"].get("error") or "none reported"}.',
        f'- XYZ error: {result["xyz_error"] or "none reported"}.',
        f'- Camera error: {result["camera"]["error"] or "none reported"}.',
        "",
        (
            "Telemetry: interrupted; replay uses recovered complete records. The original is preserved; "
            "the end may be missing. See [recovery details](telemetry_recovery.json)."
            if result.get("telemetry_recovery")
            else "Telemetry: complete gzip stream."
        ),
        "",
        f'- Depth error: {result["replay"].get("depth_error") or "none reported"}.',
        "Exported videos and frames start at the first camera frame after A (policy start) and keep the footage after the stop. Viser synchronizes indexed exposures.",
        (
            "Depth: metric frames recorded."
            if result["replay"].get("depth")
            else "Depth: not recorded in this run."
        ),
        "Camera/odometry clocks and dynamic XYZ accuracy are not independently qualified.",
        "",
    ]
    (review / "README.md").write_text("\n".join(lines))
    if not quick:
        print(f"Review ready: {review}/README.md", flush=True)
        if result["camera"]["mp4"]:
            print(f'Video: {result["camera"]["mp4"]}', flush=True)
    if result["capture_error"]:
        print(f'Recording warning: {result["capture_error"]}', flush=True)
    if result["camera"]["error"]:
        print(f'Camera warning: {result["camera"]["error"]}', flush=True)
    if result["replay"].get("depth_error"):
        print(f'Depth warning: {result["replay"]["depth_error"]}', flush=True)
    return spec


if __name__ == "__main__":
    main()
