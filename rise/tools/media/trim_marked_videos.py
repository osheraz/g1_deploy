"""Export saved camera intervals to logs/real_world_trimmed; keep originals.

    .venv/bin/python g1_deploy/rise/tools/media/trim_marked_videos.py

No robot, model, viewer, or odometry processing. Bounds are original camera
frame indices, including the displayed end frame, not offsets into an MP4.
"""

from rise.paths import RESULTS_ROOT, ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(ROOT))
from core.recording.episode import load_native_recovery
from core.recording.camera import RecordedCamera
from core.recording.annotations import validate_interval


def json_file(path):
    return json.loads(path.read_text())


def camera_bounds(annotation, state, camera):
    start, end = validate_interval(annotation, state)
    query = state["playback_monotonic_s"][[start, end]]
    first, last = (int(i) for i in camera.indices_at(query))
    if first < 0 or last < first:
        raise ValueError("Marked interval has no camera coverage")
    for key, i in (("start", first), ("end", last)):
        point = annotation[key]
        if (
            point.get("camera_frame_index") != i
            or abs(point.get("camera_exposure_monotonic_s", float("inf")) - camera.times[i]) > 1e-7
        ):
            raise ValueError(f"{key} does not match its recorded camera exposure")
    return first, last


def plan_run(run, root, output):
    mark = run / "review/recovery_interval.json"
    annotation = json_file(mark)
    telemetry = run / "review/telemetry_recovered.jsonl.gz"
    if not telemetry.is_file():
        telemetry = run / "telemetry.jsonl.gz"
    state, _, _ = load_native_recovery(telemetry)
    camera = RecordedCamera(run / "sensors/camera")
    first, last = camera_bounds(annotation, state, camera)
    with np.load(camera.directory / "frames.npz") as index:
        fps = float(index["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid recorded camera rate")
    depth = run / "sensors/depth"
    inputs = [telemetry, camera.raw, camera.directory / "frames.npz"]
    if (depth / "frames.jsonl").is_file():
        inputs += [depth / "frames.jsonl", depth / "frames.bin", depth / "metadata.json"]
    runtime = run / "model_trace/planner/runtime.json"
    if runtime.is_file():
        inputs.append(runtime)
    signature = dict(
        annotation_sha256=hashlib.sha256(mark.read_bytes()).hexdigest(),
        inputs={str(p.relative_to(run)): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs},
    )
    return dict(
        run=run,
        label=str(run.relative_to(root)),
        output=output / run.relative_to(root),
        annotation=annotation,
        camera=camera,
        first=first,
        last=last,
        frames=last - first + 1,
        fps=fps,
        signature=signature,
    )


def encode(destination, input_args, write, expected_frames):
    pending = destination.with_suffix(".pending.mp4")
    with destination.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-threads",
                "1",
                *input_args,
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
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(pending),
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
        )
        try:
            write(process.stdin)
            process.stdin.close()
            process.wait(timeout=600)
            if process.returncode:
                raise RuntimeError(f'Encoder failed: {destination.with_suffix(".log")}')
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,width,height,duration",
                    "-of",
                    "json",
                    str(pending),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            if int(stream["nb_frames"]) != expected_frames:
                raise ValueError(
                    f'Wrong frame count for {destination}: {stream["nb_frames"]} != {expected_frames}'
                )
            pending.replace(destination)
            return stream
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            pending.unlink(missing_ok=True)


def export_run(plan):
    run, output, camera = plan["run"], plan["output"], plan["camera"]
    first, last, fps, count = (plan[k] for k in ("first", "last", "fps", "frames"))
    previous = output / "manifest.json"
    if previous.is_file():
        saved = json_file(previous)
        if saved.get("signature") == plan["signature"] and all(
            (output / name).is_file() for name in saved["videos"]
        ):
            return saved
    output.mkdir(parents=True, exist_ok=True)

    def camera_bytes(pipe):
        remaining = int(camera.offsets[last] + camera.sizes[last] - camera.offsets[first])
        with camera.raw.open("rb") as raw:
            raw.seek(int(camera.offsets[first]))
            while remaining:
                data = raw.read(min(1 << 20, remaining))
                if not data:
                    raise ValueError("Camera dump ended inside the marked interval")
                pipe.write(data)
                remaining -= len(data)

    videos = {
        "external.mp4": encode(
            output / "external.mp4", ["-f", "mjpeg", "-framerate", str(fps)], camera_bytes, count
        )
    }
    depth_dir = run / "sensors/depth"
    if (depth_dir / "frames.jsonl").is_file():
        from rise.recording.depth import RecordedDepth

        depth = RecordedDepth(depth_dir)
        indices = depth.indices_at(camera.times[first : last + 1])
        runtime = run / "model_trace/planner/runtime.json"
        view = (
            json_file(runtime).get("sampling_contract", {}).get("depth_condition", {})
            if runtime.is_file()
            else {}
        )
        low, high = (
            (float(view["min_depth_m"]), float(view["max_depth_m"]))
            if view.get("enabled")
            else (0.1, 2.0)
        )
        height, width = depth.view_shape

        def depth_bytes(pipe):
            cached = None
            image = np.full((height, width, 3), (100, 0, 130), np.uint8)
            for index in indices:
                if int(index) != cached:
                    cached = int(index)
                    if cached < 0:
                        image = np.full((height, width, 3), (100, 0, 130), np.uint8)
                    else:
                        d, valid = depth.frame(cached)
                        gray = (np.clip((d - low) / max(high - low, 1e-3), 0.0, 1.0) * 255).astype(
                            np.uint8
                        )
                        image = np.stack([gray] * 3, axis=-1)
                        image[~valid] = (100, 0, 130)
                pipe.write(image.tobytes())

        videos["depth_aligned.mp4"] = encode(
            output / "depth_aligned.mp4",
            ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps)],
            depth_bytes,
            count,
        )
    result = dict(
        run=plan["label"],
        source=str(run),
        signature=plan["signature"],
        first_camera_frame=first,
        last_camera_frame_inclusive=last,
        frames=count,
        fps=fps,
        camera_exposure_monotonic_s=camera.times[first : last + 1].tolist(),
        videos=videos,
        annotation=plan["annotation"],
        note="Original JPEG frames at saved start/end, inclusive. "
        "Depth uses the latest capture at each camera exposure. Original recordings and marks are unchanged. "
        "Clip labels are operator boundaries, not verified success or unassisted-standing outcomes.",
    )
    pending = output / "manifest.pending.json"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.replace(previous)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = ROOT / "logs/real_world_trimmed"
    runs = sorted(
        p.parent
        for p in root.rglob("telemetry.jsonl.gz")
        if "hidden" not in p.relative_to(root).parts
    )
    missing = [
        str(run.relative_to(root))
        for run in runs
        if not (run / "review/recovery_interval.json").is_file()
    ]
    if not runs or missing:
        parser.error("No recordings found" if not runs else "Missing marks: " + ", ".join(missing))
    # Validate all marks before any export. Validation does not load XYZ or depth.
    plans = [plan_run(run, root, output) for run in runs]
    print(f"Validated {len(plans)}/{len(runs)} saved intervals. Exporting to {output}", flush=True)
    import torch

    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(export_run, plan): plan for plan in plans}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f'{i}/{len(plans)} {result["run"]}: {result["frames"]} frames', flush=True)
    lines = [
        "# Trimmed recorded videos",
        "",
        "Saved operator start/end marks; originals preserved. The final marked camera frame is included.",
        "",
        "| Recording | Camera | Depth |",
        "|---|---|---|",
    ]
    for plan in plans:
        label = plan["label"]
        result = json_file(plan["output"] / "manifest.json")
        depth = (
            f"[video]({label}/depth_aligned.mp4)"
            if "depth_aligned.mp4" in result["videos"]
            else "Not recorded"
        )
        lines.append(f"| {label} | [video]({label}/external.mp4) | {depth} |")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Done: {output}/README.md", flush=True)


if __name__ == "__main__":
    main()
