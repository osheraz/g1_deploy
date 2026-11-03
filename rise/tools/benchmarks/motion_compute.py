"""Sequential sensor-only timing comparison. Never passes motor-control flags."""

import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import psutil

from rise.paths import ROOT, PACKAGE_ROOT


def role(pid, parent, command):
    if pid == parent:
        return "control"
    for name, pattern in [
        ("capture", "record_lidar_live.py"),
        ("pelvis", "pelvis_odometry.py"),
        ("dlio", "dlio_odom_node"),
        ("driver", "livox_ros_driver2"),
        ("sweeps", "livox_cloud_window.py"),
        ("imu_si", "livox_imu_si.py"),
        ("camera", "v4l2-ctl"),
        ("planner", "spawn_main"),
        ("ipc", "resource_tracker"),
    ]:
        if pattern in command:
            return name
    return "other"


def stats(values):
    x = np.array(values, float)
    return (
        None
        if not len(x)
        else dict(p50=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))
    )


def summarize(output, samples):
    with gzip.open(output / "telemetry.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    controls = [r for r in rows if r.get("event") == "control"]
    if not controls:
        raise RuntimeError(f"No control ticks in {output}")
    start, end = controls[0]["monotonic_s"], controls[-1]["monotonic_s"]
    costs = {}
    last = {}
    for sample in samples:
        for proc in sample["processes"]:
            previous = last.get(proc["pid"])
            last[proc["pid"]] = (sample["t"], proc["cpu_s"])
            if previous is None or previous[0] < start or sample["t"] > end:
                continue
            item = costs.setdefault(
                proc["role"], dict(cpu_s=0.0, observed_process_seconds=0.0, peak_rss_mb=0.0)
            )
            item["cpu_s"] += max(0, proc["cpu_s"] - previous[1])
            item["observed_process_seconds"] += sample["t"] - previous[0]
            item["peak_rss_mb"] = max(item["peak_rss_mb"], proc["rss_mb"])
    for item in costs.values():
        item["cpu_cores_average"] = item["cpu_s"] / (end - start)
    poses = [r["pelvis_odometry"] for r in controls if r.get("pelvis_odometry")]
    complete = json.loads((output / "COMPLETE.json").read_text())
    if complete["commands_issued"] != 0:
        raise RuntimeError("Unexpected motor output in sensor-only benchmark")
    return dict(
        session=str(output),
        ticks=len(controls),
        duration_s=end - start,
        control_work_ms=stats([r["control_wall_s"] * 1000 for r in controls]),
        planner_ms=stats([r["sample_wall_s"] * 1000 for r in rows if r.get("event") == "plan"]),
        state_to_proposal_ms=stats(
            [
                (r["proposal_ready_s"] - r["observation_capture_s"]) * 1000
                for r in controls
                if r.get("proposal_ready_s") is not None
            ]
        ),
        cpu=costs,
        commands_issued=complete.get("commands_issued"),
        late_ticks=complete.get("publication_late_ticks"),
        odometry_available_fraction=(
            None if not poses else float(np.mean([p["available"] for p in poses]))
        ),
        bridge_ms=stats(
            [p["processing_s"] * 1000 for p in poses if p.get("processing_s") is not None]
        ),
        pose_age_ms=stats([p["age_s"] * 1000 for p in poses if p.get("age_s") is not None]),
        note="Sequential stationary trials, not motor latency. CPU estimates sample process counters during control only.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--model", choices=["diffusion", "flow"], default="diffusion")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "recording", "live"],
        default=["baseline", "recording", "live"],
    )
    p.add_argument(
        "--external-camera", help="Also record this USB camera in the recording/live modes"
    )
    a = p.parse_args()
    if not 5 <= a.seconds <= 60 or not 1 <= a.repeats <= 5:
        p.error("Use 5..60 seconds and 1..5 repeats")
    a.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for repeat in range(a.repeats):
        for mode in a.modes:
            output = a.output / f"{repeat+1}_{mode}"
            cmd = [
                sys.executable,
                str(PACKAGE_ROOT / "deploy.py"),
                "--backend",
                "tensorrt",
                "--model",
                a.model,
                "--duration-s",
                str(a.seconds),
                "--output",
                str(output),
                "--record-label",
                "stationary_hoist_benchmark",
            ]
            cmd += ["--no-record-sensors"] if mode == "baseline" else ["--record-sensors"]
            if mode != "baseline" and a.external_camera:
                cmd += ["--external-camera", a.external_camera]
            if mode == "live":
                cmd += ["--live-odometry"]
            print(f"Starting {repeat+1}/{a.repeats}: {mode}; no motor output.", flush=True)
            samples = []
            with (a.output / f"{repeat+1}_{mode}.log").open("w") as log:
                proc = subprocess.Popen(
                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    parent = psutil.Process(proc.pid)
                    deadline = time.monotonic() + a.seconds + 180
                    while proc.poll() is None:
                        values = []
                        for child in [parent, *parent.children(recursive=True)]:
                            try:
                                cpu = child.cpu_times()
                                command = " ".join(child.cmdline())
                                values.append(
                                    dict(
                                        pid=child.pid,
                                        role=role(child.pid, proc.pid, command),
                                        command=command,
                                        cpu_s=cpu.user + cpu.system,
                                        rss_mb=child.memory_info().rss / 1e6,
                                    )
                                )
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        samples.append(dict(t=time.monotonic(), processes=values))
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "See benchmark log; deployment exceeded its time allowance"
                            )
                        time.sleep(0.5)
                    if proc.returncode:
                        raise RuntimeError(f"{mode} failed; see {log.name}")
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGINT)
                        proc.wait()
                    (a.output / f"{repeat+1}_{mode}_resources.json").write_text(
                        json.dumps(samples) + "\n"
                    )
            report = summarize(output, samples)
            report.update(mode=mode, repeat=repeat + 1, model=a.model)
            if mode != "baseline":
                saved = json.loads((output / "sensors/recording_result.json").read_text())
                if not saved.get("complete"):
                    raise RuntimeError(f"Incomplete recording in {output}: {saved}")
                capture = json.loads((output / "sensors/raw/summary.json").read_text())
                report["recorded_streams"] = {
                    k: capture[k] for k in ("scans", "lidar_imu", "lowstate", "torso_imu")
                }
            reports.append(report)
            (a.output / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
