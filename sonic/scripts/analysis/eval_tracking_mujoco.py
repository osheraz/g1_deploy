#!/usr/bin/env python3
"""Tracking-accuracy / stability gate for SONIC references in MuJoCo.

Runs sim2sim_sonic.py headless on a scenario (or analyzes an existing
--record-dir session) and reports one metrics table per run:

    python scripts/eval_tracking_mujoco.py --clip sonic/model/motion_walk_straight.npz
    python scripts/eval_tracking_mujoco.py --planner-axes 0,1,0 --steps 500
    python scripts/eval_tracking_mujoco.py --record /path/to/session

Metrics: per-joint tracking error (lower/upper split), pelvis tilt, height,
fall, achieved-vs-commanded speed, heading drift, action smoothness, compute
timings. --json appends a machine-readable line for cross-run comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[3]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from sonic.model.robot_profile import get_robot_profile  # noqa: E402

LOWER = list(range(12))            # our-23 order: legs
UPPER = list(range(12, 23))        # waist yaw + arms
NAMES = ["Lhip_p", "Lhip_r", "Lhip_y", "Lknee", "Lank_p", "Lank_r",
         "Rhip_p", "Rhip_r", "Rhip_y", "Rknee", "Rank_p", "Rank_r",
         "waist",
         "Lsh_p", "Lsh_r", "Lsh_y", "Lelb", "Lwr",
         "Rsh_p", "Rsh_r", "Rsh_y", "Relb", "Rwr"]


def pct(x, q):
    return float(np.percentile(np.asarray(x, np.float64), q)) if len(x) else float("nan")


def analyze(record_dir: Path) -> dict:
    d = np.load(record_dir / "state.npz")
    meta = json.loads((record_dir / "meta.json").read_text()) \
        if (record_dir / "meta.json").exists() else {}
    n = len(d["time_s"])
    dt = float(np.mean(np.diff(d["time_s"]))) if n > 1 else 0.02

    q = np.asarray(d["q"], np.float64)                        # (T,23) measured
    ref_il = np.asarray(d["reference_q_il"], np.float64)      # (T,29)
    profile = get_robot_profile(meta.get("robot_profile", "g1_23dof"))
    ref23 = ref_il[:, profile.isaaclab_from_physical]
    valid = ~np.isnan(ref23).any(axis=1)
    err = np.degrees(np.abs(q[valid] - ref23[valid]))

    grav = np.asarray(d["projected_gravity"], np.float64)
    tilt = np.degrees(np.arccos(np.clip(-grav[:, 2], -1, 1)))
    base = np.asarray(d["base_position"], np.float64)
    height = base[:, 2]
    fell = bool(height.min() < 0.4)

    # command following: horizontal speed while a walk was commanded
    vel = np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1) / max(dt, 1e-6)
    speed_cmd = np.asarray(d["nav_speed"], np.float64)
    moving = speed_cmd[:-1] > 0
    achieved = float(np.mean(vel[moving])) if moving.any() else 0.0
    commanded = float(np.mean(speed_cmd[:-1][moving])) if moving.any() else 0.0

    # heading drift while nothing commanded a turn (facing ~ constant)
    quat = np.asarray(d["base_quat_wxyz"], np.float64)
    w, x, y, z = quat.T
    yaw = np.degrees(np.unwrap(np.arctan2(2 * (w * z + x * y),
                                          1 - 2 * (y * y + z * z))))

    act = np.asarray(d["action"], np.float64)
    act_rate = np.abs(np.diff(act, axis=0)).max(axis=1) if len(act) > 1 else [0.0]

    per_joint_p95 = np.percentile(err, 95, axis=0) if err.size else np.full(23, np.nan)
    worst = int(np.argmax(per_joint_p95)) if err.size else -1

    out = {
        "record": str(record_dir),
        "rows": int(n),
        "duration_s": round(n * dt, 2),
        "fell": fell,
        "track_mean_deg": round(float(err.mean()), 2) if err.size else None,
        "track_p95_deg": round(pct(err.max(axis=1), 95), 2) if err.size else None,
        "track_max_deg": round(float(err.max()), 2) if err.size else None,
        "lower_p95_deg": round(pct(err[:, LOWER].max(axis=1), 95), 2) if err.size else None,
        "upper_p95_deg": round(pct(err[:, UPPER].max(axis=1), 95), 2) if err.size else None,
        "worst_joint": NAMES[worst] if worst >= 0 else None,
        "worst_joint_p95_deg": round(float(per_joint_p95[worst]), 2) if worst >= 0 else None,
        "tilt_p95_deg": round(pct(tilt, 95), 2),
        "tilt_max_deg": round(float(tilt.max()), 2),
        "height_min_m": round(float(height.min()), 3),
        "speed_cmd_mps": round(commanded, 3),
        "speed_achieved_mps": round(achieved, 3),
        "travel_m": round(float(np.linalg.norm(base[-1, :2] - base[0, :2])), 2),
        "yaw_drift_deg": round(float(yaw[-1] - yaw[0]), 1),
        "action_rate_p95": round(pct(act_rate, 95), 3),
        "policy_ms_p99": round(pct(d["policy_total_ms"], 99), 2),
        "planner_ms_p99": round(pct(d["planner_total_ms"], 99), 2),
        "adaptation": meta.get("adaptation"),
        "model_profile": meta.get("model_profile"),
    }
    return out


def run_scenario(args, record_dir: Path) -> None:
    cmd = [sys.executable, str(DEPLOY_ROOT / "sonic" / "sim.py"),
           "--mjcf", args.mjcf, "--record-dir", str(record_dir),
           "--steps", str(args.steps)]
    if args.config:
        cmd += ["--config", args.config]
    if args.model_profile:
        cmd += ["--model-profile", args.model_profile]
    if args.clip:
        cmd += ["--clip", args.clip]
    else:
        cmd += ["--pico", "joystick", "--planner", "--axes", args.planner_axes]
        if not args.analog_speed:
            cmd += ["--native-gamepad"]
        if args.axes_until:
            cmd += ["--axes-until", str(args.axes_until)]
        if args.slow:
            cmd += ["--slow"]
    print("running:", " ".join(cmd[1:]))
    res = subprocess.run(cmd, capture_output=True, text=True,
                         cwd=str(DEPLOY_ROOT), timeout=1800)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise SystemExit(f"rollout failed (exit {res.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=None,
                    help="analyze this existing session instead of running one")
    ap.add_argument("--clip", default=None, help="motion-clip scenario")
    ap.add_argument("--planner-axes", default="0,1,0",
                    help="joystick scenario: fixed lx,ly,rx")
    ap.add_argument("--axes-until", type=int, default=None)
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--analog-speed", action="store_true",
                    help="stick-proportional speed instead of native fixed gaits")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-profile", default=None)
    ap.add_argument("--mjcf", default=str(
        Path(__file__).resolve().parents[4]
        / "research/repos/GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_23dof.xml"
    ))
    ap.add_argument("--json", default=None, help="append the metrics dict to this file")
    args = ap.parse_args()

    if args.record:
        record_dir = Path(args.record).expanduser()
    else:
        record_dir = Path(tempfile.mkdtemp(prefix="sonic_eval_"))
        run_scenario(args, record_dir)

    m = analyze(record_dir)
    width = max(len(k) for k in m)
    for k, v in m.items():
        print(f"  {k:<{width}}  {v}")
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    main()
