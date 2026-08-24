#!/usr/bin/env python3
"""Score commanded vs actual yaw rate, and walk-plus-turn jitter."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("/data/clhsieh/logs/e_skin/experiments/20260818_gait_yaw_rate")
NATIVE_YAW_GAIN = 2.0
JOYSTICK_YAW_GAIN = 1.5


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _yaw_from_facing(facing: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(facing[:, 1], facing[:, 0]))


def _yaw_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, np.float64).T
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rate(values: np.ndarray, dt: float) -> np.ndarray:
    return np.diff(values) / dt


def analyze(recording: Path) -> dict:
    meta = json.loads((recording / "meta.json").read_text())
    state = np.load(recording / "state.npz", allow_pickle=False)
    dt = float(meta["control_dt"])
    axes = np.asarray(state["axes"], np.float64)
    facing = np.asarray(state["nav_facing"], np.float64)
    mode = np.asarray(state["mode"], np.int64)
    base = np.asarray(state["base_position"], np.float64)
    quat = np.asarray(state["base_quat_wxyz"], np.float64)
    gravity = np.asarray(state["projected_gravity"], np.float64)
    parts = recording.resolve().parts
    teleop = "native-gamepad" if "native-gamepad" in parts else "joystick"
    yaw_cell = "yaw_1p0" if "yaw_1p0" in parts else "yaw_auto"
    scale = parts[parts.index(teleop) + 1]
    command = parts[parts.index(teleop) + 2]
    seed = recording.name

    active = np.linalg.norm(np.nan_to_num(axes), axis=1) > 0.2
    indices = np.flatnonzero(active)
    if len(indices) < 25:
        raise ValueError("recording has no active stick window")
    pad = int(round(1.0 / dt))
    if len(indices) > 2 * pad:
        sl = slice(indices[0] + pad, indices[-1] + 1 - pad)
    else:
        sl = slice(indices[0], indices[-1] + 1)

    rx = float(np.nanmedian(axes[sl, 2]))
    ly = float(np.nanmedian(axes[sl, 1]))
    applied = float(meta.get("yaw_command_scale", np.nan))
    derived = float(meta.get("derived_yaw_command_scale", np.nan))
    gain = NATIVE_YAW_GAIN if teleop == "native-gamepad" else JOYSTICK_YAW_GAIN
    expected = -gain * applied * rx

    cmd_yaw = _yaw_from_facing(facing[sl])
    pelvis_yaw = _yaw_from_quat(quat[sl])
    cmd_rate = _rate(cmd_yaw, dt)
    pelvis_rate = _rate(pelvis_yaw, dt)
    dxy = np.diff(base[sl, :2], axis=0)
    forward = np.linalg.norm(dxy, axis=1) / dt
    current_facing = facing[sl.start:sl.stop - 1, :2]
    norms = np.linalg.norm(current_facing, axis=1, keepdims=True)
    current_facing = current_facing / np.clip(norms, 1e-9, None)
    signed = np.sum(dxy * current_facing, axis=1) / dt
    tilt = np.degrees(np.arccos(np.clip(-gravity[sl, 2], -1.0, 1.0)))
    walking = bool(np.any(mode[sl] == 1))
    smooth = int(round(0.5 / dt))

    def _cv(values: np.ndarray) -> float:
        if values.size < smooth + 1:
            return float("nan")
        kernel = np.ones(smooth) / smooth
        filt = np.convolve(values, kernel, mode="valid")
        mean = float(np.mean(filt))
        return float(np.std(filt) / max(abs(mean), 1e-6))

    return {
        "recording": str(recording),
        "trial": str(meta.get("trial_label", recording.name)),
        "teleop": teleop,
        "scale": scale,
        "command": command,
        "yaw_cell": yaw_cell,
        "seed": seed,
        "fell": bool(meta.get("fell", False)),
        "yaw_command_scale": applied,
        "derived_yaw_command_scale": derived,
        "yaw_overridden": bool(meta.get("yaw_command_scale_overridden", False)),
        "rx_median": rx,
        "ly_median": ly,
        "expected_cmd_yaw_rate_rad_s": expected,
        "cmd_yaw_rate_median_rad_s": float(np.median(cmd_rate)),
        "cmd_yaw_rate_ratio": float(
            np.median(cmd_rate) / expected if abs(expected) > 1e-6 else np.nan
        ),
        "pelvis_yaw_rate_median_rad_s": float(np.median(pelvis_rate)),
        "pelvis_to_cmd_ratio": float(
            np.median(pelvis_rate) / np.median(cmd_rate)
            if abs(np.median(cmd_rate)) > 1e-6 else np.nan
        ),
        "cmd_yaw_rate_cv": _cv(cmd_rate),
        "pelvis_yaw_rate_cv": _cv(pelvis_rate),
        "heading_change_deg": float(np.degrees(cmd_yaw[-1] - cmd_yaw[0])),
        "speed_median_mps": float(np.median(forward)),
        "signed_speed_median_mps": float(np.median(signed)),
        "speed_cv": _cv(forward),
        "near_stop_pct": float(100.0 * np.mean(np.abs(signed) < 0.05)),
        "tilt_p95_deg": float(np.percentile(tilt, 95)),
        "walking": walking,
        "duration_s": float((sl.stop - sl.start) * dt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recordings = sorted(
        path.parent for path in args.input.rglob("state.npz")
        if (path.parent / "meta.json").is_file()
    )
    if not recordings:
        raise SystemExit(f"no recordings found under {args.input}")

    rows = []
    failures = []
    for recording in recordings:
        try:
            rows.append(analyze(recording))
        except (KeyError, ValueError) as error:
            failures.append({"recording": str(recording), "reason": str(error)})

    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "run_metrics.csv", rows)
    _write_csv(output / "analysis_failures.csv", failures)

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["teleop"], row["scale"], row["command"], row["yaw_cell"])].append(row)

    def _ok(sel: list[dict]) -> str:
        if any(row["fell"] for row in sel):
            return "fall"
        ratios = [row["cmd_yaw_rate_ratio"] for row in sel if np.isfinite(row["cmd_yaw_rate_ratio"])]
        if ratios and (min(ratios) < 0.85 or max(ratios) > 1.15):
            return "yaw_mismatch"
        if any(row["command"].startswith("fwd") or row["command"] == "forward" for row in sel):
            if max(row["near_stop_pct"] for row in sel) > 5.0:
                return "near_stop"
            if max(row["speed_cv"] for row in sel) > 0.45:
                return "speed_jitter"
        if max(row["pelvis_yaw_rate_cv"] for row in sel) > 1.5 and abs(
            float(np.median([row["expected_cmd_yaw_rate_rad_s"] for row in sel]))
        ) > 0.2:
            return "yaw_jitter"
        return "ok"

    aggregate = []
    print(
        f"{'teleop':16s} {'scale':8s} {'cmd':8s} {'yaw':8s} n "
        f"cmd_rate  ratio  pelvis  p/c   spd    cv    stop  tilt  verdict"
    )
    for key, selected in sorted(grouped.items()):
        teleop, scale, command, yaw_cell = key
        verdict = _ok(selected)
        row = {
            "teleop": teleop,
            "scale": scale,
            "command": command,
            "yaw_cell": yaw_cell,
            "runs": len(selected),
            "falls": sum(bool(item["fell"]) for item in selected),
            "yaw_command_scale": float(np.median([item["yaw_command_scale"] for item in selected])),
            "expected_cmd_yaw_rate_rad_s": float(
                np.median([item["expected_cmd_yaw_rate_rad_s"] for item in selected])
            ),
            "cmd_yaw_rate_median_rad_s": float(
                np.median([item["cmd_yaw_rate_median_rad_s"] for item in selected])
            ),
            "cmd_yaw_rate_ratio_median": float(
                np.median([item["cmd_yaw_rate_ratio"] for item in selected])
            ),
            "pelvis_yaw_rate_median_rad_s": float(
                np.median([item["pelvis_yaw_rate_median_rad_s"] for item in selected])
            ),
            "pelvis_to_cmd_ratio_median": float(
                np.median([item["pelvis_to_cmd_ratio"] for item in selected])
            ),
            "speed_median_mps": float(np.median([item["speed_median_mps"] for item in selected])),
            "speed_cv_worst": max(item["speed_cv"] for item in selected),
            "near_stop_worst_pct": max(item["near_stop_pct"] for item in selected),
            "pelvis_yaw_rate_cv_worst": max(item["pelvis_yaw_rate_cv"] for item in selected),
            "tilt_p95_worst_deg": max(item["tilt_p95_deg"] for item in selected),
            "verdict": verdict,
        }
        aggregate.append(row)
        print(
            f"{teleop:16s} {scale:8s} {command:8s} {yaw_cell:8s} {len(selected)} "
            f"{row['cmd_yaw_rate_median_rad_s']:+.3f} "
            f"{row['cmd_yaw_rate_ratio_median']:5.2f} "
            f"{row['pelvis_yaw_rate_median_rad_s']:+.3f} "
            f"{row['pelvis_to_cmd_ratio_median']:4.2f} "
            f"{row['speed_median_mps']:5.2f} "
            f"{row['speed_cv_worst']:5.2f} "
            f"{row['near_stop_worst_pct']:5.1f} "
            f"{row['tilt_p95_worst_deg']:5.1f} "
            f"{verdict}"
        )
    _write_csv(output / "aggregate.csv", aggregate)
    print(f"\nanalyzed {len(rows)} recordings, {len(failures)} failures")


if __name__ == "__main__":
    main()
