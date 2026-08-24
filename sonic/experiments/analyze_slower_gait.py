#!/usr/bin/env python3
"""Score slower-gait candidates on walk swing, turn rate, and stand hold."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("/data/clhsieh/logs/e_skin/experiments/20260818_slower_gait")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _yaw_from_facing(facing: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(facing[:, 1], facing[:, 0]))


def _yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, np.float64).T
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _swing_bouts(contact: np.ndarray, z: np.ndarray, dt: float) -> tuple[float, float, float]:
    enter = np.flatnonzero((~contact[1:]) & contact[:-1]) + 1
    leave = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
    durs = []
    clears = []
    for start in enter:
        later = leave[leave > start]
        if not len(later):
            break
        end = int(later[0])
        if end > start:
            durs.append((end - start) * dt)
            clears.append(float(np.max(z[start:end])))
    if not durs:
        return float("nan"), float("nan"), float(100.0 * np.mean(~contact))
    return float(np.median(durs)), float(np.median(clears)), float(100.0 * np.mean(~contact))


def analyze(recording: Path) -> dict:
    meta = json.loads((recording / "meta.json").read_text())
    state = np.load(recording / "state.npz", allow_pickle=False)
    dt = float(meta["control_dt"])
    mode = np.asarray(state["mode"], np.int64)
    axes = np.asarray(state["axes"], np.float64)
    facing = np.asarray(state["nav_facing"], np.float64)
    base = np.asarray(state["base_position"], np.float64)
    quat = np.asarray(state["base_quat_wxyz"], np.float64)
    gravity = np.asarray(state["projected_gravity"], np.float64)
    q = np.asarray(state["q_true"] if "q_true" in state.files else state["q"], np.float64)
    left_c = np.asarray(state["left_foot_contact"], bool)
    right_c = np.asarray(state["right_foot_contact"], bool)
    left_z = np.asarray(state["left_foot_position"], np.float64)[:, 2]
    right_z = np.asarray(state["right_foot_position"], np.float64)[:, 2]
    tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
    parts = recording.resolve().parts
    candidate = parts[-3]
    command = parts[-2]
    seed = recording.name
    model = str(meta.get("model_profile") or "")
    if not model and len(parts) >= 5 and parts[-4] in (
        "low-latency", "original-release", "sonic-v1.1",
    ):
        model = parts[-4]

    active = np.linalg.norm(np.nan_to_num(axes), axis=1) > 0.2
    indices = np.flatnonzero(active)
    if len(indices) < 25:
        raise ValueError("no active stick window")
    pad = int(round(1.0 / dt))
    sl = slice(indices[0] + pad, indices[-1] + 1 - pad) if len(indices) > 2 * pad else slice(indices[0], indices[-1] + 1)
    duration = max((sl.stop - sl.start) * dt, 1e-6)

    dxy = np.diff(base[sl, :2], axis=0)
    planar = np.linalg.norm(dxy, axis=1) / dt
    current = facing[sl.start:sl.stop - 1, :2]
    current = current / np.clip(np.linalg.norm(current, axis=1, keepdims=True), 1e-9, None)
    signed = np.sum(dxy * current, axis=1) / dt
    cmd_rate = np.diff(_yaw_from_facing(facing[sl])) / dt
    pelvis_rate = np.diff(_yaw_from_quat(quat[sl])) / dt
    left_dur, left_clr, left_sw = _swing_bouts(left_c[sl], left_z[sl], dt)
    right_dur, right_clr, right_sw = _swing_bouts(right_c[sl], right_z[sl], dt)
    landings = int(np.sum(left_c[sl][1:] & ~left_c[sl][:-1]) + np.sum(right_c[sl][1:] & ~right_c[sl][:-1]))
    rx = float(np.nanmedian(axes[sl, 2]))
    yaw_walk = float(meta.get("yaw_command_scale", 1.0))
    yaw_turn = float(meta.get("yaw_inplace_scale", yaw_walk))
    stick_xy = np.linalg.norm(np.nan_to_num(axes[sl, :2]), axis=1)
    turning_in_place = float(np.median(stick_xy)) < 0.2
    expected_yaw = -2.0 * (yaw_turn if turning_in_place else yaw_walk) * rx

    init = (mode == 0) & (np.arange(len(mode)) > 100) & (np.arange(len(mode)) < 250)
    final = (mode == 0) & (np.arange(len(mode)) >= sl.stop + int(round(1.0 / dt)))

    def _idle(mask: np.ndarray, prefix: str) -> dict:
        if not np.any(mask):
            return {
                f"{prefix}_duration_s": 0.0,
                f"{prefix}_tilt_p95_deg": float("nan"),
                f"{prefix}_drift_m": float("nan"),
                f"{prefix}_double_support_pct": float("nan"),
            }
        xy = base[mask, :2]
        return {
            f"{prefix}_duration_s": float(mask.sum() * dt),
            f"{prefix}_tilt_p95_deg": float(np.percentile(tilt[mask], 95)),
            f"{prefix}_drift_m": float(np.linalg.norm(xy[-1] - xy[0])),
            f"{prefix}_double_support_pct": float(100.0 * np.mean(left_c[mask] & right_c[mask])),
        }

    return {
        "recording": str(recording),
        "trial": str(meta.get("trial_label", recording.name)),
        "model": model,
        "candidate": candidate,
        "command": command,
        "seed": seed,
        "fell": bool(meta.get("fell", False)),
        "yaw_command_scale": float(meta.get("yaw_command_scale", np.nan)),
        "yaw_inplace_scale": float(meta.get("yaw_inplace_scale", np.nan)),
        "speed_median_mps": float(np.median(planar)),
        "signed_speed_median_mps": float(np.median(signed)),
        "speed_p10_mps": float(np.percentile(signed, 10)),
        "near_stop_pct": float(100.0 * np.mean(np.abs(signed) < 0.05)),
        "backward_pct": float(100.0 * np.mean(signed < -0.02)),
        "cadence": float(landings / duration),
        "double_support_pct": float(100.0 * np.mean(left_c[sl] & right_c[sl])),
        "swing_dur_median_s": float(np.nanmean([left_dur, right_dur])),
        "swing_clear_median_m": float(np.nanmean([left_clr, right_clr])),
        "swing_pct": float(np.mean([left_sw, right_sw])),
        "swing_pct_left": left_sw,
        "swing_pct_right": right_sw,
        "tilt_p95_deg": float(np.percentile(tilt[sl], 95)),
        "cmd_yaw_rate_rad_s": float(np.median(cmd_rate)),
        "expected_yaw_rate_rad_s": expected_yaw,
        "cmd_yaw_ratio": float(
            np.median(cmd_rate) / expected_yaw if abs(expected_yaw) > 1e-6 else np.nan
        ),
        "pelvis_yaw_rate_rad_s": float(np.median(pelvis_rate)),
        "hip_ptp_deg": float(np.mean(np.degrees(np.ptp(q[sl][:, [0, 6]], axis=0)))),
        "knee_ptp_deg": float(np.mean(np.degrees(np.ptp(q[sl][:, [3, 9]], axis=0)))),
        "ankle_ptp_deg": float(np.mean(np.degrees(np.ptp(q[sl][:, [4, 10]], axis=0)))),
        "walking": bool(np.any(mode[sl] == 1)),
        **_idle(init, "stand_init"),
        **_idle(final, "stand_final"),
    }


def _verdict(rows: list[dict]) -> str:
    if any(row["fell"] for row in rows):
        return "fall"
    command = rows[0]["command"]
    stand_tilt = max(row["stand_final_tilt_p95_deg"] for row in rows)
    stand_drift = max(row["stand_final_drift_m"] for row in rows)
    if stand_tilt > 8.0 or stand_drift > 0.10:
        return "stand_drift"
    if command == "turn_cw":
        ratios = [row["cmd_yaw_ratio"] for row in rows if np.isfinite(row["cmd_yaw_ratio"])]
        if ratios and (min(ratios) < 0.85 or max(ratios) > 1.15):
            return "yaw_mismatch"
        if min(row["cadence"] for row in rows) < 1.05:
            return "cadence_break"
        return "ok"
    if command == "fwd_cw":
        # Facing-aligned near-stop is a tight-circle artifact, even on sag035.
        if min(row["cadence"] for row in rows) < 1.05:
            return "cadence_break"
        if min(row["swing_dur_median_s"] for row in rows) < 0.12:
            return "swing_collapse"
        if min(row["speed_median_mps"] for row in rows) < 0.08:
            return "near_stop"
        if min(row["swing_pct"] for row in rows) < 5.0:
            return "inside_foot_starve"
        return "ok"
    if max(row["near_stop_pct"] for row in rows) > 5.0:
        return "near_stop"
    if max(row["backward_pct"] for row in rows) > 1.0:
        return "backward"
    if min(row["cadence"] for row in rows) < 1.05:
        return "cadence_break"
    if min(row["swing_dur_median_s"] for row in rows) < 0.20:
        return "swing_collapse"
    if min(row["speed_p10_mps"] for row in rows) < 0.06 and command == "forward":
        return "p10_collapse"
    return "ok"


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

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", ""), row["candidate"], row["command"])].append(row)

    aggregate = []
    print(
        f"{'model':18s} {'cand':28s} {'cmd':8s} n spd   p10   cvstop cad  swD   "
        f"clr  yawR  standT verdict"
    )
    for (model, candidate, command), selected in sorted(grouped.items()):
        verdict = _verdict(selected)
        speeds = [row["speed_median_mps"] for row in selected]
        p10s = [row["speed_p10_mps"] for row in selected]
        stops = [row["near_stop_pct"] for row in selected]
        cads = [row["cadence"] for row in selected]
        swings = [row["swing_dur_median_s"] for row in selected]
        clears = [row["swing_clear_median_m"] for row in selected]
        tilts = [row["stand_final_tilt_p95_deg"] for row in selected]
        row = {
            "model": model,
            "candidate": candidate,
            "command": command,
            "runs": len(selected),
            "falls": sum(bool(item["fell"]) for item in selected),
            "speed_median_mps": float(np.median(speeds)),
            "speed_min_mps": min(speeds),
            "p10_min_mps": min(p10s),
            "near_stop_worst_pct": max(stops),
            "backward_worst_pct": max(row["backward_pct"] for row in selected),
            "cadence_min": min(cads),
            "swing_dur_min_s": min(swings),
            "swing_clear_min_m": min(clears),
            "cmd_yaw_ratio_median": float(np.nanmedian(
                [item["cmd_yaw_ratio"] for item in selected]
            )),
            "stand_final_tilt_worst_deg": max(tilts),
            "stand_final_drift_worst_m": max(item["stand_final_drift_m"] for item in selected),
            "verdict": verdict,
        }
        aggregate.append(row)
        print(
            f"{model:18s} {candidate:28s} {command:8s} {len(selected)} "
            f"{row['speed_median_mps']:.3f} {row['p10_min_mps']:+.3f} "
            f"{row['near_stop_worst_pct']:5.1f} {row['cadence_min']:.2f} "
            f"{row['swing_dur_min_s']:.2f} {1000*row['swing_clear_min_m']:5.1f} "
            f"{row['cmd_yaw_ratio_median']:5.2f} "
            f"{row['stand_final_tilt_worst_deg']:5.1f} {verdict}"
        )
    _write_csv(output / "aggregate.csv", aggregate)
    print(f"\nanalyzed {len(rows)} recordings, {len(failures)} failures")


if __name__ == "__main__":
    main()
