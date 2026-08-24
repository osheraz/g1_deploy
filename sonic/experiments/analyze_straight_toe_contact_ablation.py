#!/usr/bin/env python3
"""Measure straightness and heel/toe touchdown geometry in MuJoCo rollouts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/"
    "20260818_forward_straight_toe_contact_ablation"
)


def _largest_segment(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("no active SLOW_WALK samples")
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return max(groups, key=len)


def _yaw_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, np.float64)
    w, x, y, z = quat.T
    return np.unwrap(np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ))


def _touchdowns(state, selected: np.ndarray, side: str) -> list[dict]:
    contact = np.asarray(state[f"{side}_foot_contact"], bool)
    corners = np.asarray(state[f"{side}_foot_corner_z"], np.float64)
    foot = np.asarray(state[f"{side}_foot_position"], np.float64)
    base = np.asarray(state["base_position"], np.float64)
    movement = np.asarray(state["nav_movement"], np.float64)[:, :2]
    events = []
    swing_samples = 0
    for index in range(1, len(contact)):
        if not selected[index]:
            swing_samples = 0
            continue
        if not contact[index - 1]:
            swing_samples += 1
        if contact[index] and not contact[index - 1] and swing_samples >= 5:
            # In the checked-in G1 MJCF the first two spheres are at x=-0.05 m
            # (heel), and the final two are at x=+0.12 m (toe).
            heel_z = float(np.mean(corners[index, :2]))
            toe_z = float(np.mean(corners[index, 2:]))
            direction = movement[index]
            direction /= max(np.linalg.norm(direction), 1e-12)
            end = min(index + 15, len(contact) - 1)
            while end > index and not np.all(contact[index:end + 1]):
                end -= 1
            relative = foot[index:end + 1, :2] - base[index:end + 1, :2]
            progress = (relative - relative[0]) @ direction
            events.append({
                "side": side,
                "index": index,
                "heel_minus_toe_z_m": heel_z - toe_z,
                "corner_range_m": float(np.ptp(corners[index])),
                "post_contact_progress_m": float(max(0.0, np.max(progress))),
            })
            swing_samples = 0
        elif contact[index]:
            swing_samples = 0
    return events


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def analyze(recording: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((recording / "meta.json").read_text())
    state = np.load(recording / "state.npz", allow_pickle=False)
    dt = float(meta["control_dt"])
    mode = np.asarray(state["mode"], np.int64)
    movement = np.asarray(state["nav_movement"], np.float64)[:, :2]
    segment = _largest_segment((mode == 1) & (np.linalg.norm(movement, axis=1) > 1e-6))
    trim = int(round(2.0 / dt))
    segment = segment[trim:-trim]
    if not len(segment):
        raise ValueError("active SLOW_WALK segment is too short after trimming")
    selected = np.zeros(len(mode), bool)
    selected[segment] = True

    direction = np.median(movement[selected], axis=0)
    direction /= max(np.linalg.norm(direction), 1e-12)
    lateral = np.array([-direction[1], direction[0]])
    base = np.asarray(state["base_position"], np.float64)[:, :2]
    relative = base[segment] - base[segment[0]]
    forward_trace = relative @ direction
    lateral_trace = relative @ lateral
    displacement = relative[-1]
    yaw = _yaw_wxyz(state["base_quat_wxyz"])[segment]
    command_yaw = float(np.arctan2(direction[1], direction[0]))
    yaw_error = np.unwrap(yaw - command_yaw)
    yaw_error -= yaw_error[0]

    events = []
    for side in ("left", "right"):
        events.extend(_touchdowns(state, selected, side))
    heel_minus_toe = [event["heel_minus_toe_z_m"] for event in events]
    progress = [event["post_contact_progress_m"] for event in events]
    trial = str(meta.get("trial_label", recording.name))
    trial_parts = trial.split("__")
    candidate = (
        trial_parts[1]
        if len(trial_parts) >= 4
        else recording.parent.name if recording.name.startswith("seed")
        else recording.name
    )
    condition = trial_parts[2] if len(trial_parts) >= 4 else "healthy"
    row = {
        "recording": str(recording),
        "trial": trial,
        "candidate": candidate,
        "condition": condition,
        "seed": int(meta.get("planner_random_seed", -1)),
        "duration_s": float(len(segment) * dt),
        "forward_displacement_m": float(displacement @ direction),
        "lateral_endpoint_m": float(displacement @ lateral),
        "lateral_abs_max_m": float(np.max(np.abs(lateral_trace))),
        "lateral_rms_m": float(np.sqrt(np.mean(lateral_trace ** 2))),
        "lateral_endpoint_per_forward": float(
            abs(displacement @ lateral) / max(abs(displacement @ direction), 1e-12)
        ),
        "heading_drift_deg": float(np.degrees(yaw_error[-1])),
        "heading_error_abs_p95_deg": float(np.percentile(np.abs(np.degrees(yaw_error)), 95)),
        "touchdowns": len(events),
        "toe_lower_median_mm": float(1000.0 * np.median(heel_minus_toe)) if events else float("nan"),
        "toe_first_gt_2mm_pct": float(100.0 * np.mean(np.asarray(heel_minus_toe) > 0.002)) if events else float("nan"),
        "toe_first_gt_5mm_pct": float(100.0 * np.mean(np.asarray(heel_minus_toe) > 0.005)) if events else float("nan"),
        "touchdown_corner_range_p90_mm": 1000.0 * _percentile(
            [event["corner_range_m"] for event in events], 90
        ),
        "post_contact_progress_p90_cm": 100.0 * _percentile(progress, 90),
        "post_contact_progress_gt_2cm_pct": float(
            100.0 * np.mean(np.asarray(progress) > 0.02)
        ) if events else float("nan"),
        "fell": bool(meta.get("fell", False)),
    }
    event_rows = [{"trial": row["trial"], **event} for event in events]
    return row, event_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["candidate"]].append(row)
    aggregate = []
    for candidate, selected in sorted(groups.items()):
        values = lambda key: np.asarray(
            [float(row[key]) for row in selected], np.float64
        )
        lateral = values("lateral_endpoint_per_forward")
        toe = values("toe_first_gt_5mm_pct")
        speed = values("forward_displacement_m") / values("duration_s")
        heading = values("heading_error_abs_p95_deg")
        aggregate.append({
            "candidate": candidate,
            "runs": len(selected),
            "conditions": len({row["condition"] for row in selected}),
            "seeds": len({int(row["seed"]) for row in selected}),
            "falls": sum(bool(row["fell"]) for row in selected),
            "lateral_per_forward_median_pct": 100.0 * np.median(lateral),
            "lateral_per_forward_p90_pct": 100.0 * np.percentile(lateral, 90),
            "lateral_per_forward_worst_pct": 100.0 * np.max(lateral),
            "toe_first_gt_5mm_mean_pct": np.mean(toe),
            "toe_first_gt_5mm_worst_pct": np.max(toe),
            "forward_speed_median_mps": np.median(speed),
            "forward_speed_min_mps": np.min(speed),
            "heading_error_p95_median_deg": np.median(heading),
            "heading_error_p95_worst_deg": np.max(heading),
        })
    return aggregate


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
    rows, events, failures = [], [], []
    for recording in recordings:
        try:
            row, event_rows = analyze(recording)
            rows.append(row)
            events.extend(event_rows)
        except (KeyError, ValueError) as error:
            failures.append({"recording": str(recording), "reason": str(error)})
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "straight_toe_metrics.csv", rows)
    _write_csv(output / "touchdown_events.csv", events)
    _write_csv(output / "analysis_failures.csv", failures)
    aggregate = _candidate_aggregate(rows)
    _write_csv(output / "candidate_aggregate.csv", aggregate)
    for row in rows:
        print(
            f"{row['trial']}: lateral={row['lateral_endpoint_m']:+.3f}m "
            f"({100 * row['lateral_endpoint_per_forward']:.1f}% of forward), "
            f"heading={row['heading_drift_deg']:+.1f}deg, "
            f"toe-first>5mm={row['toe_first_gt_5mm_pct']:.1f}%, "
            f"post-contact-p90={row['post_contact_progress_p90_cm']:.1f}cm"
        )
    print("\nCandidate aggregate:")
    for row in aggregate:
        print(
            f"{row['candidate']}: falls={row['falls']}/{row['runs']}, "
            "lateral median/p90/worst="
            f"{row['lateral_per_forward_median_pct']:.1f}/"
            f"{row['lateral_per_forward_p90_pct']:.1f}/"
            f"{row['lateral_per_forward_worst_pct']:.1f}%, "
            f"toe-first mean/worst={row['toe_first_gt_5mm_mean_pct']:.1f}/"
            f"{row['toe_first_gt_5mm_worst_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
