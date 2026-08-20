#!/usr/bin/env python3
"""Rank small-gait reference-scale candidates on speed, consistency, and contact."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_straight_toe_contact_ablation import analyze as analyze_straight
from analyze_slow_walk_training_free import analyze as analyze_dynamics


DEFAULT_INPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/20260818_small_consistent_gait"
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _worst(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def _min(values: list[float]) -> float:
    return float(np.min(values)) if values else float("nan")


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
            straight, _ = analyze_straight(recording)
            dynamics, _, _ = analyze_dynamics(recording)
        except (KeyError, ValueError) as error:
            failures.append({"recording": str(recording), "reason": str(error)})
            continue
        rows.append({
            **straight,
            "actual_forward_mean_mps": dynamics["actual_forward_mean_mps"],
            "actual_forward_p10_mps": dynamics["actual_forward_p10_mps"],
            "half_second_speed_cv": dynamics["half_second_speed_cv"],
            "near_stop_pct": dynamics["near_stop_pct"],
            "backward_motion_pct": dynamics["backward_motion_pct"],
            "tilt_p95_deg": dynamics["tilt_p95_deg"],
            "height_p05_m": dynamics["height_p05_m"],
            "leg_error_rms_rad": dynamics["leg_error_rms_rad"],
            "double_support_pct": dynamics["double_support_pct"],
            "cadence_landings_per_s": dynamics["cadence_landings_per_s"],
            "leg_speed_rms_rad_s": dynamics["leg_speed_rms_rad_s"],
        })

    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "run_metrics.csv", rows)
    _write_csv(output / "analysis_failures.csv", failures)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["condition"])].append(row)

    aggregate = []
    for (candidate, condition), selected in sorted(grouped.items()):
        speeds = [float(row["actual_forward_mean_mps"]) for row in selected]
        cvs = [float(row["half_second_speed_cv"]) for row in selected]
        p10s = [float(row["actual_forward_p10_mps"]) for row in selected]
        laterals = [100.0 * float(row["lateral_endpoint_per_forward"]) for row in selected]
        toes = [float(row["toe_first_gt_5mm_pct"]) for row in selected]
        tilts = [float(row["tilt_p95_deg"]) for row in selected]
        stops = [float(row["near_stop_pct"]) for row in selected]
        cadences = [float(row["cadence_landings_per_s"]) for row in selected]
        supports = [float(row["double_support_pct"]) for row in selected]
        aggregate.append({
            "candidate": candidate,
            "condition": condition,
            "runs": len(selected),
            "falls": sum(bool(row["fell"]) for row in selected),
            "speed_median_mps": _median(speeds),
            "speed_min_mps": _min(speeds),
            "speed_p10_min_mps": _min(p10s),
            "speed_cv_median": _median(cvs),
            "speed_cv_worst": _worst(cvs),
            "near_stop_worst_pct": _worst(stops),
            "lateral_median_pct": _median(laterals),
            "lateral_worst_pct": _worst(laterals),
            "toe_first_mean_pct": float(np.mean(toes)) if toes else float("nan"),
            "toe_first_worst_pct": _worst(toes),
            "tilt_p95_median_deg": _median(tilts),
            "tilt_p95_worst_deg": _worst(tilts),
            "cadence_median": _median(cadences),
            "double_support_median_pct": _median(supports),
        })
    _write_csv(output / "candidate_aggregate.csv", aggregate)

    print("Per-run:")
    for row in rows:
        print(
            f"{row['trial']}: speed={row['actual_forward_mean_mps']:.3f} "
            f"p10={row['actual_forward_p10_mps']:.3f} "
            f"cv={row['half_second_speed_cv']:.3f} "
            f"stop={row['near_stop_pct']:.1f}% "
            f"lat={100 * row['lateral_endpoint_per_forward']:.1f}% "
            f"toe5={row['toe_first_gt_5mm_pct']:.1f}% "
            f"tilt={row['tilt_p95_deg']:.2f} "
            f"ds={row['double_support_pct']:.1f}% "
            f"cad={row['cadence_landings_per_s']:.2f} "
            f"fell={row['fell']}"
        )
    print("\nAggregate:")
    for row in aggregate:
        print(
            f"{row['candidate']}/{row['condition']}: "
            f"speed {row['speed_median_mps']:.3f}/{row['speed_min_mps']:.3f} "
            f"cv {row['speed_cv_median']:.3f}/{row['speed_cv_worst']:.3f} "
            f"p10min {row['speed_p10_min_mps']:.3f} "
            f"stop {row['near_stop_worst_pct']:.1f}% "
            f"lat {row['lateral_median_pct']:.1f}/{row['lateral_worst_pct']:.1f}% "
            f"toe {row['toe_first_mean_pct']:.1f}/{row['toe_first_worst_pct']:.1f}% "
            f"tilt {row['tilt_p95_median_deg']:.2f}/{row['tilt_p95_worst_deg']:.2f} "
            f"falls {row['falls']}/{row['runs']}"
        )


if __name__ == "__main__":
    main()
