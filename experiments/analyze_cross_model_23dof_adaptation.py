#!/usr/bin/env python3
"""Analyze and rank cross-model fault-injected 29-to-23 experiments."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_slow_walk_training_free import analyze


DEFAULT_INPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/"
    "20260816_cross_model_23dof_adaptation"
)
DEFAULT_OUTPUT = Path(
    "/home/clhsieh/e_skin/docs/experiments/mujoco/"
    "2026-08-16_cross_model_23dof_adaptation"
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _identity(recording: Path, root: Path) -> tuple[str, str, str, str, int]:
    rel = recording.relative_to(root)
    if len(rel.parts) < 5:
        raise ValueError(f"unexpected cross-model recording path: {rel}")
    stage, model, candidate, condition, seed_text = rel.parts[:5]
    return stage, model, candidate, condition, int(seed_text.removeprefix("seed"))


def _joint_value(joints: list[dict], name: str, column: str) -> float:
    matches = [float(row[column]) for row in joints if row["joint"] == name]
    return matches[0] if matches else float("nan")


def _score_run(row: dict, native: dict | None) -> float:
    if native is None:
        return float("nan")
    def excess(name: str, floor: float) -> float:
        value = float(row[name])
        target = float(native[name])
        return max(0.0, value - target) / max(abs(target), floor)

    speed_target = max(abs(float(native["actual_forward_mean_mps"])), 0.20)
    terms = (
        abs(float(row["actual_forward_mean_mps"]) - float(native["actual_forward_mean_mps"])) / speed_target,
        excess("half_second_speed_cv", 0.20),
        excess("tilt_p95_deg", 4.0),
        excess("leg_error_rms_rad", 0.20),
        excess("post_contact_progress_p90_m", 0.02),
        abs(float(row["double_support_pct"]) - float(native["double_support_pct"])) / 25.0,
        abs(float(row["cadence_landings_per_s"]) - float(native["cadence_landings_per_s"]))
        / max(float(native["cadence_landings_per_s"]), 0.5),
    )
    score = float(np.mean(terms))
    if float(row["actual_forward_mean_mps"]) < 0.20:
        score += (0.20 - float(row["actual_forward_mean_mps"])) / 0.10
    if bool(row["fell"]):
        score += 10.0
    if not bool(row["capture_succeeded"]):
        score += 0.5
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    recordings = sorted(
        path.parent for path in args.input.rglob("state.npz")
        if (path.parent / "meta.json").is_file()
        and "benchmark" not in path.parts
    )
    if not recordings:
        raise SystemExit(f"no cross-model recordings found under {args.input}")

    planned_cells: dict[tuple[str, str, str], set[tuple[str, int]]] = defaultdict(set)
    for plan_path in args.input.glob("plan_*.json"):
        try:
            plan_rows = json.loads(plan_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for plan_row in plan_rows:
            try:
                stage = Path(plan_row["record"]).relative_to(args.input).parts[0]
                key = (stage, plan_row["model"], plan_row["candidate"])
                planned_cells[key].add(
                    (plan_row["condition"], int(plan_row["seed"]))
                )
            except (KeyError, ValueError, IndexError):
                continue

    summaries: list[dict] = []
    all_joints: list[dict] = []
    all_landings: list[dict] = []
    failures: list[dict] = []
    for recording in recordings:
        stage, model, candidate, condition, seed = _identity(recording, args.input)
        try:
            summary, joints, landings = analyze(recording)
        except (ValueError, KeyError) as exc:
            failures.append({
                "recording": str(recording), "stage": stage, "model": model,
                "candidate": candidate, "condition": condition, "seed": seed,
                "reason": str(exc),
            })
            continue
        state = np.load(recording / "state.npz", allow_pickle=False)
        capture_succeeded = bool(
            "stand_capture_count" in state.files
            and np.max(np.asarray(state["stand_capture_count"], np.int64)) > 0
        )
        identity = {
            "stage": stage, "model": model, "candidate": candidate,
            "condition": condition, "seed": seed,
        }
        summary.update(identity)
        summary["capture_succeeded"] = capture_succeeded
        summary["left_knee_error_rms_rad"] = _joint_value(joints, "left_knee", "error_rms_rad")
        summary["right_knee_error_rms_rad"] = _joint_value(joints, "right_knee", "error_rms_rad")
        summary["left_ankle_pitch_error_rms_rad"] = _joint_value(joints, "left_ankle_pitch", "error_rms_rad")
        summary["right_ankle_pitch_error_rms_rad"] = _joint_value(joints, "right_ankle_pitch", "error_rms_rad")
        summaries.append(summary)
        all_joints.extend([{**row, **identity} for row in joints])
        all_landings.extend([{**row, **identity} for row in landings])

    native = {
        (row["model"], row["condition"], int(row["seed"])): row
        for row in summaries if row["candidate"] == "native29"
    }
    native_healthy = {
        (row["model"], int(row["seed"])): row
        for row in summaries
        if row["candidate"] == "native29" and row["condition"] == "healthy"
    }
    for row in summaries:
        reference = native.get(
            (row["model"], row["condition"], int(row["seed"])),
            native_healthy.get((row["model"], int(row["seed"])))
            or native_healthy.get((row["model"], 1234)),
        )
        row["native29_relative_score"] = _score_run(row, reference)

    # Baseline, search, and finalist stages intentionally overlap so a stage
    # can be resumed independently. Rank one result per matched experimental
    # cell, preferring the most rigorous stage, instead of double-weighting a
    # deterministic seed simply because it was rerun.
    stage_rank = {"baseline": 0, "search": 1, "finalists": 2}
    selected_by_cell: dict[tuple[str, str, str, int], dict] = {}
    for row in summaries:
        key = (
            row["model"], row["candidate"], row["condition"], int(row["seed"])
        )
        current = selected_by_cell.get(key)
        if current is None or stage_rank.get(row["stage"], -1) > stage_rank.get(
            current["stage"], -1
        ):
            selected_by_cell[key] = row
    # Compare a candidate only on its most rigorous completed protocol. This
    # prevents an extra baseline-only condition from making a candidate with
    # otherwise identical search coverage appear better or worse.
    available_by_stage: dict[tuple[str, str, str], set[tuple[str, int]]] = defaultdict(set)
    for row in summaries:
        if row["candidate"] == "native29":
            continue
        available_by_stage[(row["stage"], row["model"], row["candidate"])].add(
            (row["condition"], int(row["seed"]))
        )
    candidate_stage: dict[tuple[str, str], str] = {}
    for stage, model, candidate in available_by_stage:
        key = (model, candidate)
        planned = planned_cells.get((stage, model, candidate))
        available = available_by_stage[(stage, model, candidate)]
        complete = not planned or planned.issubset(available)
        if not complete:
            continue
        current = candidate_stage.get(key)
        if current is None or stage_rank.get(stage, -1) > stage_rank.get(current, -1):
            candidate_stage[key] = stage
    # If an interrupted campaign has no complete stage, retain its most
    # advanced partial data rather than dropping the candidate entirely.
    partial_stage: dict[tuple[str, str], str] = {}
    for stage, model, candidate in available_by_stage:
        key = (model, candidate)
        current = partial_stage.get(key)
        if current is None or stage_rank.get(stage, -1) > stage_rank.get(current, -1):
            partial_stage[key] = stage
    for key, stage in partial_stage.items():
        candidate_stage.setdefault(key, stage)
    group_cells: dict[tuple[str, str, str, int], dict] = {}
    for row in summaries:
        key = (row["model"], row["candidate"])
        if row["candidate"] != "native29" and row["stage"] == candidate_stage[key]:
            group_cells[
                (row["model"], row["candidate"], row["condition"], int(row["seed"]))
            ] = row
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in group_cells.values():
        grouped[(row["model"], row["candidate"])].append(row)
    selected_ids = {id(row) for row in group_cells.values()}
    selected_ids.update(
        id(row) for row in selected_by_cell.values() if row["candidate"] == "native29"
    )
    for row in summaries:
        row["ranking_selected"] = id(row) in selected_ids
    aggregate: list[dict] = []
    for (model, candidate), rows in grouped.items():
        scores = np.asarray([row["native29_relative_score"] for row in rows], float)
        finite_scores = scores[np.isfinite(scores)]
        aggregate.append({
            "model": model,
            "candidate": candidate,
            "protocol_stage": candidate_stage[(model, candidate)],
            "runs": len(rows),
            "conditions": len({row["condition"] for row in rows}),
            "seeds": len({row["seed"] for row in rows}),
            "falls": sum(bool(row["fell"]) for row in rows),
            "capture_failures": sum(not bool(row["capture_succeeded"]) for row in rows),
            "fall_rate": float(np.mean([bool(row["fell"]) for row in rows])),
            "capture_failure_rate": float(np.mean([
                not bool(row["capture_succeeded"]) for row in rows
            ])),
            "speed_median_mps": float(np.median([row["actual_forward_mean_mps"] for row in rows])),
            "speed_min_mps": float(np.min([row["actual_forward_mean_mps"] for row in rows])),
            "speed_cv_median": float(np.median([row["half_second_speed_cv"] for row in rows])),
            "speed_cv_worst": float(np.max([row["half_second_speed_cv"] for row in rows])),
            "tilt_p95_median_deg": float(np.median([row["tilt_p95_deg"] for row in rows])),
            "tilt_p95_worst_deg": float(np.max([row["tilt_p95_deg"] for row in rows])),
            "leg_error_median_rad": float(np.median([row["leg_error_rms_rad"] for row in rows])),
            "leg_error_worst_rad": float(np.max([row["leg_error_rms_rad"] for row in rows])),
            "early_progress_p90_median_m": float(np.nanmedian([row["post_contact_progress_p90_m"] for row in rows])),
            "early_progress_p90_worst_m": float(np.nanmax([row["post_contact_progress_p90_m"] for row in rows])),
            "left_knee_error_worst_rad": float(np.max([row["left_knee_error_rms_rad"] for row in rows])),
            "left_ankle_error_worst_rad": float(np.max([row["left_ankle_pitch_error_rms_rad"] for row in rows])),
            "relative_score_median": float(np.median(finite_scores)) if len(finite_scores) else float("nan"),
            "relative_score_worst": float(np.max(finite_scores)) if len(finite_scores) else float("nan"),
            "robust_selection_score": (
                float(np.median(finite_scores) + np.max(finite_scores))
                if len(finite_scores) else float("nan")
            ),
        })
    aggregate.sort(key=lambda row: (
        row["model"], row["fall_rate"],
        -stage_rank.get(row["protocol_stage"], -1),
        row["robust_selection_score"], row["capture_failure_rate"],
    ))

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "run_metrics.csv", summaries)
    _write_csv(args.output / "joint_metrics.csv", all_joints)
    _write_csv(args.output / "landing_events.csv", all_landings)
    _write_csv(args.output / "candidate_aggregate.csv", aggregate)
    _write_csv(args.output / "failed_runs.csv", failures)
    (args.output / "recordings.json").write_text(
        json.dumps([str(path) for path in recordings], indent=2) + "\n"
    )
    for model in sorted({row["model"] for row in aggregate}):
        print(f"\n{model}")
        selected = [row for row in aggregate if row["model"] == model]
        for row in selected[:12]:
            print(
                f"  {row['candidate']:<28} runs={row['runs']:>3} "
                f"fall={row['falls']} capfail={row['capture_failures']} "
                f"score={row['robust_selection_score']:.3f} "
                f"speed={row['speed_median_mps']:.3f}/{row['speed_min_mps']:.3f} "
                f"tilt={row['tilt_p95_median_deg']:.2f}/{row['tilt_p95_worst_deg']:.2f} "
                f"err={row['leg_error_median_rad']:.3f}/{row['leg_error_worst_rad']:.3f}"
            )


if __name__ == "__main__":
    main()
