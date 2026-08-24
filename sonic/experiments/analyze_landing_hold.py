#!/usr/bin/env python3
"""Score the legacy landing-hold attempt against walk/turn/stand health."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_slower_gait import analyze as analyze_gait
from analyze_slower_gait import _verdict


DEFAULT_INPUT = Path("/data/clhsieh/logs/e_skin/experiments/20260819_landing_hold")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _landings(state, sl: slice, dt: float) -> list[dict]:
    base = np.asarray(state["base_position"], np.float64)
    facing = np.asarray(state["nav_facing"], np.float64)
    rows = []
    for side in ("left", "right"):
        contact = np.asarray(state[f"{side}_foot_contact"], bool)
        foot = np.asarray(state[f"{side}_foot_position"], np.float64)
        corners = np.asarray(state[f"{side}_foot_corner_z"], np.float64)
        swing = 0
        for index in range(max(sl.start, 1), sl.stop):
            if not contact[index - 1]:
                swing += 1
            if contact[index] and not contact[index - 1] and swing >= 5:
                end = min(index + 15, sl.stop - 1)
                while end > index and not np.all(contact[index:end + 1]):
                    end -= 1
                direction = facing[index, :2]
                direction = direction / max(np.linalg.norm(direction), 1e-12)
                relative = foot[index:end + 1, :2] - base[index:end + 1, :2]
                progress = (relative - relative[0]) @ direction
                heel = float(np.mean(corners[index, :2]))
                toe = float(np.mean(corners[index, 2:]))
                rows.append({
                    "side": side,
                    "post_contact_progress_m": float(max(0.0, np.max(progress))),
                    "heel_minus_toe_z_m": heel - toe,
                    "swing_before_s": swing * dt,
                })
                swing = 0
            elif contact[index]:
                swing = 0
    return rows


def analyze(recording: Path) -> dict:
    row = analyze_gait(recording)
    meta = json.loads((recording / "meta.json").read_text())
    state = np.load(recording / "state.npz", allow_pickle=False)
    dt = float(meta["control_dt"])
    axes = np.asarray(state["axes"], np.float64)
    active = np.linalg.norm(np.nan_to_num(axes), axis=1) > 0.2
    indices = np.flatnonzero(active)
    pad = int(round(1.0 / dt))
    sl = (
        slice(indices[0] + pad, indices[-1] + 1 - pad)
        if len(indices) > 2 * pad else slice(indices[0], indices[-1] + 1)
    )
    events = _landings(state, sl, dt)
    progress = [event["post_contact_progress_m"] for event in events]
    toes = [event["heel_minus_toe_z_m"] for event in events]
    hold = (
        np.asarray(state["sonic_landing_hold"], bool)[sl]
        if "sonic_landing_hold" in state.files else None
    )
    row.update({
        "landing_hold_enabled": bool(meta.get("landing_hold", False)),
        "landings": len(events),
        "post_contact_progress_median_cm": (
            100.0 * float(np.median(progress)) if progress else float("nan")
        ),
        "post_contact_progress_p90_cm": (
            100.0 * float(np.percentile(progress, 90)) if progress else float("nan")
        ),
        "post_contact_gt_2cm_pct": (
            float(100.0 * np.mean(np.asarray(progress) > 0.02)) if progress else float("nan")
        ),
        "toe_first_gt_5mm_pct": (
            float(100.0 * np.mean(np.asarray(toes) > 0.005)) if toes else float("nan")
        ),
        "hold_duty_pct": (
            float("nan") if hold is None else float(100.0 * np.mean(np.any(hold, axis=1)))
        ),
    })
    return row


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

    rows = [analyze(recording) for recording in recordings]
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "run_metrics.csv", rows)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", ""), row["candidate"], row["command"])].append(row)

    print(
        f"{'model':18s} {'cand':32s} {'cmd':8s} n spd   cad  swD  "
        f"p90cm >2cm  toe5 hold  verdict"
    )
    aggregate = []
    for (model, candidate, command), selected in sorted(grouped.items()):
        verdict = _verdict(selected)
        progress = [row["post_contact_progress_p90_cm"] for row in selected]
        early = [row["post_contact_gt_2cm_pct"] for row in selected]
        toes = [row["toe_first_gt_5mm_pct"] for row in selected]
        holds = [row["hold_duty_pct"] for row in selected]
        row = {
            "model": model,
            "candidate": candidate,
            "command": command,
            "runs": len(selected),
            "speed_median_mps": float(np.median([item["speed_median_mps"] for item in selected])),
            "cadence_min": min(item["cadence"] for item in selected),
            "swing_dur_min_s": min(item["swing_dur_median_s"] for item in selected),
            "post_contact_p90_cm": float(np.nanmedian(progress)),
            "post_contact_gt_2cm_worst_pct": float(np.nanmax(early)),
            "toe_first_gt_5mm_worst_pct": float(np.nanmax(toes)),
            "hold_duty_median_pct": float(np.nanmedian(holds)),
            "verdict": verdict,
        }
        aggregate.append(row)
        print(
            f"{model:18s} {candidate:32s} {command:8s} {len(selected)} "
            f"{row['speed_median_mps']:.3f} {row['cadence_min']:.2f} "
            f"{row['swing_dur_min_s']:.2f} {row['post_contact_p90_cm']:5.1f} "
            f"{row['post_contact_gt_2cm_worst_pct']:5.1f} "
            f"{row['toe_first_gt_5mm_worst_pct']:5.1f} "
            f"{row['hold_duty_median_pct']:5.1f} {verdict}"
        )
    _write_csv(output / "aggregate.csv", aggregate)
    print(f"\nanalyzed {len(rows)} recordings")


if __name__ == "__main__":
    main()
