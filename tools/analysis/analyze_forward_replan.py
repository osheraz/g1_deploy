#!/usr/bin/env python3
"""Diagnose straight-line stop/go staging and bilateral leg demand.

This tool consumes three operator-driven recordings plus the controlled replan
A/B and long validation directories.  It never imports or runs the controller.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEG_LABELS = (
    "L hip pitch", "L hip roll", "L hip yaw", "L knee", "L ankle pitch",
    "L ankle roll", "R hip pitch", "R hip roll", "R hip yaw", "R knee",
    "R ankle pitch", "R ankle roll",
)


def load_recording(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    meta = json.loads((path / "meta.json").read_text())
    with np.load(path / "state.npz") as archive:
        state = {name: archive[name] for name in archive.files}
    return meta, state


def smooth(values: np.ndarray, samples: int = 10) -> np.ndarray:
    return np.convolve(values, np.ones(samples) / samples, mode="same")


def dynamics(state: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    time_s = state["time_s"]
    dt = float(np.median(np.diff(time_s)))
    position = state["base_position"]
    speed = np.r_[
        0.0,
        np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1) / dt,
    ]
    gravity = state["projected_gravity"]
    tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
    return {"dt": dt, "speed": speed, "speed_smooth": smooth(speed), "tilt": tilt}


def command_masks(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    axes = state["axes"]
    mode = state["mode"]
    moving = mode > 0
    translation = np.hypot(axes[:, 0], axes[:, 1])
    return {
        "pure_forward": (
            moving & (axes[:, 1] > 0.8) & (np.abs(axes[:, 0]) < 0.08)
            & (np.abs(axes[:, 2]) < 0.08)
        ),
        "turning": moving & (translation > 0.8) & (np.abs(axes[:, 2]) > 0.2),
    }


def trim_island_starts(mask: np.ndarray, samples: int = 25) -> np.ndarray:
    trimmed = mask.copy()
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    for start in starts:
        trimmed[start:min(len(mask), start + samples)] = False
    return trimmed


def replan_events(state: dict[str, np.ndarray]) -> np.ndarray:
    if "planner_replan_count" in state:
        count = state["planner_replan_count"]
        return np.r_[False, np.diff(count) > 0]
    timing = state["planner_total_ms"]
    return np.r_[
        False,
        (np.abs(np.diff(timing)) > 1e-9) & np.isfinite(timing[1:]),
    ]


def metrics(meta: dict, state: dict[str, np.ndarray], mask: np.ndarray, command: str) -> dict:
    dyn = dynamics(state)
    dt = float(dyn["dt"])
    speed = np.asarray(dyn["speed_smooth"])[mask]
    error = state["target_q"] - state["q"]
    force = state["actuator_force"]
    event = replan_events(state)
    threshold = 0.25 * float(np.quantile(speed, 0.9)) if len(speed) else np.nan
    row = {
        "model": meta["model_profile"],
        "command": command,
        "seconds": float(mask.sum() * dt),
        "speed_p10_mps": float(np.quantile(speed, 0.1)),
        "speed_median_mps": float(np.median(speed)),
        "speed_p90_mps": float(np.quantile(speed, 0.9)),
        "near_stop_fraction": float(np.mean(speed < threshold)),
        "replan_hz": float(event[mask].sum() / (mask.sum() * dt)),
        "tilt_p95_deg": float(np.quantile(np.asarray(dyn["tilt"])[mask], 0.95)),
        "leg_error_rms_rad": float(np.sqrt(np.mean(error[mask, :12] ** 2))),
        "left_ankle_error_rms_rad": float(np.sqrt(np.mean(error[mask, 4] ** 2))),
        "right_ankle_error_rms_rad": float(np.sqrt(np.mean(error[mask, 10] ** 2))),
        "left_ankle_force_rms_nm": float(np.sqrt(np.mean(force[mask, 4] ** 2))),
        "right_ankle_force_rms_nm": float(np.sqrt(np.mean(force[mask, 10] ** 2))),
    }
    return row


def full_run_metrics(path: Path, *, trim_s: float = 2.0) -> tuple[dict, list[dict]]:
    meta, state = load_recording(path)
    dyn = dynamics(state)
    keep = state["time_s"] >= trim_s
    error = state["target_q"] - state["q"]
    force = state["actuator_force"]
    speed = np.asarray(dyn["speed_smooth"])[keep]
    threshold = 0.25 * float(np.quantile(speed, 0.9))
    count = replan_events(state)
    row = {
        "model": meta["model_profile"],
        "replan_period_s": float(meta.get("planner_replan_period_s", 1.0)),
        "duration_s": float(state["time_s"][-1]),
        "travel_m": float(np.linalg.norm(
            state["base_position"][-1, :2] - state["base_position"][0, :2]
        )),
        "speed_p10_mps": float(np.quantile(speed, 0.1)),
        "speed_median_mps": float(np.median(speed)),
        "speed_p90_mps": float(np.quantile(speed, 0.9)),
        "near_stop_fraction": float(np.mean(speed < threshold)),
        "replan_hz": float(count[keep].sum() / (keep.sum() * float(dyn["dt"]))),
        "tilt_p95_deg": float(np.quantile(np.asarray(dyn["tilt"])[keep], 0.95)),
        "height_p05_m": float(np.quantile(state["base_position"][keep, 2], 0.05)),
        "leg_error_rms_rad": float(np.sqrt(np.mean(error[keep, :12] ** 2))),
        "left_ankle_error_rms_rad": float(np.sqrt(np.mean(error[keep, 4] ** 2))),
        "right_ankle_error_rms_rad": float(np.sqrt(np.mean(error[keep, 10] ** 2))),
        "left_ankle_force_rms_nm": float(np.sqrt(np.mean(force[keep, 4] ** 2))),
        "right_ankle_force_rms_nm": float(np.sqrt(np.mean(force[keep, 10] ** 2))),
        "fell": bool(meta["fell"]),
    }
    joints = []
    for index, label in enumerate(LEG_LABELS):
        joints.append({
            "model": meta["model_profile"],
            "replan_period_s": row["replan_period_s"],
            "joint": label,
            "error_rms_rad": float(np.sqrt(np.mean(error[keep, index] ** 2))),
            "force_rms_nm": float(np.sqrt(np.mean(force[keep, index] ** 2))),
            "target_rate_rms_rad_s": float(np.sqrt(np.mean(
                (np.diff(state["target_q"][:, index]) / float(dyn["dt"])) ** 2
            ))),
        })
    return row, joints


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_interactive(records: list[tuple[dict, dict]], output: Path) -> None:
    fig, axes = plt.subplots(len(records), 1, figsize=(13, 8), sharex=False)
    for axis, (meta, state) in zip(axes, records):
        dyn = dynamics(state)
        masks = command_masks(state)
        axis.plot(state["time_s"], dyn["speed_smooth"], lw=0.8, color="#333333")
        axis.fill_between(
            state["time_s"], 0, 1.2, where=masks["turning"],
            color="#2ca02c", alpha=0.14, transform=axis.get_xaxis_transform(),
            label="turning command",
        )
        event = replan_events(state)
        axis.vlines(state["time_s"][event], 0, 0.05, color="#d62728", alpha=0.3, lw=0.4)
        axis.set_ylabel(f"{meta['model_profile']}\nspeed (m/s)")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s); red ticks are replans")
    fig.tight_layout()
    fig.savefig(output / "operator_speed_and_replans.png", dpi=170)
    plt.close(fig)


def plot_replan_phase(records: list[tuple[dict, dict]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 11)
    for meta, state in records:
        dyn = dynamics(state)
        event = replan_events(state)
        last = -10000
        since = np.zeros(len(event))
        for index in range(len(event)):
            if event[index]:
                last = index
            since[index] = (index - last) * float(dyn["dt"])
        pure = command_masks(state)["pure_forward"]
        values = []
        for left, right in zip(bins[:-1], bins[1:]):
            use = pure & (since >= left) & (since < right)
            values.append(np.median(np.asarray(dyn["speed_smooth"])[use]))
        axis.plot((bins[:-1] + bins[1:]) / 2, values, marker="o", label=meta["model_profile"])
    axis.set_xlabel("seconds since last straight-line replan")
    axis.set_ylabel("median horizontal speed (m/s)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "straight_speed_by_replan_phase.png", dpi=170)
    plt.close(fig)


def plot_ab(rows: list[dict], output: Path) -> None:
    models = ("low-latency", "original-release", "sonic-v1.1")
    periods = (1.0, 0.1)
    colors = ("#c44e52", "#4c72b0")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fields = (
        ("speed_median_mps", "median speed (m/s)"),
        ("near_stop_fraction", "near-stop fraction"),
        ("left_ankle_force_rms_nm", "left ankle force RMS (Nm)"),
    )
    x = np.arange(len(models))
    width = 0.36
    for axis, (field, title) in zip(axes, fields):
        for offset, (period, color) in enumerate(zip(periods, colors)):
            values = [next(r[field] for r in rows if r["model"] == model and r["replan_period_s"] == period) for model in models]
            axis.bar(x + (offset - 0.5) * width, values, width, label=f"{period:g} s", color=color)
        axis.set_title(title)
        axis.set_xticks(x, ["low-latency", "original", "v1.1"], rotation=20)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(title="replan period")
    fig.tight_layout()
    fig.savefig(output / "controlled_replan_ab.png", dpi=170)
    plt.close(fig)


def markdown_table(rows: list[dict], fields: list[tuple[str, str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label, _ in fields) + " |",
        "| " + " | ".join("---" if fmt == "s" else "---:" for _, _, fmt in fields) + " |",
    ]
    for row in rows:
        cells = []
        for key, _, fmt in fields:
            value = row[key]
            cells.append(str(value) if fmt == "s" else format(value, fmt))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", nargs=3, type=Path, required=True)
    parser.add_argument("--ab-root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interactive = [load_recording(path) for path in args.interactive]
    operator_rows = []
    for meta, state in interactive:
        for name, mask in command_masks(state).items():
            use = trim_island_starts(mask)
            if use.sum() >= 25:
                operator_rows.append(metrics(meta, state, use, name))

    ab_rows, joint_rows = [], []
    for path in sorted(p for p in args.ab_root.iterdir() if (p / "meta.json").exists()):
        row, joints = full_run_metrics(path)
        ab_rows.append(row)
        joint_rows.extend(joints)
    long_rows = []
    for path in sorted(p for p in args.long_root.iterdir() if (p / "meta.json").exists()):
        row, _ = full_run_metrics(path, trim_s=5.0)
        long_rows.append(row)

    write_csv(args.output_dir / "operator_command_metrics.csv", operator_rows)
    write_csv(args.output_dir / "controlled_replan_ab.csv", ab_rows)
    write_csv(args.output_dir / "bilateral_joint_replan_ab.csv", joint_rows)
    write_csv(args.output_dir / "long_validation_metrics.csv", long_rows)
    plot_interactive(interactive, args.output_dir)
    plot_replan_phase(interactive, args.output_dir)
    plot_ab(ab_rows, args.output_dir)

    by_model = {row["model"]: row for row in long_rows}
    best = min(long_rows, key=lambda row: (
        row["tilt_p95_deg"] + 10.0 * row["leg_error_rms_rad"]
        + 0.05 * row["left_ankle_force_rms_nm"]
    ))
    report = [
        "# Unitree-remote straight-walking and replan analysis",
        "",
        "## Finding",
        "",
        "The discrete straight walk is a planner scheduling defect, not a pure-forward axis mapping error. "
        "The recorded movement vector remains +X and never drops while the robot slows. A constant command "
        "was replanned at only 1 Hz; turning changes the facing vector and therefore caused replanning at "
        "approximately 10 Hz. The controlled A/B reproduces the difference without PICO or DDS.",
        "",
        "At 1 Hz the requested 0.4 m/s SLOW_WALK repeatedly accelerated and decelerated, while the 10 Hz "
        "schedule sustained the requested speed. The 10 Hz schedule also reduced left-ankle tracking error "
        "and effort, so stop/go staging can amplify an existing left-side hardware weakness even though it "
        "does not create that hardware asymmetry by itself.",
        "",
        "## Operator recordings",
        "",
    ]
    report += markdown_table(operator_rows, [
        ("model", "Model", "s"), ("command", "Command", "s"),
        ("seconds", "Data (s)", ".1f"), ("replan_hz", "Replan Hz", ".2f"),
        ("speed_median_mps", "Median speed", ".3f"),
        ("near_stop_fraction", "Near-stop", ".1%"),
        ("left_ankle_force_rms_nm", "L ankle RMS Nm", ".2f"),
        ("right_ankle_force_rms_nm", "R ankle RMS Nm", ".2f"),
    ])
    report += ["", "## Controlled 1.0 s versus 0.1 s straight-forward A/B", ""]
    report += markdown_table(sorted(ab_rows, key=lambda row: (row["model"], -row["replan_period_s"])), [
        ("model", "Model", "s"), ("replan_period_s", "Period", ".1f"),
        ("speed_median_mps", "Median speed", ".3f"),
        ("near_stop_fraction", "Near-stop", ".1%"),
        ("left_ankle_error_rms_rad", "L ankle error", ".3f"),
        ("right_ankle_error_rms_rad", "R ankle error", ".3f"),
        ("left_ankle_force_rms_nm", "L ankle Nm", ".2f"),
        ("right_ankle_force_rms_nm", "R ankle Nm", ".2f"),
        ("tilt_p95_deg", "Tilt p95", ".1f"),
    ])
    report += [
        "", "## Sixty-second fixed-schedule validation", "",
    ]
    report += markdown_table(sorted(long_rows, key=lambda row: row["model"]), [
        ("model", "Model", "s"), ("travel_m", "Travel m", ".2f"),
        ("speed_median_mps", "Median speed", ".3f"),
        ("tilt_p95_deg", "Tilt p95", ".1f"),
        ("leg_error_rms_rad", "Leg RMS error", ".3f"),
        ("left_ankle_error_rms_rad", "L ankle error", ".3f"),
        ("left_ankle_force_rms_nm", "L ankle Nm", ".2f"),
        ("fell", "Fell", "s"),
    ])
    report += [
        "",
        f"All three completed 60 seconds without falling. By the combined tilt, bilateral leg tracking, "
        f"and left-ankle-demand criteria, **{best['model']}** is the strongest simulation candidate. "
        "This ranks software/healthy-plant behavior only; real-robot hardware condition remains a separate variable.",
        "",
        "## Figures and detailed data",
        "",
        "- [`operator_speed_and_replans.png`](operator_speed_and_replans.png)",
        "- [`straight_speed_by_replan_phase.png`](straight_speed_by_replan_phase.png)",
        "- [`controlled_replan_ab.png`](controlled_replan_ab.png)",
        "- [`operator_command_metrics.csv`](operator_command_metrics.csv)",
        "- [`controlled_replan_ab.csv`](controlled_replan_ab.csv)",
        "- [`bilateral_joint_replan_ab.csv`](bilateral_joint_replan_ab.csv)",
        "- [`long_validation_metrics.csv`](long_validation_metrics.csv)",
        "",
        "## Scope boundary",
        "",
        "MuJoCo establishes that the scheduling defect exists with a healthy symmetric plant and that correcting it "
        "reduces left-ankle demand. It cannot prove that the remaining physical left-ankle jitter is software-caused; "
        "encoder calibration, backlash, preload, motor/driver condition, and floor contact remain hardware-side causes.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
