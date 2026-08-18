#!/usr/bin/env python3
"""Analyze deterministic SONIC 23-DoF slow-walk adaptation recordings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Metrics remain available in lean deployment envs.
    plt = None


DEFAULT_INPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/"
    "20260813_training_free_23dof_slow_walk"
)
DEFAULT_OUTPUT = Path(
    "/home/clhsieh/e_skin/docs/experiments/mujoco/"
    "2026-08-13_training_free_23dof_slow_walk"
)
MISSING_IL = np.array([5, 8, 25, 26, 27, 28], np.int64)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tilt_deg(projected_gravity) -> np.ndarray:
    gravity = np.asarray(projected_gravity, np.float64)
    return np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))


def _largest_segment(mask: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("recording has no SLOW_WALK samples")
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, breaks)
    selected = max(groups, key=len)
    return int(selected[0]), int(selected[-1]) + 1


def _landing_events(state, selected: np.ndarray, dt: float) -> list[dict]:
    base = np.asarray(state["base_position"], np.float64)
    movement = np.asarray(state["nav_movement"], np.float64)
    rows = []
    for side in ("left", "right"):
        contact = np.asarray(state[f"{side}_foot_contact"], bool)
        foot = np.asarray(state[f"{side}_foot_position"], np.float64)
        corners = np.asarray(state[f"{side}_foot_corner_z"], np.float64)
        swing_count = 0
        for index in range(1, len(contact)):
            if not selected[index]:
                swing_count = 0
                continue
            if not contact[index - 1]:
                swing_count += 1
            if contact[index] and not contact[index - 1] and swing_count >= 5:
                end = index
                while end + 1 < len(contact) and contact[end + 1] and end < index + 15:
                    end += 1
                direction = movement[index, :2]
                direction /= max(np.linalg.norm(direction), 1e-12)
                relative_xy = foot[index:end + 1, :2] - base[index:end + 1, :2]
                relative_progress = (relative_xy - relative_xy[0]) @ direction
                world_xy = foot[index:end + 1, :2]
                rows.append({
                    "side": side,
                    "index": index,
                    "post_contact_forward_progress_m": float(
                        max(0.0, np.max(relative_progress))
                    ),
                    "post_contact_world_slip_m": float(
                        np.max(np.linalg.norm(world_xy - world_xy[0], axis=1))
                    ),
                    "touchdown_vertical_speed_mps": float(
                        (foot[index, 2] - foot[index - 1, 2]) / dt
                    ),
                    "touchdown_corner_range_m": float(np.ptp(corners[index])),
                    "touchdown_min_corner_z_m": float(np.min(corners[index])),
                })
                swing_count = 0
            elif contact[index]:
                swing_count = 0
    return rows


def _stats(values) -> tuple[float, float, float]:
    values = np.asarray(values, np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 90)), float(np.max(values))


def _meta_vector(meta: dict, name: str, size: int, neutral: float) -> np.ndarray:
    value = np.asarray(meta.get(name, [neutral] * size), np.float64)
    if value.shape != (size,):
        return np.full(size, neutral, np.float64)
    return value


def analyze(recording: Path) -> tuple[dict, list[dict], list[dict]]:
    meta = json.loads((recording / "meta.json").read_text())
    state = np.load(recording / "state.npz", allow_pickle=False)
    dt = float(meta["control_dt"])
    mode = np.asarray(state["mode"], np.int64)
    slow_mask = mode == 1
    analysis_segment = "slow-walk-mode"
    if not np.any(slow_mask):
        # The planner intentionally keeps a pure facing change in semantic
        # IDLE while still installing turn-in-place motion.  For those trials,
        # the scheduled non-neutral stick interval is the active segment.
        axes = np.asarray(state["axes"], np.float64)
        slow_mask = np.linalg.norm(axes, axis=1) > 1e-6
        analysis_segment = "non-neutral-command"
    begin, end = _largest_segment(slow_mask)
    trim = int(round(2.0 / dt))
    begin_steady, end_steady = begin + trim, end - trim
    if end_steady <= begin_steady:
        raise ValueError(f"slow-walk segment too short in {recording}")
    selected = np.zeros(len(mode), bool)
    selected[begin_steady:end_steady] = True

    base = np.asarray(state["base_position"], np.float64)
    velocity_xy = np.diff(base[:, :2], axis=0, prepend=base[:1, :2]) / dt
    movement = np.asarray(state["nav_movement"], np.float64)[:, :2]
    direction_norm = np.linalg.norm(movement, axis=1, keepdims=True)
    movement_unit = movement / np.maximum(direction_norm, 1e-12)
    forward = np.sum(velocity_xy * movement_unit, axis=1)
    planar = np.linalg.norm(velocity_xy, axis=1)
    # Half-second moving mean distinguishes step-cycle stop/go from solver noise.
    window = max(1, int(round(0.5 / dt)))
    smooth_forward = np.convolve(forward, np.ones(window) / window, mode="same")
    q = np.asarray(state["q"], np.float64)
    qd = np.asarray(state["qd"], np.float64)
    target = np.asarray(state["target_q"], np.float64)
    force = np.asarray(state["actuator_force"], np.float64)
    error = target - q
    tilt = _tilt_deg(state["projected_gravity"])
    left_contact = np.asarray(state["left_foot_contact"], bool)
    right_contact = np.asarray(state["right_foot_contact"], bool)
    landings = _landing_events(state, selected, dt)
    progress = [event["post_contact_forward_progress_m"] for event in landings]
    slip = [event["post_contact_world_slip_m"] for event in landings]
    impact = [abs(event["touchdown_vertical_speed_mps"]) for event in landings]
    progress_med, progress_p90, progress_max = _stats(progress)
    slip_med, slip_p90, slip_max = _stats(slip)
    impact_med, impact_p90, impact_max = _stats(impact)

    reference = np.asarray(
        state[
            "sonic_reference_q_il_adapted"
            if "sonic_reference_q_il_adapted" in state.files
            else "reference_q_il"
        ],
        np.float64,
    )
    raw_action = np.asarray(state["sonic_action_il_raw"], np.float64)
    target_rate = np.diff(target, axis=0, prepend=target[:1]) / dt
    reference_rate = np.diff(reference, axis=0, prepend=reference[:1]) / dt
    command_speed = float(meta.get("planner_speed_override") or np.nan)
    physical_dof = len(meta["joint_names"])
    motor_scale = _meta_vector(meta, "motor_action_scale", physical_dof, 1.0)
    motor_clip = _meta_vector(meta, "motor_action_abs_limit", physical_dof, np.inf)
    motor_ema = _meta_vector(meta, "motor_action_ema", physical_dof, 1.0)
    motor_rate = _meta_vector(meta, "motor_target_rate_limit", physical_dof, np.inf)
    plant_gain = _meta_vector(meta, "plant_gain_scale", physical_dof, 1.0)
    plant_delay = _meta_vector(meta, "plant_target_delay_ms", physical_dof, 0.0)
    encoder_offset = _meta_vector(
        meta, "joint_encoder_offset_deg", physical_dof, 0.0
    )
    reference_joint_scale = _meta_vector(
        meta, "reference_joint_scale", physical_dof, 1.0
    )
    ankle_pitch_idx = np.array([4, 10], np.int64)
    steady_duration = float(selected.sum() * dt)
    mean_forward = float(np.mean(forward[selected]))
    inferred_adaptation = (
        "diagnostic-ablation"
        if (
            bool(meta.get("mask_missing_reference", False))
            or bool(meta.get("mask_missing_action_history", False))
            or str(meta.get("reduced_proprioception", "zero")) != "zero"
            or float(meta.get("action_ema", 1.0)) != 1.0
            or np.any(motor_scale != 1.0)
            or np.any(np.isfinite(motor_clip))
            or np.any(motor_ema != 1.0)
            or np.any(np.isfinite(motor_rate))
            or str(meta.get("action_history_source", "raw")) != "raw"
        )
        else "official-parity"
    )
    common = {
        "recording": str(recording),
        "trial": str(meta["trial_label"]),
        "stage": recording.parent.name,
        "model": str(meta["model_profile"]),
        "embodiment": str(meta["embodiment"]),
        "ground_friction": float(meta.get("ground_friction", np.nan)),
        "adaptation_profile": str(
            meta.get("adaptation_profile", inferred_adaptation)
        ),
        "command_speed_mps": command_speed,
        "reduced_proprioception": str(meta.get("reduced_proprioception", "zero")),
        "shadow_settling_time_s": float(meta.get("shadow_settling_time_s", 0.12)),
        "action_ema": float(meta.get("action_ema", 1.0)),
        "mask_missing_reference": bool(meta.get("mask_missing_reference", False)),
        "mask_missing_action_history": bool(meta.get("mask_missing_action_history", False)),
        "missing_adaptation_scope": str(meta.get("missing_adaptation_scope", "all")),
        "leg_action_scale_mean": float(np.mean(motor_scale[:12])),
        "ankle_pitch_action_scale": float(np.mean(motor_scale[ankle_pitch_idx])),
        "ankle_pitch_action_clip": float(np.mean(motor_clip[ankle_pitch_idx])),
        "ankle_pitch_action_ema": float(np.mean(motor_ema[ankle_pitch_idx])),
        "ankle_pitch_target_rate_limit": float(np.mean(motor_rate[ankle_pitch_idx])),
        "action_history_source": str(meta.get("action_history_source", "raw")),
        "reference_leg_scale": float(meta.get("reference_leg_scale", 1.0)),
        "reference_root_height_scale": float(
            meta.get("reference_root_height_scale", 1.0)
        ),
        "reference_sagittal_scale_mean": float(np.mean(
            reference_joint_scale[[0, 3, 4, 6, 9, 10]]
            if physical_dof >= 12 else reference_joint_scale
        )),
        "left_knee_encoder_offset_deg": float(
            encoder_offset[3] if physical_dof > 3 else np.nan
        ),
        "left_ankle_pitch_plant_gain": float(
            plant_gain[4] if physical_dof > 4 else np.nan
        ),
        "left_ankle_pitch_plant_delay_ms": float(
            plant_delay[4] if physical_dof > 4 else np.nan
        ),
        "fell": bool(meta.get("fell", False)),
        "analysis_segment": analysis_segment,
    }
    summary = {
        **common,
        "steady_duration_s": steady_duration,
        "actual_forward_mean_mps": mean_forward,
        "actual_planar_mean_mps": float(np.mean(planar[selected])),
        "actual_forward_p10_mps": float(np.percentile(forward[selected], 10)),
        "actual_forward_p90_mps": float(np.percentile(forward[selected], 90)),
        "half_second_speed_std_mps": float(np.std(smooth_forward[selected])),
        "half_second_speed_cv": float(np.std(smooth_forward[selected]) / max(abs(mean_forward), 1e-6)),
        "backward_motion_pct": float(100 * np.mean(forward[selected] < -0.02)),
        "near_stop_pct": float(100 * np.mean(np.abs(forward[selected]) < 0.05)),
        "height_p05_m": float(np.percentile(base[selected, 2], 5)),
        "height_median_m": float(np.median(base[selected, 2])),
        "tilt_p95_deg": float(np.percentile(tilt[selected], 95)),
        "tilt_max_deg": float(np.max(tilt[selected])),
        "leg_error_rms_rad": float(np.sqrt(np.mean(error[selected, :12] ** 2))),
        "leg_speed_rms_rad_s": float(np.sqrt(np.mean(qd[selected, :12] ** 2))),
        "leg_target_rate_rms_rad_s": float(np.sqrt(np.mean(target_rate[selected, :12] ** 2))),
        "leg_reference_rate_rms_rad_s": float(np.sqrt(np.mean(reference_rate[selected][:, [0,1,3,4,6,7,9,10,13,14,17,18]] ** 2))),
        "double_support_pct": float(100 * np.mean(left_contact[selected] & right_contact[selected])),
        "single_support_pct": float(100 * np.mean(left_contact[selected] ^ right_contact[selected])),
        "flight_pct": float(100 * np.mean(~left_contact[selected] & ~right_contact[selected])),
        "landing_events": len(landings),
        "cadence_landings_per_s": float(len(landings) / steady_duration),
        "post_contact_progress_median_m": progress_med,
        "post_contact_progress_p90_m": progress_p90,
        "post_contact_progress_max_m": progress_max,
        "post_contact_progress_gt_2cm_pct": float(100 * np.mean(np.asarray(progress) > 0.02)) if progress else float("nan"),
        "post_contact_slip_median_m": slip_med,
        "post_contact_slip_p90_m": slip_p90,
        "post_contact_slip_max_m": slip_max,
        "touchdown_vertical_speed_median_mps": impact_med,
        "touchdown_vertical_speed_p90_mps": impact_p90,
        "touchdown_vertical_speed_max_mps": impact_max,
        "missing_waist_action_rms": float(np.sqrt(np.mean(raw_action[selected][:, [5, 8]] ** 2))),
        "missing_waist_reference_rms_rad": float(np.sqrt(np.mean(reference[selected][:, [5, 8]] ** 2))),
        "missing_wrist_action_rms": float(np.sqrt(np.mean(raw_action[selected][:, [25, 26, 27, 28]] ** 2))),
        "planner_replans_during_steady": int(
            np.asarray(state["planner_replan_count"])[end_steady - 1]
            - np.asarray(state["planner_replan_count"])[begin_steady]
        ),
    }
    capture_count = np.asarray(state["stand_capture_count"], np.int64)
    initial_idle = (
        (mode == 0)
        & (capture_count > 0)
        & (np.arange(len(mode)) < max(begin - int(round(0.5 / dt)), 0))
    )
    final_idle = (mode == 0) & (
        np.arange(len(mode)) >= max(end + int(round(2.0 / dt)), len(mode) - int(round(2.0 / dt)))
    )
    for label, idle_mask in (("initial_idle", initial_idle), ("final_idle", final_idle)):
        if np.any(idle_mask):
            idle_xy = base[idle_mask, :2]
            summary.update({
                f"{label}_duration_s": float(idle_mask.sum() * dt),
                f"{label}_tilt_p95_deg": float(np.percentile(tilt[idle_mask], 95)),
                f"{label}_leg_error_rms_rad": float(np.sqrt(np.mean(error[idle_mask, :12] ** 2))),
                f"{label}_leg_speed_rms_rad_s": float(np.sqrt(np.mean(qd[idle_mask, :12] ** 2))),
                f"{label}_double_support_pct": float(100 * np.mean(left_contact[idle_mask] & right_contact[idle_mask])),
                f"{label}_base_drift_m": float(np.linalg.norm(idle_xy[-1] - idle_xy[0])),
            })
        else:
            summary.update({
                f"{label}_duration_s": 0.0,
                f"{label}_tilt_p95_deg": float("nan"),
                f"{label}_leg_error_rms_rad": float("nan"),
                f"{label}_leg_speed_rms_rad_s": float("nan"),
                f"{label}_double_support_pct": float("nan"),
                f"{label}_base_drift_m": float("nan"),
            })

    joints = []
    names = list(meta["joint_names"])
    for index, name in enumerate(names):
        joints.append({
            **common,
            "joint_index": index,
            "joint": name.removesuffix("_joint"),
            "error_rms_rad": float(np.sqrt(np.mean(error[selected, index] ** 2))),
            "error_p95_rad": float(np.percentile(np.abs(error[selected, index]), 95)),
            "speed_rms_rad_s": float(np.sqrt(np.mean(qd[selected, index] ** 2))),
            "target_rate_rms_rad_s": float(np.sqrt(np.mean(target_rate[selected, index] ** 2))),
            "abs_force_p95_nm": float(np.percentile(np.abs(force[selected, index]), 95)),
        })
    landing_rows = [{**common, **event} for event in landings]
    return summary, joints, landing_rows


def _add_native29_scores(summaries: list[dict]) -> None:
    """Score 0.4-m/s candidates against native 29-D stability and gait timing.

    Lower-than-native instability is never penalized.  Gait timing is matched
    symmetrically so a frozen or heavily damped robot cannot win the score.
    The floors keep tiny IDLE numerical noise from dominating the comparison.
    """
    references = [
        row for row in summaries
        if row["trial"] == "native29_mu1p00_s0p40"
    ] or [
        row for row in summaries if row["trial"] == "native29_s0p40"
    ]
    if not references:
        for row in summaries:
            row["native29_stability_excess_score"] = float("nan")
            row["native29_gait_mismatch_score"] = float("nan")
            row["native29_total_score"] = float("nan")
        return
    reference = references[0]
    lower_better = (
        ("half_second_speed_cv", 0.10),
        ("tilt_p95_deg", 2.0),
        ("leg_error_rms_rad", 0.10),
        ("leg_speed_rms_rad_s", 0.20),
        ("leg_target_rate_rms_rad_s", 0.50),
        ("post_contact_progress_p90_m", 0.01),
        ("post_contact_slip_p90_m", 0.02),
        ("touchdown_vertical_speed_p90_mps", 0.20),
        ("final_idle_tilt_p95_deg", 1.0),
        ("final_idle_leg_speed_rms_rad_s", 0.02),
        ("final_idle_base_drift_m", 0.005),
    )
    for row in summaries:
        excess = []
        for name, floor in lower_better:
            value, target = float(row[name]), float(reference[name])
            denominator = max(abs(target), floor)
            excess.append(max(0.0, value - target) / denominator)
        stability = float(np.mean(excess))
        speed_target = max(abs(float(reference["actual_forward_mean_mps"])), 0.05)
        cadence_target = max(float(reference["cadence_landings_per_s"]), 0.25)
        gait_terms = (
            abs(float(row["actual_forward_mean_mps"])
                - float(reference["actual_forward_mean_mps"])) / speed_target,
            abs(float(row["double_support_pct"])
                - float(reference["double_support_pct"])) / 25.0,
            abs(float(row["cadence_landings_per_s"])
                - float(reference["cadence_landings_per_s"])) / cadence_target,
            max(0.0, float(reference["height_p05_m"])
                - float(row["height_p05_m"])) / 0.05,
        )
        gait = float(np.mean(gait_terms))
        if bool(row["fell"]):
            stability += 10.0
        row["native29_stability_excess_score"] = stability
        row["native29_gait_mismatch_score"] = gait
        row["native29_total_score"] = stability + gait


def _plots(summaries: list[dict], output: Path) -> None:
    if plt is None:
        return
    speed_rows = sorted(
        [row for row in summaries if row["stage"] == "speed"],
        key=lambda row: row["command_speed_mps"],
    )
    if speed_rows:
        x = np.array([row["command_speed_mps"] for row in speed_rows])
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        panels = (
            ("actual_forward_mean_mps", "Actual forward speed", "m/s"),
            ("half_second_speed_cv", "Stop/go speed variation", "CV"),
            ("tilt_p95_deg", "Pelvis tilt p95", "degrees"),
            ("post_contact_progress_p90_m", "Post-contact progression p90", "m"),
        )
        for axis, (column, title, ylabel) in zip(axes.flat, panels):
            axis.plot(x, [row[column] for row in speed_rows], marker="o")
            axis.set_xlabel("planner target_vel (m/s)")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(alpha=0.3)
        fig.suptitle("Original Release / 23-DoF / native zero-fill speed sweep")
        fig.savefig(output / "speed_sweep.png", dpi=180)
        plt.close(fig)

    by_trial = {row["trial"]: row for row in summaries}
    comparison_names = (
        "native_scope_s0p40",
        "simple_mu1p00_s0p40",
        "native29_mu1p00_s0p40",
    )
    if all(name in by_trial for name in comparison_names):
        rows = [by_trial[name] for name in comparison_names]
        labels = ("23D parity", "23D selected", "29D native")
        panels = (
            ("actual_forward_mean_mps", "Actual forward speed", "m/s"),
            ("tilt_p95_deg", "Pelvis tilt p95", "degrees"),
            ("leg_target_rate_rms_rad_s", "Leg target rate RMS", "rad/s"),
            ("double_support_pct", "Double support", "% time"),
            ("cadence_landings_per_s", "Landing cadence", "landings/s"),
            ("missing_waist_action_rms", "Raw missing-waist demand", "normalized RMS"),
        )
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for axis, (column, title, ylabel) in zip(axes.flat, panels):
            axis.bar(labels, [row[column] for row in rows])
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", rotation=20)
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle("Original Release at target_vel=0.4; matched floor friction=1.0")
        fig.savefig(output / "selected_adaptation_comparison.png", dpi=180)
        plt.close(fig)

    # Final validation uses an explicitly matched floor friction for both
    # embodiments.  (The stock 29-D XML otherwise defaults to 0.5 while the
    # 23-D XML defaults to 1.0.)
    robustness = [
        row for row in summaries
        if row["stage"] in ("finalist_friction", "native29_robustness")
    ]
    if robustness:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        panels = (
            ("actual_forward_mean_mps", "Actual forward speed", "m/s"),
            ("tilt_p95_deg", "Pelvis tilt p95", "degrees"),
            ("leg_target_rate_rms_rad_s", "Leg target rate RMS", "rad/s"),
            ("touchdown_vertical_speed_p90_mps", "Touchdown speed p90", "m/s"),
        )
        for prefix, label in (("simple_mu", "23D selected"),
                              ("native29_mu", "29D native")):
            selected = sorted(
                [row for row in robustness
                 if str(row["trial"]).startswith(prefix)],
                key=lambda row: float(row["ground_friction"]),
            )
            if not selected:
                continue
            x = [float(row["ground_friction"]) for row in selected]
            for axis, (column, title, ylabel) in zip(axes.flat, panels):
                axis.plot(x, [row[column] for row in selected], marker="o", label=label)
                axis.set_xlabel("floor sliding friction")
                axis.set_ylabel(ylabel)
                axis.set_title(title)
                axis.grid(alpha=0.3)
        axes.flat[0].legend()
        fig.suptitle("Selected 23-DoF versus native 29-DoF contact robustness")
        fig.savefig(output / "friction_robustness.png", dpi=180)
        plt.close(fig)

    finalist_speed = [
        row for row in summaries if row["stage"] == "finalist_speed"
    ]
    if finalist_speed:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        panels = (
            ("actual_forward_mean_mps", "Actual forward speed", "m/s"),
            ("half_second_speed_cv", "Stop/go speed variation", "CV"),
            ("tilt_p95_deg", "Pelvis tilt p95", "degrees"),
            ("double_support_pct", "Double support", "% time"),
        )
        datasets = (
            ([row for row in finalist_speed
              if str(row["trial"]).startswith("simple_s")], "23D selected"),
            ([row for row in summaries
              if (str(row["trial"]).startswith("native29_mu1_s0p")
                  or str(row["trial"]) in
                  ("native29_s0p30", "native29_mu1p00_s0p40"))],
             "29D native"),
        )
        for candidates, label in datasets:
            selected = sorted(
                candidates,
                key=lambda row: float(row["command_speed_mps"]),
            )
            if not selected:
                continue
            x = [float(row["command_speed_mps"]) for row in selected]
            for axis, (column, title, ylabel) in zip(axes.flat, panels):
                axis.plot(x, [row[column] for row in selected], marker="o", label=label)
                axis.set_xlabel("planner target_vel (m/s)")
                axis.set_ylabel(ylabel)
                axis.set_title(title)
                axis.grid(alpha=0.3)
        axes.flat[0].legend()
        fig.suptitle("Selected 23-DoF adaptation versus matched native 29-DoF")
        fig.savefig(output / "finalist_speed.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    recordings = sorted(
        path.parent for path in args.input.rglob("state.npz")
        if (path.parent / "meta.json").is_file()
    )
    if not recordings:
        raise SystemExit(f"no recordings found under {args.input}")
    summaries, joints, landings, failures = [], [], [], []
    for recording in recordings:
        try:
            summary, joint_rows, landing_rows = analyze(recording)
        except ValueError as exc:
            meta = json.loads((recording / "meta.json").read_text())
            failures.append({
                "recording": str(recording),
                "trial": str(meta.get("trial_label", recording.name)),
                "stage": recording.parent.name,
                "fell": bool(meta.get("fell", False)),
                "rows": int(meta.get("rows", 0)),
                "reason": str(exc),
            })
            continue
        summaries.append(summary)
        joints.extend(joint_rows)
        landings.extend(landing_rows)
    _add_native29_scores(summaries)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "run_metrics.csv", summaries)
    _write_csv(args.output / "joint_metrics.csv", joints)
    _write_csv(args.output / "landing_events.csv", landings)
    _write_csv(args.output / "failed_runs.csv", failures)
    (args.output / "recordings.json").write_text(
        json.dumps([str(path) for path in recordings], indent=2) + "\n"
    )
    _plots(summaries, args.output)
    columns = (
        "trial", "command_speed_mps", "actual_forward_mean_mps",
        "half_second_speed_cv", "tilt_p95_deg", "height_p05_m",
        "leg_error_rms_rad", "post_contact_progress_p90_m",
        "post_contact_progress_gt_2cm_pct", "double_support_pct",
    )
    for row in sorted(summaries, key=lambda value: (value["stage"], value["trial"])):
        print(" ".join(f"{column}={row[column]}" for column in columns))


if __name__ == "__main__":
    main()
