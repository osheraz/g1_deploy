#!/usr/bin/env python3
"""Analyze physical post-calibration trials and a MuJoCo knee-offset sweep."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_ROOT = Path("/data/clhsieh/logs/e_skin/teleop")
SIM_ROOT = Path(
    "/data/clhsieh/logs/e_skin/experiments/20260815_knee_encoder_offset_sweep"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/experiments/physical/"
    "2026-08-15_post_left_knee_offset_diagnosis"
)
PRE_SESSIONS = ("20260814T155904Z-bluefin-1775909",)
POST_SESSIONS = (
    "20260814T214954Z-bluefin-1816953",
    "20260814T215804Z-bluefin-1817621",
    "20260814T222710Z-bluefin-1819712",
)
LEG_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
    "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rolling_rms(values: np.ndarray, width: int) -> np.ndarray:
    if values.ndim == 2:
        values = np.linalg.norm(values, axis=1)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.sqrt(np.convolve(values ** 2, kernel, mode="same"))


def _gravity_from_wxyz(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat.T
    return np.column_stack((
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ))


def _tilt_deg(gravity: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))


def _physical_events(log_path: Path) -> list[dict]:
    """Approximate planner event time from its once-per-second timing marker.

    The legacy compact C++ CSV did not store planner mode.  Its text log emits
    one ``Loop timing`` line per control second, so mode boundaries are
    recoverable to about one second.  Transition guards below are wider than
    that uncertainty.
    """
    elapsed_s = 0
    events = []
    pattern = re.compile(
        r"Replanning with mode: (\w+).*movement: \[([^]]+)\].*facing: \[([^]]+)\]"
    )
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("Loop timing"):
            elapsed_s += 1
        match = pattern.search(line)
        if not match:
            continue
        try:
            movement = np.asarray(
                [float(value) for value in match.group(2).split(",")], np.float64
            )
            facing = np.asarray(
                [float(value) for value in match.group(3).split(",")], np.float64
            )
        except ValueError:
            # A few stdout lines were interleaved at byte level.  The next
            # 10-Hz replan contains the same command and is clean.
            continue
        events.append({
            "elapsed_s": float(elapsed_s),
            "mode": match.group(1),
            "movement": movement,
            "facing": facing,
        })
    return events


def _load_physical(session: str) -> dict:
    directory = PHYSICAL_ROOT / session
    csv_path = directory / "terminal1_robot_diagnostics/robot_diagnostics.csv"
    log_path = next(directory.glob("terminal1_robot_controller_*.log"))
    scalar_names = (
        "time_ms", "playing", "imu_qw", "imu_qx", "imu_qy", "imu_qz",
        "imu_wx", "imu_wy", "imu_wz",
    )
    vector_groups = {
        "q": [f"q_{index}" for index in range(12)],
        "qd": [f"dq_{index}" for index in range(12)],
        "target": [f"q_cmd_{index}" for index in range(12)],
        "error": [f"q_error_{index}" for index in range(12)],
        "force": [f"tau_est_{index}" for index in range(12)],
    }
    columns = list(scalar_names)
    for names in vector_groups.values():
        columns.extend(names)
    values = {name: [] for name in columns}
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            for name in columns:
                values[name].append(float(row[name]))
    data = {name: np.asarray(values[name], np.float64) for name in scalar_names}
    for group, names in vector_groups.items():
        data[group] = np.column_stack([values[name] for name in names])
    data["time_s"] = data.pop("time_ms") / 1000.0
    data["events"] = _physical_events(log_path)
    return data


def _assign_physical_phases(data: dict) -> dict[str, np.ndarray]:
    time_s = data["time_s"]
    playing = data["playing"] > 0.5
    active_indices = np.flatnonzero(playing)
    if not len(active_indices):
        return {}
    control_start = time_s[active_indices[0]]
    relative = time_s - control_start
    mode = np.full(len(time_s), "OFF", dtype="U16")
    movement = np.zeros((len(time_s), 3), dtype=np.float64)
    facing = np.zeros((len(time_s), 3), dtype=np.float64)
    event_times = []
    previous_semantic = None
    events = sorted(data["events"], key=lambda item: item["elapsed_s"])
    for index, event in enumerate(events):
        start = event["elapsed_s"]
        stop = events[index + 1]["elapsed_s"] if index + 1 < len(events) else np.inf
        selected = playing & (relative >= start) & (relative < stop)
        mode[selected] = event["mode"]
        movement[selected] = event["movement"]
        facing[selected] = event["facing"]
        movement_norm = float(np.linalg.norm(event["movement"][:2]))
        facing_norm = float(np.linalg.norm(event["facing"][:2]))
        alignment = float(
            np.dot(event["movement"][:2], event["facing"][:2])
            / max(movement_norm * facing_norm, 1e-6)
        )
        semantic = (
            event["mode"],
            movement_norm > 0.9,
            alignment > 0.98,
        )
        if semantic != previous_semantic:
            event_times.append(start)
            previous_semantic = semantic

    # Remove at least one second around mode/command boundaries because text
    # logging provides only second-level alignment with the 50-Hz CSV.
    away_from_event = np.ones(len(time_s), dtype=bool)
    for event_time in event_times:
        away_from_event &= np.abs(relative - event_time) >= 1.25

    target_rate = np.vstack((
        np.zeros((1, 12)),
        np.diff(data["target"], axis=0) / np.maximum(
            np.diff(time_s, prepend=time_s[0] - 0.02)[:, None][1:], 1e-4
        ),
    ))
    command_activity = _rolling_rms(target_rate, 25)
    movement_norm = np.linalg.norm(movement[:, :2], axis=1)
    facing_norm = np.linalg.norm(facing[:, :2], axis=1)
    alignment = np.sum(movement[:, :2] * facing[:, :2], axis=1) / np.maximum(
        movement_norm * facing_norm, 1e-6
    )
    return {
        "IDLE": playing & (mode == "IDLE") & away_from_event & (command_activity < 1.0),
        "SLOW_WALK_FORWARD": (
            playing
            & (mode == "SLOW_WALK")
            & away_from_event
            & (movement_norm > 0.9)
            & (facing_norm > 0.9)
            & (alignment > 0.98)
        ),
    }


def _phase_metrics(
    *,
    source: str,
    session: str,
    phase: str,
    selected: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    target: np.ndarray,
    force: np.ndarray,
    gravity: np.ndarray,
    q_true: np.ndarray | None = None,
) -> dict | None:
    count = int(np.count_nonzero(selected))
    if count < 25:
        return None
    error = target - q
    tilt = _tilt_deg(gravity)
    row = {
        "source": source,
        "session": session,
        "phase": phase,
        "samples": count,
        "duration_s": count * 0.02,
        "left_knee_measured_median_deg": float(np.degrees(np.median(q[selected, 3]))),
        "right_knee_measured_median_deg": float(np.degrees(np.median(q[selected, 9]))),
        "left_knee_target_median_deg": float(np.degrees(np.median(target[selected, 3]))),
        "right_knee_target_median_deg": float(np.degrees(np.median(target[selected, 9]))),
        "knee_measured_delta_deg": float(np.degrees(np.median(q[selected, 3] - q[selected, 9]))),
        "knee_target_delta_deg": float(np.degrees(np.median(target[selected, 3] - target[selected, 9]))),
        "left_knee_error_median_deg": float(np.degrees(np.median(error[selected, 3]))),
        "right_knee_error_median_deg": float(np.degrees(np.median(error[selected, 9]))),
        "left_knee_error_rms_deg": float(np.degrees(np.sqrt(np.mean(error[selected, 3] ** 2)))),
        "right_knee_error_rms_deg": float(np.degrees(np.sqrt(np.mean(error[selected, 9] ** 2)))),
        "left_knee_speed_rms_deg_s": float(np.degrees(np.sqrt(np.mean(qd[selected, 3] ** 2)))),
        "right_knee_speed_rms_deg_s": float(np.degrees(np.sqrt(np.mean(qd[selected, 9] ** 2)))),
        "left_knee_force_rms": float(np.sqrt(np.mean(force[selected, 3] ** 2))),
        "right_knee_force_rms": float(np.sqrt(np.mean(force[selected, 9] ** 2))),
        "ankle_pitch_measured_delta_deg": float(np.degrees(np.median(q[selected, 4] - q[selected, 10]))),
        "left_ankle_pitch_error_rms_deg": float(np.degrees(np.sqrt(np.mean(error[selected, 4] ** 2)))),
        "right_ankle_pitch_error_rms_deg": float(np.degrees(np.sqrt(np.mean(error[selected, 10] ** 2)))),
        "left_ankle_pitch_force_rms": float(np.sqrt(np.mean(force[selected, 4] ** 2))),
        "right_ankle_pitch_force_rms": float(np.sqrt(np.mean(force[selected, 10] ** 2))),
        "tilt_median_deg": float(np.median(tilt[selected])),
        "tilt_p95_deg": float(np.percentile(tilt[selected], 95)),
        "gravity_xy_rms": float(np.sqrt(np.mean(gravity[selected, :2] ** 2))),
    }
    if q_true is not None:
        row["knee_true_delta_deg"] = float(
            np.degrees(np.median(q_true[selected, 3] - q_true[selected, 9]))
        )
    else:
        row["knee_true_delta_deg"] = float("nan")
    return row


def _physical_rows() -> list[dict]:
    rows = []
    for state, sessions in (("pre_adjustment", PRE_SESSIONS), ("post_adjustment", POST_SESSIONS)):
        for session in sessions:
            data = _load_physical(session)
            gravity = _gravity_from_wxyz(np.column_stack((
                data["imu_qw"], data["imu_qx"], data["imu_qy"], data["imu_qz"]
            )))
            for phase, selected in _assign_physical_phases(data).items():
                row = _phase_metrics(
                    source="physical",
                    session=session,
                    phase=phase,
                    selected=selected,
                    q=data["q"],
                    qd=data["qd"],
                    target=data["target"],
                    force=data["force"],
                    gravity=gravity,
                )
                if row is not None:
                    row["hardware_state"] = state
                    rows.append(row)
    return rows


def _stable_sim_mask(mode: np.ndarray, phase_id: int, dt: float) -> np.ndarray:
    selected = mode == phase_id
    guard = int(round(1.0 / dt))
    for boundary in np.flatnonzero(mode[1:] != mode[:-1]) + 1:
        selected[max(0, boundary - guard): min(len(selected), boundary + guard)] = False
    selected[:guard] = False
    return selected


def _simulation_rows(sim_root: Path) -> list[dict]:
    rows = []
    for state_path in sorted(sim_root.glob("*/state.npz")):
        directory = state_path.parent
        meta = json.loads((directory / "meta.json").read_text())
        state = np.load(state_path, allow_pickle=False)
        mode = np.asarray(state["mode"], np.int64)
        dt = float(meta["control_dt"])
        offsets = np.asarray(meta["joint_encoder_offset_deg"], np.float64)
        q = np.asarray(state["q_measured"], np.float64)
        q_true = np.asarray(state["q_true"], np.float64)
        qd = np.asarray(state["qd"], np.float64)
        target = np.asarray(state["target_q"], np.float64)
        force = np.asarray(state["actuator_force"], np.float64)
        gravity = np.asarray(state["projected_gravity"], np.float64)
        for phase, phase_id in (("IDLE", 0), ("SLOW_WALK_FORWARD", 1)):
            selected = _stable_sim_mask(mode, phase_id, dt)
            row = _phase_metrics(
                source="mujoco",
                session=directory.name,
                phase=phase,
                selected=selected,
                q=q,
                qd=qd,
                target=target,
                force=force,
                gravity=gravity,
                q_true=q_true,
            )
            if row is not None:
                row["hardware_state"] = "healthy_mujoco_with_sensor_bias"
                row["left_offset_deg"] = float(offsets[3])
                row["right_offset_deg"] = float(offsets[9])
                row["fell"] = bool(meta.get("fell", False))
                row["stand_capture_success"] = bool(
                    "stand_capture_count" in state
                    and np.max(state["stand_capture_count"]) > 0
                )
                base = np.asarray(state["base_position"], np.float64)
                consecutive = np.flatnonzero(selected)[1:]
                consecutive = consecutive[
                    np.diff(np.flatnonzero(selected)) == 1
                ]
                previous = consecutive - 1
                row["base_path_length_m"] = float(np.linalg.norm(
                    base[consecutive, :2] - base[previous, :2], axis=1
                ).sum()) if len(consecutive) else 0.0
                selected_indices = np.flatnonzero(selected)
                row["base_net_displacement_m"] = float(np.linalg.norm(
                    base[selected_indices[-1], :2] - base[selected_indices[0], :2]
                ))
                row["base_height_min_m"] = float(base[selected, 2].min())
                row["left_foot_contact_pct"] = float(
                    np.mean(state["left_foot_contact"][selected]) * 100.0
                )
                row["right_foot_contact_pct"] = float(
                    np.mean(state["right_foot_contact"][selected]) * 100.0
                )
                row["double_support_pct"] = float(np.mean(
                    state["left_foot_contact"][selected]
                    & state["right_foot_contact"][selected]
                ) * 100.0)
                rows.append(row)
    return rows


def _aggregate(rows: list[dict], phase: str, hardware_state: str) -> dict | None:
    selected = [
        row for row in rows
        if row["phase"] == phase and row["hardware_state"] == hardware_state
    ]
    if not selected:
        return None
    weights = np.asarray([row["samples"] for row in selected], np.float64)
    result = {"phase": phase, "hardware_state": hardware_state, "runs": len(selected)}
    for key in selected[0]:
        if key in {"source", "session", "phase", "hardware_state", "samples", "duration_s"}:
            continue
        try:
            values = np.asarray([float(row[key]) for row in selected], np.float64)
        except (TypeError, ValueError):
            continue
        finite = np.isfinite(values)
        if np.any(finite):
            result[key] = float(np.average(values[finite], weights=weights[finite]))
    result["samples"] = int(weights.sum())
    result["duration_s"] = float(weights.sum() * 0.02)
    return result


def _match_rows(physical: list[dict], simulation: list[dict]) -> list[dict]:
    """Rank simulations by bilateral dynamic fingerprints, not absolute pose.

    This is a hypothesis-ranking aid, not encoder calibration: LowState alone
    cannot reveal true joint angle.  Each feature is normalized by the robust
    spread across the MuJoCo sweep so different units do not dominate.
    """
    def asymmetry(row: dict, left: str, right: str) -> float:
        left_value = float(row[left])
        right_value = float(row[right])
        return (left_value - right_value) / max(left_value + right_value, 1e-9)

    feature_functions = (
        lambda row: float(row["knee_measured_delta_deg"]),
        lambda row: float(row["knee_target_delta_deg"]),
        lambda row: asymmetry(
            row, "left_knee_error_rms_deg", "right_knee_error_rms_deg"
        ),
        lambda row: asymmetry(
            row, "left_knee_speed_rms_deg_s", "right_knee_speed_rms_deg_s"
        ),
        lambda row: asymmetry(
            row, "left_knee_force_rms", "right_knee_force_rms"
        ),
        lambda row: asymmetry(
            row,
            "left_ankle_pitch_error_rms_deg",
            "right_ankle_pitch_error_rms_deg",
        ),
        lambda row: asymmetry(
            row, "left_ankle_pitch_force_rms", "right_ankle_pitch_force_rms"
        ),
        lambda row: float(row["gravity_xy_rms"]),
    )
    observed = _aggregate(physical, "SLOW_WALK_FORWARD", "post_adjustment")
    candidates = [row for row in simulation if row["phase"] == "SLOW_WALK_FORWARD"]
    if observed is None or not candidates:
        return []
    matrix = np.asarray([
        [function(row) for function in feature_functions]
        for row in candidates
    ])
    center = np.median(matrix, axis=0)
    spread = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    spread = np.where(spread > 1e-6, spread, np.std(matrix, axis=0))
    spread = np.where(spread > 1e-6, spread, 1.0)
    observed_vector = np.asarray([
        function(observed) for function in feature_functions
    ])
    output = []
    for candidate, vector in zip(candidates, matrix):
        # Clip individual residuals: real hardware/friction can push a metric
        # outside the healthy-plant sweep and should not monopolize the rank.
        residual = np.clip((vector - observed_vector) / spread, -4.0, 4.0)
        output.append({
            "rank": 0,
            "left_offset_deg": candidate["left_offset_deg"],
            "right_offset_deg": candidate["right_offset_deg"],
            "differential_offset_left_minus_right_deg": (
                candidate["left_offset_deg"] - candidate["right_offset_deg"]
            ),
            "distance": float(np.sqrt(np.mean(residual ** 2))),
            "fell": candidate["fell"],
            "stand_capture_success": candidate["stand_capture_success"],
            "sim_knee_true_delta_deg": candidate["knee_true_delta_deg"],
            "sim_knee_measured_delta_deg": candidate["knee_measured_delta_deg"],
            "observed_knee_measured_delta_deg": observed["knee_measured_delta_deg"],
        })
    output.sort(key=lambda row: (
        row["fell"], not row["stand_capture_success"], row["distance"]
    ))
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    return output


def _plots(output: Path, physical: list[dict], simulation: list[dict]) -> None:
    slow = [row for row in simulation if row["phase"] == "SLOW_WALK_FORWARD"]
    left_offsets = sorted({round(float(row["left_offset_deg"])) for row in slow})
    right_offsets = sorted({round(float(row["right_offset_deg"])) for row in slow})
    if not left_offsets or not right_offsets:
        return
    lookup = {
        (
            round(float(row["left_offset_deg"])),
            round(float(row["right_offset_deg"])),
        ): row
        for row in slow
    }
    metrics = (
        ("knee_true_delta_deg", "True L-R knee angle (deg)"),
        ("gravity_xy_rms", "Pelvis gravity XY RMS"),
        ("left_knee_force_rms", "Left knee force RMS (MuJoCo)"),
        ("right_knee_force_rms", "Right knee force RMS (MuJoCo)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        matrix = np.full((len(left_offsets), len(right_offsets)), np.nan)
        for row_index, left in enumerate(left_offsets):
            for column_index, right in enumerate(right_offsets):
                if (left, right) in lookup:
                    matrix[row_index, column_index] = lookup[(left, right)][metric]
        image = ax.imshow(matrix, origin="lower", cmap="coolwarm", aspect="equal")
        ax.set_xticks(range(len(right_offsets)), right_offsets)
        ax.set_yticks(range(len(left_offsets)), left_offsets)
        ax.set_xlabel("right-knee encoder offset (deg)")
        ax.set_ylabel("left-knee encoder offset (deg)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(output / "mujoco_offset_grid.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for phase, ax in zip(("IDLE", "SLOW_WALK_FORWARD"), axes):
        rows = [row for row in physical if row["phase"] == phase]
        labels = [f"{'pre' if row['hardware_state'].startswith('pre') else 'post'}\n{row['session'][9:15]}" for row in rows]
        x = np.arange(len(rows))
        ax.bar(x - 0.18, [row["left_knee_force_rms"] for row in rows], 0.36, label="left")
        ax.bar(x + 0.18, [row["right_knee_force_rms"] for row in rows], 0.36, label="right")
        ax.set_xticks(x, labels)
        ax.set_ylabel("knee estimated torque RMS (Nm)")
        ax.set_title(phase.replace("_", " "))
        ax.legend()
    fig.savefig(output / "physical_knee_effort.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=SIM_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    physical = _physical_rows()
    simulation = _simulation_rows(args.sim_root)
    matches = _match_rows(physical, simulation)
    _write_csv(args.output / "physical_phase_metrics.csv", physical)
    _write_csv(args.output / "mujoco_offset_sweep_metrics.csv", simulation)
    _write_csv(args.output / "qualitative_match_ranking.csv", matches)
    left_only_matches = [
        dict(row)
        for row in matches
        if abs(float(row["right_offset_deg"])) < 1e-6
        and bool(row["stand_capture_success"])
        and not bool(row["fell"])
    ]
    left_only_matches.sort(key=lambda row: float(row["distance"]))
    for rank, row in enumerate(left_only_matches, 1):
        row["rank"] = rank
    _write_csv(
        args.output / "left_only_capture_qualified_match_ranking.csv",
        left_only_matches,
    )
    left_only_metrics = sorted(
        (
            row for row in simulation
            if abs(float(row["right_offset_deg"])) < 1e-6
        ),
        key=lambda row: (row["phase"], float(row["left_offset_deg"])),
    )
    _write_csv(
        args.output / "left_only_offset_metrics.csv", left_only_metrics
    )
    aggregates = []
    for phase in ("IDLE", "SLOW_WALK_FORWARD"):
        for hardware_state in ("pre_adjustment", "post_adjustment"):
            result = _aggregate(physical, phase, hardware_state)
            if result is not None:
                aggregates.append(result)
    _write_csv(args.output / "physical_pre_post_aggregate.csv", aggregates)
    _plots(args.output, physical, simulation)
    print(
        json.dumps(
            {
                "physical_rows": len(physical),
                "simulation_rows": len(simulation),
                "match_rows": len(matches),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
