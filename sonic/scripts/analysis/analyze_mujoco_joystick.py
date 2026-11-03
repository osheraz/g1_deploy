#!/usr/bin/env python3
"""Compare recorded SONIC joystick runs in MuJoCo, including both legs.

The launcher writes one ``state.npz`` and ``meta.json`` per run. This analyzer
accepts individual run directories or a parent directory, removes the initial
one-second handover and 0.5 seconds around gait changes, and writes aggregate
phase metrics, per-joint metrics, bilateral comparisons, and heat maps.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODE_NAMES = {
    0: "IDLE",
    1: "SLOW_WALK",
    2: "WALK",
    3: "RUN",
}
LEG_LABELS = (
    "L hip pitch",
    "L hip roll",
    "L hip yaw",
    "L knee",
    "L ankle pitch",
    "L ankle roll",
    "R hip pitch",
    "R hip roll",
    "R hip yaw",
    "R knee",
    "R ankle pitch",
    "R ankle roll",
)


def _discover(inputs: list[Path]) -> list[Path]:
    recordings: set[Path] = set()
    for candidate in inputs:
        candidate = candidate.expanduser().resolve()
        if (candidate / "state.npz").is_file() and (candidate / "meta.json").is_file():
            recordings.add(candidate)
            continue
        for state in candidate.rglob("state.npz"):
            if (state.parent / "meta.json").is_file():
                recordings.add(state.parent)
    return sorted(recordings)


def _stable_mask(mode: np.ndarray, dt: float, blend: bool) -> np.ndarray:
    stable = np.ones(len(mode), dtype=bool)
    if blend:
        stable[: min(len(stable), int(round(1.0 / dt)))] = False
    guard = int(round(0.5 / dt))
    for index in np.flatnonzero(mode[1:] != mode[:-1]) + 1:
        stable[max(0, index - guard) : min(len(stable), index + guard)] = False
    return stable


def _tilt_deg(projected_gravity: np.ndarray) -> np.ndarray:
    return np.degrees(
        np.arccos(np.clip(-np.asarray(projected_gravity, np.float64)[:, 2], -1.0, 1.0))
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_metrics(recording: Path) -> tuple[list[dict], list[dict], dict]:
    meta = json.loads((recording / "meta.json").read_text())
    if meta.get("schema") != "e_skin.mujoco_joystick.v1":
        raise ValueError(f"unsupported recording schema in {recording}: {meta.get('schema')}")
    state = np.load(recording / "state.npz", allow_pickle=False)
    mode = np.asarray(state["mode"], np.int64)
    dt = float(meta["control_dt"])
    stable = _stable_mask(mode, dt, bool(meta.get("blend", True)))
    q = np.asarray(state["q"], np.float64)
    qd = np.asarray(state["qd"], np.float64)
    target = np.asarray(state["target_q"], np.float64)
    force = np.asarray(state["actuator_force"], np.float64)
    error = target - q
    lower = np.asarray(meta["dof_lower"], np.float64)
    upper = np.asarray(meta["dof_upper"], np.float64)
    base = np.asarray(state["base_position"], np.float64)
    tilt = _tilt_deg(state["projected_gravity"])
    names = list(meta["joint_names"])

    def timing_p95(name: str, selected: np.ndarray) -> float:
        if name not in state:
            return float("nan")
        values = np.asarray(state[name], np.float64)[selected]
        values = values[np.isfinite(values)]
        return float(np.percentile(values, 95)) if len(values) else float("nan")

    trial_label = str(meta.get("trial_label", "interactive"))
    if trial_label == "interactive":
        for known in ("slow_forward", "slow_lateral", "walk_forward", "walk_lateral", "idle"):
            if recording.name.endswith("_" + known):
                trial_label = known
                break
    common = {
        "recording": recording.name,
        "recording_path": str(recording),
        "model": meta["model_profile"],
        "embodiment": meta["embodiment"],
        "ground_friction": f"{float(meta.get('ground_friction', 1.0)):g}",
        "trial": trial_label,
        "fell": bool(meta.get("fell", False)),
    }

    phases: list[dict] = []
    joints: list[dict] = []
    for mode_id in sorted(set(mode.tolist())):
        selected = stable & (mode == mode_id)
        indices = np.flatnonzero(selected)
        if not len(indices):
            continue
        gait = MODE_NAMES.get(int(mode_id), f"MODE_{int(mode_id)}")
        consecutive = indices[1:][np.diff(indices) == 1]
        previous = consecutive - 1
        delta = base[consecutive, :2] - base[previous, :2]
        dx = float(delta[:, 0].sum()) if len(delta) else 0.0
        dy = float(delta[:, 1].sum()) if len(delta) else 0.0
        path = float(np.linalg.norm(delta, axis=1).sum()) if len(delta) else 0.0
        leg_error = error[selected, :12]
        leg_qd = qd[selected, :12]
        phases.append(
            {
                **common,
                "mode": gait,
                "samples": int(selected.sum()),
                "duration_s": float(selected.sum() * dt),
                "height_min_m": float(base[selected, 2].min()),
                "height_p05_m": float(np.percentile(base[selected, 2], 5)),
                "height_median_m": float(np.median(base[selected, 2])),
                "tilt_p95_deg": float(np.percentile(tilt[selected], 95)),
                "tilt_max_deg": float(tilt[selected].max()),
                "path_length_m": path,
                "net_dx_m": dx,
                "net_dy_m": dy,
                "leg_error_rms_rad": float(np.sqrt(np.mean(leg_error**2))),
                "leg_speed_rms_rad_s": float(np.sqrt(np.mean(leg_qd**2))),
                "any_leg_target_outside_pct": float(
                    np.mean(
                        np.any(
                            (target[selected, :12] < lower[:12])
                            | (target[selected, :12] > upper[:12]),
                            axis=1,
                        )
                    )
                    * 100.0
                ),
                "encoder_p95_ms": timing_p95("encoder_inference_ms", selected),
                "decoder_p95_ms": timing_p95("decoder_inference_ms", selected),
                "policy_p95_ms": timing_p95("policy_total_ms", selected),
                "planner_p95_ms": timing_p95("planner_total_ms", selected),
                "control_work_p95_ms": timing_p95("control_work_ms", selected),
            }
        )
        for joint_index in range(12):
            joint_error = np.abs(error[selected, joint_index])
            joint_target = target[selected, joint_index]
            joints.append(
                {
                    **common,
                    "mode": gait,
                    "joint_index": joint_index,
                    "joint": names[joint_index].removesuffix("_joint"),
                    "side": "left" if joint_index < 6 else "right",
                    "samples": int(selected.sum()),
                    "error_median_rad": float(np.median(joint_error)),
                    "error_p95_rad": float(np.percentile(joint_error, 95)),
                    "error_rms_rad": float(np.sqrt(np.mean(joint_error**2))),
                    "speed_rms_rad_s": float(np.sqrt(np.mean(qd[selected, joint_index] ** 2))),
                    "abs_force_p95_nm": float(
                        np.percentile(np.abs(force[selected, joint_index]), 95)
                    ),
                    "target_min_rad": float(joint_target.min()),
                    "target_max_rad": float(joint_target.max()),
                    "target_outside_limit_pct": float(
                        np.mean(
                            (joint_target < lower[joint_index])
                            | (joint_target > upper[joint_index])
                        )
                        * 100.0
                    ),
                }
            )
    return phases, joints, meta


def _aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, members in sorted(groups.items()):
        weights = np.asarray([float(row["samples"]) for row in members])
        result = dict(zip(keys, key_values))
        result["runs"] = len(members)
        result["samples"] = int(weights.sum())
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in members])
            result[metric] = float(np.average(values, weights=weights))
        output.append(result)
    return output


def _bilateral_rows(joints: list[dict]) -> list[dict]:
    indexed = {
        (
            row["model"],
            row["embodiment"],
            row["ground_friction"],
            row["trial"],
            row["mode"],
            int(row["joint_index"]),
        ): row
        for row in joints
    }
    output = []
    for model, embodiment, friction, trial, mode, left_index in sorted(indexed):
        if left_index >= 6:
            continue
        left = indexed[(model, embodiment, friction, trial, mode, left_index)]
        right = indexed.get((model, embodiment, friction, trial, mode, left_index + 6))
        if right is None:
            continue
        output.append(
            {
                "model": model,
                "embodiment": embodiment,
                "ground_friction": friction,
                "trial": trial,
                "mode": mode,
                "joint_pair": left["joint"].removeprefix("left_"),
                "left_error_p95_rad": left["error_p95_rad"],
                "right_error_p95_rad": right["error_p95_rad"],
                "left_minus_right_error_p95_rad": (
                    float(left["error_p95_rad"]) - float(right["error_p95_rad"])
                ),
                "left_force_p95_nm": left["abs_force_p95_nm"],
                "right_force_p95_nm": right["abs_force_p95_nm"],
                "left_outside_pct": left["target_outside_limit_pct"],
                "right_outside_pct": right["target_outside_limit_pct"],
            }
        )
    return output


def _figures(output: Path, joints: list[dict]) -> list[Path]:
    paths = []
    for embodiment in sorted({row["embodiment"] for row in joints}):
        frictions = sorted(
            {row["ground_friction"] for row in joints if row["embodiment"] == embodiment}
        )
        for friction in frictions:
            trials = sorted(
                {
                    row["trial"]
                    for row in joints
                    if row["embodiment"] == embodiment and row["ground_friction"] == friction
                }
            )
            for trial in trials:
                selected = [
                    row
                    for row in joints
                    if row["embodiment"] == embodiment
                    and row["ground_friction"] == friction
                    and row["trial"] == trial
                ]
                if not selected:
                    continue
                modes = sorted({row["mode"] for row in selected})
                mode = "+".join(modes)
                models = sorted({row["model"] for row in selected})
                matrix = np.full((len(models), 12), np.nan)
                for row in selected:
                    matrix[models.index(row["model"]), int(row["joint_index"])] = float(
                        row["error_p95_rad"]
                    )
                fig, ax = plt.subplots(
                    figsize=(12, 2.1 + 0.55 * len(models)), constrained_layout=True
                )
                image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0)
                ax.set_xticks(range(12), LEG_LABELS, rotation=40, ha="right")
                ax.set_yticks(range(len(models)), models)
                ax.set_title(
                    f"{embodiment} {trial} ({mode}, friction={friction}): "
                    "command tracking error p95 (rad)"
                )
                fig.colorbar(image, ax=ax, label="absolute error p95 (rad)")
                friction_token = friction.replace(".", "p")
                path = output / f"leg_error_{embodiment}_mu{friction_token}_{trial.lower()}.png"
                fig.savefig(path, dpi=180)
                plt.close(fig)
                paths.append(path)
    return paths


def _markdown(
    recordings: list[Path], phases: list[dict], joints: list[dict], figures: list[Path]
) -> str:
    timing_available = any(
        np.isfinite(float(row["policy_p95_ms"])) and np.isfinite(float(row["control_work_p95_ms"]))
        for row in phases
    )
    lines = [
        "# MuJoCo joystick comparison",
        "",
        f"Recordings analyzed: {len(recordings)}",
        "",
        "The first second of policy handover and 0.5 seconds around every gait change are excluded. "
        "Values combine repeated runs using sample-count weighting.",
        "",
        "## Phase summary",
        "",
    ]
    if timing_available:
        lines.extend(
            (
                "| Model | Embodiment | Friction | Trial | Mode | Runs | Data (s) | Fell | Tilt p95 (deg) | Height p05 (m) | Leg error RMS (rad) | Any leg target outside (%) | Policy/control work p95 (ms) |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            )
        )
    else:
        lines.extend(
            (
                "| Model | Embodiment | Friction | Trial | Mode | Runs | Data (s) | Fell | Tilt p95 (deg) | Height p05 (m) | Leg error RMS (rad) | Any leg target outside (%) |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
            )
        )
    for row in phases:
        line = (
            f"| {row['model']} | {row['embodiment']} | {row['ground_friction']} | "
            f"{row['trial']} | {row['mode']} | {row['runs']} | "
            f"{row['samples'] * 0.02:.2f} | {'yes' if row['fell'] else 'no'} | "
            f"{row['tilt_p95_deg']:.2f} | {row['height_p05_m']:.3f} | "
            f"{row['leg_error_rms_rad']:.3f} | {row['any_leg_target_outside_pct']:.1f}"
        )
        if timing_available:
            line += f" | {row['policy_p95_ms']:.2f} / {row['control_work_p95_ms']:.2f} |"
        else:
            line += " |"
        lines.append(line)
    lines.extend(("", "## All joints on both legs", ""))
    for embodiment in sorted({row["embodiment"] for row in joints}):
        cases = sorted(
            {
                (row["ground_friction"], row["trial"])
                for row in joints
                if row["embodiment"] == embodiment
            }
        )
        for friction, trial in cases:
            selected = [
                row
                for row in joints
                if row["embodiment"] == embodiment
                and row["ground_friction"] == friction
                and row["trial"] == trial
            ]
            if not selected:
                continue
            modes = sorted({row["mode"] for row in selected})
            mode = "+".join(modes)
            models = sorted({row["model"] for row in selected})
            indexed = {(row["model"], int(row["joint_index"])): row for row in selected}
            lines.extend(
                (
                    f"### {embodiment}: {trial} ({mode}, friction={friction})",
                    "",
                    "Absolute target-to-position error p95 in radians:",
                    "",
                    "| Joint | " + " | ".join(models) + " |",
                    "| --- | " + " | ".join("---:" for _ in models) + " |",
                )
            )
            for index, label in enumerate(LEG_LABELS):
                values = [indexed.get((model, index)) for model in models]
                lines.append(
                    "| "
                    + label
                    + " | "
                    + " | ".join(
                        f"{float(row['error_p95_rad']):.3f}" if row else "—" for row in values
                    )
                    + " |"
                )
            lines.append("")
    if figures:
        lines.extend(("## Figures", ""))
        for path in figures:
            lines.append(f"- [`{path.name}`]({path.name})")
        lines.append("")
    lines.extend(
        (
            "## Interpretation boundary",
            "",
            "MuJoCo isolates checkpoint, embodiment projection, planner, gains, and simulated plant behavior. "
            "It cannot reproduce repaired-hardware alignment, exact real-floor friction, preload, motor temperature, calibration, "
            "Ethernet/DDS timing, or floor-contact differences. A simulator/robot discrepancy is therefore useful "
            "evidence of a plant or interface mismatch, not proof of one specific hardware failure.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="recording directories or parent directories"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    recordings = _discover(args.inputs)
    if not recordings:
        raise SystemExit("no e_skin.mujoco_joystick.v1 recordings found")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    phase_rows: list[dict] = []
    joint_rows: list[dict] = []
    for recording in recordings:
        phases, joints, _ = _run_metrics(recording)
        phase_rows.extend(phases)
        joint_rows.extend(joints)
    phase_metrics = (
        "height_min_m",
        "height_p05_m",
        "height_median_m",
        "tilt_p95_deg",
        "tilt_max_deg",
        "path_length_m",
        "net_dx_m",
        "net_dy_m",
        "leg_error_rms_rad",
        "leg_speed_rms_rad_s",
        "any_leg_target_outside_pct",
        "encoder_p95_ms",
        "decoder_p95_ms",
        "policy_p95_ms",
        "planner_p95_ms",
        "control_work_p95_ms",
    )
    joint_metrics = (
        "error_median_rad",
        "error_p95_rad",
        "error_rms_rad",
        "speed_rms_rad_s",
        "abs_force_p95_nm",
        "target_min_rad",
        "target_max_rad",
        "target_outside_limit_pct",
    )
    phases = _aggregate(
        phase_rows,
        ("model", "embodiment", "ground_friction", "trial", "mode", "fell"),
        phase_metrics,
    )
    joints = _aggregate(
        joint_rows,
        (
            "model",
            "embodiment",
            "ground_friction",
            "trial",
            "mode",
            "joint_index",
            "joint",
            "side",
            "fell",
        ),
        joint_metrics,
    )
    bilateral = _bilateral_rows(joints)
    _write_csv(output / "phase_metrics.csv", phases)
    _write_csv(output / "joint_metrics.csv", joints)
    _write_csv(output / "bilateral_metrics.csv", bilateral)
    figures = _figures(output, joints)
    (output / "REPORT.md").write_text(_markdown(recordings, phases, joints, figures))
    (output / "recordings.json").write_text(
        json.dumps([str(path) for path in recordings], indent=2) + "\n"
    )
    print(f"Analyzed {len(recordings)} recording(s): {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
