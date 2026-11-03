#!/usr/bin/env python3
"""Summarize timing, tracking, pelvis tilt, and unavailable SONIC actions.

This consumes the diagnostic ``state.npz`` written by ``sonic/deploy.py``.  It
does not certify a robot run; it makes the deployment-PC evidence needed to
separate inference latency, loop/DDS jitter, and mechanical tracking visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _finite(x) -> np.ndarray:
    x = np.asarray(x, np.float64).reshape(-1)
    return x[np.isfinite(x)]


def _stats(x, scale=1.0) -> str:
    x = _finite(x) * scale
    if not len(x):
        return "no finite samples"
    p50, p95, p99 = np.percentile(x, [50, 95, 99])
    return f"p50={p50:.3f} p95={p95:.3f} p99={p99:.3f} max={x.max():.3f}"


def _rpy_wxyz(q) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(-1, 4)
    w, x, y, z = q.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    state_path = path / "state.npz" if path.is_dir() else path
    state = dict(np.load(state_path, allow_pickle=False))
    meta_path = state_path.with_name("meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return state, meta


def report(path: Path) -> str:
    state, meta = _load(path)
    if not state:
        raise ValueError("episode contains no state arrays")
    n = len(next(iter(state.values())))
    dt = float(meta.get("control_dt", 0.02))
    lines = [f"SONIC episode: {path} ({n} control rows, target {1 / dt:.1f} Hz)"]

    providers = {
        key: meta.get(key, [])
        for key in ("encoder_providers", "decoder_providers", "planner_providers")
    }
    lines.append(f"providers: {providers}")

    timing = (
        ("loop period", "loop_period_s", 1e3, "ms"),
        ("control compute", "control_compute_s", 1e3, "ms"),
        ("LowState age", "low_state_age_s", 1e3, "ms"),
        ("policy total", "policy_total_ms", 1.0, "ms"),
        ("encoder", "encoder_inference_ms", 1.0, "ms"),
        ("decoder", "decoder_inference_ms", 1.0, "ms"),
        ("planner inference", "planner_inference_ms", 1.0, "ms"),
        ("planner total", "planner_total_ms", 1.0, "ms"),
        ("PICO input age", "teleop_age_s", 1e3, "ms"),
    )
    lines.append("timing:")
    for label, key, scale, unit in timing:
        if key in state:
            lines.append(f"  {label:18s} {_stats(state[key], scale)} {unit}")

    if "loop_period_s" in state:
        period = _finite(state["loop_period_s"])
        late = np.mean(period > 1.25 * dt) * 100.0
        severe = np.mean(period > 2.0 * dt) * 100.0
        lines.append(
            f"  loop intervals >{1.25 * dt * 1e3:.0f}ms: {late:.2f}%  "
            f">{2 * dt * 1e3:.0f}ms: {severe:.2f}%"
        )
    if "low_state_age_s" in state:
        age = _finite(state["low_state_age_s"])
        lines.append(
            f"  LowState age >1 tick: {np.mean(age > dt) * 100:.2f}%  "
            f">2 ticks: {np.mean(age > 2 * dt) * 100:.2f}%"
        )

    tracking_key = (
        "tracking_error_q_physical"
        if "tracking_error_q_physical" in state
        else "tracking_error_q_23"
    )
    if tracking_key in state:
        error_deg = np.degrees(np.abs(np.asarray(state[tracking_key])))
        per_step_max = np.max(error_deg, axis=1)
        per_joint_p99 = np.percentile(error_deg, 99, axis=0)
        worst = np.argsort(per_joint_p99)[-5:][::-1]
        lines.append(f"tracking max/step: {_stats(per_step_max)} deg")
        lines.append(
            "tracking p99 worst joints: "
            + ", ".join(f"j{j}={per_joint_p99[j]:.2f}deg" for j in worst)
        )

    if "imu_quat_wxyz" in state:
        rpy = _rpy_wxyz(state["imu_quat_wxyz"])
        lines.append(f"absolute pelvis roll:  {_stats(np.abs(np.degrees(rpy[:, 0])))} deg")
        lines.append(f"absolute pelvis pitch: {_stats(np.abs(np.degrees(rpy[:, 1])))} deg")

    saturation_key = (
        "command_saturated_physical"
        if "command_saturated_physical" in state
        else "command_saturated_23"
    )
    if saturation_key in state:
        saturated = np.asarray(state[saturation_key], bool)
        lines.append(
            f"command saturation: {np.mean(np.any(saturated, axis=1)) * 100:.2f}% "
            "of rows have >=1 limited joint"
        )
    if "sonic_missing_action_raw" in state:
        missing = np.asarray(state["sonic_missing_action_raw"], np.float64)
        names = meta.get("sonic_missing_action_order", [f"missing_{i}" for i in range(6)])
        lines.append("unavailable 29-DoF action demand (absolute p99/max):")
        for i, name in enumerate(names):
            v = np.abs(missing[:, i])
            lines.append(f"  {name:18s} p99={np.percentile(v, 99):.3f} max={np.max(v):.3f}")

    lines.extend(
        (
            "interpretation boundary:",
            "  - High encoder/decoder/policy time points to inference/runtime setup.",
            "  - Normal inference with late loop periods or old LowState points to host scheduling, DDS, or network jitter.",
            "  - Normal timing with growing tracking/tilt, saturation, or large unavailable waist demand points away from a Python rewrite and toward control/embodiment mismatch.",
            "  - These local timestamps do not measure robot-clock-to-PC one-way latency; add synchronized wire timestamps before claiming a network latency value.",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="episode directory or state.npz")
    args = parser.parse_args()
    print(report(args.episode.expanduser()))


if __name__ == "__main__":
    main()
