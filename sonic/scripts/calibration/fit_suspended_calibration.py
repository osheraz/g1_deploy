#!/usr/bin/env python3
"""Fit a bounded monotone inverse map from a suspended calibration log."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonic.calibration.joint_mapping import (
    JointCalibrationProfile, JointCurve, monotone_inverse_curve,
)
from sonic.calibration.protocol import MOTOR_NAMES, PRESENT_23


def _load(path: Path):
    with path.expanduser().open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or rows[0].get("schema") not in {
        "e_skin.suspended_calibration_log.v1",
        "e_skin.suspended_calibration_log.v2",
        "e_skin.suspended_calibration_log.v3",
    }:
        raise ValueError("not an e_skin suspended calibration log")
    phase = np.asarray([row["phase"] for row in rows])
    active = np.asarray([int(row["active_motor"]) for row in rows])
    direction = np.asarray([int(row["direction"]) for row in rows])
    target = np.asarray([[float(row[f"target_model_{i}"]) for i in range(29)] for row in rows])
    command = np.asarray([[float(row[f"command_hardware_{i}"]) for i in range(29)] for row in rows])
    measured = np.asarray([[float(row[f"q_measured_{i}"]) for i in range(29)] for row in rows])
    captured = None
    if all(f"captured_neutral_{i}" in rows[0] for i in range(29)):
        captured = np.asarray([
            [float(row[f"captured_neutral_{i}"]) for i in range(29)] for row in rows
        ])
        # Review-gated rows use block metadata rather than a single active_motor.
        block_kind = np.asarray([row.get("block_kind", "") for row in rows])
        single = block_kind == "single_joint"
        delta = np.abs(command - captured)
        inferred = np.argmax(np.nan_to_num(delta, nan=-1.0), axis=1)
        nonzero = np.nanmax(delta, axis=1) > 1e-8
        active = np.where(single & nonzero, inferred, active)
        inferred_direction = np.sign(
            command[np.arange(len(command)), inferred]
            - captured[np.arange(len(command)), inferred]
        ).astype(np.int64)
        direction = np.where(single & nonzero, inferred_direction, direction)
    external = None
    if all(f"q_external_{i}" in rows[0] for i in range(29)):
        external = np.asarray([
            [float(row[f"q_external_{i}"]) for i in range(29)] for row in rows
        ])
    return phase, active, direction, target, command, measured, captured, external


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coordinate-basis", choices=("mujoco_truth", "external_truth", "encoder_only"),
        required=True,
    )
    parser.add_argument("--max-correction", type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.max_correction <= .35:
        raise SystemExit("--max-correction must be in (0, 0.35]")
    (
        phase, active, direction, target, command, encoder_measured, captured,
        external,
    ) = _load(args.log)
    if args.coordinate_basis == "external_truth":
        if external is None:
            raise SystemExit(
                "external_truth requires q_external_0..q_external_28 columns; "
                "merge a synchronized external measurement first"
            )
        measured = external
    else:
        measured = encoder_measured
    joints: dict[int, JointCurve] = {}
    fit_summary = {}
    for motor in PRESENT_23:
        mask = (
            (active == motor)
            & (np.char.find(phase.astype(str), "dwell") >= 0)
            & np.isfinite(measured[:, motor])
        )
        if captured is not None:
            mask &= np.abs(command[:, motor] - captured[:, motor]) > 1e-8
        if np.sum(mask) < 10:
            continue
        def fit_branch(branch):
            # Reduce every held target to a robust median.  The last 60% of a
            # dwell is naturally represented more often in the repeated CSV;
            # medians reject its transition edge and contact-free vibration.
            levels = np.unique(np.round(command[branch, motor], 6))
            cmd_median, measured_median = [], []
            for level in levels:
                select = branch & (np.abs(command[:, motor] - level) < 5e-7)
                if np.sum(select) < 2:
                    continue
                cmd_median.append(float(np.median(command[select, motor])))
                measured_median.append(float(np.median(measured[select, motor])))
            try:
                return monotone_inverse_curve(
                    np.asarray(cmd_median), np.asarray(measured_median)
                )
            except ValueError:
                return None

        # The combined curve owns the complete measured range. Directional
        # branches only contribute a local hysteresis residual where they were
        # actually sampled; they never extrapolate an endpoint over the other
        # half of the joint range.
        base = fit_branch(mask)
        if base is None:
            continue
        branches = {
            label: fit_branch(mask & (direction == sign))
            for label, sign in (("increasing", 1), ("decreasing", -1))
        }
        knots = base[0]
        base_command = base[1]
        mapped = {}
        for label, branch in branches.items():
            if branch is None:
                mapped[label] = base_command.copy()
                continue
            branch_base = np.interp(branch[0], base[0], base[1])
            residual = branch[1] - branch_base
            local = np.interp(knots, branch[0], residual, left=0.0, right=0.0)
            mapped[label] = base_command + local
        inc = np.maximum.accumulate(mapped["increasing"])
        dec = np.maximum.accumulate(mapped["decreasing"])
        joints[int(motor)] = JointCurve(knots, inc, dec, args.max_correction)
        raw_error = measured[mask, motor] - target[mask, motor]
        fit_summary[str(motor)] = {
            "name": MOTOR_NAMES[motor],
            "samples": int(np.sum(mask)),
            "measured_span_rad": [float(knots[0]), float(knots[-1])],
            "raw_dwell_rmse_rad": float(np.sqrt(np.mean(raw_error**2))),
            "direction_hysteresis_median_rad": float(np.median(np.abs(inc - dec))),
        }
    profile = JointCalibrationProfile(
        embodiment="g1_23dof",
        coordinate_basis=args.coordinate_basis,
        # Promotion is always a separate reviewed action.  In particular,
        # encoder-only sweeps cannot see encoder zero, link deflection, or sole
        # geometry and therefore may never self-authorize deployment.
        approved_for_deployment=False,
        joints=joints,
        metadata={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_log": str(args.log.expanduser().absolute()),
            "measurement_source": (
                "q_external" if args.coordinate_basis == "external_truth"
                else "q_measured"
            ),
            "fit": fit_summary,
            "warning": (
                "provisional profile; do not apply to a physical walking controller "
                "until geometric/load validation and explicit promotion"
            ),
        },
    )
    output = profile.save(args.output)
    print(f"PROFILE FIT: {output}")
    print(f"joints={len(joints)} coordinate_basis={args.coordinate_basis} "
          "approved_for_deployment=false")


if __name__ == "__main__":
    main()
