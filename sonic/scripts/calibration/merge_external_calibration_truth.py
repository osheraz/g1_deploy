#!/usr/bin/env python3
"""Synchronize externally measured model-coordinate joint angles to a run log.

The external CSV must contain ``time_s`` and ``q_external_0`` through
``q_external_28``. Missing/occluded joints may be NaN. Times are seconds from
the calibration start; ``--time-offset-s`` handles a known clock offset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="suspended calibration state.csv")
    parser.add_argument("external", type=Path, help="external joint-angle CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-offset-s", type=float, default=0.0)
    parser.add_argument("--max-gap-s", type=float, default=0.075)
    args = parser.parse_args()
    if not 0 < args.max_gap_s <= 0.5:
        raise SystemExit("--max-gap-s must be in (0, 0.5]")

    log_fields, log_rows = read_csv(args.log)
    external_fields, external_rows = read_csv(args.external)
    required = ["time_s", *[f"q_external_{i}" for i in range(29)]]
    missing = [name for name in required if name not in external_fields]
    if missing:
        raise SystemExit(f"external CSV is missing columns: {', '.join(missing)}")
    if not log_rows or log_rows[0].get("schema") != "e_skin.suspended_calibration_log.v1":
        raise SystemExit("input log is not an e_skin suspended calibration log")
    if any(name.startswith("q_external_") for name in log_fields):
        raise SystemExit("input log already contains external-truth columns")

    # Physical logs contain actual_time_s so camera/mocap measurements align
    # to when LowState was sampled rather than to the ideal schedule. MuJoCo
    # and older logs fall back to deterministic time_s.
    log_time_field = "actual_time_s" if "actual_time_s" in log_fields else "time_s"
    log_time = np.asarray([float(row[log_time_field]) for row in log_rows])
    ext_time = np.asarray([float(row["time_s"]) for row in external_rows])
    ext_time += args.time_offset_s
    if len(ext_time) < 2 or not np.all(np.diff(ext_time) > 0):
        raise SystemExit("external time_s must be strictly increasing with at least two rows")
    external = np.asarray(
        [[float(row[f"q_external_{i}"]) for i in range(29)] for row in external_rows]
    )
    nearest_right = np.searchsorted(ext_time, log_time, side="left")
    right = np.clip(nearest_right, 0, len(ext_time) - 1)
    left = np.clip(nearest_right - 1, 0, len(ext_time) - 1)
    nearest_gap = np.minimum(np.abs(log_time - ext_time[left]), np.abs(log_time - ext_time[right]))
    close = nearest_gap <= args.max_gap_s

    merged = np.full((len(log_rows), 29), np.nan)
    for motor in range(29):
        valid = np.isfinite(external[:, motor])
        if np.sum(valid) < 2:
            continue
        values = np.interp(log_time, ext_time[valid], external[valid, motor])
        in_range = (log_time >= ext_time[valid][0]) & (log_time <= ext_time[valid][-1])
        merged[close & in_range, motor] = values[close & in_range]

    output = args.output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [*log_fields, *[f"q_external_{i}" for i in range(29)]]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for k, row in enumerate(log_rows):
            out = dict(row)
            for motor in range(29):
                value = merged[k, motor]
                out[f"q_external_{motor}"] = "nan" if not np.isfinite(value) else f"{value:.9f}"
            writer.writerow(out)
    coverage = 100.0 * np.mean(np.isfinite(merged), axis=0)
    print(f"MERGED: {output}")
    print(f"log time source: {log_time_field}")
    print(
        "external coverage by motor (%): "
        + ", ".join(f"{motor}={coverage[motor]:.1f}" for motor in range(29) if coverage[motor] > 0)
    )


if __name__ == "__main__":
    main()
