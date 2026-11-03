#!/usr/bin/env python3
"""Fail closed unless a physical calibration plan matches MuJoCo validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonic.calibration.gated_protocol import FULL_SCREENING_AMPLITUDE, load_review_protocol
from sonic.calibration.protocol import PRESENT_23, SCREENING_AMPLITUDE

VALIDATION = ROOT / "configs/calibration/review_gated_validation.json"
CPP = REPO / "sonic_cpp/src/g1/g1_deploy_onnx_ref/src/g1_suspended_calibration.cpp"
GAIN_CONFIG = ROOT / "configs/g1_23dof_sonic.yaml"


def parse_cpp_array(source: str, name: str, value_type: str = "double") -> np.ndarray:
    match = re.search(
        rf"constexpr\s+std::array<{value_type},\s*[^>]+>\s+{name}\s*=\s*\{{(.*?)\}};",
        source,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"cannot find C++ array {name}")
    values = [item.strip() for item in match.group(1).replace("\n", " ").split(",")]
    values = [item for item in values if item]
    return np.asarray([float(item) for item in values], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("pilot", "screening"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    expected = json.loads(VALIDATION.read_text())
    contract = expected["protocols"][args.protocol]
    plan = load_review_protocol(args.plan)
    plan_hash = sha256(args.plan)
    if plan.name != args.protocol:
        raise SystemExit(f"PARITY ERROR: plan says {plan.name}, expected {args.protocol}")
    if plan_hash != contract["protocol_sha256"]:
        raise SystemExit(
            "PARITY ERROR: generated protocol has not passed the recorded full "
            f"MuJoCo collision run ({plan_hash})"
        )
    if len(plan.blocks) != contract["blocks"]:
        raise SystemExit("PARITY ERROR: block count changed")
    if sum(len(block.local_time_s) for block in plan.blocks) != contract["plan_samples"]:
        raise SystemExit("PARITY ERROR: plan sample count changed")
    if abs(plan.motion_duration_s - contract["motion_duration_s"]) > 1e-8:
        raise SystemExit("PARITY ERROR: plan duration changed")

    cfg = yaml.safe_load(GAIN_CONFIG.read_text())
    source = CPP.read_text()
    cpp_kp = parse_cpp_array(source, "kKp")
    cpp_kd = parse_cpp_array(source, "kKd")
    cpp_initial = parse_cpp_array(source, "kInitialPose")
    if not np.array_equal(cpp_kp, np.asarray(cfg["kps"], dtype=np.float64)):
        raise SystemExit("PARITY ERROR: physical C++ Kp differs from MuJoCo config")
    if not np.array_equal(cpp_kd, np.asarray(cfg["kds"], dtype=np.float64)):
        raise SystemExit("PARITY ERROR: physical C++ Kd differs from MuJoCo config")
    if not np.array_equal(cpp_initial, np.asarray(cfg["default_angles"], dtype=np.float64)):
        raise SystemExit(
            "PARITY ERROR: physical calibration initial pose differs from "
            "the SONIC/MuJoCo default pose"
        )
    cpp_present = parse_cpp_array(source, "kPresent", "int").astype(np.int64)
    if not np.array_equal(cpp_present, PRESENT_23):
        raise SystemExit("PARITY ERROR: physical C++ 23-DoF motor mapping changed")
    dt_match = re.search(r"constexpr double kControlDt\s*=\s*([^;]+);", source)
    if not dt_match or float(dt_match.group(1)) != expected["control_dt_s"]:
        raise SystemExit("PARITY ERROR: physical C++ target period is not 50 Hz")
    writer_match = re.search(r"const auto period\s*=\s*std::chrono::microseconds\((\d+)\);", source)
    if not writer_match or int(writer_match.group(1)) != 2000:
        raise SystemExit("PARITY ERROR: physical LowCmd writer period is not 2 ms")

    review_full = parse_cpp_array(source, "kReviewFullAmplitude")
    expected_full = np.zeros(29, dtype=np.float64)
    for motor, value in FULL_SCREENING_AMPLITUDE.items():
        expected_full[motor] = value
    if not np.array_equal(review_full, expected_full):
        raise SystemExit("PARITY ERROR: C++ full-screening envelope differs from Python")
    cpp_small = parse_cpp_array(source, "kScreeningAmplitude")
    expected_small = np.zeros(29, dtype=np.float64)
    for motor, value in SCREENING_AMPLITUDE.items():
        expected_small[motor] = value
    if not np.array_equal(cpp_small, expected_small):
        raise SystemExit("PARITY ERROR: C++ pilot source envelope differs from Python")

    print(
        "PHYSICAL PARITY OK: "
        f"protocol={args.protocol} sha256={plan_hash} blocks={len(plan.blocks)} "
        f"samples={contract['plan_samples']} target_hz=50 writer_hz=500 "
        "physical_dof=23 gains=identical initial_pose=identical "
        "collision_validation=zero-contact"
    )


if __name__ == "__main__":
    main()
