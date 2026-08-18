#!/usr/bin/env python3
"""Run a deterministic, motor-free bilateral knee encoder-offset sweep.

The experiment uses the Original Release SONIC model, the 23-DoF MuJoCo
embodiment, the official-parity adaptation, and native gamepad semantics.  Each
trial has 5 s IDLE, 20 s forward SLOW_WALK at the official 0.4 m/s request,
then 5 s IDLE.  The sweep is a 5x5 grid of signed left/right knee measurement
offsets.  No command is sent to a physical robot.

Offset convention:
    measured_q = true_mujoco_q + offset

The same measured value is supplied to SONIC proprioception and the low-level
PD comparison, so this models a real encoder-zero error rather than changing a
plot after simulation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "g1_deploy/scripts/sim2sim_sonic.py"
PYTHON = Path("/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python")
TRT = Path("/data/clhsieh/software/nvidia/tensorrt/10.13.3.9")
DEFAULT_OUTPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/20260815_knee_encoder_offset_sweep"
)
OFFSETS_DEG = (-20, -10, 0, 10, 20)


def _tag(value: int) -> str:
    return f"p{value}" if value > 0 else f"m{abs(value)}" if value < 0 else "z0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=list(OFFSETS_DEG),
        help="signed degree values used for both axes of the Cartesian grid",
    )
    parser.add_argument(
        "--left-offsets",
        type=int,
        nargs="+",
        default=None,
        help="optional left-knee values; overrides --offsets for the left axis",
    )
    parser.add_argument(
        "--right-offsets",
        type=int,
        nargs="+",
        default=None,
        help="optional right-knee values; overrides --offsets for the right axis",
    )
    args = parser.parse_args()

    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")
    left_offsets = args.offsets if args.left_offsets is None else args.left_offsets
    right_offsets = args.offsets if args.right_offsets is None else args.right_offsets
    if any(abs(value) > 30 for value in (*left_offsets, *right_offsets)):
        raise SystemExit("this diagnostic sweep is intentionally limited to +/-30 deg")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "others/GR00T-WholeBodyControl")
    env["PYTHONNOUSERSITE"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            (
                str(TRT / "lib"),
                "/usr/local/cuda/lib64",
                "/usr/local/lib",
                env.get("LD_LIBRARY_PATH", ""),
            ),
        )
    )

    matrix = [
        (left, right)
        for left in left_offsets
        for right in right_offsets
    ]
    for index, (left, right) in enumerate(matrix, 1):
        name = f"left_{_tag(left)}_right_{_tag(right)}"
        record = args.output / name
        complete = (record / "state.npz").is_file() and (record / "meta.json").is_file()
        if complete and not args.force:
            print(f"[{index:02d}/{len(matrix)}] skip complete {name}")
            continue

        command = [
            str(PYTHON),
            str(SIM),
            "--config",
            str(ROOT / "g1_deploy/configs/g1_23dof_sonic.yaml"),
            "--model-profile",
            "original-release",
            "--mjcf",
            str(ROOT / "sonic_cpp/g1/scene_23dof.xml"),
            "--pico",
            "joystick",
            "--planner",
            "--native-gamepad",
            "--joystick-mode",
            "slow_walk",
            "--axes=0,1,0",
            "--axes-from-step",
            "250",
            "--axes-until",
            "1250",
            "--capture-at-step",
            "100",
            "--steps",
            "1500",
            "--trial-label",
            name,
            "--record-dir",
            str(record),
            "--joint-encoder-offset-deg",
            f"left_knee={left}",
            "--joint-encoder-offset-deg",
            f"right_knee={right}",
        ]
        print(
            f"[{index:02d}/{len(matrix)}] {name}: "
            f"left={left:+d} deg, right={right:+d} deg"
        )
        subprocess.run(command, cwd=ROOT / "g1_deploy", env=env, check=True)


if __name__ == "__main__":
    main()
