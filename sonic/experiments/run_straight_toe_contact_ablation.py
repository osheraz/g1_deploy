#!/usr/bin/env python3
"""Run matched Original Release straight-walk/contact adaptation ablations.

The experiment is MuJoCo-only.  It never initializes Unitree DDS.  Every cell
uses the same 23-DoF plant, V2 planner command, seed, timing, and support-pose
capture; only the tracker-reference mask/scale and optional legacy ankle
output shaping change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "g1_deploy/sonic/scripts/sim2sim_sonic.py"
PYTHON = Path("/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python")
TRT = Path("/data/clhsieh/software/nvidia/tensorrt/10.13.3.9")
DEFAULT_OUTPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/"
    "20260818_forward_straight_toe_contact_ablation"
)
DEFAULT_SEEDS = (1234, 4321, 2026)

WRIST_MASKS = (
    "--mask-missing-reference",
    "--mask-missing-action-history",
    "--missing-adaptation-scope", "wrist",
)
ALL_MASKS = (
    "--mask-missing-reference",
    "--mask-missing-action-history",
    "--missing-adaptation-scope", "all",
)
ANKLE_SHAPING = (
    "--motor-ema", "ankle_pitch=0.95",
    "--motor-scale", "ankle_roll=0.95",
)

CANDIDATES: dict[str, tuple[str, ...]] = {
    "parity": (),
    "wrist_masks": WRIST_MASKS,
    "wrist_masks_hip080": (
        *WRIST_MASKS, "--reference-joint-scale", "hip_pitch=0.80",
    ),
    "wrist_masks_ankle": (*WRIST_MASKS, *ANKLE_SHAPING),
    "wrist_masks_hip080_ankle": (
        *WRIST_MASKS,
        "--reference-joint-scale", "hip_pitch=0.80",
        *ANKLE_SHAPING,
    ),
    "all_masks": ALL_MASKS,
    "all_masks_hip095": (
        *ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.95",
    ),
    "all_masks_hip090": (
        *ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.90",
    ),
    "all_masks_hip085": (
        *ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.85",
    ),
    "all_masks_hip080": (
        *ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.80",
    ),
    "all_masks_ankle": (*ALL_MASKS, *ANKLE_SHAPING),
}


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "others/GR00T-WholeBodyControl")
    env["PYTHONNOUSERSITE"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(TRT / "lib"),
        "/usr/local/cuda/lib64",
        "/usr/local/lib",
        env.get("LD_LIBRARY_PATH", ""),
    )))
    return env


def _command(
    candidate: str,
    condition: str,
    knee_offset_deg: float,
    seed: int,
    steps: int,
    axes: str,
    record: Path,
) -> list[str]:
    condition_flags = (
        () if knee_offset_deg == 0.0 else
        ("--joint-encoder-offset-deg", f"left_knee={knee_offset_deg:g}")
    )
    return [
        str(PYTHON), str(SIM),
        "--config", str(ROOT / "g1_deploy/configs/g1_23dof_sonic.yaml"),
        "--model-profile", "original-release",
        "--mjcf", str(ROOT / "sonic_cpp/g1/scene_23dof.xml"),
        "--pico", "joystick",
        "--planner",
        "--native-gamepad",
        "--joystick-mode", "slow_walk",
        f"--axes={axes}",
        "--axes-from-step", "250",
        "--axes-until", str(steps - 250),
        "--planner-speed", "0.4",
        "--planner-seed", str(seed),
        "--capture-at-step", "100",
        "--steps", str(steps),
        "--trial-label",
        f"original-release__{candidate}__{condition}__seed{seed}",
        "--record-dir", str(record),
        *CANDIDATES[candidate],
        *condition_flags,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", nargs="+", choices=tuple(CANDIDATES),
        default=list(CANDIDATES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--knee-offsets-deg",
        nargs="+",
        type=float,
        default=[0.0],
        help="signed left-knee encoder biases to test; measured=true+offset",
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument(
        "--axes", default="0,1,0",
        help="synthetic native-gamepad axes; 0,1,0 is forward",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1000:
        raise SystemExit("--steps must be at least 1000")
    if any(seed < 0 for seed in args.seeds):
        raise SystemExit("planner seeds must be non-negative")
    if any(not -90.0 <= value <= 90.0 for value in args.knee_offsets_deg):
        raise SystemExit("--knee-offsets-deg values must be within [-90, 90]")
    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")

    plan = []
    for seed in args.seeds:
        for candidate in args.candidates:
            for knee_offset_deg in args.knee_offsets_deg:
                sign = "p" if knee_offset_deg >= 0 else "m"
                magnitude = f"{abs(knee_offset_deg):g}".replace(".", "p")
                condition = (
                    "healthy" if knee_offset_deg == 0.0
                    else f"left_knee_{sign}{magnitude}deg"
                )
                record = args.output / candidate / condition / f"seed{seed}"
                plan.append({
                    "candidate": candidate,
                    "condition": condition,
                    "knee_offset_deg": knee_offset_deg,
                    "seed": seed,
                    "record": str(record),
                    "command": _command(
                        candidate,
                        condition,
                        knee_offset_deg,
                        seed,
                        args.steps,
                        args.axes,
                        record,
                    ),
                })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"PLAN: {len(plan)} trials, candidates={','.join(args.candidates)}, "
        f"seeds={','.join(map(str, args.seeds))}, "
        f"knee_offsets_deg={','.join(map(str, args.knee_offsets_deg))}, "
        f"steps={args.steps}",
        flush=True,
    )
    if args.dry_run:
        return

    env = _environment()
    for index, row in enumerate(plan, 1):
        record = Path(row["record"])
        complete = (record / "state.npz").is_file() and (record / "meta.json").is_file()
        if complete and not args.force:
            print(
                f"[{index}/{len(plan)}] skip complete {row['candidate']} "
                f"{row['condition']} seed={row['seed']}"
            )
            continue
        print(
            f"[{index}/{len(plan)}] {row['candidate']} "
            f"{row['condition']} seed={row['seed']}",
            flush=True,
        )
        record.mkdir(parents=True, exist_ok=True)
        with (record / "console.log").open("w") as stream:
            result = subprocess.run(
                row["command"], cwd=ROOT / "g1_deploy", env=env,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            tail = (record / "console.log").read_text(errors="replace").splitlines()[-40:]
            print("\n".join(tail), file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, row["command"])


if __name__ == "__main__":
    main()
