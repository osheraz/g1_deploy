#!/usr/bin/env python3
"""Search training-free reference scales for a smaller, consistent Slow Walk.

The experiment is MuJoCo-only. Every cell keeps official Slow Walk 0.4,
corrected forward axes, all-six missing-axis masks, and direct decoder/motor
output. Only the encoder-facing gait-reference scale changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sonic.common.environment import experiment_environment

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "g1_deploy/sonic/scripts/sim2sim_sonic.py"
PYTHON = Path("/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python")
DEFAULT_OUTPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/20260818_small_consistent_gait"
)
DEFAULT_SEEDS = (1234, 4321, 2026)

ALL_MASKS = (
    "--mask-missing-reference",
    "--mask-missing-action-history",
    "--missing-adaptation-scope", "all",
)

CANDIDATES: dict[str, tuple[str, ...]] = {
    "hip100": ALL_MASKS,
    "hip085": (*ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.85"),
    "hip080": (*ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.80"),
    "sag085": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.85"),
    "sag080": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.80"),
    "sag075": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.75"),
    "hip085_knee085": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.85",
        "--reference-joint-scale", "knee=0.85",
    ),
    "knee085": (*ALL_MASKS, "--reference-joint-scale", "knee=0.85"),
    "leg085": (*ALL_MASKS, "--reference-leg-scale", "0.85"),
    "hip085_root090": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.85",
        "--reference-root-height-scale", "0.90",
    ),
    "sag085_root090": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.85",
        "--reference-root-height-scale", "0.90",
    ),
    "sag070": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.70"),
    "legs090": (*ALL_MASKS, "--reference-leg-scale", "0.90"),
    "hip080_knee090_ank090": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.80",
        "--reference-joint-scale", "knee=0.90",
        "--reference-joint-scale", "ankle_pitch=0.90",
    ),
    "hip075_knee085_ank090": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.75",
        "--reference-joint-scale", "knee=0.85",
        "--reference-joint-scale", "ankle_pitch=0.90",
    ),
    "sag065": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.65"),
    "sag060": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.60"),
    "sag055": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.55"),
    "sag050": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.50"),
    "sag045": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.45"),
    "sag040": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.40"),
    "sag038": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.38"),
    "sag035": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.35"),
    "sag030": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.30"),
    "sag025": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.25"),
    "sag020": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.20"),
    "hip050": (*ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.50"),
    "hip040": (*ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.40"),
    "hip035": (*ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.35"),
    "hipank040": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.40",
        "--reference-joint-scale", "ankle_pitch=0.40",
    ),
    "hipknee040": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.40",
        "--reference-joint-scale", "knee=0.40",
    ),
    "hipank035": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.35",
        "--reference-joint-scale", "ankle_pitch=0.35",
    ),
    "hipknee035": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.35",
        "--reference-joint-scale", "knee=0.35",
    ),
}




def _condition_name(knee_offset_deg: float, ankle_gain: float) -> str:
    sign = "p" if knee_offset_deg >= 0 else "m"
    magnitude = f"{abs(knee_offset_deg):g}".replace(".", "p")
    knee = (
        "healthy" if knee_offset_deg == 0.0
        else f"left_knee_{sign}{magnitude}deg"
    )
    if abs(ankle_gain - 1.0) <= 1e-12:
        return knee
    gain_tag = f"{ankle_gain:g}".replace(".", "p")
    if knee == "healthy":
        return f"ankle_g{gain_tag}"
    return f"{knee}_ankle_g{gain_tag}"


def _command(
    candidate: str,
    condition: str,
    knee_offset_deg: float,
    ankle_gain: float,
    seed: int,
    steps: int,
    axes: str,
    record: Path,
    planner_speed: float,
) -> list[str]:
    condition_flags: list[str] = []
    if knee_offset_deg != 0.0:
        condition_flags.extend(
            ("--joint-encoder-offset-deg", f"left_knee={knee_offset_deg:g}")
        )
    if abs(ankle_gain - 1.0) > 1e-12:
        condition_flags.extend(
            ("--plant-gain-scale", f"left_ankle_pitch={ankle_gain:g}")
        )
    speed_tag = f"{planner_speed:g}".replace(".", "p")
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
        "--planner-speed", f"{planner_speed:g}",
        "--planner-seed", str(seed),
        "--capture-at-step", "100",
        "--steps", str(steps),
        "--trial-label",
        f"original-release__{candidate}__{condition}__p{speed_tag}__seed{seed}",
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
        help="signed left-knee encoder biases; measured=true+offset",
    )
    parser.add_argument(
        "--ankle-gains",
        nargs="+",
        type=float,
        default=[1.0],
        help="left ankle-pitch plant gains; 1.0 is the healthy ankle",
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--axes", default="0,1,0")
    parser.add_argument("--planner-speed", type=float, default=0.4)
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
    if any(not 0.2 <= value <= 1.5 for value in args.ankle_gains):
        raise SystemExit("--ankle-gains values must be within [0.2, 1.5]")
    if not 0.0 < args.planner_speed <= 1.5:
        raise SystemExit("--planner-speed must be in (0, 1.5]")
    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")

    speed_tag = f"{args.planner_speed:g}".replace(".", "p")
    plan = []
    for seed in args.seeds:
        for candidate in args.candidates:
            for knee_offset_deg in args.knee_offsets_deg:
                for ankle_gain in args.ankle_gains:
                    condition = _condition_name(knee_offset_deg, ankle_gain)
                    record = args.output / candidate / condition / f"seed{seed}"
                    if abs(args.planner_speed - 0.4) > 1e-9:
                        record = (
                            args.output / candidate / f"p{speed_tag}"
                            / condition / f"seed{seed}"
                        )
                    plan.append({
                        "candidate": candidate,
                        "condition": condition,
                        "knee_offset_deg": knee_offset_deg,
                        "ankle_gain": ankle_gain,
                        "planner_speed": args.planner_speed,
                        "seed": seed,
                        "record": str(record),
                        "command": _command(
                            candidate, condition, knee_offset_deg,
                            ankle_gain, seed, args.steps, args.axes,
                            record, args.planner_speed,
                        ),
                    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"PLAN: {len(plan)} trials, candidates={','.join(args.candidates)}, "
        f"seeds={','.join(map(str, args.seeds))}, "
        f"knee_offsets_deg={','.join(map(str, args.knee_offsets_deg))}, "
        f"ankle_gains={','.join(map(str, args.ankle_gains))}, "
        f"planner_speed={args.planner_speed:g}, steps={args.steps}",
        flush=True,
    )
    if args.dry_run:
        return

    env = experiment_environment(ROOT)
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
