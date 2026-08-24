#!/usr/bin/env python3
"""Cross-model, fault-injected SONIC 29-to-23 adaptation experiments.

Every run is headless MuJoCo with the real V2 planner and selected atomic
encoder/decoder profile. No DDS participant or physical motor publisher is
created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "g1_deploy/scripts/sim2sim_sonic.py"
PYTHON = Path("/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python")
TRT = Path("/data/clhsieh/software/nvidia/tensorrt/10.13.3.9")
DEFAULT_OUTPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/"
    "20260816_cross_model_23dof_adaptation"
)
MODELS = ("low-latency", "original-release", "sonic-v1.1")
DEFAULT_SEEDS = (1234, 4321, 2026)

MILD_FAULT = (
    "--joint-encoder-offset-deg", "left_knee=-5",
    "--plant-gain-scale", "left_ankle_pitch=0.70",
    "--plant-target-delay-ms", "left_ankle_pitch=20",
)
MASKS = ("--mask-missing-reference", "--mask-missing-action-history")

CONDITIONS = {
    "healthy": (),
    "knee_m5": ("--joint-encoder-offset-deg", "left_knee=-5"),
    "knee_m10": ("--joint-encoder-offset-deg", "left_knee=-10"),
    "knee_p5": ("--joint-encoder-offset-deg", "left_knee=5"),
    "knee_p10": ("--joint-encoder-offset-deg", "left_knee=10"),
    "ankle_g55": ("--plant-gain-scale", "left_ankle_pitch=0.55"),
    "delay40": (
        "--joint-encoder-offset-deg", "left_knee=-5",
        "--plant-gain-scale", "left_ankle_pitch=0.70",
        "--plant-target-delay-ms", "left_ankle_pitch=40",
    ),
    "mild": MILD_FAULT,
    "lowfric": ("--ground-friction", "0.4"),
    "mild_lowfric": (*MILD_FAULT, "--ground-friction", "0.4"),
}

CANDIDATES = {
    "parity": (),
    "refmask": ("--mask-missing-reference",),
    "histmask": ("--mask-missing-action-history",),
    "masks": MASKS,
    "masks_hip065": (*MASKS, "--reference-joint-scale", "hip_pitch=0.65"),
    "masks_hip070": (*MASKS, "--reference-joint-scale", "hip_pitch=0.70"),
    "masks_hip075": (*MASKS, "--reference-joint-scale", "hip_pitch=0.75"),
    "masks_hip080": (*MASKS, "--reference-joint-scale", "hip_pitch=0.80"),
    "masks_hip085": (*MASKS, "--reference-joint-scale", "hip_pitch=0.85"),
    "masks_hip090": (*MASKS, "--reference-joint-scale", "hip_pitch=0.90"),
    "masks_hip095": (*MASKS, "--reference-joint-scale", "hip_pitch=0.95"),
    "masks_knee085": (*MASKS, "--reference-joint-scale", "knee=0.85"),
    "masks_knee110": (*MASKS, "--reference-joint-scale", "knee=1.10"),
    "masks_ankle090": (*MASKS, "--reference-joint-scale", "ankle_pitch=0.90"),
    "masks_hip080_knee090": (
        *MASKS,
        "--reference-joint-scale", "hip_pitch=0.80",
        "--reference-joint-scale", "knee=0.90",
    ),
    "masks_hip080_knee110": (
        *MASKS,
        "--reference-joint-scale", "hip_pitch=0.80",
        "--reference-joint-scale", "knee=1.10",
    ),
    "masks_hip080_ankle090": (
        *MASKS,
        "--reference-joint-scale", "hip_pitch=0.80",
        "--reference-joint-scale", "ankle_pitch=0.90",
    ),
    "masks_sagittal085": (
        *MASKS, "--reference-joint-scale", "sagittal=0.85"
    ),
    "parity_hip080": ("--reference-joint-scale", "hip_pitch=0.80"),
    "waist_masks_hip080": (
        *MASKS, "--missing-adaptation-scope", "waist",
        "--reference-joint-scale", "hip_pitch=0.80",
    ),
    "wrist_masks_hip080": (
        *MASKS, "--missing-adaptation-scope", "wrist",
        "--reference-joint-scale", "hip_pitch=0.80",
    ),
    # Targeted follow-up around the Low-Latency screening winner. These isolate
    # which wrist-channel mask and hip amplitude actually produce the gain.
    "wrist_masks": (*MASKS, "--missing-adaptation-scope", "wrist"),
    "wrist_masks_hip085": (
        *MASKS, "--missing-adaptation-scope", "wrist",
        "--reference-joint-scale", "hip_pitch=0.85",
    ),
    "wrist_masks_hip090": (
        *MASKS, "--missing-adaptation-scope", "wrist",
        "--reference-joint-scale", "hip_pitch=0.90",
    ),
    "wrist_masks_hip095": (
        *MASKS, "--missing-adaptation-scope", "wrist",
        "--reference-joint-scale", "hip_pitch=0.95",
    ),
    "wrist_refmask_hip080": (
        "--mask-missing-reference", "--missing-adaptation-scope", "wrist",
        "--reference-joint-scale", "hip_pitch=0.80",
    ),
    "wrist_histmask_hip080": (
        "--mask-missing-action-history", "--missing-adaptation-scope", "wrist",
        "--reference-joint-scale", "hip_pitch=0.80",
    ),
}

BASELINE_CANDIDATES = ("parity", "masks", "masks_hip080")
BASELINE_CONDITIONS = (
    "healthy", "knee_m10", "knee_p10", "mild", "lowfric", "mild_lowfric"
)
SEARCH_CANDIDATES = tuple(CANDIDATES)
SEARCH_CONDITIONS = ("healthy", "knee_m10", "knee_p10", "mild", "lowfric")


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
    *,
    model: str,
    embodiment: str,
    candidate: str,
    condition: str,
    seed: int,
    record: Path,
    steps: int,
) -> list[str]:
    suffix = "23dof" if embodiment == "g1_23dof" else "29dof"
    flags = () if candidate == "native29" else CANDIDATES[candidate]
    condition_flags = CONDITIONS[condition]
    return [
        str(PYTHON), str(SIM),
        "--config", str(ROOT / f"g1_deploy/configs/{embodiment}_sonic.yaml"),
        "--model-profile", model,
        "--mjcf", str(ROOT / f"sonic_cpp/g1/scene_{suffix}.xml"),
        "--pico", "joystick",
        "--planner",
        "--native-gamepad",
        "--joystick-mode", "slow_walk",
        # GamepadManagerTeleop uses positive left-stick Y for forward.  The
        # former negative value requested backward travel while facing +X.
        "--axes=0,1,0",
        "--axes-from-step", "550",
        "--axes-until", str(steps - 350),
        "--planner-speed", "0.4",
        "--planner-seed", str(seed),
        "--capture-at-step", "200",
        "--steps", str(steps),
        "--trial-label", f"{model}__{candidate}__{condition}__seed{seed}",
        "--record-dir", str(record),
        *flags,
        *condition_flags,
    ]


def _plan(args) -> list[dict]:
    rows: list[dict] = []
    if args.stage == "baseline":
        seeds = args.seeds or (1234,)
        for model in args.models:
            for seed in seeds:
                for condition in ("healthy", "lowfric"):
                    rows.append({
                        "model": model, "embodiment": "g1_29dof",
                        "candidate": "native29", "condition": condition,
                        "seed": seed,
                    })
                for candidate in BASELINE_CANDIDATES:
                    for condition in BASELINE_CONDITIONS:
                        rows.append({
                            "model": model, "embodiment": "g1_23dof",
                            "candidate": candidate, "condition": condition,
                            "seed": seed,
                        })
    elif args.stage == "search":
        seeds = args.seeds or (1234,)
        candidates = args.candidates or SEARCH_CANDIDATES
        unknown = sorted(set(candidates) - set(CANDIDATES))
        if unknown:
            raise SystemExit("unknown search candidate(s): " + ", ".join(unknown))
        for model in args.models:
            for seed in seeds:
                for candidate in candidates:
                    for condition in SEARCH_CONDITIONS:
                        rows.append({
                            "model": model, "embodiment": "g1_23dof",
                            "candidate": candidate, "condition": condition,
                            "seed": seed,
                        })
    else:
        if not args.candidates:
            raise SystemExit("--stage finalists requires --candidates")
        seeds = args.seeds or DEFAULT_SEEDS
        for model in args.models:
            for seed in seeds:
                # Retain a matched native-embodiment reference for every seed so
                # finalist scores do not silently fall back to seed 1234.
                for condition in ("healthy", "lowfric"):
                    rows.append({
                        "model": model, "embodiment": "g1_29dof",
                        "candidate": "native29", "condition": condition,
                        "seed": seed,
                    })
                for candidate in args.candidates:
                    if candidate not in CANDIDATES:
                        raise SystemExit(f"unknown finalist candidate: {candidate}")
                    for condition in CONDITIONS:
                        rows.append({
                            "model": model, "embodiment": "g1_23dof",
                            "candidate": candidate, "condition": condition,
                            "seed": seed,
                        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "search", "finalists"), required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1200:
        raise SystemExit("--steps must be at least 1200 for capture, gait, and final IDLE")
    if any(seed < 0 for seed in (args.seeds or ())):
        raise SystemExit("planner seeds must be non-negative")
    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")

    plan = _plan(args)
    records = []
    for row in plan:
        record = (
            args.output / args.stage / row["model"] / row["candidate"]
            / row["condition"] / f"seed{row['seed']}"
        )
        command = _command(**row, record=record, steps=args.steps)
        records.append({**row, "record": str(record), "command": command})
    args.output.mkdir(parents=True, exist_ok=True)
    model_tag = "-".join(args.models)
    selection_tag = ""
    if args.candidates:
        digest = hashlib.sha256("\n".join(args.candidates).encode()).hexdigest()[:8]
        selection_tag = f"_selected-{digest}"
    (args.output / f"plan_{args.stage}_{model_tag}{selection_tag}.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(
        f"PLAN: stage={args.stage} trials={len(records)} "
        f"models={','.join(args.models)} steps={args.steps}",
        flush=True,
    )
    if args.dry_run:
        return

    env = _environment()
    for index, row in enumerate(records, 1):
        record = Path(row["record"])
        complete = (record / "state.npz").is_file() and (record / "meta.json").is_file()
        if complete and not args.force:
            print(f"[{index}/{len(records)}] skip {row['model']} {row['candidate']} {row['condition']} seed={row['seed']}")
            continue
        print(
            f"[{index}/{len(records)}] {row['model']} {row['candidate']} "
            f"{row['condition']} seed={row['seed']}",
            flush=True,
        )
        record.mkdir(parents=True, exist_ok=True)
        console_log = record / "console.log"
        with console_log.open("w") as stream:
            result = subprocess.run(
                row["command"],
                cwd=ROOT / "g1_deploy",
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            tail = console_log.read_text(errors="replace").splitlines()[-40:]
            print("\n".join(tail), file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, row["command"])


if __name__ == "__main__":
    main()
