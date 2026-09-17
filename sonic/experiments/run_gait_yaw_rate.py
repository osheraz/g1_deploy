#!/usr/bin/env python3
"""MuJoCo sweep: gait scale × command × teleop, plus unlocked-yaw A/B.

Official Slow Walk 0.4 stays fixed. The measured quantity is whether stick
yaw rate follows the smallest sagittal reference scale, and whether
walk-plus-turn stays a walk instead of jittering.
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
DEFAULT_OUTPUT = Path("/data/clhsieh/logs/e_skin/experiments/20260818_gait_yaw_rate")
DEFAULT_SEEDS = (1234, 4321)

ALL_MASKS = (
    "--mask-missing-reference",
    "--mask-missing-action-history",
    "--missing-adaptation-scope", "all",
)

SCALES: dict[str, tuple[str, ...]] = {
    "hip100": ALL_MASKS,
    "hip085": (*ALL_MASKS, "--reference-joint-scale", "hip_pitch=0.85"),
    "sag085": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.85"),
    "sag050": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.50"),
    "sag040": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.40"),
    "sag035": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.35"),
}

COMMANDS: dict[str, str] = {
    "forward": "0,1,0",
    "turn_cw": "0,0,1",
    "turn_ccw": "0,0,-1",
    "fwd_cw": "0,1,1",
    "fwd_ccw": "0,1,-1",
}

TELEOPS = ("native-gamepad", "joystick")


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "research/repos/GR00T-WholeBodyControl")
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
    scale: str,
    command: str,
    teleop: str,
    seed: int,
    steps: int,
    record: Path,
    yaw_override: float | None,
) -> list[str]:
    argv = [
        str(PYTHON), str(SIM),
        "--config", str(ROOT / "g1_deploy/configs/g1_23dof_sonic.yaml"),
        "--model-profile", "original-release",
        "--mjcf", str(ROOT / "sonic_cpp/g1/scene_23dof.xml"),
        "--pico", "joystick",
        "--planner",
        "--joystick-mode", "slow_walk",
        f"--axes={COMMANDS[command]}",
        "--axes-from-step", "250",
        "--axes-until", str(steps - 250),
        "--planner-seed", str(seed),
        "--capture-at-step", "100",
        "--steps", str(steps),
        "--trial-label",
        (
            f"original-release__{scale}__{command}__{teleop}__"
            f"yaw{'auto' if yaw_override is None else f'{yaw_override:g}'.replace('.', 'p')}"
            f"__seed{seed}"
        ),
        "--record-dir", str(record),
        *SCALES[scale],
    ]
    if teleop == "native-gamepad":
        argv.extend(("--native-gamepad", "--planner-speed", "0.4"))
    if yaw_override is not None:
        argv.extend(("--yaw-command-scale", f"{yaw_override:g}"))
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument("--commands", nargs="+", choices=tuple(COMMANDS), default=list(COMMANDS))
    parser.add_argument("--teleops", nargs="+", choices=TELEOPS, default=["native-gamepad"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--include-unlocked", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1000:
        raise SystemExit("--steps must be at least 1000")
    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")

    plan = []
    for seed in args.seeds:
        for teleop in args.teleops:
            for scale in args.scales:
                for command in args.commands:
                    record = (
                        args.output / teleop / scale / command / "yaw_auto" / f"seed{seed}"
                    )
                    plan.append({
                        "scale": scale,
                        "command": command,
                        "teleop": teleop,
                        "yaw_override": None,
                        "seed": seed,
                        "record": str(record),
                        "command_line": _command(
                            scale=scale, command=command, teleop=teleop,
                            seed=seed, steps=args.steps, record=record,
                            yaw_override=None,
                        ),
                    })
    if args.include_unlocked:
        for seed in args.seeds:
            for scale in [name for name in ("sag040", "sag035") if name in args.scales]:
                for command in [name for name in ("fwd_cw", "fwd_ccw") if name in args.commands]:
                    record = (
                        args.output / "native-gamepad" / scale / command
                        / "yaw_1p0" / f"seed{seed}"
                    )
                    plan.append({
                        "scale": scale,
                        "command": command,
                        "teleop": "native-gamepad",
                        "yaw_override": 1.0,
                        "seed": seed,
                        "record": str(record),
                        "command_line": _command(
                            scale=scale, command=command, teleop="native-gamepad",
                            seed=seed, steps=args.steps, record=record,
                            yaw_override=1.0,
                        ),
                    })

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"PLAN: {len(plan)} trials, scales={','.join(args.scales)}, "
        f"commands={','.join(args.commands)}, teleops={','.join(args.teleops)}, "
        f"seeds={','.join(map(str, args.seeds))}, "
        f"unlocked={args.include_unlocked}",
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
                f"[{index}/{len(plan)}] skip complete {row['scale']} "
                f"{row['command']} {row['teleop']} seed={row['seed']}"
            )
            continue
        print(
            f"[{index}/{len(plan)}] {row['scale']} {row['command']} "
            f"{row['teleop']} yaw={row['yaw_override'] or 'auto'} "
            f"seed={row['seed']}",
            flush=True,
        )
        record.mkdir(parents=True, exist_ok=True)
        with (record / "console.log").open("w") as stream:
            result = subprocess.run(
                row["command_line"], cwd=ROOT / "g1_deploy", env=env,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            tail = (record / "console.log").read_text(errors="replace").splitlines()[-40:]
            print("\n".join(tail), file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, row["command_line"])


if __name__ == "__main__":
    main()
