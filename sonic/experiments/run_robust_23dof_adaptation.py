#!/usr/bin/env python3
"""Deterministic SONIC robustness experiments for 29-to-23 adaptation.

No DDS channel is opened and no robot command is published.  Every trial uses
the real Python planner/tracker stack in closed-loop MuJoCo.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "g1_deploy/sonic/scripts/sim2sim_sonic.py"
PYTHON = Path(
    "/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python"
)
TRT = Path("/data/clhsieh/software/nvidia/tensorrt/10.13.3.9")
DEFAULT_OUTPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/"
    "20260815_robust_23dof_adaptation"
)


def _trial(name: str, *flags: str, embodiment: str = "g1_23dof") -> dict:
    return {"name": name, "flags": list(flags), "embodiment": embodiment}


MILD_FAULT = (
    "--joint-encoder-offset-deg", "left_knee=-5",
    "--plant-gain-scale", "left_ankle_pitch=0.70",
    "--plant-target-delay-ms", "left_ankle_pitch=20",
)
MASK_BOTH = ("--mask-missing-reference", "--mask-missing-action-history")


def _stages() -> dict[str, list[dict]]:
    fault_calibration = [
        _trial("healthy"),
        _trial("left_knee_m5", "--joint-encoder-offset-deg", "left_knee=-5"),
        _trial("left_knee_m10", "--joint-encoder-offset-deg", "left_knee=-10"),
        *[
            _trial(
                f"left_ankle_gain_{int(gain * 100):03d}",
                "--plant-gain-scale", f"left_ankle_pitch={gain:g}",
            )
            for gain in (0.85, 0.70, 0.55)
        ],
        *[
            _trial(
                f"left_ankle_delay_{delay:03d}ms",
                "--plant-target-delay-ms", f"left_ankle_pitch={delay:g}",
            )
            for delay in (20, 40, 60)
        ],
        _trial("combined_m5_g70_d0", *MILD_FAULT[:-2]),
        _trial("combined_m5_g70_d20", *MILD_FAULT),
        _trial(
            "combined_m5_g70_d40",
            *MILD_FAULT[:-2],
            "--plant-target-delay-ms", "left_ankle_pitch=40",
        ),
        _trial(
            "combined_m5_g70_d20_mu040",
            *MILD_FAULT, "--ground-friction", "0.4",
        ),
    ]

    reference_sweep = []
    for condition, fault in (
        ("healthy", ()),
        ("fault", MILD_FAULT),
    ):
        for masks, mask_flags in (("parity", ()), ("masks", MASK_BOTH)):
            for scale in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
                tag = f"{scale:.2f}".replace(".", "p")
                reference_sweep.append(_trial(
                    f"{condition}_{masks}_leg{tag}",
                    *mask_flags,
                    "--reference-leg-scale", f"{scale:g}",
                    *fault,
                ))
    for scale in (0.60, 0.70, 0.80, 0.90, 1.00):
        tag = f"{scale:.2f}".replace(".", "p")
        reference_sweep.append(_trial(
            f"lowfric_masks_leg{tag}",
            *MASK_BOTH,
            "--reference-leg-scale", f"{scale:g}",
            "--ground-friction", "0.4",
        ))
    for scale in (0.60, 0.80, 1.00):
        tag = f"{scale:.2f}".replace(".", "p")
        reference_sweep.append(_trial(
            f"native29_leg{tag}",
            "--reference-leg-scale", f"{scale:g}",
            embodiment="g1_29dof",
        ))

    root_height_sweep = []
    for condition, fault in (("healthy", ()), ("fault", MILD_FAULT)):
        for leg_scale in (0.60, 0.70, 0.80):
            for root_scale in (0.50, 0.75, 1.00):
                lt = f"{leg_scale:.2f}".replace(".", "p")
                rt = f"{root_scale:.2f}".replace(".", "p")
                root_height_sweep.append(_trial(
                    f"{condition}_leg{lt}_root{rt}",
                    *MASK_BOTH,
                    "--reference-leg-scale", f"{leg_scale:g}",
                    "--reference-root-height-scale", f"{root_scale:g}",
                    *fault,
                ))
    sagittal_sweep = []
    for condition, fault in (
        ("healthy", ()),
        ("fault", MILD_FAULT),
        ("lowfric", ("--ground-friction", "0.4")),
    ):
        for scale in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
            tag = f"{scale:.2f}".replace(".", "p")
            sagittal_sweep.append(_trial(
                f"{condition}_sagittal{tag}",
                *MASK_BOTH,
                "--reference-joint-scale", f"sagittal={scale:g}",
                *fault,
            ))
    for condition, fault in (("healthy", ()), ("fault", MILD_FAULT)):
        sagittal_sweep.extend([
            _trial(
                f"{condition}_hippitch0p70",
                *MASK_BOTH,
                "--reference-joint-scale", "hip_pitch=0.70",
                *fault,
            ),
            _trial(
                f"{condition}_hip070_knee085_ankle090",
                *MASK_BOTH,
                "--reference-joint-scale", "hip_pitch=0.70",
                "--reference-joint-scale", "knee=0.85",
                "--reference-joint-scale", "ankle_pitch=0.90",
                *fault,
            ),
        ])
    finalists = []
    conditions = (
        ("healthy", ()),
        ("mild", MILD_FAULT),
        ("delay40", ()),
        ("knee_m10", ("--joint-encoder-offset-deg", "left_knee=-10")),
        ("ankle_g55", ("--plant-gain-scale", "left_ankle_pitch=0.55")),
        ("lowfric", ("--ground-friction", "0.4")),
        (
            "mild_lowfric",
            (*MILD_FAULT, "--ground-friction", "0.4"),
        ),
    )
    # Construct the stronger-delay case separately to keep each CLI pair
    # explicit and reviewable.
    conditions = tuple(
        (
            name,
            (
                "--joint-encoder-offset-deg", "left_knee=-5",
                "--plant-gain-scale", "left_ankle_pitch=0.70",
                "--plant-target-delay-ms", "left_ankle_pitch=40",
            ) if name == "delay40" else flags,
        )
        for name, flags in conditions
    )
    candidates = [
        ("baseline", MASK_BOTH),
        *[
            (
                f"hip{scale:.2f}".replace(".", "p"),
                (*MASK_BOTH, "--reference-joint-scale", f"hip_pitch={scale:g}"),
            )
            for scale in (0.70, 0.75, 0.80, 0.85, 0.90)
        ],
        (
            "parity_hip0p75",
            ("--reference-joint-scale", "hip_pitch=0.75"),
        ),
        (
            "allleg0p80",
            (*MASK_BOTH, "--reference-leg-scale", "0.80"),
        ),
        (
            "hip0p75_root0p75",
            (
                *MASK_BOTH,
                "--reference-joint-scale", "hip_pitch=0.75",
                "--reference-root-height-scale", "0.75",
            ),
        ),
    ]
    for condition, fault in conditions:
        for candidate, adaptation in candidates:
            finalists.append(_trial(
                f"{condition}_{candidate}", *adaptation, *fault
            ))
    return {
        "fault-calibration": fault_calibration,
        "reference-sweep": reference_sweep,
        "root-height-sweep": root_height_sweep,
        "sagittal-sweep": sagittal_sweep,
        "finalists": finalists,
    }


def main() -> None:
    stages = _stages()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=stages, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "research/repos/GR00T-WholeBodyControl")
    env["PYTHONNOUSERSITE"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        str(TRT / "lib"),
        "/usr/local/cuda/lib64",
        "/usr/local/lib",
        env.get("LD_LIBRARY_PATH", ""),
    ]))

    trials = stages[args.stage]
    for index, trial in enumerate(trials, 1):
        record = args.output / args.stage / trial["name"]
        complete = (
            (record / "state.npz").is_file()
            and (record / "meta.json").is_file()
        )
        if complete and not args.force:
            print(f"[{index}/{len(trials)}] skip complete {trial['name']}")
            continue
        embodiment = trial["embodiment"]
        suffix = "23dof" if embodiment == "g1_23dof" else "29dof"
        command = [
            str(PYTHON), str(SIM),
            "--config", str(ROOT / f"g1_deploy/configs/{embodiment}_sonic.yaml"),
            "--model-profile", "original-release",
            "--mjcf", str(ROOT / f"sonic_cpp/g1/scene_{suffix}.xml"),
            "--pico", "joystick",
            "--planner",
            "--native-gamepad",
            "--joystick-mode", "slow_walk",
            "--axes=0,1,0",
            "--axes-from-step", "250",
            "--axes-until", "1250",
            "--planner-speed", "0.4",
            "--capture-at-step", "100",
            "--steps", "1500",
            "--trial-label", trial["name"],
            "--record-dir", str(record),
            *trial["flags"],
        ]
        print(
            f"[{index}/{len(trials)}] {trial['name']} "
            f"({embodiment})",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT / "g1_deploy", env=env, check=True)


if __name__ == "__main__":
    main()
