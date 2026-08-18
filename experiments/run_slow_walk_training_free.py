#!/usr/bin/env python3
"""Run deterministic, motor-free SONIC slow-walk experiments in MuJoCo.

Each trial starts with five seconds of IDLE, requests a support-qualified stand
capture, walks forward for twenty seconds, and returns to IDLE for five
seconds. Existing complete recordings are skipped unless ``--force`` is used.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "g1_deploy/scripts/sim2sim_sonic.py"
PYTHON = Path(
    "/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python"
)
TRT = Path("/data/clhsieh/software/nvidia/tensorrt/10.13.3.9")
DEFAULT_OUTPUT = Path(
    "/data/clhsieh/logs/e_skin/experiments/"
    "20260813_training_free_23dof_slow_walk"
)


def _speed_name(speed: float) -> str:
    return f"{speed:.2f}".replace(".", "p")


def _trial(
    name: str,
    speed: float,
    *extra: str,
    embodiment: str = "g1_23dof",
    axes: str = "0,1,0",
) -> dict:
    return {
        "name": name,
        "speed": speed,
        "extra": list(extra),
        "embodiment": embodiment,
        "axes": axes,
    }


def _stable_trial(name: str, *extra: str, speed: float = 0.40) -> dict:
    """Original Release 23-D observation adaptation used by shaping sweeps."""
    return _trial(
        name,
        speed,
        "--mask-missing-reference",
        "--mask-missing-action-history",
        *extra,
    )


def _finalist_trial(
    name: str,
    *extra: str,
    speed: float = 0.40,
    axes: str = "0,1,0",
    yaw_ema: bool = False,
) -> dict:
    shaping = [
        "--motor-ema", "ankle_pitch=0.95",
        "--motor-scale", "ankle_roll=0.95",
    ]
    if yaw_ema:
        shaping.extend(("--motor-ema", "hip_yaw=0.95"))
    return _trial(
        name,
        speed,
        "--mask-missing-reference",
        "--mask-missing-action-history",
        *shaping,
        *extra,
        axes=axes,
    )


def _stages() -> dict[str, list[dict]]:
    speeds = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    observer = []
    for speed in (0.20, 0.30, 0.40):
        tag = _speed_name(speed)
        observer.extend([
            _trial(f"zero_s{tag}", speed, "--reduced-proprioception", "zero"),
            _trial(f"reference_s{tag}", speed, "--reduced-proprioception", "reference"),
            _trial(
                f"shadow06_s{tag}", speed,
                "--reduced-proprioception", "shadow",
                "--shadow-settling-time", "0.06",
            ),
            _trial(
                f"shadow12_s{tag}", speed,
                "--reduced-proprioception", "shadow",
                "--shadow-settling-time", "0.12",
            ),
            _trial(
                f"shadow24_s{tag}", speed,
                "--reduced-proprioception", "shadow",
                "--shadow-settling-time", "0.24",
            ),
        ])
    semantic = []
    for speed in (0.20, 0.30, 0.40):
        tag = _speed_name(speed)
        semantic.extend([
            _trial(f"native_s{tag}", speed),
            _trial(
                f"mask_ref_s{tag}", speed,
                "--mask-missing-reference",
            ),
            _trial(
                f"mask_history_s{tag}", speed,
                "--mask-missing-action-history",
            ),
            _trial(
                f"mask_both_s{tag}", speed,
                "--mask-missing-reference",
                "--mask-missing-action-history",
            ),
        ])
    smoothing = []
    for speed in (0.20, 0.30, 0.40):
        tag = _speed_name(speed)
        for alpha in (0.85, 0.70, 0.50):
            smoothing.append(_trial(
                f"ema{int(alpha * 100):02d}_s{tag}", speed,
                "--action-ema", f"{alpha:g}",
            ))
    winner_smoothing = []
    for alpha in (1.0, 0.90, 0.80, 0.70, 0.50):
        winner_smoothing.append(_trial(
            f"mask_both_ema{int(alpha * 100):03d}_s0p40",
            0.40,
            "--mask-missing-reference",
            "--mask-missing-action-history",
            "--action-ema", f"{alpha:g}",
        ))
    scope = [
        _trial("native_scope_s0p40", 0.40),
    ]
    for mask in ("reference", "history", "both"):
        flags = []
        if mask in ("reference", "both"):
            flags.append("--mask-missing-reference")
        if mask in ("history", "both"):
            flags.append("--mask-missing-action-history")
        for scope_name in ("waist", "wrist", "all"):
            scope.append(_trial(
                f"mask_{mask}_{scope_name}_s0p40",
                0.40,
                *flags,
                "--missing-adaptation-scope", scope_name,
            ))
    threshold = []
    for speed in (0.32, 0.34, 0.36, 0.38, 0.40):
        tag = _speed_name(speed)
        threshold.extend([
            _trial(f"native_s{tag}", speed),
            _trial(
                f"mask_both_s{tag}", speed,
                "--mask-missing-reference",
                "--mask-missing-action-history",
            ),
        ])
    robustness = []
    for friction in (0.30, 0.40, 0.60, 1.00):
        tag = _speed_name(friction)
        robustness.extend([
            _trial(
                f"native_mu{tag}_s0p40", 0.40,
                "--ground-friction", f"{friction:g}",
            ),
            _trial(
                f"mask_both_mu{tag}_s0p40", 0.40,
                "--ground-friction", f"{friction:g}",
                "--mask-missing-reference",
                "--mask-missing-action-history",
            ),
        ])
    scale = [
        _stable_trial("baseline_s0p40"),
        *[
            _stable_trial(
                f"leg_scale_{str(value).replace('.', 'p')}_s0p40",
                "--motor-scale", f"legs={value:g}",
            )
            for value in (0.75, 0.85, 0.90, 0.95, 1.05, 1.10)
        ],
        *[
            _stable_trial(
                f"ankle_pitch_scale_{str(value).replace('.', 'p')}_s0p40",
                "--motor-scale", f"ankle_pitch={value:g}",
            )
            for value in (0.70, 0.80, 0.90, 1.10)
        ],
    ]
    single_shaping = [
        *[
            _stable_trial(
                f"ankle_pitch_ema_{str(value).replace('.', 'p')}_s0p40",
                "--motor-ema", f"ankle_pitch={value:g}",
            )
            for value in (0.95, 0.90, 0.80, 0.60)
        ],
        *[
            _stable_trial(
                f"ankle_pitch_rate_{value:g}_s0p40".replace(".", "p"),
                "--motor-rate", f"ankle_pitch={value:g}",
            )
            for value in (4.0, 6.0, 8.0, 10.0, 12.0)
        ],
        *[
            _stable_trial(
                f"ankle_pitch_clip_{value:g}_s0p40".replace(".", "p"),
                "--motor-clip", f"ankle_pitch={value:g}",
            )
            for value in (1.75, 2.25, 2.75, 3.25)
        ],
        _stable_trial(
            "executed_history_s0p40",
            "--action-history-source", "executed",
        ),
    ]
    group_refinement = [
        _stable_trial(
            "ankle_ema095_baseline_s0p40",
            "--motor-ema", "ankle_pitch=0.95",
        ),
    ]
    for group in (
        "hip_pitch", "hip_roll", "hip_yaw", "knee",
        "ankle_pitch", "ankle_roll",
    ):
        for value in (0.90, 0.95, 1.05):
            group_refinement.append(_stable_trial(
                f"ankle_ema095_{group}_scale_"
                f"{str(value).replace('.', 'p')}_s0p40",
                "--motor-ema", "ankle_pitch=0.95",
                "--motor-scale", f"{group}={value:g}",
            ))
    for group in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll"):
        group_refinement.append(_stable_trial(
            f"ankle_ema095_{group}_ema095_s0p40",
            "--motor-ema", "ankle_pitch=0.95",
            "--motor-ema", f"{group}=0.95",
        ))
    group_refinement.append(_stable_trial(
        "ankle_ema095_executed_history_s0p40",
        "--motor-ema", "ankle_pitch=0.95",
        "--action-history-source", "executed",
    ))
    fine_tune = []
    for pitch_ema in (0.93, 0.95, 0.97, 0.99):
        for roll_scale in (0.92, 0.95, 0.98, 1.00):
            fine_tune.append(_stable_trial(
                "pitch_ema_"
                f"{str(pitch_ema).replace('.', 'p')}_roll_scale_"
                f"{str(roll_scale).replace('.', 'p')}_s0p40",
                "--motor-ema", f"ankle_pitch={pitch_ema:g}",
                "--motor-scale", f"ankle_roll={roll_scale:g}",
            ))
    for roll_scale in (0.92, 0.95, 0.98, 1.00):
        fine_tune.append(_stable_trial(
            "pitch_ema_0p95_yaw_ema_0p95_roll_scale_"
            f"{str(roll_scale).replace('.', 'p')}_s0p40",
            "--motor-ema", "ankle_pitch=0.95",
            "--motor-ema", "hip_yaw=0.95",
            "--motor-scale", f"ankle_roll={roll_scale:g}",
        ))
    finalist_friction = []
    for friction in (0.30, 0.40, 0.60, 1.00):
        tag = _speed_name(friction)
        finalist_friction.extend((
            _finalist_trial(
                f"simple_mu{tag}_s0p40",
                "--ground-friction", f"{friction:g}",
            ),
            _finalist_trial(
                f"yaw_ema_mu{tag}_s0p40",
                "--ground-friction", f"{friction:g}",
                yaw_ema=True,
            ),
        ))
    finalist_speed = []
    for speed in (0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.50):
        tag = _speed_name(speed)
        finalist_speed.extend((
            _finalist_trial(f"simple_s{tag}", speed=speed),
            _finalist_trial(f"yaw_ema_s{tag}", speed=speed, yaw_ema=True),
        ))
    for speed in (0.32, 0.34, 0.36, 0.38, 0.50):
        finalist_speed.append(_trial(
            f"native29_mu1_s{_speed_name(speed)}",
            speed,
            "--ground-friction", "1.0",
            embodiment="g1_29dof",
        ))
    maneuvers = {
        "forward_turn_left": "0,1,-0.5",
        "forward_turn_right": "0,1,0.5",
        "turn_left": "0,0,-0.5",
        "turn_right": "0,0,0.5",
        "strafe_left": "-0.5,0,0",
        "strafe_right": "0.5,0,0",
    }
    finalist_maneuvers = []
    for name, axes in maneuvers.items():
        finalist_maneuvers.extend((
            _finalist_trial(f"simple_{name}", axes=axes),
            _finalist_trial(f"yaw_ema_{name}", axes=axes, yaw_ema=True),
            _trial(
                f"native29_mu1_{name}", 0.40,
                "--ground-friction", "1.0",
                embodiment="g1_29dof", axes=axes,
            ),
        ))
    return {
        "speed": [
            _trial(f"zero_s{_speed_name(speed)}", speed)
            for speed in speeds
        ],
        "observer": observer,
        "semantic": semantic,
        "smoothing": smoothing,
        "winner_smoothing": winner_smoothing,
        "scope": scope,
        "threshold": threshold,
        "robustness": robustness,
        "scale": scale,
        "single_shaping": single_shaping,
        "group_refinement": group_refinement,
        "fine_tune": fine_tune,
        "finalist_friction": finalist_friction,
        "finalist_speed": finalist_speed,
        "finalist_maneuvers": finalist_maneuvers,
        "native29": [
            _trial(
                f"native29_s{_speed_name(speed)}", speed,
                "--ground-friction", "1.0",
                embodiment="g1_29dof",
            )
            for speed in (0.20, 0.30, 0.40)
        ],
        "native29_robustness": [
            _trial(
                f"native29_mu{_speed_name(friction)}_s0p40",
                0.40,
                "--ground-friction", f"{friction:g}",
                embodiment="g1_29dof",
            )
            for friction in (0.30, 0.40, 0.60, 1.00)
        ],
    }


def main() -> None:
    stages = _stages()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(stages), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "others/GR00T-WholeBodyControl")
    env["PYTHONNOUSERSITE"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        str(TRT / "lib"),
        "/usr/local/cuda/lib64",
        "/usr/local/lib",
        env.get("LD_LIBRARY_PATH", ""),
    ]))

    for index, trial in enumerate(stages[args.stage], 1):
        record = args.output / args.stage / trial["name"]
        complete = (record / "state.npz").is_file() and (record / "meta.json").is_file()
        if complete and not args.force:
            print(f"[{index}/{len(stages[args.stage])}] skip complete {record}")
            continue
        embodiment = trial["embodiment"]
        config = ROOT / f"g1_deploy/configs/{embodiment}_sonic.yaml"
        mjcf = ROOT / f"sonic_cpp/g1/scene_{'23dof' if embodiment == 'g1_23dof' else '29dof'}.xml"
        command = [
            str(PYTHON), str(SIM),
            "--config", str(config),
            "--model-profile", "original-release",
            "--mjcf", str(mjcf),
            "--pico", "joystick",
            "--planner",
            "--native-gamepad",
            "--joystick-mode", "slow_walk",
            f"--axes={trial['axes']}",
            "--axes-from-step", "250",
            "--axes-until", "1250",
            "--planner-speed", f"{trial['speed']:g}",
            "--capture-at-step", "100",
            "--steps", "1500",
            "--trial-label", trial["name"],
            "--record-dir", str(record),
            *trial["extra"],
        ]
        print(
            f"[{index}/{len(stages[args.stage])}] {trial['name']}: "
            f"speed={trial['speed']:g}, embodiment={embodiment}"
        )
        subprocess.run(command, cwd=ROOT / "g1_deploy", env=env, check=True)


if __name__ == "__main__":
    main()
