#!/usr/bin/env python3
"""Search slower-than-0.35 walks that keep swing, then re-test turn and stand.

Official Slow Walk stays 0.4. Collapse at sagittal 0.30 is lost swing time
and clearance, not tracking error. Candidates either preserve ankle/knee
clearance or keep dvel_fut larger than the position scale.
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
DEFAULT_OUTPUT = Path("/data/clhsieh/logs/e_skin/experiments/20260818_slower_gait")
DEFAULT_SEEDS = (1234, 4321)

ALL_MASKS = (
    "--mask-missing-reference",
    "--mask-missing-action-history",
    "--missing-adaptation-scope", "all",
)

MODELS = ("low-latency", "original-release", "sonic-v1.1")

CANDIDATES: dict[str, tuple[str, ...]] = {
    "unscaled": ALL_MASKS,
    "sag035": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.35"),
    "sag030": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.30"),
    "sag028": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.28"),
    "sag025": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.25"),
    "sag030_dvel070": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-dvel-scale", "sagittal=0.70",
    ),
    "sag028_dvel070": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-dvel-scale", "sagittal=0.70",
    ),
    "sag025_dvel080": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-dvel-scale", "sagittal=0.80",
    ),
    "sag030_dvel100": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-dvel-scale", "sagittal=1.00",
    ),
    "hip025_kn040_ank100": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.25",
        "--reference-joint-scale", "knee=0.40",
        "--reference-joint-scale", "ankle_pitch=1.00",
    ),
    "hip028_kn045_ank100": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.28",
        "--reference-joint-scale", "knee=0.45",
        "--reference-joint-scale", "ankle_pitch=1.00",
    ),
    "hip025_kn050_ank100": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.25",
        "--reference-joint-scale", "knee=0.50",
        "--reference-joint-scale", "ankle_pitch=1.00",
    ),
    "hip030_kn040_ank115": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.30",
        "--reference-joint-scale", "knee=0.40",
        "--reference-joint-scale", "ankle_pitch=1.15",
    ),
    "hip025_kn040_ank085": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.25",
        "--reference-joint-scale", "knee=0.40",
        "--reference-joint-scale", "ankle_pitch=0.85",
    ),
    "hip025_kn040_ank100_dvel070": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.25",
        "--reference-joint-scale", "knee=0.40",
        "--reference-joint-scale", "ankle_pitch=1.00",
        "--reference-joint-dvel-scale", "hip_pitch=0.70",
        "--reference-joint-dvel-scale", "knee=0.70",
        "--reference-joint-dvel-scale", "ankle_pitch=1.00",
    ),
    "hip020_kn040_ank110_dvel075": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.20",
        "--reference-joint-scale", "knee=0.40",
        "--reference-joint-scale", "ankle_pitch=1.10",
        "--reference-joint-dvel-scale", "hip_pitch=0.75",
        "--reference-joint-dvel-scale", "knee=0.75",
        "--reference-joint-dvel-scale", "ankle_pitch=1.10",
    ),
    "sag032": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.32"),
    "sag033": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.33"),
    "sag030_kn6": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "knee=6",
    ),
    "sag030_kn8_ank4": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "knee=8",
        "--reference-joint-offset-deg", "ankle_pitch=4",
    ),
    "sag028_kn8": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
    ),
    "sag028_kn10_ank5": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=10",
        "--reference-joint-offset-deg", "ankle_pitch=5",
    ),
    "sag025_kn10_ank6": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-offset-deg", "knee=10",
        "--reference-joint-offset-deg", "ankle_pitch=6",
    ),
    "sag030_ank6": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "ankle_pitch=6",
    ),
    "hip028_kn045_ank100_kn5": (
        *ALL_MASKS,
        "--reference-joint-scale", "hip_pitch=0.28",
        "--reference-joint-scale", "knee=0.45",
        "--reference-joint-scale", "ankle_pitch=1.00",
        "--reference-joint-offset-deg", "knee=5",
    ),
    "sag034": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.34"),
    "sag028_kn8_yaw020": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.20",
    ),
    "sag028_kn8_yaw015": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
    ),
    "sag030_kn6_yaw022": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.22",
    ),
    "sag025_kn12_ank8": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-offset-deg", "knee=12",
        "--reference-joint-offset-deg", "ankle_pitch=8",
    ),
    "sag028_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag030_kn6_w018_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.18",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag028_flex060": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-flex-scale", "knee=0.60",
    ),
    "sag025_flex070": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-flex-scale", "knee=0.70",
    ),
    "sag028_dvel042": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-dvel-scale", "sagittal=0.42",
    ),
    "sag028_kn6_dvel040": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=6",
        "--reference-joint-dvel-scale", "sagittal=0.40",
    ),
    "sag026_flex065_kn5": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.26",
        "--reference-joint-flex-scale", "knee=0.65",
        "--reference-joint-offset-deg", "knee=5",
    ),
    "sag033_kn4": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.33",
        "--reference-joint-offset-deg", "knee=4",
    ),
    "sag027_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.27",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag026_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.26",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag025_kn10_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag028_kn8_w012_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.12",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag028_kn10_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag029_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.29",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag028_kn8_ank3_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--reference-joint-offset-deg", "ankle_pitch=3",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    # Cadence path: keep sagittal foot separation above the ~0.17 m foot
    # length and slow the official Slow Walk clock to the 0.35 floor.
    "p35_sag050": (
        *ALL_MASKS,
        "--planner-speed", "0.35",
        "--reference-joint-scale", "sagittal=0.50",
    ),
    "p35_sag040": (
        *ALL_MASKS,
        "--planner-speed", "0.35",
        "--reference-joint-scale", "sagittal=0.40",
    ),
    "p35_sag040_kn6_w018_t035": (
        *ALL_MASKS,
        "--planner-speed", "0.35",
        "--reference-joint-scale", "sagittal=0.40",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.18",
        "--yaw-inplace-scale", "0.35",
    ),
    "p35_sag033_kn6_w018_t035": (
        *ALL_MASKS,
        "--planner-speed", "0.35",
        "--reference-joint-scale", "sagittal=0.33",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.18",
        "--yaw-inplace-scale", "0.35",
    ),
    "p35_sag028_kn8_w015_t035": (
        *ALL_MASKS,
        "--planner-speed", "0.35",
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    # Falsify "missing lateral lean": hip roll is already unscaled and ~21°.
    "sag025_kn10_roll130_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-scale", "hip_roll=1.30",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag027_kn8_roll130_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.27",
        "--reference-joint-scale", "hip_roll=1.30",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag026_kn8_roll115_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.26",
        "--reference-joint-scale", "hip_roll=1.15",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag025_kn10_roll120_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-scale", "hip_roll=1.20",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag026_kn8_wide4_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.26",
        "--reference-joint-offset-deg", "knee=8",
        "--reference-joint-offset-deg", "left_hip_roll=4",
        "--reference-joint-offset-deg", "right_hip_roll=-4",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "stealth": (
        *ALL_MASKS,
        "--joystick-mode", "stealth_walk",
    ),
    "injured": (
        *ALL_MASKS,
        "--joystick-mode", "injured_walk",
    ),
    "stealth_sag070": (
        *ALL_MASKS,
        "--joystick-mode", "stealth_walk",
        "--reference-joint-scale", "sagittal=0.70",
    ),
    "sag070": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.70"),
    "sag050": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.50"),
    "sag023": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.23"),
    "sag020": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.20"),
    "sag018": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.18"),
    "sag025_kn10_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.25",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag023_kn10_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.23",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag020_kn12_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.20",
        "--reference-joint-offset-deg", "knee=12",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag055": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.55"),
    "sag045": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.45"),
    "sag040": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.40"),
    "sag026": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.26"),
    "sag024": (*ALL_MASKS, "--reference-joint-scale", "sagittal=0.24"),
    "sag026_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.26",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag024_kn10_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.24",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag050_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.50",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag045_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.45",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag040_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.40",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag048_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.48",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag047_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.47",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag046_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.46",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag042_kn6_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.42",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag035_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.35",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag033_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.33",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag030_kn6_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag028_kn8_w018_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.18",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag028_kn8_w020_t040": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.20",
        "--yaw-inplace-scale", "0.40",
    ),
    "sag045_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.45",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag044_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.44",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag029_kn6_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.29",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag029_kn8_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.29",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag031_kn6_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.31",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    "sag028_kn10_w015_t035": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=10",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
    ),
    # Legacy landing-hold attempt. Reproduction only; not the operator recipe.
    "unscaled_hold": (*ALL_MASKS, "--landing-hold"),
    "sag035_hold": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.35",
        "--landing-hold",
    ),
    "sag033_hold": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.33",
        "--landing-hold",
    ),
    "sag028_kn8_w015_t035_hold": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold",
    ),
    "sag030_kn6_w015_t035_hold": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold",
    ),
    "sag050_w015_t035_hold": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.50",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold",
    ),
    "sag044_w015_t035_hold": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.44",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold",
    ),
    "unscaled_dvel": (*ALL_MASKS, "--landing-hold-dvel"),
    "sag035_dvel": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.35",
        "--landing-hold-dvel",
    ),
    "sag033_dvel": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.33",
        "--landing-hold-dvel",
    ),
    "sag028_kn8_w015_t035_dvel": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.28",
        "--reference-joint-offset-deg", "knee=8",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold-dvel",
    ),
    "sag030_kn6_w015_t035_dvel": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.30",
        "--reference-joint-offset-deg", "knee=6",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold-dvel",
    ),
    "sag050_w015_t035_dvel": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.50",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold-dvel",
    ),
    "sag044_w015_t035_dvel": (
        *ALL_MASKS,
        "--reference-joint-scale", "sagittal=0.44",
        "--yaw-command-scale", "0.15",
        "--yaw-inplace-scale", "0.35",
        "--landing-hold-dvel",
    ),
}

COMMANDS: dict[str, str] = {
    "forward": "0,1,0",
    "turn_cw": "0,0,1",
    "fwd_cw": "0,1,1",
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
    candidate: str, command: str, seed: int, steps: int, record: Path,
    model: str,
) -> list[str]:
    return [
        str(PYTHON), str(SIM),
        "--config", str(ROOT / "g1_deploy/configs/g1_23dof_sonic.yaml"),
        "--model-profile", model,
        "--mjcf", str(ROOT / "sonic_cpp/g1/scene_23dof.xml"),
        "--pico", "joystick",
        "--planner",
        "--native-gamepad",
        *(
            () if "--joystick-mode" in CANDIDATES[candidate]
            else ("--joystick-mode", "slow_walk")
        ),
        f"--axes={COMMANDS[command]}",
        "--axes-from-step", "250",
        "--axes-until", str(steps - 250),
        *(
            () if (
                "--planner-speed" in CANDIDATES[candidate]
                or "--joystick-mode" in CANDIDATES[candidate]
            )
            else ("--planner-speed", "0.4")
        ),
        "--planner-seed", str(seed),
        "--capture-at-step", "100",
        "--steps", str(steps),
        "--trial-label",
        f"{model}__{candidate}__{command}__seed{seed}",
        "--record-dir", str(record),
        *CANDIDATES[candidate],
        *(
            ("--legacy-reproduction",)
            if any(
                flag in {"--landing-hold", "--landing-hold-dvel"}
                for flag in CANDIDATES[candidate]
            )
            else ()
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", choices=tuple(CANDIDATES),
                        default=list(CANDIDATES))
    parser.add_argument("--commands", nargs="+", choices=tuple(COMMANDS),
                        default=list(COMMANDS))
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=["original-release"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1000:
        raise SystemExit("--steps must be at least 1000")
    if not PYTHON.is_file():
        raise SystemExit(f"deployment Python is missing: {PYTHON}")

    nest_model = len(args.models) > 1 or args.models != ["original-release"]
    plan = []
    for seed in args.seeds:
        for model in args.models:
            for candidate in args.candidates:
                for command in args.commands:
                    record = (
                        args.output / model / candidate / command / f"seed{seed}"
                        if nest_model
                        else args.output / candidate / command / f"seed{seed}"
                    )
                    plan.append({
                        "model": model,
                        "candidate": candidate,
                        "command": command,
                        "seed": seed,
                        "record": str(record),
                        "command_line": _command(
                            candidate, command, seed, args.steps, record, model
                        ),
                    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"PLAN: {len(plan)} trials, models={','.join(args.models)}, "
        f"candidates={','.join(args.candidates)}, "
        f"commands={','.join(args.commands)}, seeds={','.join(map(str, args.seeds))}",
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
                f"[{index}/{len(plan)}] skip complete {row['model']} "
                f"{row['candidate']} {row['command']} seed={row['seed']}"
            )
            continue
        print(
            f"[{index}/{len(plan)}] {row['model']} {row['candidate']} "
            f"{row['command']} seed={row['seed']}",
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
