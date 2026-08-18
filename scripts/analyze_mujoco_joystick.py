#!/usr/bin/env python3
"""Compatibility entrypoint for tools/analysis/analyze_mujoco_joystick.py."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/analysis/analyze_mujoco_joystick.py"),
    run_name="__main__",
)
