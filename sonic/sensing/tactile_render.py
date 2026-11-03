"""Tactile rendering for the deploy side.

Thin re-export of ``jq_tactile_driver.render`` -- one set of taxel maps for the
Qt visualizer and viser alike. This module used to carry its own copy, which
drifted to a stale 72-entry palm map and rendered the palm wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from jq_tactile_driver.render import (  # noqa: E402,F401
    GAIN,
    render_hand_canvas,
    render_sensor_panels,
)

__all__ = ["render_sensor_panels", "render_hand_canvas", "GAIN"]
