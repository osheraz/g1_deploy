#!/usr/bin/env python3
"""Generate the shared MuJoCo/physical review-gated 23-DoF protocol."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonic.calibration.gated_protocol import build_review_protocol, save_review_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("pilot", "screening"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = build_review_protocol(args.protocol)
    path = save_review_protocol(protocol, args.output)
    print(
        f"PROTOCOL OK: {path} embodiment=23dof blocks={len(protocol.blocks)} "
        f"motion_duration_s={protocol.motion_duration_s:.2f}"
    )


if __name__ == "__main__":
    main()
