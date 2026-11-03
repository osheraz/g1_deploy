#!/usr/bin/env python3
"""Generate and validate the plan consumed by both calibration runners."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonic.calibration.protocol import build_plan, save_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("screening", "full"), default="screening")
    parser.add_argument("--scope", choices=("legs", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(protocol=args.protocol, scope=args.scope)
    output = save_plan(plan, args.output)
    print(f"PLAN OK: {output}")
    print(
        f"protocol={plan.protocol} scope={plan.scope} samples={len(plan.time_s)} "
        f"duration_s={plan.duration_s:.2f}"
    )


if __name__ == "__main__":
    main()
