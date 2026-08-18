#!/usr/bin/env bash
# Fixed C++ physical preset for the corrected-forward 23-DoF gait.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--arm-real-robot] [--hip-pitch-scale[=SCALE]] [--interface NAME] [--probe-hosts|--clean-ports|--clean-team-ports]"
            echo "Fixed stack: C++, Original Release, forward-stable-23, Unitree joystick, physical G1 23-DoF."
            echo "Without --arm-real-robot, performs a no-DDS preflight only."
            echo "--hip-pitch-scale without a value uses the knee-bias-robust MuJoCo default 0.85."
            echo "Omitting it keeps the base bilateral hip-pitch reference scale at 1.00."
            exit 0
            ;;
        --model-profile|--embodiment|--adaptation|--adaptation-profile)
            echo "ERROR: this preset locks Original Release / g1_23dof / forward-stable-23" >&2
            exit 2
            ;;
    esac
done
exec "$SCRIPT_DIR/../robot_cpp.sh" \
    --model-profile original-release \
    --embodiment g1_23dof \
    --adaptation forward-stable-23 \
    "$@"
