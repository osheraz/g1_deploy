#!/usr/bin/env bash
# Fixed C++ physical preset for the historical robust 23-DoF gait.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--arm-real-robot] [--interface NAME] [--probe-hosts|--clean-ports|--clean-team-ports]"
            echo "Fixed stack: C++, Original Release, robust-small-gait-23, Unitree joystick, physical G1 23-DoF."
            echo "Without --arm-real-robot, performs a no-DDS preflight only."
            exit 0
            ;;
        --model-profile|--embodiment|--adaptation|--adaptation-profile|--hip-pitch-scale|--hip-pitch-scale=*)
            echo "ERROR: this preset locks Original Release / g1_23dof / robust-small-gait-23" >&2
            exit 2
            ;;
    esac
done
exec "$SCRIPT_DIR/../robot_cpp.sh" \
    --model-profile original-release \
    --embodiment g1_23dof \
    --adaptation robust-small-gait-23 \
    "$@"
