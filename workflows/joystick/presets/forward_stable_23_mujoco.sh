#!/usr/bin/env bash
# Fixed interactive MuJoCo preset for the corrected-forward 23-DoF gait.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--hip-pitch-scale[=SCALE]] [--unitree-remote] [--interface IFACE] [--ground-friction MU] [--joint-encoder-offset-deg JOINT=DEGREES] [--planner-replan-period SECONDS] [--idle-hold|--no-idle-hold] [--record-root DIR] [--no-record]"
            echo "Fixed stack: Python MuJoCo, Original Release, forward-stable-23, G1 23-DoF."
            exit 0
            ;;
        --model-profile|--embodiment|--adaptation)
            echo "ERROR: this preset locks Original Release / g1_23dof / forward-stable-23" >&2
            exit 2
            ;;
    esac
done
exec "$SCRIPT_DIR/../mujoco.sh" \
    --model-profile original-release \
    --embodiment g1_23dof \
    --adaptation forward-stable-23 \
    "$@"
