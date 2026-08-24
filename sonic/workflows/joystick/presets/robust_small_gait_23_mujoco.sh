#!/usr/bin/env bash
# Withdrawn legacy MuJoCo preset. Blocked from live use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# shellcheck source=../../../../sonic_cpp/teleop/core.common.sh
source "$ROOT/sonic_cpp/teleop/core.common.sh"

LEGACY_REPRODUCTION=0
FORWARD_ARGS=()
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 --legacy-reproduction [mujoco.sh flags]"
            echo "LEGACY: Original Release / g1_23dof / robust-small-gait-23."
            echo "Withdrawn backward-selected profile. Blocked unless --legacy-reproduction."
            echo "Current recipe: forward_stable_23_mujoco.sh --slow-gait"
            exit 0
            ;;
        --legacy-reproduction)
            LEGACY_REPRODUCTION=1
            FORWARD_ARGS+=("$arg")
            ;;
        --model-profile|--embodiment|--adaptation)
            echo "ERROR: this preset locks Original Release / g1_23dof / robust-small-gait-23" >&2
            exit 2
            ;;
        *)
            FORWARD_ARGS+=("$arg")
            ;;
    esac
done
if (( ! LEGACY_REPRODUCTION )); then
    legacy_sonic_attempt_error "robust-small-gait-23"
    echo "MuJoCo reproduction only: add --legacy-reproduction." >&2
    echo "Current recipe: $ROOT/g1_deploy/workflows/joystick/presets/forward_stable_23_mujoco.sh --slow-gait" >&2
    exit 2
fi
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../mujoco.sh" \
    --model-profile original-release \
    --embodiment g1_23dof \
    --adaptation robust-small-gait-23 \
    "${FORWARD_ARGS[@]}"
