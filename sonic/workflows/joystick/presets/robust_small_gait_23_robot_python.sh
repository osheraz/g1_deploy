#!/usr/bin/env bash
# Withdrawn legacy physical Python preset. Blocked; no override.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# shellcheck source=../../../../sonic_cpp/teleop/core.common.sh
source "$ROOT/sonic_cpp/teleop/core.common.sh"

legacy_sonic_attempt_error "robust-small-gait-23"
echo "Physical Python launch is refused. There is no override." >&2
echo "Current recipe: $ROOT/g1_deploy/sonic/workflows/joystick/presets/forward_stable_23_robot_cpp.sh --slow-gait" >&2
exit 2
