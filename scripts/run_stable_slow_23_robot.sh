#!/usr/bin/env bash
# Withdrawn legacy physical entry point. Blocked; no override.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../../sonic_cpp/teleop/common.sh
source "$ROOT/sonic_cpp/teleop/common.sh"

legacy_sonic_attempt_error "stable-slow-23"
echo "Physical Python launch is refused. There is no override." >&2
echo "Current recipe: $ROOT/g1_deploy/scripts/run_planner_joystick_robot_cpp.sh --adaptation forward-stable-23 --slow-gait" >&2
exit 2
