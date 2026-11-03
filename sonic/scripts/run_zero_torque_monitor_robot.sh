#!/usr/bin/env bash
# True-passive G1 joint monitor: tau=kp=kd=0. Physical robot only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INTERFACE="enp132s0"
TARGET="zero"
RATE_HZ="2"
ARM=0

while (( $# > 0 )); do
    case "$1" in
        --interface) INTERFACE="${2:?--interface requires a value}"; shift 2 ;;
        --target) TARGET="${2:?--target requires zero or launch}"; shift 2 ;;
        --rate-hz) RATE_HZ="${2:?--rate-hz requires a value}"; shift 2 ;;
        --arm-real-robot) ARM=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--interface IFACE] [--target zero|launch] [--rate-hz HZ] --arm-real-robot"
            exit 0 ;;
        *) echo "ERROR: unsupported option: $1" >&2; exit 2 ;;
    esac
done

(( ARM )) || { echo "ERROR: --arm-real-robot is required" >&2; exit 2; }
[[ -t 0 ]] || { echo "ERROR: keep this utility in a foreground terminal" >&2; exit 2; }
[[ "$INTERFACE" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "ERROR: invalid interface" >&2; exit 2; }
[[ "$TARGET" == zero || "$TARGET" == launch ]] || {
    echo "ERROR: --target must be zero or launch" >&2; exit 2;
}

echo "TRUE ZERO TORQUE: tau=0, kp=0, kd=0. Robot must be rigidly suspended."
echo "Standalone Unitree DDS utility. Target is displayed only."
echo "Select or Ctrl-C exits without re-enabling another controller."

env HAS_ROS2=0 cmake -S "$ROOT/sonic_cpp" -B "$ROOT/sonic_cpp/build" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/sonic_cpp/build" --target g1_zero_torque_monitor -j2 >/dev/null

exec "$ROOT/sonic_cpp/target/release/g1_zero_torque_monitor" \
    --interface "$INTERFACE" \
    --target "$TARGET" \
    --rate-hz "$RATE_HZ" \
    --arm-real-robot
