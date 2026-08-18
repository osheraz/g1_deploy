#!/usr/bin/env bash
# Review-gated physical calibration. Defaults to protocol/build dry-run: no DDS.
set -euo pipefail

ROOT="/home/clhsieh/e_skin"
PYTHON="/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python"
INTERFACE="enp132s0"
PROTOCOL="pilot"
OUTPUT=""
ARM=0

while (( $# > 0 )); do
    case "$1" in
        --interface) INTERFACE="${2:?--interface requires a value}"; shift 2 ;;
        --protocol) PROTOCOL="${2:?--protocol requires pilot or screening}"; shift 2 ;;
        --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
        --arm-real-robot) ARM=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--protocol pilot|screening] [--interface IFACE] [--output DIR] [--arm-real-robot]"
            echo "Without --arm-real-robot: validates/builds only and opens no DDS channel."
            exit 0 ;;
        *) echo "ERROR: unsupported option: $1" >&2; exit 2 ;;
    esac
done

[[ "$PROTOCOL" == pilot || "$PROTOCOL" == screening ]] || {
    echo "ERROR: --protocol must be pilot or screening" >&2; exit 2;
}
[[ "$INTERFACE" =~ ^[A-Za-z0-9_.:-]+$ ]] || {
    echo "ERROR: invalid interface token" >&2; exit 2;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 2; }

if [[ -z "$OUTPUT" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT="/data/clhsieh/logs/e_skin/calibration/review_gated_robot/${stamp}-${PROTOCOL}"
fi
mkdir -p "$OUTPUT"

"$PYTHON" "$ROOT/g1_deploy/scripts/generate_review_gated_calibration_protocol.py" \
    --protocol "$PROTOCOL" --output "$OUTPUT"

"$PYTHON" "$ROOT/g1_deploy/scripts/verify_review_gated_physical_parity.py" \
    --protocol "$PROTOCOL" --plan "$OUTPUT/protocol.csv"

env HAS_ROS2=0 cmake -S "$ROOT/sonic_cpp" -B "$ROOT/sonic_cpp/build" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/sonic_cpp/build" --target g1_suspended_calibration -j2 >/dev/null

BINARY="$ROOT/sonic_cpp/target/release/g1_suspended_calibration"
PLAN="$OUTPUT/protocol.csv"
STATE="$OUTPUT/state.csv"
"$BINARY" --plan "$PLAN" --output "$STATE"

if (( ! ARM )); then
    echo "DRY RUN COMPLETE: protocol validated; DDS was not opened; no motor command was sent."
    echo "Protocol: $PLAN"
    exit 0
fi

[[ -t 0 ]] || {
    echo "ERROR: physical calibration must remain in a foreground terminal" >&2
    exit 2
}

echo "PHYSICAL 23-DOF REVIEW-GATED CALIBRATION"
echo "Robot must be secured by both shoulder attachments with feet and arms clear."
echo "On launch, the controller automatically moves smoothly to the NVIDIA initial pose."
echo "Select aborts that startup transition into damping; F1+F3 is used only after it is ready."
if [[ "$PROTOCOL" == screening ]]; then
    echo "EXPANDED SCREENING: isolated leg peaks reach hip pitch 0.45, hip roll/yaw 0.30,"
    echo "knee 0.60, ankle pitch 0.35, and ankle roll 0.18 rad."
    echo "The coupled inward hip-roll block is separately capped at 0.144 rad per side."
    echo "DYNAMIC/OOD BLOCKS: isolated/coupled sweeps reach 2.2 Hz and at most 2.0 rad/s target rate."
    echo "The final block uses rapid asymmetric direction reversals to expose lag/backlash."
    echo "Run the fast pilot first and visually approve every displayed block."
fi
echo "F1+F3 captures a 1.5 s steady joint/IMU baseline."
echo "At each review gate: hold A 0.30 s to run, B 0.30 s to skip; Select aborts to damping."
echo "During a block, B cancels it with a smooth neutral return; Select aborts immediately."

"$BINARY" --plan "$PLAN" --output "$STATE" \
    --interface "$INTERFACE" --arm-real-robot
