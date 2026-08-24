#!/usr/bin/env bash
# Physical measurement only. This launcher never starts SONIC or locomotion.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/data/clhsieh/environments/python/venvs/e-skin-4090-verify/bin/python"
INTERFACE="enp132s0"
PROTOCOL="screening"
SCOPE="all"
ARM=0
ALLOW_FULL=0
OUTPUT=""
while (( $# > 0 )); do
    case "$1" in
        --interface) INTERFACE="${2:?--interface requires a value}"; shift 2 ;;
        --protocol) PROTOCOL="${2:?--protocol requires screening or full}"; shift 2 ;;
        --scope) SCOPE="${2:?--scope requires legs or all}"; shift 2 ;;
        --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
        --arm-real-robot) ARM=1; shift ;;
        --allow-full-range) ALLOW_FULL=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--protocol screening|full] [--scope legs|all] [--interface IFACE] [--output DIR] [--arm-real-robot] [--allow-full-range]"
            echo "Default is a no-DDS screening/all dry run. Full physical range requires both --arm-real-robot and --allow-full-range."
            exit 0 ;;
        *) echo "ERROR: unsupported option: $1" >&2; exit 2 ;;
    esac
done

[[ "$PROTOCOL" == screening || "$PROTOCOL" == full ]] || {
    echo "ERROR: --protocol must be screening or full" >&2; exit 2;
}
[[ "$SCOPE" == legs || "$SCOPE" == all ]] || {
    echo "ERROR: --scope must be legs or all" >&2; exit 2;
}
[[ "$INTERFACE" =~ ^[A-Za-z0-9_.:-]+$ ]] || {
    echo "ERROR: invalid interface token" >&2; exit 2;
}
if [[ "$PROTOCOL" == full && "$ARM" == 1 && "$ALLOW_FULL" != 1 ]]; then
    echo "ERROR: full-range physical calibration requires --allow-full-range after a reviewed screening run" >&2
    exit 2
fi
[[ -x "$PYTHON" ]] || { echo "ERROR: missing calibration Python: $PYTHON" >&2; exit 2; }

if [[ -z "$OUTPUT" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT="/data/${USER}/logs/e_skin/calibration/${stamp}-g1_23dof-${PROTOCOL}-${SCOPE}"
fi
mkdir -p "$OUTPUT"
PLAN="$OUTPUT/plan.csv"
STATE="$OUTPUT/state.csv"
PROFILE="$OUTPUT/provisional_encoder_only_profile.json"

"$PYTHON" "$ROOT/g1_deploy/sonic/scripts/generate_suspended_calibration_plan.py" \
    --protocol "$PROTOCOL" --scope "$SCOPE" --output "$PLAN"

env HAS_ROS2=0 cmake -S "$ROOT/sonic_cpp" -B "$ROOT/sonic_cpp/build" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/sonic_cpp/build" --target g1_suspended_calibration -j2 >/dev/null
BINARY="$ROOT/sonic_cpp/target/release/g1_suspended_calibration"
"$BINARY" --plan "$PLAN" --output "$STATE"

if (( ! ARM )); then
    echo "DRY RUN COMPLETE: no DDS was opened and no motor command was sent."
    echo "Plan: $PLAN"
    exit 0
fi
[[ -t 0 ]] || {
    echo "ERROR: physical calibration must remain in a foreground terminal" >&2
    exit 2
}

echo "PHYSICAL CALIBRATION: no SONIC and no walking. Robot must be rigidly suspended."
echo "After fresh LowState appears, hold F1+F3 for 0.30 s to capture/start. Select aborts into damping."
set +e
"$BINARY" --plan "$PLAN" --output "$STATE" --interface "$INTERFACE" --arm-real-robot
status=$?
set -e
if [[ -s "$STATE" ]]; then
    echo "Recorded: $STATE"
fi
if (( status == 0 )) && [[ -s "$STATE" ]]; then
    "$PYTHON" "$ROOT/g1_deploy/sonic/scripts/fit_suspended_calibration.py" "$STATE" \
        --coordinate-basis encoder_only --output "$PROFILE"
    echo "Provisional encoder-only fit (not deployable): $PROFILE"
elif (( status != 0 )); then
    echo "Calibration did not complete (status $status); partial data is preserved and no profile was fitted." >&2
fi
exit "$status"
