#!/usr/bin/env bash
# Run the ninfer test suite with GPU-aware scheduling.
#
#   scripts/run-tests.sh                serial ctest (the default; what CI gates on)
#   scripts/run-tests.sh -j             parallel: GPU tests one-per-free-device, CPU tests wide
#   scripts/run-tests.sh -j 4           parallel with an explicit CPU-job count
#   scripts/run-tests.sh -j --gpu-devices "0,1"   explicit free devices (default: both GPUs
#                                                 when nothing listens on 8080, else device 1)
#
# Scheduling: tests carry labels - `gpu_serial` (large device allocations, strictly one at a
# time), `gpu` (small kernel tests, one per free device) and everything else (CPU/host, wide).
# The runs execute as concurrent ctest invocations, each pinned to a device through
# NINFER_TEST_CUDA_DEVICE (the test registrations read it via ENVIRONMENT_MODIFICATION).
set -uo pipefail

JOBS_CPU=$(nproc)
GPU_DEVICES=""
BUILD_DIR=build
CONFIG_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -j) JOBS_CPU=$(nproc); GPU_AUTO=1; shift ;;
        -j*) JOBS_CPU="${1#-j}"; GPU_AUTO=1; shift ;;
        --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        *) CONFIG_ARGS+=("$1"); shift ;;
    esac
done

cd "$BUILD_DIR"

# Free GPUs: a device serving the production endpoint is not ours to share.
detect_gpus() {
    local count
    count=$(nvidia-smi -L 2>/dev/null | wc -l)
    if curl -fsS -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "1"
        return
    fi
    if [[ "$count" -ge 2 ]]; then echo "0,1"; else echo "0"; fi
}
[[ -z "$GPU_DEVICES" ]] && GPU_DEVICES=$(detect_gpus)

IFS=',' read -r -a DEVICES <<< "$GPU_DEVICES"
echo "GPU devices: ${DEVICES[*]}  CPU jobs: $JOBS_CPU"

# The GPU test names, then the serial subset split out of them.
mapfile -t GPU_TESTS < <(ctest -L gpu -N 2>/dev/null | grep -E "^  Test #" |
                         sed -E 's/^  Test #[0-9]+: ([A-Za-z0-9_]+).*/\1/')
mapfile -t SERIAL_TESTS < <(ctest -L gpu_serial -N 2>/dev/null | grep -E "^  Test #" |
                            sed -E 's/^  Test #[0-9]+: ([A-Za-z0-9]+).*/\1/')

declare -A is_serial
for t in "${SERIAL_TESTS[@]}"; do is_serial[$t]=1; done

# Shard the parallelizable GPU tests round-robin over the devices (strictly one test per
# device per moment; two on one card is the contention flake).
declare -a SHARD_NAMES
declare -a SHARD_DEVICE
for d in "${!DEVICES[@]}"; do SHARD_NAMES[$d]=""; SHARD_DEVICE[$d]="${DEVICES[$d]}"; done
n=0
for t in "${GPU_TESTS[@]}"; do
    if [[ -n "${is_serial[$t]:-}" ]]; then continue; fi
    d=$((n % ${#DEVICES[@]}))
    SHARD_NAMES[$d]+="$t;"
    n=$((n + 1))
done

pids=()
for d in "${!DEVICES[@]}"; do
    names="${SHARD_NAMES[$d]%;}"
    [[ -z "$names" ]] && continue
    echo "GPU shard device ${SHARD_DEVICE[$d]}: $(tr ';' ' ' <<< "$names")"
    NINFER_TEST_CUDA_DEVICE="${SHARD_DEVICE[$d]}" \
        ctest -j 1 -R "^(${names//;/|})$" --output-on-failure \
        > "gpu-shard-$d.log" 2>&1 &
    pids+=($!)
done
# The serial tier: large allocations, one at a time, on the first device.
if [[ "${#SERIAL_TESTS[@]}" -gt 0 ]]; then
    serial_names=$(IFS='|'; echo "${SERIAL_TESTS[*]}")
    echo "GPU serial device ${DEVICES[0]}: $serial_names"
    NINFER_TEST_CUDA_DEVICE="${DEVICES[0]}" \
        ctest -j 1 -R "^($serial_names)$" --output-on-failure \
        > "gpu-serial.log" 2>&1 &
    pids+=($!)
fi
# CPU tests: wide.
CPU_COUNT=$(ctest -N 2>/dev/null | tail -1 | awk '{print $3}')
echo "CPU tests: $CPU_COUNT, -j $JOBS_CPU"
ctest -LE "^gpu$" -j "$JOBS_CPU" --output-on-failure > "cpu-parallel.log" 2>&1 &
pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "=== run-tests done (fail=$fail)"
exit "$fail"
