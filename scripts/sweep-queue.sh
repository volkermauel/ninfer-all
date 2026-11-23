#!/usr/bin/env bash
# Full-scale queue-path measurement sweep on the 5090 (device 0).
# Holds the GPU window lock, stops production once, runs the load matrix across the
# before/after binaries and the flag configs, restores production, verifies health.
# Results land in ~/ninfer/queue-sweep/<config>.json (one JSON per load point).
set -uo pipefail

LOGS=~/git/ninfer-queue/logs
OUT=~/ninfer/queue-sweep
BIN_BASE=~/git/ninfer-queue/bin/serve-base-34a5fa4d
BIN_NEW=~/git/ninfer-queue/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
LOCK=/tmp/ninfer-gpu-window.lock
PORT=8080

mkdir -p "$LOGS" "$OUT" ~/git/ninfer-queue/bin

one_load() { # config_name  concurrency  total  max_tokens  gen_note
    local config=$1 conc=$2 total=$3 mt=$4
    echo "--- load $config: ${conc}-on-8, total $total, max_tokens $mt"
    python3 ~/git/ninfer-queue/scripts/measure_queue.py "$PORT" "$conc" "$total" "$mt" 1400 600 \
        > "$OUT/${config}-${conc}on8-mt${mt}.json"
    cat "$OUT/${config}-${conc}on8-mt${mt}.json"
}

run_config() { # tag  binary  extra_flags...
    local tag=$1 bin=$2; shift 2
    echo "=== config $tag"
    CUDA_VISIBLE_DEVICES=0 "$bin" "$MODEL" --model-id "q-$tag" --host 127.0.0.1 \
        --port "$PORT" --max-context 131072 --max-concurrency 8 --kv-dtype rk4v4 \
        --kv-paging --kv-page-group 64 --kv-page-dedup --spec dspark --draft-min-p 0.5 \
        --kv-page-budget 1048576 --pending-timeout-ms 10000 \
        "$@" > "$LOGS/serve-$tag.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 150); do
        curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2
    done
    grep -m1 "capacity |" "$LOGS/serve-$tag.log"
    # warmup
    python3 ~/git/ninfer-queue/scripts/measure_queue.py "$PORT" 1 1 256 1400 600 >/dev/null
    one_load "$tag" 12 12 32768
    one_load "$tag" 16 16 32768
    one_load "$tag" 8 8 256
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    sleep 3
}

flock -x -w 120 "$LOCK" bash -s << 'EOF'
set -uo pipefail
echo "=== window open $(date -Is)"
sudo -n systemctl stop ninfer-quant
sleep 3
systemctl is-active ninfer-quant || true

# BASE: tuning HEAD 34a5fa4d (pre-queue-path changes; no new flags exist there)
run_config base-upfront-fifo ~/git/ninfer-queue/bin/serve-base-34a5fa4d

# NEW: the queue-path branch binary across the flag matrix
run_config new-upfront-fifo ~/git/ninfer-queue/build/apps/ninfer-serve
run_config new-chunked-fifo ~/git/ninfer-queue/build/apps/ninfer-serve \
    --kv-reserve-mode chunked --kv-reserve-gen-chunk 256
run_config new-upfront-shortest ~/git/ninfer-queue/build/apps/ninfer-serve \
    --kv-queue-order shortest
run_config new-chunked-shortest ~/git/ninfer-queue/build/apps/ninfer-serve \
    --kv-reserve-mode chunked --kv-reserve-gen-chunk 256 --kv-queue-order shortest

sudo -n systemctl start ninfer-quant
for _ in $(seq 1 60); do
    curl -fsS -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2
done
echo "=== production health: $(curl -fsS -m 3 http://127.0.0.1:8080/health || echo FAILED)"
echo "=== window closed $(date -Is)"
EOF
echo "=== sweep complete"
