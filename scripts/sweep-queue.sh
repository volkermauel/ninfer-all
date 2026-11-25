#!/usr/bin/env bash
# Full-scale queue-path measurement sweep on the 5090 (device 0). One pass per config:
# server up (rk4v4, 8 lanes, 131072, dspark, paging), warmup, then the pressure loads:
#   A: 12-on-8, ~8K-token prompts, max_tokens 4096 (declared 64 groups/lane x12 = 768 groups;
#      the 1,792-MiB budget holds 2,212 groups: upfront must refuse ~3 lanes at 10 s timeouts,
#      chunked grants 12 x 129 = 1,548 and serves all 12)
#   B: 16-on-8, same shape (16 x 129 = 2,064 <= 2,212 chunked; 2,268 > 2,212 upfront)
#   C: 8-on-8 parity at the auto-full budget, 1.4K prompts, 256 gen
set -uo pipefail
LOGS=~/git/ninfer-queue/logs
OUT=~/ninfer/queue-sweep
BIN_NEW=~/git/ninfer-queue/build/apps/ninfer-serve
BIN_BASE=~/git/ninfer-queue/bin/serve-base-34a5fa4d
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
PORT=8080

mkdir -p "$LOGS" "$OUT"

one_load() { # config  conc  total  max_tokens  prompt_tokens
    local config=$1 conc=$2 total=$3 mt=$4 pt=$5
    python3 ~/git/ninfer-queue/scripts/measure_queue.py "$PORT" "$conc" "$total" "$mt" "$pt" 600 \
        > "$OUT/${config}-${conc}on8-pt${pt}-mt${mt}.json"
    echo "  $(cat "$OUT/${config}-${conc}on8-mt${mt}.json" 2>/dev/null || cat "$OUT/${config}-${conc}on8-pt${pt}-mt${mt}.json")"
}

run_config() { # tag  binary  budget_mib  extra_flags...
    local tag=$1 bin=$2 budget=$3; shift 3
    echo "=== config $tag (budget ${budget} MiB)"
    CUDA_VISIBLE_DEVICES=0 "$bin" "$MODEL" --model-id inkling-2-mini --host 127.0.0.1 \
        --port "$PORT" --max-context 131072 --max-concurrency 8 --kv-dtype rk4v4 \
        --kv-paging --kv-page-group 64 --kv-page-dedup --spec dspark --draft-min-p 0.5 \
        --kv-page-budget "$budget" --pending-timeout-ms 10000 \
        "$@" > "$LOGS/serve-$tag.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 150); do
        curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2
    done
    grep -m1 "capacity |" "$LOGS/serve-$tag.log"
    python3 ~/git/ninfer-queue/scripts/measure_queue.py "$PORT" 1 1 256 1400 600 >/dev/null
    one_load "$tag" 12 12 4096 8000
    one_load "$tag" 16 16 4096 8000
    one_load "$tag" 8 8 256 1400
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    sleep 3
}

CONFIG=${1:-all}
case "$CONFIG" in
base)    run_config base-upfront-fifo "$BIN_BASE" 1792 ;;
new)     run_config new-upfront-fifo   "$BIN_NEW" 1792
         run_config new-chunked-fifo   "$BIN_NEW" 1792 --kv-reserve-mode chunked --kv-reserve-gen-chunk 256 ;;
order)   run_config new-upfront-shortest  "$BIN_NEW" 1792 --kv-queue-order shortest
         run_config new-chunked-shortest  "$BIN_NEW" 1792 --kv-reserve-mode chunked --kv-reserve-gen-chunk 256 --kv-queue-order shortest ;;
parity)  run_config parity-upfront "$BIN_NEW" 1048576 ;;
*) run_config base-upfront-fifo "$BIN_BASE" 1792
   run_config new-upfront-fifo   "$BIN_NEW" 1792
   run_config new-chunked-fifo   "$BIN_NEW" 1792 --kv-reserve-mode chunked --kv-reserve-gen-chunk 256
   run_config new-upfront-shortest  "$BIN_NEW" 1792 --kv-queue-order shortest
   run_config new-chunked-shortest  "$BIN_NEW" 1792 --kv-reserve-mode chunked --kv-reserve-gen-chunk 256 --kv-queue-order shortest
   run_config parity-upfront     "$BIN_NEW" 1048576
   # The grow-on-demand delta at the floor-pool pressure point: 32768 contexts, pool floor
   # 512 groups, 12/16 lanes of 1.4K prompts + 4096 declared generation (86 groups a lane
   # upfront vs 26 chunked). Upfront admits ~5; chunked serves all.
   run_config delta-32768-upfront-fifo  "$BIN_NEW" 289 --max-context 32768
   run_config delta-32768-chunked-fifo  "$BIN_NEW" 289 --max-context 32768 --kv-reserve-mode chunked --kv-reserve-gen-chunk 256 ;;

esac
echo "=== sweep config '$CONFIG' complete"
