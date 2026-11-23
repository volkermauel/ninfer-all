#!/usr/bin/env bash
# The page-group size sweep (INNER: run inside paged-window.sh, which holds the GPU lock and has
# stopped production): 8x131072 rk4v4 paging at --kv-page-group 32/64/128/256/512/1024, decode
# tok/s at concurrency 1 and 8 (~1,373-token prompts, 256 gen), ~10K-token prefill ms, the
# capacity line and the live pages line per size.
set -uo pipefail
LOGS=~/git/ninfer-all-paged/logs
BIN=~/git/ninfer-all-paged/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
BENCH=~/git/ninfer-all-paged/scripts/paged_bench.py
PORT=${PORT:-8084}
mkdir -p "$LOGS"

for P in 32 64 128 256 512 1024; do
  echo "== page group $P =="
  LOG="$LOGS/sweep-P$P.log"
  CUDA_VISIBLE_DEVICES=0 "$BIN" "$MODEL" --model-id sweep --host 127.0.0.1 \
      --port "$PORT" --max-context 131072 --max-concurrency 8 --kv-dtype rk4v4 \
      --kv-paging --kv-page-budget 1048576 --kv-page-group "$P" \
      --pending-timeout-ms 60000 > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 2
  done
  if [ "$ok" = 1 ]; then
    grep -m1 "capacity |" "$LOG"
    python3 "$BENCH" "$PORT" 1 8
    python3 "$BENCH" "$PORT" prefill10000
    grep "pages |" "$LOG" | tail -1
  else
    echo "page group $P: the server did not start (see $LOG)"
    tail -3 "$LOG"
  fi
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  sleep 3
done
echo "sweep done"
