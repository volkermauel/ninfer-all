#!/usr/bin/env bash
# The dedup A/B (INNER: run inside paged-window.sh): 8x131072 rk4v4 paging, sessions sharing a
# common ~1,000-token prefix with divergent suffixes, --no-kv-page-dedup against the default
# (dedup on). Pages lines, hit rate, decode/prefill, and the greedy EN/DE/ZH gate captured per
# variant for a byte-for-byte diff.
set -uo pipefail
LOGS=~/git/ninfer-all-paged/logs
BIN=~/git/ninfer-all-paged/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
BENCH=~/git/ninfer-all-paged/scripts/paged_bench.py
GATE=~/git/ninfer-all-paged/scripts/paged-gate.sh
PORT=${PORT:-8084}
mkdir -p "$LOGS"

run_variant() {
  local NAME="$1"; shift
  echo "== dedup $NAME =="
  LOG="$LOGS/dedup-$NAME.log"
  CUDA_VISIBLE_DEVICES=0 "$BIN" "$MODEL" --model-id sweep --host 127.0.0.1 \
      --port "$PORT" --max-context 131072 --max-concurrency 8 --kv-dtype rk4v4 \
      --kv-paging --kv-page-budget 1048576 --pending-timeout-ms 60000 "$@" \
      > "$LOG" 2>&1 &
  local PID=$!
  local ok=0
  for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 2
  done
  if [ "$ok" != 1 ]; then
    echo "dedup $NAME: the server did not start (see $LOG)"
    tail -3 "$LOG"
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
    return
  fi
  grep -m1 "capacity |" "$LOG"
  for N in 4 8 16; do python3 "$BENCH" "$PORT" dedup "$N"; done
  python3 "$BENCH" "$PORT" 1 8
  bash "$GATE" "$PORT" "$LOGS/gate-$NAME.txt"
  grep "pages |" "$LOG" | tail -3
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  sleep 3
}

run_variant off --no-kv-page-dedup
run_variant on
echo "dedup A/B done"
diff "$LOGS/gate-off.txt" "$LOGS/gate-on.txt" > "$LOGS/gate-dedup-diff.txt" 2>&1 &&
  echo "gate: dedup on/off byte-identical (EN/DE/ZH)" ||
  { echo "gate: OUTPUTS DIFFER"; cat "$LOGS/gate-dedup-diff.txt" | head -10; }
