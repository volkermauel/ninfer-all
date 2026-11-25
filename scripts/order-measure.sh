#!/usr/bin/env bash
# Order-policy measurement: 8 lanes saturated by 8 long requests (8K tokens, 256 gen), while
# 16 short requests (100 tokens, 16 gen) arrive — fifo runs the longs first (the shorts wait
# behind them); shortest admits the shorts first. Metric: the shorts' mean wall time and the
# count of shorts completed within 5 s of their submission.
set -uo pipefail
LOGS=~/git/ninfer-queue/logs
BIN=~/git/ninfer-queue/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
ORDER=${1:-fifo}
CUDA_VISIBLE_DEVICES=1 "$BIN" "$MODEL" --model-id inkling-2-mini --host 127.0.0.1 \
    --port 8086 --max-context 32768 --max-concurrency 8 --kv-dtype rk4v4 --kv-paging \
    --kv-page-group 64 --kv-page-dedup --kv-queue-order "$ORDER" --kv-reserve-mode chunked \
    --kv-reserve-gen-chunk 256 --kv-page-budget 2048 --pending-timeout-ms 60000 \
    > "$LOGS/serve-ord-$ORDER.log" 2>&1 &
PID=$!
for _ in $(seq 1 120); do curl -fsS http://127.0.0.1:8086/health >/dev/null 2>&1 && break; sleep 2; done

python3 - "$ORDER" << 'PYEOF'
import json, sys, threading, time, urllib.request
order = sys.argv[1]
PARA = ("The lighthouse keeper recorded the weather every hour in the station log. "
        "Wind speed and direction, barometric pressure, visibility and the state of the sea "
        "were entered in careful columns, together with the movements of ships passing the cape.")
def body_of(n, gen):
    prompt = PARA * max(1, n // 64)
    return json.dumps({"model": "inkling-2-mini", "max_tokens": gen, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
results = {}
def ask(tag, body, deadline):
    t0 = time.perf_counter()
    r = urllib.request.Request("http://127.0.0.1:8086/v1/chat/completions", data=body,
                               headers={"Authorization": "Bearer sk-alliancemedicalrp",
                                        "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(r, timeout=deadline).read()
        results[tag] = time.perf_counter() - t0
    except Exception as e:
        results[tag] = -1.0
long_bodies = [body_of(8000, 256) for _ in range(8)]
short_bodies = [body_of(100, 16) for _ in range(16)]
def fire(body, tag):
    t = threading.Thread(target=ask, args=(tag, body, 600)); t.start(); return t
# 8 long requests saturate the lanes, submitted first
long_ts = [fire(b, f"long{i}") for i, b in enumerate(long_bodies)]
time.sleep(0.5)
# then 16 short requests arrive
short_ts = [fire(b, f"short{i}") for i, b in enumerate(short_bodies)]
for t in long_ts + short_ts: t.join()
short_walls = [v for k, v in sorted(results.items()) if k.startswith("short") and v > 0]
fast = sum(1 for v in short_walls if v < 5.0)
print("%s: shorts completed %d/16, mean wall %.2f s, under-5s %d" % (
    order, len(short_walls), sum(short_walls) / max(1, len(short_walls)), fast))
PYEOF
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "ord-$ORDER server exit: $?"
