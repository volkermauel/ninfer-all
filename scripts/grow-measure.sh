#!/usr/bin/env bash
# The grow-on-demand measurement: 32-on-8 (four waves of 8) of 8K-token prompts with 32-token
# generations against a 541-group pool (budget 300 MiB). Upfront needs 32 x 23 = 736 groups
# (waves 2+ cannot be granted: the queued lanes expire at the 6 s deadline -> 503s); chunked
# needs 32 x 4 = 128 groups for the first waves and grows per round -> far more served.
set -uo pipefail
LOGS=~/git/ninfer-queue/logs
BIN=~/git/ninfer-queue/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer

run_mode() { # mode
    local mode=$1; shift
    local extra=()
    if [[ "$mode" == chunked ]]; then extra=(--kv-reserve-mode chunked --kv-reserve-gen-chunk 32); fi
    CUDA_VISIBLE_DEVICES=0 "$BIN" "$MODEL" --model-id inkling-2-mini --host 127.0.0.1 \
        --port 8080 --max-context 32768 --max-concurrency 8 --kv-dtype rk4v4 \
        --kv-paging --kv-page-group 64 --kv-page-dedup --kv-page-budget 600 \
        --pending-timeout-ms 6000 --kv-queue-timeout-ms 6000 ${extra+"${extra[@]}"} \
        > "$LOGS/serve-grow-$mode.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
    grep -m1 "capacity |" "$LOGS/serve-grow-$mode.log"
    python3 - 8 32 4096 "$mode" << 'PYEOF'
import http.client
import json
import queue
import sys
import threading
import time
conc, total, gen, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
PARA = ("The lighthouse keeper recorded the weather every hour in the station log. "
        "Wind speed and direction, barometric pressure, visibility and the state of the sea "
        "were entered in careful columns, together with the movements of ships passing the cape.")
prompt = PARA * 125  # ~8K tokens
body = json.dumps({"model": "inkling-2-mini", "max_tokens": gen, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
headers = {"Authorization": "Bearer sk-alliancemedicalrp", "Content-Type": "application/json"}
results, work, lock = [], queue.Queue(), threading.Lock()
def put(n):
    for i in range(n): work.put(i)
def worker():
    while True:
        try: idx = work.get_nowait()
        except queue.Empty: return
        rec = {"idx": idx, "ok": False}
        t0 = time.perf_counter()
        try:
            c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=300)
            c.request("POST", "/v1/chat/completions", body, headers)
            r = c.getresponse(); data = r.read()
            rec["status"] = r.status; rec["wall_s"] = round(time.perf_counter() - t0, 2)
            if r.status == 200:
                j = json.loads(data); rec["pred_n"] = j.get("usage", {}).get("completion_tokens", 0)
                rec["ok"] = True
            elif r.status == 503:
                rec["retry_after"] = r.getheader("Retry-After")
            results.append(rec); c.close()
        except Exception as e:
            rec["err"] = repr(e)[:80]; results.append(rec)
put(conc)
threads = [threading.Thread(target=worker) for _ in range(conc)]
for t in threads: t.start()
scheduled = conc
def feeder():
    global scheduled
    while scheduled < total:
        time.sleep(0.7)
        with lock:
            if work.qsize() < conc: put(conc); scheduled += conc
feeder_thread = threading.Thread(target=feeder); feeder_thread.start()
for t in threads: t.join()
feeder_thread.join()
ok = [r for r in results if r.get("ok")]
t503 = [r for r in results if r.get("status") == 503]
gen_tok = sum(r.get("pred_n", 0) for r in ok)
retry_afters = sorted({r.get("retry_after") for r in t503 if r.get("retry_after")})
walls = sorted(r.get("wall_s", 0) for r in results)
print(json.dumps({"mode": mode, "admitted": len(ok), "expired_503": len(t503),
                  "other": len(results) - len(ok) - len(t503),
                  "gen_tokens": gen_tok, "retry_after_values": retry_afters,
                  "wall_p50_s": walls[len(walls)//2] if walls else 0,
                  "wall_p90_s": walls[int(len(walls)*0.9)] if walls else 0}), flush=True)
PYEOF
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    sleep 3
}

run_mode upfront
run_mode chunked
echo "=== grow-on-demand measurement complete"
