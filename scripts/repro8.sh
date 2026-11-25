#!/usr/bin/env bash
# Reproduce the 8-lane concurrent-prefill crash: 8 chat requests together, paging on.
set -uo pipefail
LOGS=~/git/ninfer-all-paged/logs
BIN=~/git/ninfer-all-paged/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
CTX=${1:-32768}
CUDA_VISIBLE_DEVICES=1 "$BIN" "$MODEL" --model-id paged-impl-r8 --host 127.0.0.1 \
    --port 8083 --max-context "$CTX" --max-concurrency 8 --kv-dtype rk4v4 --kv-paging \
    --kv-page-budget 1048576 > "$LOGS/serve-r8-$CTX.log" 2>&1 &
PID=$!
for _ in $(seq 1 150); do curl -fsS http://127.0.0.1:8083/health >/dev/null 2>&1 && break; sleep 2; done
grep -m1 "capacity |" "$LOGS/serve-r8-$CTX.log"
python3 - > /dev/null << 'PYEOF'
import json, urllib.request, threading
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]
def ask(i):
    n = 200 + i * 5
    prompt = ("Remember this list of facts: " +
              " ".join(f"{words[j % 10]}-{j} is item {j}." for j in range(n)) +
              "\nReply with the single word OK when you have read them.")
    body = json.dumps({"max_tokens": 256, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.Request("http://127.0.0.1:8083/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=900).read().decode())
        print(f"lane {i}: OK {d['choices'][0]['message']['content'][:30]!r}")
    except Exception as e:
        print(f"lane {i}: ERROR {e}")
ts = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
[t.start() for t in ts]
[t.join() for t in ts]
print("round 1 done; firing round 2 on the reused lanes")
ts = [threading.Thread(target=ask, args=(i + 8,)) for i in range(8)]
[t.start() for t in ts]
[t.join() for t in ts]
print("round 2 done")
import os, time
time.sleep(3)
PYEOF
kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "server exit code: $?"
tail -4 "$LOGS/serve-r8-$CTX.log"
