#!/usr/bin/env bash
# Full-scale 5090 verification: 8x131072 paging under sustained 8-lane load.
set -uo pipefail
LOGS=~/git/ninfer-all-paged/logs
BIN=~/git/ninfer-all-paged/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
CTX=131072
ROUNDS=${1:-4}
CUDA_VISIBLE_DEVICES=0 "$BIN" "$MODEL" --model-id paged-full-rk4v4 --host 127.0.0.1 \
    --port 8080 --max-context "$CTX" --max-concurrency 8 --kv-dtype rk4v4 --kv-paging \
    --kv-page-budget 1048576 --pending-timeout-ms 60000 > "$LOGS/serve-full-rk4v4.log" 2>&1 &
PID=$!
for _ in $(seq 1 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
grep -m1 "capacity |" "$LOGS/serve-full-rk4v4.log"
python3 - "$ROUNDS" << 'PYEOF'
import json, sys, time, urllib.request, threading
rounds = int(sys.argv[1])
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]
def ask(i):
    n = 200 + i * 5
    prompt = ("Remember this list of facts: " +
              " ".join(f"{words[j % 10]}-{j} is item {j}." for j in range(n)) +
              "\nReply with the single word OK when you have read them.")
    body = json.dumps({"max_tokens": 256, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=900).read().decode())
        print(f"lane {i}: OK {d['choices'][0]['message']['content'][:20]!r}")
    except Exception as e:
        print(f"lane {i}: ERROR {e}")
for round_index in range(rounds):
    ts = [threading.Thread(target=ask, args=(i + round_index * 8,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print(f"round {round_index + 1} done")
PYEOF
echo "load done"
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "server exit: $?"
grep -ac "IllegalAddress" "$LOGS/serve-full-rk4v4.log" 2>/dev/null
grep -m2 "pages |" "$LOGS/serve-full-rk4v4.log" | tail -2
