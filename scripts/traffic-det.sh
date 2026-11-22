#!/usr/bin/env bash
# Deterministic-order traffic: submit 8 requests with a fixed stagger, three rounds, so the
# engine's admission/prefill/decode interleaving is reproducible.
set -uo pipefail
for _ in $(seq 1 20); do curl -fsS http://127.0.0.1:8083/health >/dev/null 2>&1 && break; sleep 2; done
python3 - << 'PYEOF'
import json, time, urllib.request, threading
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]
errors = []
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
        print(f"lane {i}: OK {d['choices'][0]['message']['content'][:20]!r}")
    except Exception as e:
        errors.append(f"lane {i}: {e}")
        print(f"lane {i}: ERROR {e}")
for round_index in range(3):
    threads = []
    for i in range(8):
        t = threading.Thread(target=ask, args=(i + round_index * 8,))
        t.start()
        threads.append(t)
        time.sleep(0.05)
    for t in threads: t.join()
    print(f"round {round_index + 1} done")
if errors:
    print(f"{len(errors)} ERRORS")
else:
    print("all clean")
PYEOF
echo "traffic finished"
