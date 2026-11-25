#!/usr/bin/env bash
# Traffic for an already-running server on 8080: 8 concurrent ~1.4K-token prompts, 256 gen
# tokens, two rounds (the second reuses the lanes).
set -uo pipefail
for _ in $(seq 1 20); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
python3 - << 'PYEOF'
import json, urllib.request, threading
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
        print(f"lane {i}: OK {d['choices'][0]['message']['content'][:30]!r}")
    except Exception as e:
        print(f"lane {i}: ERROR {e}")
ts = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
[t.start() for t in ts]
[t.join() for t in ts]
print("round 1 done")
ts = [threading.Thread(target=ask, args=(i + 8,)) for i in range(8)]
[t.start() for t in ts]
[t.join() for t in ts]
print("round 2 done")
PYEOF
echo "traffic finished"
