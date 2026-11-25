#!/usr/bin/env bash
# Byte-identity gate for the serving surface: greedy EN/DE/ZH prompts against a running server,
# one line per language, saved for a byte-for-byte diff between dedup on and off (and dense).
# Usage: paged-gate.sh PORT OUTFILE
set -uo pipefail
PORT="${1:-8083}"
OUT="${2:-/tmp/opencode/gate.txt}"
python3 - "$PORT" "$OUT" << 'PYEOF'
import json, sys, http.client
port, out = int(sys.argv[1]), sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "sweep"
prompts = {
    "EN": "Explain in one short paragraph why the sky is blue, then give one everyday example.",
    "DE": "Erkläre in einem kurzen Absatz, warum der Himmel blau ist, und nenne ein Beispiel.",
    "ZH": "用一段话解释天空为什么是蓝色的，然后给出一个日常例子。",
}
with open(out, "w", encoding="utf-8") as f:
    for lang, content in prompts.items():
        body = json.dumps({"model": model, "max_tokens": 256, "temperature": 0,
                           "enable_thinking": False,
                           "enable_thinking": False,
                           "messages": [{"role": "user", "content": content}]}).encode()
        r = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        r.request("POST", "/v1/chat/completions", body,
                  {"Authorization": "Bearer sk-bench", "Content-Type": "application/json"})
        resp = r.getresponse()
        data = json.loads(resp.read())
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        reason = data.get("choices", [{}])[0].get("finish_reason", "")
        f.write(f"--- {lang} ({reason}, {len(text)} chars)\n{text}\n")
        print(f"{lang}: {len(text)} chars, finish {reason}")
PYEOF
