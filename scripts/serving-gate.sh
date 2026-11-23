#!/usr/bin/env bash
# Serving gate: greedy EN/DE byte-identical between the pre-queue-change lineage (ddaf9bf9
# binary) and the queue-path binary in both reserve modes. The three servers run sequentially
# on one 3090 (each 16384x8 plan fits alone; three at once do not).
set -uo pipefail
LOGS=~/git/ninfer-queue/logs
BASE=~/git/ninfer-all-paged/build/apps/ninfer-serve
NEW=~/git/ninfer-queue/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
GATE_PID=0

wait_up() {
    for _ in $(seq 1 150); do curl -fsS "http://127.0.0.1:8083/health" >/dev/null 2>&1 && return 0; sleep 2; done
    return 1
}

ask() { # port prompt_file  -> prints the reply's content
    python3 - "$1" "$2" << 'PYEOF'
import json, sys, urllib.request
port, path = sys.argv[1], sys.argv[2]
prompt = open(path).read()
body = json.dumps({"model": "inkling-2-mini", "max_tokens": 256, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                           headers={"Authorization": "Bearer sk-alliancemedicalrp",
                                    "Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=300).read().decode())
print(d["choices"][0]["message"]["content"])
PYEOF
}

serve_and_capture() { # tag  binary  extra_flags...
    local tag=$1 bin=$2; shift 2
    CUDA_VISIBLE_DEVICES=1 "$bin" "$MODEL" --model-id inkling-2-mini --host 127.0.0.1 \
        --port 8083 --max-context 16384 --max-concurrency 8 --kv-dtype rk4v4 \
        --kv-paging --spec dspark --draft-min-p 0.5 --kv-page-budget 1048576 \
        "$@" > "$LOGS/serve-gate-$tag.log" 2>&1 &
    GATE_PID=$!
    wait_up || { echo "FAIL server $tag"; kill $GATE_PID 2>/dev/null; return 1; }
    for f in en de; do
        ask 8083 "/tmp/gate-$f.txt" > "/tmp/gate-$f-$tag.txt"
    done
    kill $GATE_PID 2>/dev/null; wait $GATE_PID 2>/dev/null
    sleep 3
}

EN='Name the four largest planets of the solar system, shortest answer.'
DE='Übersetze ins Deutsche, nur die Übersetzung: Guten Morgen, das Wetter ist heute wunderschön.'
printf '%s' "$EN" > /tmp/gate-en.txt
printf '%s' "$DE" > /tmp/gate-de.txt

echo "control (ddaf9bf9 lineage, upfront+fifo)"
serve_and_capture base "$BASE"
echo "new (upfront+fifo)"
serve_and_capture new-upfront "$NEW" --kv-reserve-mode upfront
echo "new (chunked, 64-token grants)"
serve_and_capture new-chunked "$NEW" --kv-reserve-mode chunked --kv-reserve-gen-chunk 64

for f in en de; do
    if cmp -s "/tmp/gate-$f-base.txt" "/tmp/gate-$f-new-upfront.txt" &&
       cmp -s "/tmp/gate-$f-base.txt" "/tmp/gate-$f-new-chunked.txt"; then
        echo "OK   $f byte-identical: base == new-upfront == new-chunked ($(head -c 60 /tmp/gate-$f-base.txt))"
    else
        echo "FAIL $f outputs differ"
        echo "  base:  $(head -c 80 "/tmp/gate-$f-base.txt" 2>/dev/null)"
        echo "  upfwd: $(head -c 80 "/tmp/gate-$f-new-upfront.txt" 2>/dev/null)"
        echo "  chunk: $(head -c 80 "/tmp/gate-$f-new-chunked.txt" 2>/dev/null)"
    fi
done
echo "=== serving gate complete"
