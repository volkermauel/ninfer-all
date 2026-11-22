#!/usr/bin/env bash
# Functional proof of --kv-paging on the RTX 3090 (device 1): uneven lanes, greedy completions
# (EN/DE/ZH, reasoning included) byte-identical to the dense run, live page reporting, clean
# over-subscription refusal. Production (device 0, port 8080) is never touched.
set -uo pipefail

LOGS=~/git/ninfer-all-paged/logs
BIN=~/git/ninfer-all-paged/build/apps/ninfer-serve
MODEL=~/ninfer/models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
mkdir -p "$LOGS"

start_server() { # name, port, extra args...
    local name=$1 port=$2; shift 2
    CUDA_VISIBLE_DEVICES=1 "$BIN" "$MODEL" --model-id "paged-impl-$name" --host 127.0.0.1 \
        --port "$port" "$@" >"$LOGS/serve-$name.log" 2>&1 &
    echo $!
}

wait_health() { # port
    local port=$1
    for _ in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
        sleep 2
    done
    return 1
}

gen_prompt() { # n outfile: a deterministic prompt of about n clauses
    python3 - "$1" > "$2" <<'EOF'
import sys
n = int(sys.argv[1])
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]
print("Remember this list of facts: " +
      " ".join(f"{words[i % 10]}-{i} is item {i}." for i in range(n)) +
      "\nReply with the single word OK when you have read them.")
EOF
}

ask_file() { # port max_tokens prompt_file outfile
    python3 - "$1" "$2" "$3" > "$4" <<'EOF'
import json, sys, urllib.request
port, tokens, path = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open(path).read()
body = json.dumps({"max_tokens": int(tokens), "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request, timeout=600).read().decode())
EOF
}

say() { # port max_tokens prompt outfile  (short prompts straight from the argument)
    python3 - "$1" "$2" "$3" > "$4" <<'EOF'
import json, sys, urllib.request
port, tokens, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"max_tokens": int(tokens), "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request, timeout=600).read().decode())
EOF
}

message_of() { # json_file: the reply's message (content and reasoning) as one string
    python3 - "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    m = d["choices"][0]["message"]
    print(repr(m.get("content")) + " reasoning " + repr(m.get("reasoning_content")))
except Exception as e:
    print("ERROR " + repr(e) + " body " + open(sys.argv[1]).read()[:160])
EOF
}

main() {
    echo "== 1. paging on, rk4v4, 4 lanes x 32K (port 8083)"
    PID=$(start_server paged-rk4v4 8083 --max-context 32768 --max-concurrency 4 \
        --kv-dtype rk4v4 --kv-paging --log-stats-interval-ms 1000)
    trap 'kill $PID 2>/dev/null' EXIT
    wait_health 8083 || { echo "FAIL server did not become healthy"; tail -5 "$LOGS/serve-paged-rk4v4.log"; exit 1; }
    grep -m1 "capacity |" "$LOGS/serve-paged-rk4v4.log"

    echo "== 2. uneven lanes: four concurrent requests, ~0.8K .. ~28K tokens, 64 .. 512 new tokens"
    gen_prompt 150 "$LOGS/p1.txt"; gen_prompt 1500 "$LOGS/p2.txt"
    gen_prompt 3200 "$LOGS/p3.txt"; gen_prompt 5200 "$LOGS/p4.txt"
    (ask_file 8083 64  "$LOGS/p1.txt" "$LOGS/r1.json") &
    (ask_file 8083 128 "$LOGS/p2.txt" "$LOGS/r2.json") &
    (ask_file 8083 256 "$LOGS/p3.txt" "$LOGS/r3.json") &
    (ask_file 8083 512 "$LOGS/p4.txt" "$LOGS/r4.json") &
    wait
    for f in r1 r2 r3 r4; do
        message_of "$LOGS/$f.json" | head -c 200 || { echo "FAIL $f:"; head -c 300 "$LOGS/$f.json"; echo; }
    done
    grep -E "pages \|" "$LOGS/serve-paged-rk4v4.log" | tail -2

    echo "== 3. greedy completions EN/DE/ZH (512 tokens: reasoning plus answer)"
    say 8083 512 "Name the four largest planets of the solar system, shortest answer." "$LOGS/en.json"
    say 8083 512 "Translate to German, output only the translation: Good morning, the weather is lovely today." "$LOGS/de.json"
    say 8083 512 "用一句话解释什么是分页（paging）。" "$LOGS/zh.json"
    for f in en de zh; do
        message_of "$LOGS/$f.json" | head -c 300
    done
    grep -E "pages \|" "$LOGS/serve-paged-rk4v4.log" | tail -2

    echo "== 4. dense control (no flag) on port 8084: the same three replies must match"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; trap - EXIT
    sleep 5
    PID2=$(start_server dense-rk4v4 8084 --max-context 32768 --max-concurrency 4 --kv-dtype rk4v4)
    trap 'kill $PID2 2>/dev/null' EXIT
    wait_health 8084 || { echo "FAIL control server"; tail -3 "$LOGS/serve-dense-rk4v4.log"; exit 1; }
    grep -m1 "capacity |" "$LOGS/serve-dense-rk4v4.log"
    say 8084 512 "Name the four largest planets of the solar system, shortest answer." "$LOGS/en-dense.json"
    say 8084 512 "Translate to German, output only the translation: Good morning, the weather is lovely today." "$LOGS/de-dense.json"
    say 8084 512 "用一句话解释什么是分页（paging）。" "$LOGS/zh-dense.json"
    for f in en de zh; do
        if [ "$(message_of "$LOGS/$f.json")" = "$(message_of "$LOGS/$f-dense.json")" ]; then
            echo "OK   $f: paged == dense"
        else
            echo "FAIL $f: paged and dense differ"
            echo "  paged: $(message_of "$LOGS/$f.json" | head -c 200)"
            echo "  dense: $(message_of "$LOGS/$f-dense.json" | head -c 200)"
        fi
    done
    kill $PID2 2>/dev/null; wait $PID2 2>/dev/null; trap - EXIT

    echo "== 5. rk8v4 with paging (restart on 8083), health + completions + live pages"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; trap - EXIT
    PID=$(start_server paged-rk8v4 8083 --max-context 32768 --max-concurrency 4 \
        --kv-dtype rk8v4 --kv-paging --log-stats-interval-ms 1000)
    trap 'kill $PID 2>/dev/null' EXIT
    wait_health 8083 || { echo "FAIL rk8v4 server"; tail -5 "$LOGS/serve-paged-rk8v4.log"; exit 1; }
    grep -m1 "capacity |" "$LOGS/serve-paged-rk8v4.log"
    say 8083 512 "Name the four largest planets of the solar system, shortest answer." "$LOGS/en-rk8v4.json"
    message_of "$LOGS/en-rk8v4.json" | head -c 300
    (ask_file 8083 256 "$LOGS/p3.txt" "$LOGS/r5.json") &
    sleep 8; grep -E "pages \|" "$LOGS/serve-paged-rk8v4.log" | tail -2; wait

    echo "== 6. over-subscription: --kv-page-budget 32 MiB (~468 groups), two ~28K-token requests"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; trap - EXIT
    PID=$(start_server paged-budget 8083 --max-context 32768 --max-concurrency 2 \
        --kv-dtype rk4v4 --kv-paging --kv-page-budget 32 --pending-timeout-ms 8000)
    trap 'kill $PID 2>/dev/null' EXIT
    wait_health 8083 || { echo "FAIL budget server"; tail -5 "$LOGS/serve-paged-budget.log"; exit 1; }
    grep -m1 "capacity |" "$LOGS/serve-paged-budget.log"
    (ask_file 8083 64 "$LOGS/p4.txt" "$LOGS/b1.json") &
    (ask_file 8083 64 "$LOGS/p4.txt" "$LOGS/b2.json") &
    wait
    for f in b1 b2; do
        echo "$f: $(head -c 220 "$LOGS/$f.json")"
    done
    grep -E "pages \|" "$LOGS/serve-paged-budget.log" | tail -1
    echo "== done"
}

main "$@"
