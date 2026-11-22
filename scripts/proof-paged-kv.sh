#!/usr/bin/env bash
# Functional proof of --kv-paging on the RTX 3090 (device 1): uneven lanes, greedy completions
# (EN/DE/ZH) byte-identical to the dense run, live page reporting, clean over-subscription refusal.
# Production (device 0, ninfer-quant, ports 8080-8082) is never touched.
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

ask() { # port, max_tokens, prompt
    curl -sS "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' -d "$(
        printf '{"max_tokens":%s,"temperature":0,"messages":[{"role":"user","content":%s}]}' \
            "$2" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$3")")"
}

gen_prompt() { # n: a deterministic prompt of about n words
    python3 - "$1" <<'EOF'
import sys
n = int(sys.argv[1])
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]
print("Remember this list of facts: " +
      " ".join(f"{words[i % 10]}-{i} is item {i}." for i in range(n)) +
      "\nReply with the single word OK when you have read them.")
EOF
}

main() {
    echo "== 1. paging on, rk4v4, 4 lanes x 32K (port 8083)"
    PID=$(start_server paged-rk4v4 8083 --max-context 32768 --max-concurrency 4 \
        --kv-dtype rk4v4 --kv-paging --log-stats-interval-ms 1000)
    trap 'kill $PID 2>/dev/null' EXIT
    wait_health 8083 || { echo "FAIL server did not become healthy"; tail -5 "$LOGS/serve-paged-rk4v4.log"; exit 1; }
    grep -m1 "capacity |" "$LOGS/serve-paged-rk4v4.log"

    echo "== 2. uneven lanes: four concurrent requests, 1K .. 24K prompts, 16 .. 512 new tokens"
    (ask 8083 16  "$(gen_prompt 200)"  > "$LOGS/r1.json") &
    (ask 8083 64  "$(gen_prompt 1500)" > "$LOGS/r2.json") &
    (ask 8083 256 "$(gen_prompt 6000)" > "$LOGS/r3.json") &
    (ask 8083 512 "$(gen_prompt 12000)" > "$LOGS/r4.json") &
    wait
    for f in r1 r2 r3 r4; do
        python3 -c "import json;d=json.load(open('$LOGS/$f.json'));print('$f:', repr(d['choices'][0]['message']['content'][:80]))" \
            || { echo "FAIL $f:"; head -c 300 "$LOGS/$f.json"; echo; }
    done

    echo "== 3. greedy completions EN/DE/ZH under load, on the paging server"
    ask 8083 64 "Translate to German, output only the translation: Good morning, the weather is lovely today." > "$LOGS/de.json"
    ask 8083 64 "用一句话解释什么是分页（paging）。" > "$LOGS/zh.json"
    ask 8083 64 "Name the four largest planets of the solar system, shortest answer." > "$LOGS/en.json"
    for f in en de zh; do
        python3 -c "import json;d=json.load(open('$LOGS/$f.json'));print('$f:', repr(d['choices'][0]['message']['content']))"
    done
    grep -E "pages \|" "$LOGS/serve-paged-rk4v4.log" | tail -3

    echo "== 4. dense control (no flag) on port 8084: the same three prompts must match byte for byte"
    PID2=$(start_server dense-rk4v4 8084 --max-context 32768 --max-concurrency 4 --kv-dtype rk4v4)
    wait_health 8084 || { echo "FAIL control server"; exit 1; }
    for f in en de zh; do
        case $f in
            en) P="Name the four largest planets of the solar system, shortest answer.";;
            de) P="Translate to German, output only the translation: Good morning, the weather is lovely today.";;
            zh) P="用一句话解释什么是分页（paging）。";;
        esac
        ask 8084 64 "$P" > "$LOGS/$f-dense.json"
        if cmp -s <(python3 -c "import json;print(json.load(open('$LOGS/$f.json'))['choices'][0]['message']['content'])") \
                  <(python3 -c "import json;print(json.load(open('$LOGS/$f-dense.json'))['choices'][0]['message']['content'])"); then
            echo "OK   $f: paged == dense"
        else
            echo "FAIL $f: paged and dense differ"
        fi
    done
    kill $PID2 2>/dev/null; wait $PID2 2>/dev/null

    echo "== 5. rk8v4 with paging on port 8083 (restart), health + one completion"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; trap - EXIT
    PID=$(start_server paged-rk8v4 8083 --max-context 32768 --max-concurrency 4 \
        --kv-dtype rk8v4 --kv-paging --log-stats-interval-ms 1000)
    trap 'kill $PID 2>/dev/null' EXIT
    wait_health 8083 || { echo "FAIL rk8v4 server"; tail -5 "$LOGS/serve-paged-rk8v4.log"; exit 1; }
    grep -m1 "capacity |" "$LOGS/serve-paged-rk8v4.log"
    ask 8083 48 "Name the four largest planets of the solar system, shortest answer." > "$LOGS/en-rk8v4.json"
    python3 -c "import json;d=json.load(open('$LOGS/en-rk8v4.json'));print('rk8v4:', repr(d['choices'][0]['message']['content']))"
    ask 8083 256 "$(gen_prompt 9000)" > "$LOGS/r5.json" &
    sleep 4; grep -E "pages \|" "$LOGS/serve-paged-rk8v4.log" | tail -2; wait

    echo "== 6. over-subscription: --kv-page-budget 96 MiB (~180 groups) with two 32K lanes"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; trap - EXIT
    PID=$(start_server paged-budget 8083 --max-context 32768 --max-concurrency 2 \
        --kv-dtype rk4v4 --kv-paging --kv-page-budget 96 --pending-timeout-ms 8000)
    trap 'kill $PID 2>/dev/null' EXIT
    wait_health 8083 || { echo "FAIL budget server"; tail -5 "$LOGS/serve-paged-budget.log"; exit 1; }
    grep -m1 "capacity |" "$LOGS/serve-paged-budget.log"
    (ask 8083 4096 "$(gen_prompt 200)" > "$LOGS/b1.json") &
    (ask 8083 4096 "$(gen_prompt 200)" > "$LOGS/b2.json") &
    wait
    for f in b1 b2; do
        head -c 200 "$LOGS/$f.json"; echo " <- $f"
    done
    echo "== done"
}

main "$@"
