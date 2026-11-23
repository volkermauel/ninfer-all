#!/usr/bin/env python3
"""Queue-path load for the paging admission measurements.

Usage: measure_queue.py PORT CONC TOTAL MAX_TOKENS PROMPT_TOKENS [TIMEOUT_S]
Fires TOTAL requests through a worker pool of CONC (the over-subscription regime when
TOTAL > CONC), each request asking MAX_TOKENS with a prompt of about PROMPT_TOKENS.
Prints one JSON object: admitted (HTTP 200), timeouts (HTTP 503), other errors,
aggregate tok/s over the whole wall, mean wall per request, and p50/p90 wall.
"""
import http.client
import json
import queue
import sys
import threading
import time

HOST = "127.0.0.1"
KEY = "sk-alliancemedicalrp"
MODEL = "inkling-2-mini"  # overridable: argv[6]

PARA = ("The lighthouse keeper recorded the weather every hour in the station log. "
        "Wind speed and direction, barometric pressure, visibility and the state of the sea "
        "were entered in careful columns, together with the movements of ships passing the cape "
        "and the small events of the island. Every entry ended with a short remark about the "
        "mood of the day, and the keeper read the growing book of weather like a diary.")


def prompt_of(tokens):
    # PARA is ~64 tokens; repeat to reach the requested size, then trim greedily.
    repeats = max(1, (tokens + len(PARA) - 1) // len(PARA))
    return PARA * repeats


def main():
    port = int(sys.argv[1])
    conc = int(sys.argv[2])
    total = int(sys.argv[3])
    max_tokens = int(sys.argv[4])
    prompt_tokens = int(sys.argv[5])
    timeout_s = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    if len(sys.argv) > 7 and sys.argv[7]:
        globals()["MODEL"] = sys.argv[7]

    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt_of(prompt_tokens)}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "enable_thinking": False}).encode()
    headers = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

    work = queue.Queue()
    for i in range(total):
        work.put(i)
    results = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    def worker():
        while True:
            try:
                idx = work.get_nowait()
            except queue.Empty:
                return
            rec = {"idx": idx, "ok": False}
            try:
                c = http.client.HTTPConnection(HOST, port, timeout=timeout_s)
                c.request("POST", "/v1/chat/completions", body, headers)
                r = c.getresponse()
                data = r.read()
                t1 = time.perf_counter()
                if r.status != 200:
                    rec.update({"status": r.status, "wall_s": t1 - t0})
                    results.append(rec)
                    return_ = None
                    c.close()
                    continue
                j = json.loads(data)
                t = j.get("timings", {})
                rec.update({"ok": True, "wall_s": t1 - t0,
                            "pred_n": t.get("predicted_n", 0),
                            "engine_tok_s": t.get("predicted_per_second", 0),
                            "prompt_n": j.get("usage", {}).get("prompt_tokens", 0)})
                results.append(rec)
                c.close()
            except Exception as e:  # noqa: BLE001 — the report wants every failure shape
                rec.update({"err": repr(e)[:160]})
                results.append(rec)

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.get("ok")]
    timeouts = [r for r in results if r.get("status") == 503]
    other_err = [r for r in results if not r.get("ok") and r.get("status") != 503
                 and not r.get("err")]
    gen = sum(r["pred_n"] for r in ok)
    walls = sorted(r.get("wall_s", 0) for r in results)
    errs = [r.get("err", "") for r in results if r.get("err")][:3]
    p50 = walls[len(walls) // 2] if walls else 0
    p90 = walls[int(len(walls) * 0.9)] if walls else 0
    print(json.dumps({
        "conc": conc, "total": total, "admitted": len(ok),
        "queue_timeouts_503": len(timeouts), "other_failures": len(other_err) +
        len([r for r in results if r.get("err")]),
        "agg_tok_s": round(gen / wall, 1) if wall else 0,
        "gen_tokens": gen, "wall_s": round(wall, 2),
        "wall_mean_s": round(sum(r.get("wall_s", 0) for r in results) / max(1, len(results)), 2),
        "wall_p50_s": round(p50, 2), "wall_p90_s": round(p90, 2),
        "errors": errs,
    }), flush=True)


if __name__ == "__main__":
    main()
