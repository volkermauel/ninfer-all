#!/usr/bin/env python3
"""bench.py-style HTTP client for the page sweep and the dedup A/B (server on 127.0.0.1).

Modes:
  paged_bench.py PORT 1 8              decode: ~1,373-token prompts, 256 gen, one line per point
  paged_bench.py PORT prefill10000     a single ~10K-token prompt, 256 gen, prefill ms focus
  paged_bench.py PORT dedup N SESSIONS_PREFIX_SUFFIX
                                       N sessions share a common prefix with divergent suffixes
Prints one JSON line per point. Metrics: aggregate tok/s, per-stream tok/s, prefill ms, prompt_n.
"""
import json, sys, threading, time, http.client

KEY = "sk-bench"
MODEL = "sweep"
GEN = 256

PARA = ("The lighthouse keeper recorded the weather every hour in the station log. "
        "Wind speed and direction, barometric pressure, visibility and the state of the sea "
        "were entered in careful columns, together with the movements of ships passing the cape "
        "and the small events of the island. Every entry ended with a short remark about the "
        "mood of the day, and the keeper read the growing book of weather like a diary.")

def request_body(prompt, gen=GEN):
    return json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": gen, "temperature": 0, "enable_thinking": False})

HEADERS = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

def one_request(port, body, results, idx):
    t0 = time.perf_counter()
    rec = {"idx": idx, "ok": False}
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        c.request("POST", "/v1/chat/completions", body, HEADERS)
        r = c.getresponse()
        data = r.read()
        t1 = time.perf_counter()
        if r.status != 200:
            rec["status"] = r.status
            rec["err"] = data[:200].decode("utf-8", "replace")
            results.append(rec)
            return
        j = json.loads(data)
        t = j.get("timings", {})
        rec.update({
            "ok": True,
            "wall_s": round(t1 - t0, 3),
            "pred_n": t.get("predicted_n", 0),
            "engine_tok_s": round(t.get("predicted_per_second", 0), 2),
            "prefill_ms": t.get("prompt_ms", t.get("prefill_ms")),
            "prompt_n": j.get("usage", {}).get("prompt_tokens", 0),
            "content": j.get("choices", [{}])[0].get("message", {}).get("content", ""),
        })
        results.append(rec)
    except Exception as e:
        rec["err"] = repr(e)[:200]
        results.append(rec)

def run_point(port, bodies, label):
    results, threads = [], []
    t0 = time.perf_counter()
    for i, body in enumerate(bodies):
        th = threading.Thread(target=one_request, args=(port, body, results, i))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    ok = [r for r in results if r.get("ok")]
    gen = sum(r["pred_n"] for r in ok)
    prefill = [r["prefill_ms"] for r in ok if r["prefill_ms"] is not None]
    point = {
        "label": label,
        "wall_s": round(wall, 2),
        "agg_tok_s": round(gen / wall, 1) if wall else 0.0,
        "streams_ok": len(ok), "streams_total": len(bodies),
        "per_stream_mean": round(sum(r["engine_tok_s"] for r in ok) / len(ok), 1) if ok else 0.0,
        "prefill_ms_mean": round(sum(prefill) / len(prefill), 1) if prefill else None,
        "prompt_n": ok[0]["prompt_n"] if ok else 0,
        "errors": [r.get("err", r.get("status")) for r in results if not r.get("ok")][:3],
    }
    print(json.dumps(point), flush=True)
    return point, ok

def main():
    port = int(sys.argv[1])
    args = sys.argv[2:]
    if not args:
        args = ["1", "8"]
    if args[0] == "prefill10000":
        prompt = PARA * 155
        bodies = [request_body(prompt)]
        run_point(port, bodies, "prefill10000")
        return
    if args[0] == "dedup":
        # dedup N: N sessions, PARA*x shared prefix + per-session divergent suffix.
        sessions = int(args[1])
        shared = PARA * 62  # ~1,000 tokens
        bodies = []
        for s in range(sessions):
            suffix = (f"Session {s}: summarize the keeper's duties in one sentence, then list "
                      f"exactly {5 + s} instruments a lighthouse station keeps, numbered.")
            bodies.append(request_body(shared + "\n\n" + suffix))
        run_point(port, bodies, f"dedup{sessions}")
        return
    for conc in args:
        prompt = PARA * 21  # ~1,373 tokens
        bodies = [request_body(prompt) for _ in range(int(conc))]
        run_point(port, bodies, f"conc{conc}")

main()
