#!/usr/bin/env python3
"""Quick decode bench for Glimmer trials. Usage: glimmer-bench.py [port] [c1,c2,...]"""
import json, re, sys, time, urllib.request, concurrent.futures as cf

PORT = sys.argv[1] if len(sys.argv) > 1 else "8210"
CONCS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,4,8").split(",")]
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/v1/models"))["data"][0]["id"]

def metrics():
    try:
        t = urllib.request.urlopen(f"http://localhost:{PORT}/metrics").read().decode()
        d = dict(re.findall(r"(spec_decode_num_accepted_tokens_total|spec_decode_num_drafts_total)\{[^}]*\} ([0-9.e+]+)", t))
        return {k: float(v) for k, v in d.items()}
    except Exception:
        return {}

def gen(i, max_tok=512):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": f"Write a detailed explanation of how photosynthesis works, variant {i}. No thinking, answer directly."}],
            "max_tokens": max_tok, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["usage"]["completion_tokens"], time.time() - t0

print(f"benching port {PORT} at c={CONCS}")
gen(0, 64)  # warmup
for c in CONCS:
    a = metrics()
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        res = list(ex.map(lambda i: gen(i), range(c)))
    wall = time.time() - t0
    b = metrics()
    toks = sum(t for t, _ in res)
    per_user = sum(t / d for t, d in res) / len(res)
    line = f"c={c}: aggregate {toks/wall:6.1f} tok/s | per-user {per_user:5.1f} tok/s"
    if a and b:
        acc = b.get("spec_decode_num_accepted_tokens_total", 0) - a.get("spec_decode_num_accepted_tokens_total", 0)
        dr = b.get("spec_decode_num_drafts_total", 0) - a.get("spec_decode_num_drafts_total", 0)
        if dr:
            line += f" | accepted/draft {acc/dr:.2f}"
    print(line)
