#!/usr/bin/env python3
"""A/B quality harness: RadixArk NVFP4 vs official FP8 (Qwen3.8-27B) on
CODING TASKS + REASONING-TRACE analysis, on the real serving stack.

Why this shape (2026-08-17): quant damage in hybrid-attention models shows up
in agentic/coding behavior and reasoning-chain quality, not in perplexity or
caption tasks. Community signature of a damaged quant: longer, loopier
reasoning with lower final accuracy. So we measure BOTH the outcome (tests
pass?) and the trace (length, repetition, whether it converges).

Usage:
  python3 patches/qwen-quant-ab.py --base http://127.0.0.1:8001 --label nvfp4
  # swap model, then:
  python3 patches/qwen-quant-ab.py --base http://127.0.0.1:8001 --label fp8
  python3 patches/qwen-quant-ab.py --compare nvfp4 fp8

Results land in scratch dir (or --outdir): per-task JSON incl. full reasoning
traces for human review, plus a summary table. Runs each task at medium AND
high effort — trace degradation under long reasoning is where quants crack.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# ── Tasks: verifiable coding problems. `check` runs in a sandboxed python
# with the model's code; task passes iff exit 0. Mix: algorithmic, debugging,
# edge-case reasoning, and code comprehension with an exact expected answer.
TASKS = [
    {
        "id": "interval-merge",
        "prompt": "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals given as [start, end] pairs (ints, possibly unsorted, possibly touching like [1,2],[2,3] which merge). Return merged intervals sorted by start. Reply with ONLY a python code block.",
        "check": """
{code}
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,2],[2,3]]) == [[1,3]]
assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]
assert merge_intervals([]) == []
assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]
""",
    },
    {
        "id": "lru-ttl",
        "prompt": "Implement a Python class TTLCache(capacity, ttl) with get(key) and put(key, value, now) and get(key, now) where `now` is an explicit float timestamp parameter (no real clocks). Entries expire when now - insert_time >= ttl. On overflow evict least-recently-USED non-expired entry (get and put both count as use). get returns None for missing/expired. Reply with ONLY a python code block.",
        "check": """
{code}
c = TTLCache(2, 10.0)
c.put('a', 1, 0.0); c.put('b', 2, 1.0)
assert c.get('a', 2.0) == 1
c.put('c', 3, 3.0)                 # evicts b (a was used at t=2)
assert c.get('b', 4.0) is None
assert c.get('a', 5.0) == 1
assert c.get('a', 10.0) is None    # expired (10-0 >= 10)
assert c.get('c', 12.9) == 3
assert c.get('c', 13.0) is None
""",
    },
    {
        "id": "debug-race",
        "prompt": "This function has a bug: it should return the indices of the two numbers adding to target, preferring the pair with the SMALLEST first index, but it fails some cases.\n\ndef two_sum_pref(nums, target):\n    seen = {}\n    best = None\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            cand = (seen[target - x], i)\n            if best is None or cand > best:\n                best = cand\n        seen[x] = i\n    return best\n\nFix ONLY the bug (comparison logic), keep the algorithm. Reply with ONLY a python code block containing the fixed function.",
        "check": """
{code}
assert two_sum_pref([2,7,11,15], 9) == (0,1)
assert two_sum_pref([3,2,4,3], 6) == (0,3)
assert two_sum_pref([1,5,3,5,1], 6) == (0,1)
assert two_sum_pref([1,2], 5) is None
""",
    },
    {
        "id": "edge-parser",
        "prompt": "Write a Python function parse_size(s) that parses human sizes like '10GB', '512 MiB', '1.5tb', '100' (bare = bytes) into integer bytes. Decimal units (KB/MB/GB/TB) are powers of 1000; binary (KiB/MiB/GiB/TiB) are powers of 1024. Case-insensitive, optional whitespace between number and unit. Raise ValueError on anything else (negative, garbage, empty). Truncate fractional bytes toward zero. Reply with ONLY a python code block.",
        "check": """
{code}
assert parse_size('10GB') == 10_000_000_000
assert parse_size('512 MiB') == 512*1024*1024
assert parse_size('1.5tb') == 1_500_000_000_000
assert parse_size('100') == 100
assert parse_size('0.5 KiB') == 512
for bad in ['', '-5GB', '10XB', 'GB', '1.2.3MB']:
    try:
        parse_size(bad); assert False, bad
    except ValueError:
        pass
""",
    },
    {
        "id": "comprehension",
        "prompt": "Read this function carefully:\n\ndef f(xs):\n    out = []\n    st = []\n    for i, x in enumerate(xs):\n        while st and xs[st[-1]] < x:\n            out.append((st.pop(), i))\n        st.append(i)\n    return out\n\nFor input xs = [2, 1, 2, 4, 3], what exactly does f return? Answer with ONLY a python code block defining ANSWER = <the exact return value>.",
        "check": """
{code}
assert ANSWER == [(1,2),(0,3),(2,3)], ANSWER
""",
    },
    {
        "id": "refactor-gen",
        "prompt": "Rewrite this to a lazy generator chunked(iterable, n) yielding lists of up to n items, working on ANY iterable (including generators, no len()), n>=1, last chunk may be short:\n\ndef chunked(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]\n\nReply with ONLY a python code block.",
        "check": """
{code}
import types
g = chunked((x for x in range(7)), 3)
assert isinstance(g, types.GeneratorType) or hasattr(g, '__next__')
assert list(g) == [[0,1,2],[3,4,5],[6]]
assert list(chunked([], 2)) == []
assert list(chunked(iter('abcd'), 1)) == [['a'],['b'],['c'],['d']]
""",
    },
]

EFFORTS = ["medium", "high"]
SAMPLES = 2  # per task per effort


def chat(base, prompt, effort, max_tokens=8192, timeout=600):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy-key"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = json.load(r)
    msg = b["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "finish": b["choices"][0].get("finish_reason"),
        "completion_tokens": b.get("usage", {}).get("completion_tokens"),
        "wall_s": round(time.time() - t0, 1),
    }


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else text


def run_check(code, check_tmpl):
    src = check_tmpl.format(code=code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
        return r.returncode == 0, (r.stderr or r.stdout)[-400:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (infinite loop?)"
    finally:
        Path(path).unlink(missing_ok=True)


def trace_metrics(reasoning):
    """Loop/quality signals on the reasoning trace."""
    words = reasoning.split()
    n = len(words)
    # repetition: fraction of duplicate 8-grams (loopy traces repeat themselves)
    grams = [" ".join(words[i:i+8]) for i in range(0, max(0, n - 8))]
    rep = 1 - (len(set(grams)) / len(grams)) if grams else 0.0
    # second-guess churn: how often it reverses course
    flips = len(re.findall(r"\b(wait|actually|hmm|no,|scratch that|let me reconsider)\b",
                           reasoning, re.I))
    return {"reasoning_chars": len(reasoning), "dup_8gram_frac": round(rep, 3),
            "self_corrections": flips}


def run_suite(base, label, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for task in TASKS:
        for effort in EFFORTS:
            for s in range(SAMPLES):
                r = chat(base, task["prompt"], effort)
                code = extract_code(r["content"])
                ok, err = run_check(code, task["check"])
                row = {
                    "task": task["id"], "effort": effort, "sample": s,
                    "pass": ok, "finish": r["finish"],
                    "completion_tokens": r["completion_tokens"],
                    "wall_s": r["wall_s"], **trace_metrics(r["reasoning"]),
                }
                results.append(row)
                # full artifacts for human trace review
                (outdir / f"{task['id']}-{effort}-{s}.json").write_text(json.dumps(
                    {**row, "reasoning": r["reasoning"], "content": r["content"],
                     "check_error": None if ok else err}, indent=1))
                print(f"  {task['id']:14} {effort:6} s{s}: "
                      f"{'PASS' if ok else 'FAIL'} tok={r['completion_tokens']} "
                      f"dup={row['dup_8gram_frac']} corr={row['self_corrections']}")
    (outdir / "summary.json").write_text(json.dumps(results, indent=1))
    agg(results, label)
    return results


def agg(results, label):
    n = len(results)
    passed = sum(r["pass"] for r in results)
    print(f"\n== {label}: {passed}/{n} passed "
          f"({100*passed/n:.0f}%) | median tokens "
          f"{sorted(r['completion_tokens'] or 0 for r in results)[n//2]} | "
          f"mean dup8 {sum(r['dup_8gram_frac'] for r in results)/n:.3f} | "
          f"mean self-corrections {sum(r['self_corrections'] for r in results)/n:.1f}")


def compare(outroot, a, b):
    ra = json.loads((outroot / a / "summary.json").read_text())
    rb = json.loads((outroot / b / "summary.json").read_text())
    print(f"\n{'':16}{a:>14}{b:>14}")
    for eff in EFFORTS:
        for name, sel in [("pass rate", lambda r: r["pass"]),
                          ("med tokens", None)]:
            xa = [r for r in ra if r["effort"] == eff]
            xb = [r for r in rb if r["effort"] == eff]
            if sel:
                print(f"{eff} {name:12}{100*sum(map(sel,xa))/len(xa):>13.0f}%"
                      f"{100*sum(map(sel,xb))/len(xb):>13.0f}%")
            else:
                ma = sorted(r['completion_tokens'] or 0 for r in xa)[len(xa)//2]
                mb = sorted(r['completion_tokens'] or 0 for r in xb)[len(xb)//2]
                print(f"{eff} {name:12}{ma:>14}{mb:>14}")
    for name, key in [("dup 8-gram", "dup_8gram_frac"), ("self-corr", "self_corrections")]:
        print(f"{name:14}{sum(r[key] for r in ra)/len(ra):>14.3f}"
              f"{sum(r[key] for r in rb)/len(rb):>14.3f}")
    print("\nPer-task disagreements (trace review targets):")
    for t in TASKS:
        pa = sum(r["pass"] for r in ra if r["task"] == t["id"])
        pb = sum(r["pass"] for r in rb if r["task"] == t["id"])
        if pa != pb:
            print(f"  {t['id']}: {a}={pa} vs {b}={pb} → read {t['id']}-*.json traces")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--label")
    ap.add_argument("--outdir", default=str(Path.home() / ".sparkstation" / "quant-ab"))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    root = Path(args.outdir)
    if args.compare:
        compare(root, *args.compare)
    else:
        if not args.label:
            ap.error("--label required for a run")
        run_suite(args.base.rstrip("/"), args.label, root / args.label)
