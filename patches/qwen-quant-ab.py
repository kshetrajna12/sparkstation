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
    # ── HARD tier (daily-driver-grade) ──────────────────────────────────
    {
        "id": "regex-lite",
        "max_tokens": 16384,
        "prompt": "Implement match(pattern, s) -> bool in Python: full-string regex matching supporting literal chars, '.' (any single char), and '*' (zero or more of the PRECEDING element). No other metachars. Implement the semantics yourself (no `re`). Reply with ONLY a python code block.",
        "check": """
{code}
import sys
sys.setrecursionlimit(100000)
assert match("a", "aa") == False
assert match("a*", "aa") == True
assert match(".*", "ab") == True
assert match("c*a*b", "aab") == True
assert match("mis*is*p*.", "mississippi") == False
assert match("mis*is*ip*.", "mississippi") == True
assert match("a*a*a*a*a*a*b", "aaaaaaaaaaaaaaaaaaab") == True
assert match("", "") == True
assert match("a*", "") == True
assert match(".", "") == False
assert match("ab*", "a") == True
assert "re." not in {code!r}
""",
    },
    {
        "id": "expr-eval",
        "max_tokens": 16384,
        "prompt": "Write a Python function evaluate(expr) that evaluates arithmetic expressions with + - * / ** parentheses and unary minus. Rules: normal precedence (** highest, then unary minus, then * /, then + -), ** is RIGHT-associative, unary minus binds tighter than ** on its LEFT operand (so -2**2 == 4 i.e. (-2)**2), / is float division, whitespace allowed anywhere, ints stay exact otherwise float. Raise ValueError on malformed input. No eval/exec/ast. Reply with ONLY a python code block.",
        "check": """
{code}
assert evaluate("2+3*4") == 14
assert evaluate("(2+3)*4") == 20
assert evaluate("2**3**2") == 512
assert evaluate("-2**2") == 4
assert evaluate("2--3") == 5
assert evaluate(" 7 / 2 ") == 3.5
assert evaluate("-(3+4)*2") == -14
assert evaluate("2*-3") == -6
for bad in ["", "2+", "(2", "2 3", "*3", "2**", "()"]:
    try:
        evaluate(bad); assert False, bad
    except ValueError:
        pass
src = {code!r}
assert "eval(" not in src.replace("evaluate(", "") and "exec(" not in src
""",
    },
    {
        "id": "edit-script",
        "max_tokens": 16384,
        "prompt": "Write a Python function edit_script(a, b) for strings a, b returning a MINIMAL-length list of operations transforming a into b using only deletions and insertions (no substitutions). Op format: ('del', i) deletes the char at index i of the CURRENT string; ('ins', i, ch) inserts ch so it lands at index i of the CURRENT string. Ops apply sequentially. Minimal length = len(a)+len(b)-2*LCS(a,b). Reply with ONLY a python code block.",
        "check": """
{code}
def lcs(a, b):
    import functools
    @functools.lru_cache(None)
    def f(i, j):
        if i == len(a) or j == len(b): return 0
        return 1+f(i+1,j+1) if a[i]==b[j] else max(f(i+1,j), f(i,j+1))
    return f(0,0)
def apply(a, ops):
    s = list(a)
    for op in ops:
        if op[0] == 'del': del s[op[1]]
        else: s.insert(op[1], op[2])
    return ''.join(s)
for a, b in [("kitten","sitting"), ("", "abc"), ("abc",""), ("same","same"),
             ("ABCABBA","CBABAC"), ("x"*30+"y", "y"+"x"*30)]:
    ops = edit_script(a, b)
    assert apply(a, ops) == b, (a, b, ops)
    assert len(ops) == len(a)+len(b)-2*lcs(a,b), (a, b, len(ops))
""",
    },
    {
        "id": "wis-schedule",
        "max_tokens": 16384,
        "prompt": "Write a Python function best_schedule(jobs) solving weighted interval scheduling: jobs is a list of (start, end, weight) with end>start, weights>0; two jobs conflict if their half-open intervals [start,end) overlap. Return (max_total_weight, chosen) where chosen is the list of job INDICES (into the input list) of one optimal non-conflicting set, sorted by start time. Must be O(n log n) — it will be called with 200k jobs, so no O(n^2). Reply with ONLY a python code block.",
        "check": """
{code}
w, ch = best_schedule([(1,3,5),(2,5,6),(4,6,5),(6,7,4),(5,8,11),(7,9,2)])
assert w == 17, w
picked = [(1,3,5),(2,5,6),(4,6,5),(6,7,4),(5,8,11),(7,9,2)]
iv = [picked[i] for i in ch]
assert sum(x[2] for x in iv) == 17
assert all(iv[k][1] <= iv[k+1][0] for k in range(len(iv)-1))
assert best_schedule([]) == (0, [])
assert best_schedule([(0,10,3),(0,10,7)])[0] == 7
import random, time
random.seed(7)
big = []
for _ in range(200000):
    s0 = random.randrange(10**6); big.append((s0, s0+random.randrange(1,50), random.randrange(1,100)))
t0 = time.time(); wbig, chbig = best_schedule(big); dt = time.time()-t0
assert dt < 20, f"too slow: {{dt}}s"
ivb = [big[i] for i in chbig]
ivb.sort()
assert all(ivb[k][1] <= ivb[k+1][0] for k in range(len(ivb)-1))
assert sum(x[2] for x in ivb) == wbig
""",
    },
    {
        "id": "token-bucket",
        "max_tokens": 16384,
        "prompt": "Implement a Python class TokenBucket(capacity, refill_rate) with method allow(now, cost=1) -> bool. Semantics: bucket starts FULL; tokens refill continuously at refill_rate per second (fractional accumulation, capped at capacity); allow spends `cost` tokens and returns True iff at least `cost` tokens are available at time `now`; a denied request spends NOTHING. `now` is a monotonically non-decreasing float passed in (no real clocks). Reply with ONLY a python code block.",
        "check": """
{code}
b = TokenBucket(3, 1.0)
assert [b.allow(0.0) for _ in range(3)] == [True, True, True]
assert b.allow(0.0) == False
assert b.allow(0.5) == False
assert b.allow(1.0) == True
assert b.allow(1.0) == False
b2 = TokenBucket(10, 2.0)
assert b2.allow(0.0, cost=10) == True
assert b2.allow(2.0, cost=5) == False
assert b2.allow(3.0, cost=5) == True
assert b2.allow(1000.0, cost=10) == True
b3 = TokenBucket(1, 0.1)
assert b3.allow(0.0) == True
assert b3.allow(5.0) == False
assert b3.allow(10.0) == True
""",
    },
    {
        "id": "real-fix-ports",
        "max_tokens": 16384,
        "prompt": "This is REAL production code from an LLM-cluster supervisor (port allocator). Spec: the port range is INCLUSIVE on both ends (8001..8100 = exactly 100 allocatable ports); allocate_port returns the lowest free port; release with full_release=True frees the port for reuse. There is exactly one bug. Find it and reply with ONLY a python code block containing the fully corrected class (minimal change).\n\nclass PortAllocator:\n    def __init__(self, start=8001, end=8100):\n        self.port_range = (start, end)\n        self.allocated_ports = {{}}\n    def allocate_port(self, model_id):\n        allocated_set = set(self.allocated_ports.values())\n        for port in range(self.port_range[0], self.port_range[1]):\n            if port not in allocated_set:\n                self.allocated_ports[model_id] = port\n                return port\n        raise RuntimeError('No available ports in range')\n    def release(self, model_id, full_release=False):\n        if full_release and model_id in self.allocated_ports:\n            del self.allocated_ports[model_id]",
        "check": """
{code}
pa = PortAllocator(8001, 8100)
ports = [pa.allocate_port(f"m{{i}}") for i in range(100)]
assert ports[0] == 8001 and ports[-1] == 8100 and len(set(ports)) == 100
try:
    pa.allocate_port("overflow"); assert False
except RuntimeError:
    pass
pa.release("m50", full_release=True)
assert pa.allocate_port("m50b") == 8051
pa.release("m0")
try:
    pa.allocate_port("m0b"); assert False, "non-full release must NOT free the port"
except RuntimeError:
    pass
""",
    },
    {
        "id": "real-fix-comps",
        "max_tokens": 16384,
        "prompt": "This is REAL code from a real-estate comps engine (simplified from production). Spec: return comps within radius_miles of (lat, lon) using flat-earth distance dy=(dlat)*69 miles, dx=(dlon)*69*cos(lat in RADIANS) miles; filter by bed count — BUT counties like Maricopa publish beds as None, and unknown bed counts must PASS bed filters (never be coerced to a failing value); result sorted by distance_miles (rounded to 2dp, included in each dict). There is exactly one bug versus this spec. Reply with ONLY a python code block containing the corrected function (minimal change).\n\nimport math\n\ndef filter_comps(rows, lat, lon, radius_miles, min_beds=None, max_beds=None):\n    def dist_miles(r):\n        dy = (r['lat'] - lat) * 69.0\n        dx = (r['lon'] - lon) * 69.0 * math.cos(math.radians(lat))\n        return math.hypot(dx, dy)\n    comps = []\n    for r in rows:\n        d = dist_miles(r)\n        if d > radius_miles:\n            continue\n        beds = r['beds'] if r['beds'] is not None else 0\n        if min_beds is not None and beds < min_beds:\n            continue\n        if max_beds is not None and beds > max_beds:\n            continue\n        comps.append({**r, 'distance_miles': round(d, 2)})\n    comps.sort(key=lambda c: c['distance_miles'])\n    return comps",
        "check": """
{code}
rows = [
    {{'id': 1, 'lat': 33.45, 'lon': -112.07, 'beds': None}},
    {{'id': 2, 'lat': 33.46, 'lon': -112.07, 'beds': 2}},
    {{'id': 3, 'lat': 33.45, 'lon': -112.06, 'beds': 4}},
    {{'id': 4, 'lat': 34.45, 'lon': -112.07, 'beds': 3}},
]
out = filter_comps(rows, 33.45, -112.07, 5.0, min_beds=3)
ids = [c['id'] for c in out]
assert 1 in ids, "None beds must pass min_beds filter (Maricopa case)"
assert 2 not in ids and 3 in ids and 4 not in ids
assert ids[0] == 1
out2 = filter_comps(rows, 33.45, -112.07, 5.0, max_beds=3)
assert [c['id'] for c in out2] == [1, 2]
assert out[0]['distance_miles'] == 0.0
import math
d3 = filter_comps(rows, 33.45, -112.07, 5.0)[1]['distance_miles']
assert abs(d3 - round(69.0*math.cos(math.radians(33.45))*0.01, 2)) < 0.02
""",
    },
    {
        "id": "real-hard-valuation",
        "max_tokens": 20000,
        "prompt": "HARD problem from a real-estate valuation engine (this is the core of a real codebase; ZHVI = Zillow-style monthly home-value index). Implement estimate_value(subject, comps, index, as_of) -> float.\n\nInputs: subject = {{'sqft': int}}; comps = list of {{'sale_price': int, 'sqft': int, 'sale_date': 'YYYY-MM'}}; index = dict mapping 'YYYY-MM' -> float (market index level, all needed months present incl. as_of); as_of = 'YYYY-MM'.\n\nGoal: estimate the subject's market value AS OF as_of. Real-world conditions your estimator MUST survive (the tests are built from them): (1) markets move — comps may be 6-18 months old while the index rose or fell 15-25%, so raw sale prices are stale; (2) occasional corrupted records — a comp may have a wildly wrong price (data-entry error, non-arms-length sale); (3) comps differ in size — value scales with sqft. Accuracy required: within 3% of true value on every scenario. Think carefully about HOW to use the index and how to aggregate robustly. Reply with ONLY a python code block.",
        "check": """
{code}
import random
def scenario(seed, drift, n=7, outlier=False):
    rng = random.Random(seed)
    months = [f"20{{24+(m//12)}}-{{(m%12)+1:02d}}" for m in range(24)]
    idx = {{}}
    level = 100.0
    for mo in months:
        idx[mo] = level
        level *= (1 + drift + rng.uniform(-0.004, 0.004))
    as_of = months[-1]
    true_ppsf_asof = 300.0 * idx[as_of] / 100.0
    comps = []
    for i in range(n):
        mo = months[rng.randrange(4, 20)]
        sqft = rng.randrange(1200, 3200)
        ppsf_at_sale = true_ppsf_asof * idx[mo] / idx[as_of]
        price = ppsf_at_sale * sqft * (1 + rng.uniform(-0.015, 0.015))
        comps.append({{'sale_price': int(price), 'sqft': sqft, 'sale_date': mo}})
    if outlier:
        comps[2] = dict(comps[2], sale_price=comps[2]['sale_price'] * 3)
    subject = {{'sqft': 2000}}
    truth = true_ppsf_asof * 2000
    return subject, comps, idx, as_of, truth

for seed, drift, outl in [(1, 0.012, False), (2, 0.015, True), (3, -0.010, False),
                          (4, 0.018, True), (5, 0.000, True), (6, 0.020, False)]:
    subj, comps, idx, as_of, truth = scenario(seed, drift, outlier=outl)
    est = estimate_value(subj, comps, idx, as_of)
    err = abs(est - truth) / truth
    assert err < 0.03, f"seed={{seed}} drift={{drift}} outlier={{outl}}: est {{est:.0f}} vs truth {{truth:.0f}} ({{100*err:.1f}}% off)"
""",
    },
    # ── ESCALATION tier: limit-finding. Not part of the standard battery;
    # run explicitly by id when arms keep passing. ────────────────────────
    {
        "id": "esc-interpreter",
        "max_tokens": 24000,
        "prompt": "Implement run(src) -> int|float for a mini-language. Grammar: program = expr; expr supports: integer/float literals; variables; let-bindings `let NAME = expr in expr` (lexically scoped, shadowing allowed); conditionals `if expr then expr else expr` (0 and 0.0 are falsey, everything else truthy); comparison ops < > <= >= == != (result 1 or 0); arithmetic + - * / with normal precedence (comparisons bind LOOSER than arithmetic); parentheses; unary minus. Whitespace/newlines insignificant. Right-to-left evaluation NOT required — evaluate normally. Raise ValueError on syntax errors or unbound variables. No eval/exec. Reply with ONLY a python code block.",
        "check": """
{code}
assert run("1+2*3") == 7
assert run("let x = 5 in x*x") == 25
assert run("let x = 2 in let x = x+1 in x*10") == 30
assert run("if 2>1 then 10 else 20") == 10
assert run("if 0 then 10 else 20") == 20
assert run("let a = 3 in if a==3 then a*2 else -1") == 6
assert run("let f = 1+1 in if f >= 2 then (let g = f*f in g+1) else 0") == 5
assert run("-(2+3)*2") == -10
assert run("1 < 2 + 3") == 1
assert run("(1<2) + 3") == 4
assert run("if 1<2 then if 0 then 1 else 2 else 3") == 2
for bad in ["let x = in 3", "x+1", "if 1 then 2", "let 3 = 4 in 5", "1 +", "()"]:
    try:
        run(bad); assert False, bad
    except ValueError:
        pass
src = {code!r}
assert "eval(" not in src and "exec(" not in src
""",
    },
    {
        "id": "esc-ot-converge",
        "max_tokens": 24000,
        "prompt": "Operational transformation for concurrent text edits. Ops: ('ins', pos, ch) and ('del', pos). apply(doc, op) is standard (insert before pos / delete at pos). Implement transform(op_a, op_b) -> op_a2: rewrite op_a so that applying op_b then op_a2 has the same effect as the (intended) op_a on the original doc. Required convergence property (TP1): for any doc and any valid concurrent ops a, b: apply(apply(doc,a), transform(b,a)) == apply(apply(doc,b), transform(a,b)). Tie-break rule when both insert at the SAME position: the op whose char has the LOWER ordinal goes first (equal chars: either order is fine since result ties). A delete transformed against a delete of the same position becomes a no-op — represent no-op as ('nop',). apply must ignore ('nop',). Reply with ONLY a python code block containing transform and apply.",
        "check": """
{code}
import random, itertools, string
def ref_apply(doc, op):
    if op[0]=='nop': return doc
    if op[0]=='ins': return doc[:op[1]] + op[2] + doc[op[1]:]
    return doc[:op[1]] + doc[op[1]+1:]
rng = random.Random(42)
fails = 0
for trial in range(400):
    doc = ''.join(rng.choice('abcdef') for _ in range(rng.randrange(1, 8)))
    def rand_op():
        if rng.random() < 0.5:
            return ('ins', rng.randrange(0, len(doc)+1), rng.choice('xyz'))
        return ('del', rng.randrange(0, len(doc)))
    a, b = rand_op(), rand_op()
    r1 = apply(apply(doc, a), transform(b, a))
    r2 = apply(apply(doc, b), transform(a, b))
    assert r1 == r2, f"diverged: doc={{doc!r}} a={{a}} b={{b}} -> {{r1!r}} vs {{r2!r}}"
    assert apply(doc, a) == ref_apply(doc, a)
""",
    },
    {
        "id": "esc-bounded-queue",
        "max_tokens": 24000,
        "prompt": "Implement class BoundedQueue(capacity) with put(item, timeout=None) and get(timeout=None) using threading primitives (threading.Lock/Condition — NOT queue.Queue). Blocking semantics: put blocks while full, get blocks while empty; timeout (seconds) -> raise TimeoutError on expiry; FIFO order; must be correct under many concurrent producers/consumers (no lost items, no duplicates, no deadlock); shutdown() wakes ALL blocked threads which then raise RuntimeError, and makes future put/get raise RuntimeError. Reply with ONLY a python code block.",
        "check": """
{code}
import threading, time, collections
q = BoundedQueue(4)
N, P, C = 500, 4, 4
produced = collections.Counter(); consumed = collections.Counter()
def prod(k):
    for i in range(N):
        item = (k, i)
        q.put(item, timeout=10); produced[item] += 1
def cons(out):
    while True:
        try:
            item = q.get(timeout=1.5)
        except TimeoutError:
            return
        except RuntimeError:
            return
        out[item] += 1
threads = [threading.Thread(target=prod, args=(k,)) for k in range(P)]
outs = [collections.Counter() for _ in range(C)]
threads += [threading.Thread(target=cons, args=(outs[j],)) for j in range(C)]
for t in threads: t.start()
for t in threads: t.join(timeout=30)
assert not any(t.is_alive() for t in threads), "deadlock or hang"
total = collections.Counter()
for o in outs: total.update(o)
assert total == produced and sum(total.values()) == N*P, "lost or duplicated items"
q2 = BoundedQueue(1)
res = []
def blocked_getter():
    try: q2.get(timeout=10)
    except RuntimeError: res.append("runtime")
    except TimeoutError: res.append("timeout")
t = threading.Thread(target=blocked_getter); t.start()
time.sleep(0.2); q2.shutdown(); t.join(timeout=5)
assert res == ["runtime"], res
try:
    q2.put(1); assert False
except RuntimeError: pass
q3 = BoundedQueue(1); q3.put('x')
t0=time.time()
try:
    q3.put('y', timeout=0.3); assert False
except TimeoutError: pass
assert 0.2 < time.time()-t0 < 2.0
""",
    },
]

# Effort TIERS, not fixed names: models name levels differently (qwen:
# low/medium/high/xhigh; DSV4: low/high/max — "medium" silently maps to LOW
# on DSV4's parser). Pass --efforts per run; position = tier, so compare
# aligns tier-to-tier: tier0 = "workhorse" (each model's sane daily level),
# tier1 = "deep". Defaults suit qwen; use --efforts low,high for DSV4.
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


import shutil
BWRAP = shutil.which("bwrap")


def run_check(code, check_tmpl):
    """Execute model-generated code in a throwaway sandbox.

    bubblewrap when available: no network, read-only /usr, tmpfs /tmp, dies
    with parent — generated code cannot touch repos, HOME, or the cluster.
    Falls back to a plain subprocess (with a loud note) if bwrap is absent.
    """
    src = check_tmpl.format(code=code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    if BWRAP:
        cmd = [BWRAP, "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
               "--symlink", "usr/lib", "/lib", "--symlink", "usr/sbin", "/sbin",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--unshare-all", "--die-with-parent", "--new-session",
               "--ro-bind", path, "/check.py", "/usr/bin/python3", "/check.py"]
    else:
        print("  !! bwrap missing — running UNSANDBOXED", file=sys.stderr)
        cmd = [sys.executable, path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
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


def run_suite(base, label, outdir, efforts):
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for task in TASKS:
        for tier, effort in enumerate(efforts):
            for s in range(SAMPLES):
                r = chat(base, task["prompt"], effort,
                         max_tokens=task.get("max_tokens", 8192), timeout=900)
                code = extract_code(r["content"])
                ok, err = run_check(code, task["check"])
                row = {
                    "task": task["id"], "effort": effort, "tier": tier, "sample": s,
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
    print(f"\n{'':20}{a:>14}{b:>14}")
    tiers = sorted({r.get("tier", 0) for r in ra})
    TIERNAME = {0: "workhorse", 1: "deep"}
    for tier in tiers:
        xa = [r for r in ra if r.get("tier", 0) == tier]
        xb = [r for r in rb if r.get("tier", 0) == tier]
        tn = f"{TIERNAME.get(tier, tier)}({xa[0]['effort']}/{xb[0]['effort']})"
        print(f"{tn[:19]:19} pass{100*sum(r['pass'] for r in xa)/len(xa):>10.0f}%"
              f"{100*sum(r['pass'] for r in xb)/len(xb):>13.0f}%")
        ma = sorted(r['completion_tokens'] or 0 for r in xa)[len(xa)//2]
        mb = sorted(r['completion_tokens'] or 0 for r in xb)[len(xb)//2]
        print(f"{'':19} med tok{ma:>11}{mb:>14}")
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
    ap.add_argument("--efforts", default="medium,high",
                    help="comma list, one per tier (dsv4: low,high)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--check-solution", nargs=2, metavar=("TASK_ID", "FILE"),
                    help="verify a solution FILE (e.g. written by a pi mission) against TASK_ID's checks")
    args = ap.parse_args()
    root = Path(args.outdir)
    if args.check_solution:
        tid, fpath = args.check_solution
        task = next((t for t in TASKS if t["id"] == tid), None)
        if task is None:
            print(json.dumps({"pass": False, "error": f"unknown task {tid}"})); sys.exit(2)
        fp = Path(fpath)
        if not fp.exists():
            print(json.dumps({"pass": False, "error": "solution file missing"})); sys.exit(1)
        ok, err = run_check(fp.read_text(), task["check"])
        print(json.dumps({"pass": ok, "error": None if ok else err[-300:]}))
        sys.exit(0 if ok else 1)
    if args.compare:
        compare(root, *args.compare)
    else:
        if not args.label:
            ap.error("--label required for a run")
        run_suite(args.base.rstrip("/"), args.label, root / args.label,
                  [e.strip() for e in args.efforts.split(",")])
