#!/bin/bash
set -euo pipefail

# Autoresearch benchmark for Qwen3.6 evaluation
# Usage: BENCH_MODEL=qwen3.8-27b ./autoresearch.sh

# Active model read from autoresearch.model file (or default to current deployment)
if [ -f autoresearch.model ]; then
    MODEL="$(cat autoresearch.model | tr -d '[:space:]')"
else
    MODEL="qwen3.8-27b"
fi
MODE="${BENCH_MODE:-${2:-chat}}"
REQUESTS="${BENCH_REQUESTS:-10}"
WARMUP="${BENCH_WARMUP:-3}"
MAX_TOKENS="${BENCH_MAX_TOKENS:-256}"

# Attribute this harness to the 'autoresearch' gateway client (per-client
# metrics/limits). The key is read from the gitignored gateway/clients.yaml so
# it never lands in git; override with SPARKSTATION_API_KEY. Falls back to
# The gateway enforces registered keys; local-ops key comes from .env.
SPARK_KEY="${SPARKSTATION_API_KEY:-$(grep -oE 'sk-spark-autoresearch-[0-9a-f]+' gateway/clients.yaml 2>/dev/null | head -1)}"
export SPARK_KEY="${SPARK_KEY:-$(grep -s ^GATEWAY_LOCAL_KEY "$(dirname "$0")/.env" | cut -d= -f2)}"

echo "Benchmarking $MODEL ($MODE) — $REQUESTS requests, $WARMUP warmup, max_tokens=$MAX_TOKENS" >&2

# Qwen3.5 / Qwen3.6 benchmark with thinking disabled
# (thinking consumes max_tokens budget → unfair throughput comparison)
if [[ "$MODEL" == *"qwen3."* || "$MODEL" == *"qwen35"* || "$MODEL" == *"qwen36"* ]] && [[ "$MODE" == "chat" ]]; then
    echo "Qwen3.x detected — direct benchmark with thinking disabled" >&2
    .venv/bin/python - "$MODEL" "$REQUESTS" "$WARMUP" "$MAX_TOKENS" <<'PYEOF'
import asyncio, os, sys, time, json, statistics, httpx

model, num_req, warmup, max_tok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
BASE_URL = "http://localhost:8000/v1"
API_KEY = os.environ.get("SPARK_KEY", "missing-key")
prompts = [
    "Explain the concept of photosynthesis in two sentences.",
    "What are the three laws of thermodynamics?",
    "Write a haiku about machine learning.",
    "Describe the difference between TCP and UDP.",
    "What is the capital of France and its population?",
    "Explain recursion to a five year old.",
    "List three benefits of regular exercise.",
    "What causes rainbows to form?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "What is the Pythagorean theorem?",
]

async def bench_one(client, prompt, max_tokens):
    start = time.perf_counter()
    ttft = None
    tokens = 0
    async with client.stream("POST", f"{BASE_URL}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120.0) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                        tokens += 1
                except: pass
    latency = (time.perf_counter() - start) * 1000
    return {"latency": latency, "ttft": ttft or latency, "tokens": tokens}

async def main():
    async with httpx.AsyncClient() as client:
        print(f"  Warming up ({warmup} requests)...", file=sys.stderr)
        for i in range(warmup):
            await bench_one(client, prompts[0], max_tok)
        print(f"  Running {num_req} requests at concurrency 1...", file=sys.stderr)
        results = []
        for i in range(num_req):
            r = await bench_one(client, prompts[i % len(prompts)], max_tok)
            results.append(r)
        lats = [r["latency"] for r in results]
        ttfts = [r["ttft"] for r in results]
        toks = [r["tokens"] for r in results]
        total_tokens = sum(toks)
        total_seconds = sum(lats) / 1000.0
        tok_per_sec = total_tokens / total_seconds if total_seconds > 0 else 0
        print(f"METRIC chat_latency_p50_ms={sorted(lats)[len(lats)//2]:.1f}")
        print(f"METRIC chat_latency_p95_ms={sorted(lats)[int(len(lats)*0.95)]:.1f}")
        print(f"METRIC chat_latency_mean_ms={statistics.mean(lats):.1f}")
        print(f"METRIC chat_latency_min_ms={min(lats):.1f}")
        print(f"METRIC chat_latency_max_ms={max(lats):.1f}")
        print(f"METRIC chat_ttft_p50_ms={sorted(ttfts)[len(ttfts)//2]:.1f}")
        print(f"METRIC chat_ttft_p95_ms={sorted(ttfts)[int(len(ttfts)*0.95)]:.1f}")
        print(f"METRIC chat_ttft_mean_ms={statistics.mean(ttfts):.1f}")
        print(f"METRIC chat_total_tokens={total_tokens}")
        print(f"METRIC chat_tokens_per_sec={tok_per_sec:.1f}")
        rps = len(results) / total_seconds if total_seconds > 0 else 0
        print(f"METRIC chat_requests_per_sec={rps:.2f}")

asyncio.run(main())
PYEOF
else
    .venv/bin/python .pi/skills/sparkstation-bench/scripts/bench.py "$MODE" \
        --model "$MODEL" --requests "$REQUESTS" --warmup "$WARMUP" \
        --max-tokens "$MAX_TOKENS" --output metrics
fi
