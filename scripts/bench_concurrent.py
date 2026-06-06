"""Concurrent throughput bench for the chat alias.

Fires N concurrent requests against /v1/chat/completions and measures:
  - aggregate tok/s   = sum(tokens_generated) / wall_clock_seconds
  - per-request latency p50/p95
  - TTFT p50/p95

For each concurrency level, runs WARMUP rounds first, then BENCH rounds.
Wall clock is the *gating* metric — captures both per-stream throughput and
how well the server batches across concurrent requests.

Usage:
    .venv/bin/python scripts/bench_concurrent.py [--model qwen3.5-35b] \\
        [--concurrencies 1,4,8,16,32] [--rounds 3] [--max-tokens 256]
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx


PROMPTS = [
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
    "Describe the water cycle.",
    "What is the speed of light in vacuum?",
    "Name the planets in our solar system in order.",
    "What is the difference between weather and climate?",
    "Explain Newton's first law of motion.",
    "How does a transistor work, briefly?",
    "What is photosynthesis, simply put?",
    "Define entropy in one sentence.",
    "What is the difference between mitosis and meiosis?",
    "Explain how an internal combustion engine works.",
    "What is the function of the mitochondria?",
    "Describe the structure of an atom.",
    "What is dark matter?",
    "Explain quantum entanglement simply.",
    "What is the function of red blood cells?",
    "Explain the greenhouse effect.",
    "How do vaccines work?",
    "What is the difference between AC and DC current?",
    "Define machine learning in one sentence.",
    "Explain the Doppler effect.",
    "What is the Big Bang theory?",
    "How do solar panels generate electricity?",
]


async def stream_one(client: httpx.AsyncClient, base_url: str, model: str,
                     prompt: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    ttft = None
    tokens = 0
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with client.stream(
        "POST", f"{base_url}/chat/completions",
        json=body,
        headers={"Authorization": "Bearer dummy-key"},
        timeout=300.0,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            try:
                chunk = json.loads(line[6:])
                if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                    tokens += 1
            except Exception:
                pass
    return {
        "latency_ms": (time.perf_counter() - start) * 1000,
        "ttft_ms": ttft if ttft is not None else (time.perf_counter() - start) * 1000,
        "tokens": tokens,
    }


async def run_round(client, base_url, model, concurrency, max_tokens):
    """Fire `concurrency` requests in parallel, return per-request metrics + wall."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]
    wall_start = time.perf_counter()
    results = await asyncio.gather(*[
        stream_one(client, base_url, model, p, max_tokens) for p in prompts
    ])
    wall_ms = (time.perf_counter() - wall_start) * 1000
    return results, wall_ms


async def bench_at(client, base_url, model, concurrency, rounds, warmup, max_tokens):
    print(f"\n=== concurrency={concurrency} (warmup={warmup}, rounds={rounds}) ===", flush=True)
    for w in range(warmup):
        _, ww = await run_round(client, base_url, model, concurrency, max_tokens)
        print(f"  warmup {w+1}/{warmup}: {ww/1000:.1f}s", flush=True)

    all_results = []
    total_wall_ms = 0.0
    for r in range(rounds):
        results, wall_ms = await run_round(client, base_url, model, concurrency, max_tokens)
        all_results.extend(results)
        total_wall_ms += wall_ms
        round_tokens = sum(r["tokens"] for r in results)
        print(f"  round {r+1}/{rounds}: {wall_ms/1000:.1f}s wall, {round_tokens} tokens, "
              f"{round_tokens / (wall_ms/1000):.1f} tok/s aggregate", flush=True)

    total_tokens = sum(r["tokens"] for r in all_results)
    lats = sorted(r["latency_ms"] for r in all_results)
    ttfts = sorted(r["ttft_ms"] for r in all_results)
    n = len(lats)
    p50 = lats[n // 2]
    p95 = lats[int(n * 0.95)] if n > 1 else lats[0]
    ttft_p50 = ttfts[n // 2]
    ttft_p95 = ttfts[int(n * 0.95)] if n > 1 else ttfts[0]
    agg_toks = total_tokens / (total_wall_ms / 1000) if total_wall_ms > 0 else 0
    return {
        "concurrency": concurrency,
        "total_requests": n,
        "total_tokens": total_tokens,
        "wall_seconds": total_wall_ms / 1000,
        "aggregate_tok_per_sec": agg_toks,
        "per_req_latency_p50_ms": p50,
        "per_req_latency_p95_ms": p95,
        "ttft_p50_ms": ttft_p50,
        "ttft_p95_ms": ttft_p95,
        "tokens_mean_per_req": total_tokens / n if n else 0,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen3.5-35b")
    p.add_argument("--concurrencies", default="1,4,8,16,32",
                   help="comma-separated list")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    levels = [int(c.strip()) for c in args.concurrencies.split(",")]
    print(f"Benching {args.model} via {args.base_url}", flush=True)
    print(f"Levels: {levels} | rounds={args.rounds} | warmup={args.warmup} | "
          f"max_tokens={args.max_tokens}", flush=True)

    summary = []
    async with httpx.AsyncClient() as client:
        for c in levels:
            r = await bench_at(client, args.base_url, args.model, c,
                               args.rounds, args.warmup, args.max_tokens)
            summary.append(r)

    print("\n\n=== SUMMARY ===", flush=True)
    hdr = ("conc", "agg tok/s", "p50 lat ms", "p95 lat ms", "ttft p50", "ttft p95", "mean toks/req", "reqs")
    print(f"{hdr[0]:>4}  {hdr[1]:>10}  {hdr[2]:>10}  {hdr[3]:>10}  {hdr[4]:>9}  {hdr[5]:>9}  {hdr[6]:>14}  {hdr[7]:>5}",
          flush=True)
    for r in summary:
        print(f"{r['concurrency']:>4}  {r['aggregate_tok_per_sec']:>10.1f}  "
              f"{r['per_req_latency_p50_ms']:>10.0f}  {r['per_req_latency_p95_ms']:>10.0f}  "
              f"{r['ttft_p50_ms']:>9.0f}  {r['ttft_p95_ms']:>9.0f}  "
              f"{r['tokens_mean_per_req']:>14.1f}  {r['total_requests']:>5}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
