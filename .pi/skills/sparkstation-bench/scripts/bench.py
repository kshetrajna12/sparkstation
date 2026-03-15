#!/usr/bin/env python3
"""
Sparkstation Inference Benchmark

Measures throughput, latency, and TTFT for chat, embedding, and vision models.
Outputs METRIC lines for autoresearch compatibility.
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

BASE_URL = os.environ.get("SPARKSTATION_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("SPARKSTATION_API_KEY", "dummy-key")
CLIP_DIRECT_URL = os.environ.get("SPARKSTATION_CLIP_URL", None)  # Auto-discovered if None

# Default prompts for benchmarking
CHAT_PROMPTS = [
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

EMBEDDING_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons.",
    "The weather today is sunny with a chance of rain.",
    "Quantum computing uses qubits instead of classical bits.",
    "Deep learning has revolutionized computer vision tasks.",
    "Natural language processing enables human-computer interaction.",
    "Reinforcement learning trains agents through reward signals.",
    "Transfer learning reduces the need for large datasets.",
    "Attention mechanisms improved sequence-to-sequence models.",
]

# Generate a valid 224x224 RGB PNG for CLIP/vision benchmarks
def _make_test_png() -> str:
    """Generate a 224x224 solid-color PNG as base64."""
    import struct
    import zlib
    width, height = 224, 224
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + b"\x80\x40\x20" * width
    def _chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _chunk(b"IDAT", zlib.compress(raw_data))
    png += _chunk(b"IEND", b"")
    return base64.b64encode(png).decode()

TINY_IMAGE_B64 = _make_test_png()


@dataclass
class RequestResult:
    """Result of a single benchmark request."""
    success: bool
    latency_ms: float
    ttft_ms: Optional[float] = None  # Time to first token (chat only)
    tokens_generated: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    mode: str
    model: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    results: list = field(default_factory=list)

    @property
    def latencies(self) -> list:
        return [r.latency_ms for r in self.results if r.success]

    @property
    def ttfts(self) -> list:
        return [r.ttft_ms for r in self.results if r.success and r.ttft_ms is not None]

    @property
    def tokens(self) -> list:
        return [r.tokens_generated for r in self.results if r.success]

    def percentile(self, values: list, p: float) -> float:
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = int(len(sorted_v) * p / 100)
        idx = min(idx, len(sorted_v) - 1)
        return sorted_v[idx]

    def summary(self) -> dict:
        lat = self.latencies
        ttft = self.ttfts
        tok = self.tokens
        total_time = max(lat) if lat else 0  # Approximate wall-clock time

        s = {
            "mode": self.mode,
            "model": self.model,
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
        }

        if lat:
            s["latency_p50_ms"] = round(self.percentile(lat, 50), 1)
            s["latency_p95_ms"] = round(self.percentile(lat, 95), 1)
            s["latency_p99_ms"] = round(self.percentile(lat, 99), 1)
            s["latency_mean_ms"] = round(statistics.mean(lat), 1)
            s["latency_min_ms"] = round(min(lat), 1)
            s["latency_max_ms"] = round(max(lat), 1)

        if ttft:
            s["ttft_p50_ms"] = round(self.percentile(ttft, 50), 1)
            s["ttft_p95_ms"] = round(self.percentile(ttft, 95), 1)
            s["ttft_mean_ms"] = round(statistics.mean(ttft), 1)

        if tok:
            total_tokens = sum(tok)
            total_seconds = sum(lat) / 1000.0  # Sum of individual latencies
            s["total_tokens"] = total_tokens
            s["tokens_per_sec"] = round(total_tokens / (total_seconds / self.concurrency), 1) if total_seconds > 0 else 0

        if lat:
            wall_seconds = sum(lat) / 1000.0 / self.concurrency
            s["requests_per_sec"] = round(self.successful_requests / wall_seconds, 2) if wall_seconds > 0 else 0

        return s


async def bench_chat_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool = True,
) -> RequestResult:
    """Send a single chat completion request and measure timing."""
    start = time.perf_counter()
    ttft = None

    try:
        if stream:
            tokens = 0
            async with client.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=120.0,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    return RequestResult(
                        success=False,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        error=f"HTTP {resp.status_code}: {body.decode()[:200]}",
                    )

                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        try:
                            chunk = json.loads(line[6:])
                            if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                tokens += 1
                        except json.JSONDecodeError:
                            pass

            latency = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=True,
                latency_ms=latency,
                ttft_ms=ttft,
                tokens_generated=tokens,
            )
        else:
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=120.0,
            )
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return RequestResult(
                    success=False,
                    latency_ms=latency,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return RequestResult(success=True, latency_ms=latency, tokens_generated=tokens)

    except Exception as e:
        return RequestResult(
            success=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )


async def bench_embedding_request(
    client: httpx.AsyncClient,
    model: str,
    text: str,
) -> RequestResult:
    """Send a single embedding request and measure timing."""
    start = time.perf_counter()

    try:
        resp = await client.post(
            f"{BASE_URL}/embeddings",
            json={"model": model, "input": text},
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30.0,
        )
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return RequestResult(
                success=False,
                latency_ms=latency,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        return RequestResult(success=True, latency_ms=latency)

    except Exception as e:
        return RequestResult(
            success=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )


async def bench_clip_request(
    client: httpx.AsyncClient,
    model: str,
    use_image: bool = True,
) -> RequestResult:
    """Send a single CLIP embedding request.

    CLIP image embeddings must go directly to the CLIP backend (not through LiteLLM)
    because LiteLLM doesn't support the [{image: ...}] input format.
    Text embeddings can go through the gateway.
    """
    start = time.perf_counter()

    try:
        if use_image:
            # Hit CLIP backend directly — discover port from supervisor
            clip_url = CLIP_DIRECT_URL or "http://127.0.0.1:8003/v1"
            resp = await client.post(
                f"{clip_url}/embeddings",
                json={
                    "model": "openai/clip-vit-large-patch14",
                    "input": [{"image": TINY_IMAGE_B64}],
                },
                timeout=30.0,
            )
        else:
            # Text embedding through gateway
            resp = await client.post(
                f"{BASE_URL}/embeddings",
                json={"model": model, "input": "a photo of a cat"},
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=30.0,
            )
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return RequestResult(
                success=False,
                latency_ms=latency,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        return RequestResult(success=True, latency_ms=latency)

    except Exception as e:
        return RequestResult(
            success=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )


async def bench_vision_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a vision (image+text) chat request."""
    start = time.perf_counter()

    try:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{TINY_IMAGE_B64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=120.0,
        )
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return RequestResult(
                success=False,
                latency_ms=latency,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return RequestResult(success=True, latency_ms=latency, tokens_generated=tokens)

    except Exception as e:
        return RequestResult(
            success=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e),
        )


async def run_benchmark(
    mode: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int = 256,
    warmup: int = 2,
    custom_prompt: Optional[str] = None,
) -> BenchmarkResult:
    """Run a benchmark with the given parameters."""

    async with httpx.AsyncClient() as client:
        # Warmup
        if warmup > 0:
            print(f"  Warming up ({warmup} requests)...", file=sys.stderr)
            for i in range(warmup):
                if mode == "chat":
                    await bench_chat_request(client, model, CHAT_PROMPTS[0], max_tokens)
                elif mode == "embedding":
                    await bench_embedding_request(client, model, EMBEDDING_TEXTS[0])
                elif mode == "clip":
                    await bench_clip_request(client, model)
                elif mode == "vision":
                    await bench_vision_request(client, model, "Describe this image briefly.", max_tokens)

        # Run benchmark
        print(f"  Running {num_requests} requests at concurrency {concurrency}...", file=sys.stderr)
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def run_one(idx: int):
            async with semaphore:
                if mode == "chat":
                    prompt = custom_prompt or CHAT_PROMPTS[idx % len(CHAT_PROMPTS)]
                    return await bench_chat_request(client, model, prompt, max_tokens)
                elif mode == "embedding":
                    text = EMBEDDING_TEXTS[idx % len(EMBEDDING_TEXTS)]
                    return await bench_embedding_request(client, model, text)
                elif mode == "clip":
                    return await bench_clip_request(client, model)
                elif mode == "vision":
                    prompt = custom_prompt or "Describe this image in detail."
                    return await bench_vision_request(client, model, prompt, max_tokens)

        tasks = [run_one(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        # Log failures
        for r in results:
            if not r.success:
                print(f"  FAILED: {r.error}", file=sys.stderr)

        return BenchmarkResult(
            mode=mode,
            model=model,
            concurrency=concurrency,
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            results=list(results),
        )


def print_summary(result: BenchmarkResult, output_format: str = "text"):
    """Print benchmark results."""
    s = result.summary()

    if output_format == "json":
        print(json.dumps(s, indent=2))
        return

    if output_format == "metrics":
        prefix = f"{s['mode']}"
        for key, value in s.items():
            if key in ("mode", "model", "total_requests", "successful", "failed", "concurrency"):
                continue
            if isinstance(value, (int, float)):
                print(f"METRIC {prefix}_{key}={value}")
        return

    # Text format
    print(f"\n{'='*60}")
    print(f"  {s['mode'].upper()} Benchmark: {s['model']}")
    print(f"  Concurrency: {s['concurrency']} | Requests: {s['successful']}/{s['total_requests']} OK")
    print(f"{'='*60}")

    if "latency_p50_ms" in s:
        print(f"  Latency  p50: {s['latency_p50_ms']:>8.1f} ms")
        print(f"           p95: {s['latency_p95_ms']:>8.1f} ms")
        print(f"           p99: {s['latency_p99_ms']:>8.1f} ms")
        print(f"          mean: {s['latency_mean_ms']:>8.1f} ms")
        print(f"           min: {s['latency_min_ms']:>8.1f} ms")
        print(f"           max: {s['latency_max_ms']:>8.1f} ms")

    if "ttft_p50_ms" in s:
        print(f"  TTFT     p50: {s['ttft_p50_ms']:>8.1f} ms")
        print(f"           p95: {s['ttft_p95_ms']:>8.1f} ms")
        print(f"          mean: {s['ttft_mean_ms']:>8.1f} ms")

    if "tokens_per_sec" in s:
        print(f"  Throughput:   {s['tokens_per_sec']:>8.1f} tok/s")
        print(f"  Total tokens: {s['total_tokens']:>8d}")

    if "requests_per_sec" in s:
        print(f"  Requests/sec: {s['requests_per_sec']:>8.2f}")

    print(f"{'='*60}\n")


async def discover_clip_url() -> Optional[str]:
    """Discover CLIP backend URL from the supervisor."""
    global CLIP_DIRECT_URL
    if CLIP_DIRECT_URL:
        return CLIP_DIRECT_URL
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:9001/models", timeout=5.0)
            if resp.status_code == 200:
                for m in resp.json():
                    if m.get("model_name") == "clip-vit":
                        CLIP_DIRECT_URL = m["api_base"]
                        return CLIP_DIRECT_URL
    except Exception:
        pass
    # Fallback
    CLIP_DIRECT_URL = "http://127.0.0.1:8003/v1"
    return CLIP_DIRECT_URL


async def discover_models() -> dict:
    """Discover running models from the gateway."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{BASE_URL}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = {}
                for m in data.get("data", []):
                    mid = m["id"]
                    if mid == "default":
                        continue
                    # Guess type from name
                    if "bge" in mid or "e5" in mid:
                        models[mid] = "embedding"
                    elif "clip" in mid:
                        models[mid] = "clip"
                    else:
                        models[mid] = "chat"
                return models
        except Exception as e:
            print(f"Failed to discover models: {e}", file=sys.stderr)
    return {}


async def run_all(args):
    """Run benchmarks on all discovered models."""
    models = await discover_models()
    if not models:
        print("ERROR: No models found. Is Sparkstation running?", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(models)} models: {dict(models)}", file=sys.stderr)

    # Auto-discover CLIP URL
    await discover_clip_url()

    all_results = []
    for model_name, model_type in models.items():
        print(f"\nBenchmarking {model_name} ({model_type})...", file=sys.stderr)
        result = await run_benchmark(
            mode=model_type,
            model=model_name,
            concurrency=args.concurrency_levels[0] if args.concurrency_levels else 1,
            num_requests=args.requests,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
        )
        all_results.append(result)

    for r in all_results:
        print_summary(r, args.output)

    # Always print metrics for all
    if args.output != "metrics":
        print("\n--- METRICS ---")
        for r in all_results:
            print_summary(r, "metrics")


async def main():
    parser = argparse.ArgumentParser(description="Sparkstation Inference Benchmark")
    parser.add_argument(
        "mode",
        choices=["chat", "embedding", "clip", "vision", "all"],
        help="Benchmark mode",
    )
    parser.add_argument("--model", default=None, help="Model alias")
    parser.add_argument(
        "--concurrency",
        default="1",
        help="Concurrent requests (comma-separated for sweep)",
    )
    parser.add_argument("--requests", type=int, default=10, help="Total requests per level")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for chat")
    parser.add_argument("--prompt", default=None, help="Custom prompt")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests")
    parser.add_argument("--base-url", default=None, help="Gateway URL")
    parser.add_argument(
        "--output",
        choices=["text", "json", "metrics"],
        default="text",
        help="Output format",
    )

    args = parser.parse_args()

    if args.base_url:
        global BASE_URL
        BASE_URL = args.base_url

    args.concurrency_levels = [int(x) for x in args.concurrency.split(",")]

    if args.mode == "all":
        await run_all(args)
        return

    # Default model selection
    if not args.model:
        defaults = {
            "chat": "qwen3-vl-30b",
            "embedding": "bge-m3",
            "clip": "clip-vit",
            "vision": "qwen3-vl-30b",
        }
        args.model = defaults.get(args.mode, "qwen3-vl-30b")

    # Auto-discover CLIP URL if needed
    if args.mode in ("clip", "all"):
        await discover_clip_url()

    # Run for each concurrency level
    for conc in args.concurrency_levels:
        result = await run_benchmark(
            mode=args.mode,
            model=args.model,
            concurrency=conc,
            num_requests=args.requests,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            custom_prompt=args.prompt,
        )
        print_summary(result, args.output)

        # Always output metrics
        if args.output != "metrics":
            print_summary(result, "metrics")


if __name__ == "__main__":
    asyncio.run(main())
