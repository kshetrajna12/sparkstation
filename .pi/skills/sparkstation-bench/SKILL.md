---
name: sparkstation-bench
description: Benchmark inference throughput and latency for Sparkstation models. Measures TTFT, tokens/sec, p50/p95/p99 latency, and embedding throughput. Use when asked to "benchmark", "measure latency", "test throughput", or "profile inference performance".
---

# Sparkstation Bench

Automated inference benchmarking for all Sparkstation model types: chat, embeddings (text + image), and vision.

## Usage

Run the benchmark script with the desired mode:

```bash
# Benchmark chat model (default: qwen3-vl-30b)
python3 .pi/skills/sparkstation-bench/scripts/bench.py chat --model qwen3-vl-30b --concurrency 1 --requests 10

# Benchmark with concurrency sweep
python3 .pi/skills/sparkstation-bench/scripts/bench.py chat --model qwen3-vl-30b --concurrency 1,2,4,8 --requests 20

# Benchmark text embeddings
python3 .pi/skills/sparkstation-bench/scripts/bench.py embedding --model bge-m3 --requests 50

# Benchmark image embeddings (CLIP)
python3 .pi/skills/sparkstation-bench/scripts/bench.py clip --model clip-vit --requests 20

# Benchmark vision (image + chat)
python3 .pi/skills/sparkstation-bench/scripts/bench.py vision --model qwen3-vl-30b --requests 5

# Full suite — benchmarks all running models
python3 .pi/skills/sparkstation-bench/scripts/bench.py all
```

## Output

Outputs structured `METRIC name=value` lines for autoresearch compatibility, plus a human-readable summary table:

```
METRIC chat_ttft_p50_ms=142.3
METRIC chat_ttft_p95_ms=210.5
METRIC chat_tokens_per_sec=45.2
METRIC chat_latency_p50_ms=890.1
METRIC chat_latency_p95_ms=1230.4
METRIC embedding_requests_per_sec=120.5
METRIC embedding_latency_p50_ms=8.3
```

## Options

- `--model`: Model alias (default: auto-detect from running models)
- `--concurrency`: Concurrent requests, comma-separated for sweep (default: 1)
- `--requests`: Total requests per concurrency level (default: 10)
- `--max-tokens`: Max tokens for chat responses (default: 256)
- `--prompt`: Custom prompt for chat benchmarks
- `--warmup`: Number of warmup requests (default: 2)
- `--base-url`: Gateway URL (default: http://localhost:8000/v1)
- `--output`: Output format: `text`, `json`, `metrics` (default: text)

## Integration with Autoresearch

This skill outputs `METRIC` lines compatible with autoresearch. To optimize inference:

1. Write `autoresearch.sh` that calls `bench.py` with your target workload
2. Modify vLLM/SGLang args in `models.yaml`
3. Restart the model and re-benchmark

## Prerequisites

- Sparkstation must be running with models loaded
- Python packages: `openai`, `httpx` (already in sparkstation venv)
