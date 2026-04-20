# Autoresearch: Qwen3.6-35B-A3B Evaluation

## Objective
Evaluate whether Qwen/Qwen3.6-35B-A3B (released 2026-04-16) should replace the currently
deployed Qwen3.5-35B-A3B MXFP4 across all Sparkstation profiles.

**Candidates:**
- `Qwen/Qwen3.6-35B-A3B` (BF16 + runtime MXFP4 quantization — matches Qwen3.5 setup)
- `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quantized FP8)

**Key architectural note:** Qwen3.6 uses the SAME `Qwen3_5MoeForConditionalGeneration`
architecture as Qwen3.5 (`model_type: qwen3_5_moe`). Our existing `vllm-qwen35-mxfp4:cu130`
Docker image works as a drop-in replacement — just change the model name.

## Metrics
- **Primary**: `tok_per_sec` (tok/s, higher is better)
- **Secondary**: `ttft_p50_ms`, `latency_p50_ms`, `latency_p95_ms`

## Baseline
Current deployed: Qwen3.5-35B-A3B MXFP4 → **~55 tok/s, TTFT p50 ~82ms**
(image-indexing profile, single-request chat, `enable_thinking=false`)

## How to Run
`./autoresearch.sh` — benchmarks chat throughput with thinking disabled.
Model alias selected via `BENCH_MODEL` env var.

## Files in Scope
- `models.yaml` — profile definitions (add test profiles for Qwen3.6 variants)
- `autoresearch.sh` — benchmark script
- `autoresearch.md` — this file

## Off Limits
- Gateway code, bench.py
- `supervisor/launchers/vllm_launcher.py` (works as-is)
- `docker/vllm-qwen35/` (image already supports qwen3_5_moe arch)

## Constraints
- All models in a profile must fit in 113GB GPU memory total
- Profile restarts are slow (~5 min); minimize unnecessary switches
- **IMPORTANT**: Restore image-indexing profile to a working state after experiments
- Current image-indexing has 5 models: qwen3.5-35b, bge-m3, clip-vit, species-detect, face-detect

## Experiment Plan
1. **Baseline re-bench**: Confirm current Qwen3.5-35B MXFP4 throughput (noise floor check)
2. **Qwen3.6 BF16 + MXFP4 runtime**: Test with existing Docker image, mxfp4 quantization
3. **Qwen3.6-FP8**: Test pre-quantized FP8 variant (smaller weights, may be faster)
4. **Compare** tok/s, TTFT, latency
5. **Sanity quality check**: Send a few reasoning/tool-calling prompts, verify outputs parse
6. **Decide**: If >= baseline AND no regressions, deploy unified on 3.6

## What's Been Tried
(Updated as experiments run)
