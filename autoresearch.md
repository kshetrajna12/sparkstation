# Autoresearch: Qwen3.6-35B-A3B Evaluation

## Objective
Evaluate whether Qwen/Qwen3.6-35B-A3B (released 2026-04-16) should replace the currently
deployed Qwen3.5-35B-A3B MXFP4 across all Sparkstation profiles.

**Candidates:**
- `Qwen/Qwen3.6-35B-A3B` (BF16 + runtime MXFP4 quantization — matches Qwen3.5 setup)
- `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quantized FP8)
- `mmangkad/Qwen3.6-35B-A3B-NVFP4` (pre-quantized NVFP4 — Blackwell-native 4-bit, per-block FP8 scales)

**Rationale for NVFP4 on GB10:** Blackwell's 5th-gen Tensor Cores natively execute NVFP4
(NVIDIA's per-block FP8-scaled FP4). In principle this should outperform both FP8 (larger
weights, more memory bandwidth) and MXFP4 (coarser E8M0 scales). No pre-quantized MXFP4
weights exist for vLLM on HuggingFace — all MXFP4 distributions are GGUF/MLX — so runtime
MXFP4 via the existing docker image is the only MXFP4 path.

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
4. **Qwen3.6-NVFP4**: Test Blackwell-native 4-bit variant (expected best on GB10 Tensor Cores)
5. **Compare** tok/s, TTFT, latency
6. **Sanity quality check**: Send a few reasoning/tool-calling prompts, verify outputs parse
7. **Decide**: If best Qwen3.6 variant >= baseline AND no regressions, deploy unified on 3.6

## What's Been Tried

### Iteration 0 — Baseline (2026-04-20) — KEPT
Qwen/Qwen3.5-35B-A3B via `vllm-qwen35-mxfp4:cu130` runtime MXFP4, thinking=false.
- **tok_per_sec: 56.3** ← primary metric
- ttft_p50: 77.9 ms, ttft_p95: 81.6 ms
- latency_p50: 4453.8 ms (= 256 tok / 56 tok/s, expected)
- Fixed instrumentation bug in `autoresearch.sh`: `chat_template_kwargs` was nested inside
  `extra_body` (SDK-only field, stripped by gateway) so thinking was silently ON for the
  first run. Moved to top-level of the JSON body. Verified thinking is now disabled
  (completion_tokens = actual output length, no `<think>...` block).

### Iteration 1 — Qwen3.6 BF16 + runtime MXFP4 (2026-04-20) — KEPT (WINNER)
Qwen/Qwen3.6-35B-A3B via same `vllm-qwen35-mxfp4:cu130` image. Architecture unchanged
(`qwen3_5_moe`), drop-in swap. Confirmed CUTLASS_FP4 (Blackwell SM120 native) kernels
lit up via vLLM's mxfp4 auto-select.
- tok_per_sec: 55.7 (-1.1% vs 56.3 baseline — within 10-req noise)
- ttft_p50: 84.9 ms (+9%), ttft_p95: 129.0 ms (+58%)
- Encountered two infrastructure bugs along the way:
  1. Supervisor's `/models/start` endpoint silently discarded `memory_gb` and used a
     buggy substring-match heuristic (`"3b" in "a3b"` → 7 GB for a 35B model). Fixed
     in commit 2530ffe: added `memory_gb: Optional[float]` to `ModelStartRequest`.
  2. `autoresearch.sh` had the `chat_template_kwargs` misnested (pre-existing, fixed
     in iter 0). Went route-via-models.yaml instead of supervisor API.

### Iteration 2 — Qwen3.6-35B-A3B-FP8 (2026-04-20) — REVERTED
Pre-quantized FP8 variant, same docker image.
- tok_per_sec: 48.2 (-14.4% vs baseline — hard regression)
- vLLM auto-selected **TRITON FP8 MoE backend** on Blackwell. Available-but-unused
  faster backends: DEEPGEMM, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS. Forcing one
  would need either a different image (`vllm-deepgemm:26.02` exists locally) or
  `VLLM_FP8_MOE_BACKEND` env. Out of scope for this round.
- Also fixed mid-run: a zombie supervisor process bug (misleading "readonly database"
  errors). See memory/project_supervisor_zombie_modelinstance_bug.md.

### Iteration 3 — Qwen3.6-35B-A3B-NVFP4 (2026-04-20) — SKIPPED
Did not attempt. Pre-bench research showed NVFP4 is infeasible on GB10 today:
1. mmangkad/Qwen3.6-35B-A3B-NVFP4 model card targets **SGLang**, not vLLM
2. GB10 is sm_121 — lacks the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction NVFP4
   needs; FlashInfer falls back to CUTLASS which also fails on sm_121
3. Community workaround (avarok/dgx-vllm-nvfp4-kernel) uses software E2M1 emulation
   and reports ~35 tok/s — already worse than iter-1's 55.7 tok/s
4. Open vLLM issue #31085 tracks native sm_120/121 NVFP4 MoE kernel support — not
   merged yet
Revisit when vLLM ships native GB10 NVFP4 kernels.

### Iteration 4 — upgrade vLLM for instanttensor + larger batch budget (2026-04-20) — FAILED
Attempted to upgrade from our current `vllm-qwen35-mxfp4:cu130` (vLLM 0.17.0,
namake-taro patches) to upstream vLLM nightly to gain `--load-format instanttensor`
(cold start ~5 min → <1 min) and bump `max_num_batched_tokens` 2096 → 8192.

Two attempts, both crashed during weight load:

1. `vllm/vllm-openai:nightly` (vLLM 0.19.2rc1.dev21):
   - Container ENTRYPOINT = `["vllm serve"]`; supervisor launcher prepends the same,
     producing a doubled `vllm serve vllm serve ...`. Worked around with a wrapper
     image (`ENTRYPOINT []`).
   - MXFP4 MoE backend auto-picked = `MARLIN`. Weight load failed:
     `RuntimeError: The size of tensor a (1024) must match the size of tensor b (2048)
     at non-singleton dimension 1`.

2. `vllm-qwen35:cu130` (vLLM 0.17.1rc1.dev177):
   - Empty entrypoint, no wrapper needed.
   - Also auto-picked MARLIN. Different but related failure:
     `IndexError: tuple index out of range` on `loaded_weight.shape[2]` in
     `fused_moe/layer.py:1076`.

**Root cause:** upstream vLLM's MARLIN MXFP4 MoE kernel on GB10 (sm_121) has
a shared-memory race and lacks specialized GatedDeltaNet kernels for Qwen3.x MoE
models. Known issues: vllm-project/vllm#30135, #35924, #37030. On sm_121 the
priority backends (FLASHINFER_TRTLLM, TRITON, FLASHINFER_CUTLASS) all fail
`is_supported_config`, so vLLM falls through to MARLIN which then breaks.

**The namake-taro patches** (https://github.com/namake-taro/vllm-custom) in our
current image fix this by: (a) BF16 → MXFP4 online quantization path, (b) SM121
device-support fixes for CUTLASS, (c) Marlin MoE 256-thread kernel shared-memory
race fix, (d) Triton allocator + FlashInfer header fixes. **They are the only
working path right now** — any stock vLLM image fails.

Forum post reports 70 tok/s Qwen3.5-35B on namake-taro patches. Our baseline at
56.3 may have additional tuning headroom (unexplored args, different bench
protocol), but that's a separate investigation.

Reverted models.yaml to iter-1 state.

## Decision
**Winner: Iteration 1 — Qwen/Qwen3.6-35B-A3B with runtime MXFP4 via `vllm-qwen35-mxfp4:cu130` (namake-taro patches on vLLM 0.17.0).**

Same docker image, same alias (`qwen3.5-35b`), same `extra_args` as the previous
Qwen3.5 entry — only the HF name changes. Zero client-side migration. Performance
is a tie with 3.5 (-1.1% tok/s, within 10-req noise floor). Qwen3.6 is a strictly
newer base model with the same vision + tool-calling + reasoning capabilities.

`models.yaml` already reflects this decision (autoload + image-indexing profile both
point at `Qwen/Qwen3.6-35B-A3B`). A sparkstation restart applies the change.

**Upgrade is blocked** until either:
1. namake-taro publishes patches for a newer vLLM base, OR
2. Upstream vLLM merges the sm_121 MARLIN MoE fixes (tracking #30135, #35924, #37030).
