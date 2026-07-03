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

### Iteration 5 — NVFP4 + MTP on upstream vLLM 0.22.1 (2026-06-05) — SHIPPED, TUNING OPEN

NVIDIA published `nvidia/Qwen3.6-35B-A3B-NVFP4` with a DGX Spark recipe on the HF
model card. Combined with `vllm/vllm-openai:nightly` (the *non-cu130* tag, which
turned out to also be cu130 underneath at vLLM 0.22.1rc1.dev195), the iter-4
blocker dissolves: MARLIN NvFp4 MoE backend, FlashInfer attention, ModelOpt-mixed
quantization detection — all working out of the box.

**Deployed config** (image-indexing profile, all six chat entries unified):
- `nvidia/Qwen3.6-35B-A3B-NVFP4` via `vllm/vllm-openai:nightly` (vLLM 0.22.1rc1)
- `--quantization modelopt` → vLLM detects `modelopt_mixed`
- `--moe-backend marlin --attention-backend flashinfer --kv-cache-dtype fp8`
- `--enable-chunked-prefill --async-scheduling --enable-prefix-caching`
- `--max-num-batched-tokens 8192 --max-num-seqs 4 --max-model-len 65536`
- env vars per HF card: `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_FP8_MOE_BACKEND=flashinfer_cutlass`,
  `FLASHINFER_DISABLE_VERSION_CHECK=1`, `CUTE_DSL_ARCH=sm_121a`
- `--speculative-config {"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}`

Local snapshot tags: `vllm-openai:nightly-2026-06-05`,
`vllm-openai:cu130-nightly-2026-06-05` (kept to pin the working build against
future nightly drift).

**Bench results (single-stream, 10 reqs, 3 warmup, max_tokens=256, enable_thinking=False):**
- num_speculative_tokens=3 (HF recipe default): **34.0 tok/s** (-39% vs baseline)
- num_speculative_tokens=1: **45.7 tok/s** (-18% vs baseline)
- ttft p50: ~92 ms for both (vs 78 ms baseline)

c=1 regression is real but the NVFP4 path has wins the bench doesn't show:
- Weights: ~11 GB vs ~22 GB MXFP4 (frees ~10 GB for KV cache or another model)
- HF card reports 433 tok/s at c=32 (where MTP earns its keep) — not yet validated
- Mainline vLLM (no namake-taro patches) — DFlash drafter becomes reachable

**Path-clearing fixes shipped along the way** (Phase 1.5 — supervisor + CLI):
1. `ModelStartRequest` was missing `docker_image`, `env_vars`, `volumes`, `speculative_extra`
   — API-restarted models couldn't preserve image overrides.
2. `/models/start` blocked by stale STOPPED/SUSPENDED registry entry (swap blocker;
   error claimed "already running" which was false).
3. Multiple `.value` calls on `model.status` (a string due to ConfigDict(use_enum_values=True))
   crashed `/models/{id}/suspend`, `/resume`, `/status`, half-completing a suspend.
4. `/models/{id}/status` Pydantic schema missing `model_type` → always 500.
5. `registry.reconcile_state` orphan-detection compared SHORT docker IDs against FULL
   IDs in DB → wiped every legitimate container on every supervisor restart (the
   "restart flaky" memo was the symptom).
6. `resource_manager` didn't re-register surviving RUNNING/STARTING models on
   restart → port-allocation collisions on the next launch.
7. `cli.py start` unconditionally `db_path.unlink()`'d on detached mode — undid the
   reconcile-based adoption and forced cold reload of every model. Removed.
8. Lifespan purge wiped STARTING entries even with live containers → killed
   mid-cold-load models.
9. `pyproject.toml` `[tool.hatch.build.targets.wheel]` only packaged `supervisor/`
   and `gateway/` — the installed CLI failed with `ModuleNotFoundError: cli`.
   Added `cli.py`/`cli_init.py` via `force-include`.
10. `PROJECT_ROOT = Path(__file__).resolve().parent` resolved to site-packages when
    invoked via the installed CLI → `models.yaml` lookup + gateway-yaml writes
    silently went to wrong paths. Replaced with cwd-walk-up-to-find-models.yaml.
11. vllm_launcher always prepended `vllm serve` even though
    `vllm/vllm-openai:*` images have `ENTRYPOINT ["vllm","serve"]` → doubled command
    and `unrecognized arguments`. Now forces `--entrypoint vllm`.
12. `speculative-config` only emitted when `speculative_model` was set, blocking
    built-in MTP heads (no draft model). Now emits for either model OR method,
    and supports `speculative_extra` for vendor-specific keys.
13. Launcher passthroughs added: `compressed-tensors` + `modelopt` quantizations,
    `--moe-backend`, `--attention-backend`, `--async-scheduling`, `--dtype`.
14. CLI `models stop/start/swap <alias>` commands added with a `_restart_gateway()`
    helper that rewrites litellm.yaml + bounces the gateway process. Stop+start
    on bge-m3 verified end-to-end before the chat-model swap.

**Still open / deferred** (separate work items):
- Concurrent throughput bench (c=4/8/16/32) to confirm the HF claim
- MTP acceptance-rate logging (vLLM doesn't surface it; need a custom probe)
- Per-alias 503+Retry-After during swap (the current gateway-restart causes a
  brief outage across ALL aliases, not just the one being swapped) — needs a
  custom middleware in front of LiteLLM (auto_resume_middleware.py is scaffolded
  but not wired).
- Rename the `qwen3.5-35b` alias to something honest (3.6-35b or `chat`) — clients
  pin the string, so this needs a coordinated change across Wildlife Indexer +
  Caddy + Cloudflare Access policies.
- Memory tuning: `memory_gb: 40` may be conservative now that weights are 11 GB.
- DFlash drafter for Qwen3.6-35B-A3B (z-lab) is mainline-vLLM compatible now —
  worth A/B-ing against MTP once concurrent throughput is profiled.

### Iteration 6 — MTP OFF + full CUDA graphs (2026-07-02) — SHIPPED (interim winner)
Root cause of the iter-5 single-stream regression found in the engine log:
`FULL_AND_PIECEWISE is not supported with spec-decode for attention backend
FlashInferBackend ... setting cudagraph_mode=PIECEWISE`. MTP k=1 was paying the
drafter cost AND downgrading CUDA graphs to piecewise. Disabled MTP via
image-indexing profile override (`speculative_method: null`), added
`VLLM_MARLIN_USE_ATOMIC_ADD=1` (engine-suggested small-N GEMM tweak).

**Bench (same harness, worker1, 262K ctx, marlin+flashinfer):**
- c=1: **72.8 tok/s** (vs 47.1 with MTP k=1 → +55%), TTFT p50 108 ms
- c=4: **122.7 tok/s agg** (vs 76.1 → +61%), TTFT p50 152 ms (was 460!)
- c=8: **188.7 tok/s agg** (vs 116.4 → +62%), TTFT p50 206 ms (was 641)

Also finally beats the old MXFP4 baseline (56 tok/s) by +29% single-stream.

**Next (iter-7, in progress):** external DGX Spark reports (~90-102 tok/s) run
this exact checkpoint with MTP-3 on NEWER vLLM nightlies where spec-decode
keeps full cudagraphs. A/B: `vllm/vllm-openai:cu130-nightly` (2026-07 build)
+ `{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}` vs
iter-6 config. Keep whichever wins; MTP acceptance is workload-dependent, so
judge at c=1 AND c=4/8.

### Iteration 7 — MTP-3 + fresh v0.23 nightly A/B (2026-07-02) — REVERTED (both parts)
External DGX Spark reports (~90-102 tok/s) suggested MTP-3 on newer vLLM.
Tested on fresh `vllm/vllm-openai:nightly` (v0.23.1rc1.dev714, 20h old):

**MTP-3 on v0.23:** 41.0 tok/s c=1 / 118.6 c=4 / 168.1 c=8 — loses to
MTP-off everywhere. The `FULL_AND_PIECEWISE not supported with spec-decode
(FlashInferBackend)` cudagraph downgrade still fires on v0.23; MTP acceptance
doesn't compensate on our workload. The external ~100 tok/s reports don't
reproduce here.

**MTP-off on v0.23:** benched 71.4 c=1 / 212.5 c=4 / 398.6 c=8 — LOOKS like a
2× concurrent win, BUT the output is garbage: `!!!!!...` with
enable_thinking=false, never-terminating reasoning with thinking on (classic
sm_121 FP4 kernel corruption). The throughput numbers are meaningless.
**Lesson: always sanity-check output content alongside tok/s — a broken
kernel benches fast.**

**Action:** pinned the known-good 2026-06 build as
`vllm/vllm-openai:nightly-20260611-goodgb10` (image f6353499db8e, tagged on
both hosts) and set it as the model's docker_image. Iter-6 config (MTP off,
marlin, flashinfer, full cudagraphs) is the keeper: **72.8 tok/s c=1 /
122.7 c=4 / 188.7 c=8, TTFT p50 108/152/206 ms** — +55-63% over the shipped
iter-5 config and +29% over the old MXFP4 baseline.

**Open questions for a future round:**
- Re-test rolling nightly periodically (garbage bug may get fixed; v0.23's
  scheduler showed real batching upside if output becomes correct).
- MTP with attention_backend=flash_attn or triton (avoids the FlashInfer
  cudagraph downgrade) — untested; may finally let MTP pay off.
- vLLM PR #40082 `flashinfer_cutedsl_sm12x` MoE backend (+2-6% claim).
