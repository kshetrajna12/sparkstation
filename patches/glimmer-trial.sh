#!/usr/bin/env bash
# Muse Glimmer 30B NVFP4 trial on primary (port 8210) — manual, outside supervisor.
# Mirrors recipes.vllm.ai/meta-models/Muse-Glimmer-30B DGX Spark guidance:
#   tp=1, no --gpu-memory-utilization/--max-num-seqs overrides beyond defaults,
#   DFlash num_speculative_tokens=15 (fixed by draft head architecture).
#
# Spec-decode sweep 2026-08-13 (num_speculative_tokens 15/8/5/3/off, benchmarked against the
# three image_metadata_indexing query shapes). 15 confirmed best — keep it. Per-user tok/s:
#
#   config | vision (conc 3) | synthesis (conc 5) | chat+tools (conc 1) | weighted 60/20/20
#   15     | 18.99           | 22.29              | 35.43               | 22.94
#   8      | 19.03           | 21.85              | 27.34               | 21.26
#   5      | 18.00           | 18.87              | 24.26               | 19.43
#   3      | 17.28           | 18.15              | 22.22               | 18.44
#   off    | 10.54           | 10.66              | 11.11               | 10.68
#
# Throughput is monotonic in num_speculative_tokens: lowering it only costs speed, and
# disabling spec decode halves throughput (-53% weighted, -69% on chat). DFlash acceptance
# varies by shape at nst=15 — chat 3.61 tok/draft, synthesis 2.32, vision 1.78 — but the
# ranking of configs is the same for all three, so one setting serves the whole mix.
# Shape-A 15-vs-8 is a tie within ~10% run-to-run noise (15 re-measured at 18.99/19.34/20.88).
# Details: scratchpad specdec-sweep.md.
set -euo pipefail

docker run -d \
  --platform linux/arm64 \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p 8210:8210 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint vllm \
  --name glimmer-trial \
  vllm/vllm-openai:muse-glimmer-arm64-cu130 \
  serve Inferact/Muse-Glimmer-30B-NVFP4-W4A4 \
  --served-model-name muse-glimmer \
  --host 0.0.0.0 --port 8210 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto \
  --speculative-config '{"method": "dflash", "model": "meta-models/Muse-Glimmer-30B-assistant", "num_speculative_tokens": 15}'
