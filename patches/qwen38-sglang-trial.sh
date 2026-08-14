#!/usr/bin/env bash
# Qwen3.8-27B RadixArk NVFP4 on SGLang — primary:8214 (2026-08-14).
# Based on the SGLang cookbook recipe for Qwen3.8-27B, adapted for DGX Spark GB10:
# mem-fraction 0.40 instead of 0.95 (unified memory), image = cu13 nightly with
# flashinfer b12x SM121 kernels. Recipe flags kept: flashinfer attention,
# chunked-prefill 8192, disable-prefill-cuda-graph, mamba-full-memory-ratio 4.59
# (Gated DeltaNet linear-attention cache sizing), qwen3 parsers.
set -euo pipefail

docker run -d \
  --platform linux/arm64 \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8214:30000 \
  -v /home/kshetrajna/.cache/huggingface:/root/.cache/huggingface \
  --name qwen38-sglang-trial \
  lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b \
  python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b \
  --mem-fraction-static 0.40 \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --context-length 65536 \
  --host 0.0.0.0 \
  --port 30000
