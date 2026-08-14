#!/usr/bin/env bash
# Qwen3.8-27B-FP8 trial — primary:8213 (day-0, 2026-08-14).
# Image: vllm/vllm-openai:muse-glimmer-arm64-cu130 (Aug 11 main build — registers
# Qwen3_5ForConditionalGeneration + Qwen3_5MTP). MTP off initially: on Qwen3.6-A3B
# spec decode forced PIECEWISE cudagraphs and lost to MTP-off (memory: qwen-serving
# lessons); retest MTP as a separate A/B once baseline is sane.
set -euo pipefail

docker run -d \
  --platform linux/arm64 \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p 8213:8213 \
  -v /home/kshetrajna/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint vllm \
  --name qwen38-trial \
  vllm/vllm-openai:muse-glimmer-arm64-cu130 \
  serve RadixArk/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8213 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --generation-config auto \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
