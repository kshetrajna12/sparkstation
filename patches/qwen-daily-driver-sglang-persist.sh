#!/usr/bin/env bash
# SGLang+DSpark — BEST all-rounder (coding/generic daily driver), synthesized
# from the full trial. Single-stream speed (DSpark+compile) + concurrency &
# safety (fp8 KV) + quality (float32 SSM) + hard memory safety (0.78 + cgroup cap).
set -euo pipefail
CFG="$HOME/.config/qwen38"; HF="$HOME/.cache/huggingface"
IMAGE="lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
KEY=$(cat "$CFG/api-key")
docker run -d --restart unless-stopped --name qwen38-best --gpus all \
  --memory 105g --memory-swap 105g --shm-size 16g --network host --ipc=host \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -v "$CFG/sglang-cache":/cache -v "$HF":/root/.cache/huggingface -v "$CFG":/out \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --trust-remote-code --model-path RadixArk/Qwen3.8-27B-NVFP4 --tp-size 1 \
    --served-model-name qwen3.8-27b \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.78 \
    --attention-backend flashinfer --chunked-prefill-size 4096 \
    --disable-prefill-cuda-graph --cuda-graph-max-bs 8 \
    --speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
    --speculative-draft-attention-backend flashinfer \
    --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant \
    --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype float32 \
    --mamba-full-memory-ratio 11.01 --max-running-requests 8 \
    --enable-torch-compile --torch-compile-max-bs 8 \
    --num-continuous-decode-steps 2 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --chat-template /out/chat-template-sglang.jinja \
    --enable-metrics \
    --api-key "$KEY" --host 0.0.0.0 --port 30000
