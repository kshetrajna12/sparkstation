#!/usr/bin/env bash
# Muse Glimmer 30B on SGLang (LMSYS reference stack) — trial on primary:8211.
# Mode: pass "lm-only" to reproduce the LMSYS text-only bench config
# (--language-model-only), anything else for full multimodal serving.
set -euo pipefail
MODE="${1:-multimodal}"

EXTRA=()
if [ "$MODE" = "lm-only" ]; then EXTRA+=(--language-model-only); fi

docker run -d \
  --platform linux/arm64 \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8211:30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --name glimmer-sglang-trial \
  lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b \
  python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/local/muse-glimmer-radixark-lmheadfix \
  --served-model-name muse-glimmer \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.38 \
  --context-length 65536 \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path meta-models/Muse-Glimmer-30B-assistant \
  --trust-remote-code \
  "${EXTRA[@]}"
