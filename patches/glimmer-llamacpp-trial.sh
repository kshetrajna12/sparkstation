#!/usr/bin/env bash
# Muse Glimmer 30B — llama.cpp NATIVE sm_121 build (28 tok/s vs 18 on the generic
# official image; GB10 needs -DCMAKE_CUDA_ARCHITECTURES=121). primary:8212.
# CRITICAL flags: --spec-type draft-dflash (defaults off) + --swa-full (SWA layers
# otherwise silently disable speculation).
set -euo pipefail
exec "$LLAMA_CPP_BUILD/bin/llama-server"  # point at your llama.cpp build \
  -m $HOME/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-GGUF/snapshots/a0532f7263ee67f1e0a5f5c5fdcd50dd62fc9aa4/muse-glimmer-30B-kquant-17gb.gguf \
  --mmproj $HOME/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-GGUF/snapshots/a0532f7263ee67f1e0a5f5c5fdcd50dd62fc9aa4/mmproj-kquant.gguf \
  --model-draft $HOME/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-GGUF/snapshots/a0532f7263ee67f1e0a5f5c5fdcd50dd62fc9aa4/dflash-kquant.gguf \
  --gpu-layers 999 --gpu-layers-draft 999 \
  --spec-draft-n-max 16 --spec-draft-n-min 0 \
  --spec-type draft-dflash \
  --spec-draft-backend-sampling \
  --swa-full \
  --ctx-size 65536 --parallel 8 \
  --jinja \
  --host 0.0.0.0 --port 8212 \
  --alias muse-glimmer
