#!/usr/bin/env bash
# dots3-note-prev NVFP4 (Frosty40, 167GB) across BOTH Sparks — vLLM + Ray TP=2.
# 2026-08-14 trial. Usage:
#   dots3-dual-spark.sh head    # on primary (192.168.100.10): Ray head + vllm serve
#   dots3-dual-spark.sh worker  # on worker1 (192.168.100.11): Ray worker join
# PREREQS: same vllm/vllm-openai:nightly image + weights in HF cache on BOTH nodes;
# all other models stopped (sparkstation stop + trial containers removed).
set -euo pipefail
ROLE="${1:?head|worker}"
IMG="vllm-dots3:ray"
# Cluster IPs live in gitignored .sparkstation.local.yaml (repo convention).
HEAD_IP="${HEAD_IP:-$(python3 -c "import yaml;print(yaml.safe_load(open('"'"'.sparkstation.local.yaml'"'"'))['"'"'cluster'"'"']['"'"'hosts'"'"']['"'"'worker1'"'"']['"'"'ip'"'"'].rsplit('"'"'.'"'"',1)[0]+'"'"'.10'"'"')" 2>/dev/null || echo MISSING-set-HEAD_IP)}"
IFACE=enp1s0f0np0

COMMON_DOCKER=(
  docker run -d --gpus all --network host --ipc=host --shm-size 32g
  --ulimit memlock=-1
  -v /home/kshetrajna/.cache/huggingface:/root/.cache/huggingface
  -e NCCL_SOCKET_IFNAME=$IFACE
  -e GLOO_SOCKET_IFNAME=$IFACE
  -e NCCL_IB_DISABLE=1
  -e VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_SM120
  -e VLLM_HOST_IP=$([ "$ROLE" = head ] && echo "$HEAD_IP" || echo "${WORKER_IP:-${HEAD_IP%.*}.11}")
)

if [ "$ROLE" = worker ]; then
  "${COMMON_DOCKER[@]}" --name dots3-ray-worker --entrypoint bash "$IMG" -c \
    "ray start --address=$HEAD_IP:6379 --num-gpus=1 --block"
else
  "${COMMON_DOCKER[@]}" --name dots3-ray-head --entrypoint bash "$IMG" -c "
    ray start --head --port=6379 --num-gpus=1 &&
    until [ \$(python3 -c 'import ray; ray.init(address=\"auto\"); print(int(ray.cluster_resources().get(\"GPU\",0)))' 2>/dev/null) = 2 ]; do sleep 5; done &&
    vllm serve Frosty40/dots3-note-prev-NVFP4 \
      --served-model-name dots3-note \
      --host 0.0.0.0 --port 8215 \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --distributed-executor-backend ray \
      --gpu-memory-utilization 0.80 \
      --max-model-len 32768 \
      --max-num-seqs 4 \
      --max-num-batched-tokens 8192 \
      --kv-cache-dtype bfloat16 \
      --attention-backend FLASH_ATTN_MLA_SPARSE \
      --generation-config auto"
fi
