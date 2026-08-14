#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 across BOTH Sparks — vLLM + Ray TP=2. 2026-08-14 tuning session.
# Usage: qwen38-dual-spark.sh head|worker [tcp|roce]   (default roce)
# Daily-driver latency experiment: single-node baseline is 23.6 tok/s c=1.
set -euo pipefail
ROLE="${1:?head|worker}"
NET="${2:-roce}"
IMG="vllm-dots3:ray"
# Cluster IPs live in gitignored .sparkstation.local.yaml (repo convention).
HEAD_IP="${HEAD_IP:-$(python3 -c "import yaml;print(yaml.safe_load(open('"'"'.sparkstation.local.yaml'"'"'))['"'"'cluster'"'"']['"'"'hosts'"'"']['"'"'worker1'"'"']['"'"'ip'"'"'].rsplit('"'"'.'"'"',1)[0]+'"'"'.10'"'"')" 2>/dev/null || echo MISSING-set-HEAD_IP)}"
IFACE=enp1s0f0np0

NCCL_ENV=(-e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE)
if [ "$NET" = "roce" ]; then
  NCCL_ENV+=(-e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3)
else
  NCCL_ENV+=(-e NCCL_IB_DISABLE=1)
fi

COMMON_DOCKER=(
  docker run -d --gpus all --network host --ipc=host --shm-size 32g
  --ulimit memlock=-1 --cap-add=IPC_LOCK
  -v /dev/infiniband:/dev/infiniband
  -v /home/kshetrajna/.cache/huggingface:/root/.cache/huggingface
  "${NCCL_ENV[@]}"
  -e VLLM_HOST_IP=$([ "$ROLE" = head ] && echo "$HEAD_IP" || echo "${WORKER_IP:-${HEAD_IP%.*}.11}")
)

if [ "$ROLE" = worker ]; then
  "${COMMON_DOCKER[@]}" --name qwen38-ray-worker --entrypoint bash "$IMG" -c \
    "ray start --address=$HEAD_IP:6379 --num-gpus=1 --block"
else
  "${COMMON_DOCKER[@]}" --name qwen38-ray-head --entrypoint bash "$IMG" -c "
    ray start --head --port=6379 --num-gpus=1 &&
    until [ \$(python3 -c 'import ray; ray.init(address=\"auto\"); print(int(ray.cluster_resources().get(\"GPU\",0)))' 2>/dev/null) = 2 ]; do sleep 5; done &&
    vllm serve RadixArk/Qwen3.8-27B-NVFP4 \
      --served-model-name qwen3.8-27b \
      --host 0.0.0.0 --port 8216 \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --distributed-executor-backend ray \
      --gpu-memory-utilization 0.55 \
      --max-model-len 65536 \
      --max-num-seqs 8 \
      --max-num-batched-tokens 8192 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --generation-config auto \
      --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 2}'"
fi
