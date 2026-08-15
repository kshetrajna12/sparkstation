#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 (official ~4.6-bit, 167GB) across BOTH Sparks — vLLM+Ray TP=2.
# 2026-08-15 coding-profile trial. Goal: validate stock-vLLM DSA sparse attention on
# sm_121 and measure minimum KV for 256K context, then work memory budget backwards.
# Usage: dsv4-dual-spark.sh head|worker [tcp|roce]   (default roce)
set -euo pipefail
ROLE="${1:?head|worker}"
NET="${2:-roce}"
IMG="vllm-dots3:ray"
# Cluster IPs live in gitignored .sparkstation.local.yaml (repo convention).
HEAD_IP="${HEAD_IP:-192.168.100.10}"
IFACE=enp1s0f0np0

NCCL_ENV=(-e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE)
if [ "$NET" = "roce" ]; then
  NCCL_ENV+=(-e NCCL_IB_DISABLE=0 -e NCCL_NET=IB -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3)
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
  "${COMMON_DOCKER[@]}" --name dsv4-ray-worker --entrypoint bash "$IMG" -c \
    "ray start --address=$HEAD_IP:6379 --num-gpus=1 --block"
else
  "${COMMON_DOCKER[@]}" --name dsv4-ray-head --entrypoint bash "$IMG" -c "
    ray start --head --port=6379 --num-gpus=1 &&
    until [ \$(python3 -c 'import ray; ray.init(address=\"auto\"); print(int(ray.cluster_resources().get(\"GPU\",0)))' 2>/dev/null) = 2 ]; do sleep 5; done &&
    vllm serve nvidia/DeepSeek-V4-Flash-NVFP4 \
      --served-model-name dsv4-flash \
      --host 0.0.0.0 --port 8217 \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --distributed-executor-backend ray \
      --gpu-memory-utilization 0.85 \
      --max-model-len 262144 \
      --max-num-seqs 4 \
      --max-num-batched-tokens 8192 \
      --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
      --reasoning-parser deepseek_v4 \
      --generation-config auto"
fi
