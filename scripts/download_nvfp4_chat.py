"""Pre-download nvidia/Qwen3.6-35B-A3B-NVFP4 into the HF cache.

Run in a tmux session so the download survives SSH disconnects:

    tmux new -d -s nvfp4 'cd /home/kshetrajna/src/github.com/sparkstation && \
        .venv/bin/python scripts/download_nvfp4_chat.py 2>&1 | tee data/nvfp4_download.log'

To monitor:
    tmux attach -t nvfp4              # attach (Ctrl-B D to detach)
    tail -f data/nvfp4_download.log   # or watch the log

Resumable: if interrupted, re-running picks up where it left off.
"""
from huggingface_hub import snapshot_download

REPO = "nvidia/Qwen3.6-35B-A3B-NVFP4"

print(f"[download_nvfp4] Fetching {REPO} (~22 GB) into ~/.cache/huggingface/hub")
print("[download_nvfp4] Resumable; safe to interrupt and rerun.")

path = snapshot_download(
    repo_id=REPO,
    # Default cache dir is ~/.cache/huggingface/hub — same dir vllm-qwen35-mxfp4
    # mounts into the model container at /root/.cache/huggingface, so the swap
    # will read directly from this cache with no extra plumbing.
    resume_download=True,
    # Skip .gguf and .pt mirror files — vLLM only needs the safetensors set.
    ignore_patterns=["*.gguf", "*.pt", "consolidated.*"],
)

print(f"[download_nvfp4] Downloaded to: {path}")
print("[download_nvfp4] You can now run `sparkstation models swap qwen3.5-35b`")
print("[download_nvfp4] once models.yaml is updated.")
