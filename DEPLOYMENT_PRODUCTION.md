# Sparkstation Production Deployment Guide

**Target Platform**: NVIDIA DGX Spark (Grace Blackwell)
**Architecture**: Docker-based model backends with Supervisor coordination
**Version**: 0.1.0
**Last Updated**: 2025-11-02

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start (Docker Mode - Recommended)](#quick-start-docker-mode---recommended)
4. [Cluster mode (multi-Spark)](#cluster-mode-multi-spark)
5. [Directory Layout](#directory-layout)
6. [Configuration](#configuration)
7. [Systemd Services](#systemd-services)
8. [Model Configuration](#model-configuration)
9. [Monitoring](#monitoring)
10. [Maintenance](#maintenance)
11. [Troubleshooting](#troubleshooting)
12. [Advanced: Subprocess Mode](#advanced-subprocess-mode)

---

## Architecture Overview

### Design Philosophy

**✅ Docker-First Approach (Recommended)**
- Official NVIDIA vLLM Docker images with Blackwell GPU support
- Simplified setup - no conda/micromamba management
- Better isolation and reproducibility
- Automatic GPU passthrough via `--gpus` flag
- One-command backend setup

**✅ Lightweight Supervisor**
- Sparkstation + LiteLLM in `uv` environment (minimal dependencies)
- Manages Docker containers for model backends
- Handles lifecycle, health checks, auto-suspend/resume

**✅ Systemd Process Management**
- Supervisor and gateway as systemd services
- Daily maintenance via systemd timer
- Clean restart, logging, and watchdog support

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DGX Spark (128 GB Unified Memory)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sparkstation (uv env) - /opt/sparkstation                      │
│  ├── Supervisor (FastAPI) - Port 9001                           │
│  ├── LiteLLM Gateway - Port 8000                                │
│  └── Manages Docker containers for model backends               │
│                                                                  │
│  Docker Containers (vLLM)                                        │
│  ├── sparkstation-{model-id-1} - Port 8001                      │
│  ├── sparkstation-{model-id-2} - Port 8002                      │
│  └── sparkstation-{model-id-3} - Port 8003                      │
│      └── Image: nvcr.io/nvidia/vllm:25.11-py3                   │
│                                                                  │
│  Shared Resources                                               │
│  ├── /var/lib/models - HuggingFace cache (shared via volumes)  │
│  ├── /var/log/sparkstation - Supervisor logs                   │
│  └── /opt/sparkstation/data - SQLite database                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Software

1. **Docker with NVIDIA Container Toolkit**
   ```bash
   # Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com | sh

   # NVIDIA Container Toolkit
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **Python 3.11+ and uv**
   ```bash
   # uv package manager
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **NVIDIA GPU Driver**
   - DGX Spark typically has CUDA 12.4 or 12.6 pre-installed
   - Verify: `nvidia-smi`

### Verify Setup

```bash
# Test Docker + NVIDIA
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start (Docker Mode - Recommended)

### 1. Clone and Install Sparkstation

```bash
# Clone repository
git clone https://github.com/kshetrajna12/sparkstation.git
cd sparkstation

# Install Sparkstation (lightweight uv environment)
uv sync

# Create data directory
mkdir -p data
```

### 2. Pull Docker Image

```bash
# Automated setup script
./scripts/setup_backends.sh

# OR manual pull
docker pull nvcr.io/nvidia/vllm:25.11-py3

# Verify
./scripts/verify_backends.sh
```

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit configuration (optional - defaults work for most cases)
# Key settings:
#   USE_DOCKER=true  (already default)
#   TOTAL_UNIFIED_MEMORY_GB=128
#   API_KEY=your-secret-key-here
```

### 4. Configure Models

Edit `models.yaml` to define auto-load models:

```yaml
autoload:
  enabled: true
  models:
    - name: "Qwen/Qwen3-VL-4B-Instruct-FP8"
      alias: "qwen3-vl-4b"
      backend: "vllm"
      quantization: "none"
      idle_timeout_minutes: 30
      auto_suspend_enabled: true
      extra_args:
        max_model_len: 8192
        gpu_memory_utilization: 0.25
```

### 5. Start Sparkstation

```bash
# Using CLI (recommended)
sparkstation start -d

# OR using scripts
./scripts/start_supervisor.sh  # Terminal 1
./scripts/start_gateway.sh     # Terminal 2
```

### 6. Verify Deployment

```bash
# Check status
sparkstation status

# Check Supervisor health
curl http://localhost:9001/health

# Check Gateway health
curl http://localhost:8000/health

# List running models
sparkstation models list
```

---

## Cluster mode (multi-Spark)

Sparkstation can span more than one DGX Spark by promoting one host to **primary** and attaching one or more **worker** hosts. All CLI, gateway, and supervisor traffic still land on the primary; the primary places selected models on workers over SSH and drives their Docker daemons directly.

### When to use it

- **Multi-agent workloads** that fan out concurrent chat requests and saturate a single Spark's KV cache.
- **Dedicated chat host**: move the largest chat model to its own worker so vision/embedding/detection models on the primary don't have to share GPU memory or throughput.
- Any deployment where physical isolation per model class is worth the extra setup cost.

### Prerequisites

- **Passwordless SSH** from the primary user to `<ssh-user>@<worker-ip>`. Validate with:
  ```bash
  ssh -o BatchMode=yes <ssh-user>@<worker-ip> true
  ```
- **Docker + NVIDIA Container Toolkit** installed on **both** hosts, and the SSH user in the `docker` group on the worker.
- **Matching Docker versions** are preferred (mixed versions may work but are not the tested configuration).
- **HuggingFace cache mirrored** to the worker before first launch, so cold-start doesn't re-download large snapshots:
  ```bash
  sparkstation cluster sync-cache worker1
  ```
  This rsyncs `/var/lib/models` from the primary to the worker.

### Public/private config split

Cluster config is intentionally split into two files so real network details never land in git:

- **`models.yaml`** (committed) — cluster role names, model specs, per-profile `host:` overrides.
- **`.sparkstation.local.yaml`** (gitignored) — maps role names to real `<worker-ip>` / `<ssh-user>` / `<hostname>` values.

Example `models.yaml` fragment:

```yaml
cluster:
  hosts:
    - name: primary
    - name: worker1

profiles:
  image-indexing:
    overrides:
      qwen3.5-35b:
        host: worker1
        memory_gb: 100
        extra_args:
          max_model_len: 32768
          max_num_seqs: 64
```

Example `.sparkstation.local.yaml` (never committed):

```yaml
cluster:
  hosts:
    - name: worker1
      ssh_user: <ssh-user>
      address: <worker-ip>
      hostname: <hostname>
```

### Swap the chat model onto a dedicated worker

With the profile override above in place:

```bash
sparkstation models swap qwen3.5-35b
```

The supervisor tears down the primary-hosted instance, launches a new container on `worker1` over SSH, updates gateway routing, and reports the new host in `sparkstation status`.

### Rollback (chat back to primary)

Edit the `image-indexing` profile in `models.yaml` — either delete the `host: worker1` line from the `qwen3.5-35b` override, or set it to `host: primary` — then re-swap:

```bash
sparkstation models swap qwen3.5-35b
```

The chat model returns to the primary host with its base spec (default `max_model_len` / `max_num_seqs`, no expanded batching room).

### Benchmark: dedicated chat worker

Moving the 35B chat model onto its own Spark, with the expanded batching config above, materially improves throughput and time-to-first-token under concurrent chat load:

| Concurrency | Throughput before | Throughput after | TTFT before | TTFT after |
|-------------|-------------------|------------------|-------------|------------|
| c=16        | —                 | —                | 5.7 s       | 1.4 s      |
| c=32        | 107 tok/s         | 159 tok/s (+49%) | —           | —          |

The TTFT collapse at c=16 (5.7 s → 1.4 s) is the largest UX win for multi-agent workflows; the c=32 throughput bump (+49%) is the sustained-load win.

---

## Directory Layout

```bash
/opt/sparkstation/              # Sparkstation installation
├── .venv/                      # uv virtual environment
├── supervisor/                 # Supervisor code
├── gateway/                    # Gateway code
├── scripts/                    # Utility scripts
├── data/                       # Runtime data
│   ├── sparkstation.db         # SQLite database
│   └── sparkstation.log        # Rotating logs
└── models.yaml                 # Model configuration

/var/lib/models/                # HuggingFace model cache (shared)
├── models--Qwen--Qwen3-VL-4B-Instruct-FP8/
└── models--openai--gpt-oss-20b/

/var/log/sparkstation/          # Sparkstation logs (if configured)
└── sparkstation.log            # Rotating log file

/etc/sparkstation/              # System configuration
├── env                         # Central environment file
└── litellm.yaml                # LiteLLM routing config
```

---

## Configuration

### Environment Variables

Key settings in `.env`:

```bash
# Backend mode (Docker is recommended)
USE_DOCKER=true
VLLM_DOCKER_IMAGE=nvcr.io/nvidia/vllm:25.11-py3

# DGX Spark Constraints
TOTAL_UNIFIED_MEMORY_GB=128
MEMORY_HARD_LIMIT_GB=110  # 85% of total
MAX_RESIDENT_MODELS=5
# Note: in cluster mode, this budget is enforced on the PRIMARY host only.
# Remote workers have their own capacity, managed on the target host —
# per-host memory tracking is a follow-up (see Monitoring for the same caveat).

# Auto-suspend
AUTO_SUSPEND_ENABLED=true
DEFAULT_IDLE_TIMEOUT_MINUTES=30

# Health Checks
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL_SECONDS=300  # 5 minutes
HEALTH_CHECK_MAX_FAILURES=3

# Auto-restart
AUTO_RESTART_ENABLED=true
AUTO_RESTART_MAX_ATTEMPTS=3
AUTO_RESTART_BACKOFF_MINUTES=1,5,15  # Exponential backoff

# Security
API_KEY=your-secret-key-here  # Optional: Enable API key auth

# LiteLLM Gateway
LITELLM_ADMIN_URL=http://127.0.0.1:8000
GATEWAY_SYNC_INTERVAL_SECONDS=60

# Logging
LOG_TO_FILE=true
LOG_FILE_PATH=./data/sparkstation.log
LOG_MAX_BYTES=10485760  # 10 MB per file
LOG_BACKUP_COUNT=5
```

### Shared Model Cache

All Docker containers share `/var/lib/models` for HuggingFace cache:

```bash
# Create shared cache directory
sudo mkdir -p /var/lib/models
sudo chown -R $USER:$USER /var/lib/models

# Pre-download models (optional but recommended)
export HF_HOME=/var/lib/models
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct-FP8
```

Docker containers automatically mount this directory via `-v /var/lib/models:/root/.cache/huggingface`.

---

## Systemd Services

### 1. Sparkstation Supervisor Service

Use existing template: `scripts/systemd/sparkstation-supervisor.service`

Install:
```bash
sudo cp scripts/systemd/sparkstation-supervisor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sparkstation-supervisor
sudo systemctl start sparkstation-supervisor
```

### 2. LiteLLM Gateway Service

Use existing template: `scripts/systemd/sparkstation-gateway.service`

Install:
```bash
sudo cp scripts/systemd/sparkstation-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sparkstation-gateway
sudo systemctl start sparkstation-gateway
```

### 3. Daily Maintenance

Enable daily maintenance (3 AM):
```bash
sudo cp scripts/systemd/sparkstation-maintenance.service /etc/systemd/system/
sudo cp scripts/systemd/sparkstation-maintenance.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sparkstation-maintenance.timer
sudo systemctl start sparkstation-maintenance.timer
```

### Restart behavior with cluster mode

When the supervisor restarts (systemctl restart, `sparkstation stop && sparkstation start`, or crash-recover), the cluster additions are backwards-compatible:

- The SQLite migration that adds the `host` column to the model-instance table is **idempotent** — existing single-host DBs are upgraded in place on first boot with no data wipe. Rolling back to a single-host build with the same DB is also safe (the column is simply unused).
- On startup, the reconcile loop iterates over every persisted instance and probes its Docker daemon on the assigned host (local socket for `primary`, SSH-tunneled Docker for remote workers). Containers missing from their target daemon are marked stopped and re-launched per the current profile.
- The recommended restart flow is still `sparkstation stop && sparkstation start --profile <name>` — `sparkstation restart` is known to race with the DB lock and is not cluster-hardened yet.

---

## Model Configuration

### Auto-Load Models (Recommended)

Edit `models.yaml`:

```yaml
autoload:
  enabled: true
  models:
    # Vision model
    - name: "Qwen/Qwen3-VL-4B-Instruct-FP8"
      alias: "qwen3-vl-4b"
      backend: "vllm"
      quantization: "none"
      idle_timeout_minutes: 30
      auto_suspend_enabled: true
      extra_args:
        max_model_len: 8192
        max_concurrent_requests: 32
        gpu_memory_utilization: 0.25

    # Reasoning model
    - name: "openai/gpt-oss-20b"
      alias: "gpt-oss-20b"
      backend: "vllm"
      quantization: "none"  # Uses built-in MXFP4
      idle_timeout_minutes: 60
      auto_suspend_enabled: true
      extra_args:
        max_model_len: 16384
        max_concurrent_requests: 32
        gpu_memory_utilization: 0.30
```

Models automatically load on supervisor startup.

### Manual Model Management

```bash
# Start a model
curl -X POST http://localhost:9001/models/start \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "model_name": "Qwen/Qwen3-VL-4B-Instruct-FP8",
    "backend": "vllm",
    "model_alias": "qwen3-vl-4b",
    "quantization": "none"
  }'

# Or via CLI
sparkstation models start Qwen/Qwen3-VL-4B-Instruct-FP8 \
  --alias qwen3-vl-4b \
  --backend vllm \
  --quantization none
```

---

## Monitoring

### Prometheus + Grafana

#### Setup Prometheus

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'sparkstation'
    static_configs:
      - targets: ['localhost:9001']
    scrape_interval: 15s
```

#### Import Grafana Dashboard

1. Open Grafana → Dashboards → Import
2. Upload `monitoring/grafana-dashboard.json`
3. Select Prometheus datasource

See `monitoring/README.md` for detailed setup.

### Available Metrics

- `unified_memory_used_bytes` - Total memory usage
- `gpu_temperature_celsius` - GPU temperature
- `gpu_power_draw_watts` - Power consumption
- `model_status` - Model status (0-4)
- `resident_models_count` - Active models
- `model_requests_total` - Request counter per model

> **Cluster note:** these metrics do not yet carry a `host` label. In a multi-Spark cluster, today's Grafana dashboards silently aggregate values across the primary and any workers (e.g. `unified_memory_used_bytes` and `gpu_*` reflect only the primary's exporter). A `host` label on all series is on the roadmap; until then, treat cluster-wide dashboards as primary-host views.

---

## Maintenance

### Daily Automated Maintenance

Runs via systemd timer at 3 AM:
- Cleanup log files older than 30 days
- Vacuum SQLite database
- Detect stale/zombie models
- Check for port leaks
- Generate resource usage snapshot

Manual run:
```bash
python scripts/maintenance.py

# Dry run
python scripts/maintenance.py --dry-run

# Check last run
sudo journalctl -u sparkstation-maintenance -n 50
```

### Model Cache Cleanup

```bash
# Check cache size
du -sh /var/lib/models

# List cached models
huggingface-cli scan-cache

# Interactive cleanup
huggingface-cli delete-cache
```

### Database Cleanup

```bash
# Full cleanup (stops all services and resets database)
sparkstation cleanup --force

# Manual database reset
rm data/sparkstation.db
sparkstation restart
```

---

## Troubleshooting

### Model fails to start

```bash
# Check logs
sparkstation models logs <model-id> -f

# Or check Supervisor logs
tail -f data/sparkstation.log

# Check Docker containers
docker ps -a --filter name=sparkstation-

# Check specific container
docker logs sparkstation-<model-id>
```

### Common Issues

**401 Unauthorized**:
- API key required - add `X-API-Key` header to requests
- Or disable auth: remove `API_KEY` from `.env`

**Insufficient memory**:
- Check `/resources` endpoint: `curl http://localhost:9001/resources`
- Reduce `MAX_RESIDENT_MODELS` in `.env`
- Enable auto-suspend: `AUTO_SUSPEND_ENABLED=true`

**Container not starting**:
- Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
- Check Docker daemon: `sudo systemctl status docker`
- Verify image exists: `docker images | grep vllm`

**Gateway returns 404**:
- Model may be suspended - first request triggers auto-resume (~15s)
- Check model status: `sparkstation status`
- Verify gateway is running: `curl http://localhost:8000/health`

### Logs

**Supervisor**:
```bash
# File logs
tail -f data/sparkstation.log

# Systemd logs
sudo journalctl -u sparkstation-supervisor -f
```

**Gateway**:
```bash
sudo journalctl -u sparkstation-gateway -f
```

**Model containers**:
```bash
sparkstation models logs <model-id> -f
```

---

## Advanced: Subprocess Mode

For users who prefer direct Python execution over Docker:

### Why Use Subprocess Mode?

- Need custom vLLM builds
- Debugging backend code
- Maximum performance (no container overhead)
- Fine-grained control over Python environment

### Prerequisites

Install micromamba (lightweight conda):
```bash
curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
sudo mv bin/micromamba /usr/local/bin/
```

### Backend Environment Setup

#### vLLM Environment

```bash
# Create directory
sudo mkdir -p /opt/backends/vllm
sudo chown $USER /opt/backends/vllm

# Create environment
micromamba create -y -p /opt/backends/vllm -c conda-forge python=3.11

# Install PyTorch with CUDA 12.4
micromamba run -p /opt/backends/vllm pip install \
  "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
micromamba run -p /opt/backends/vllm pip install "vllm==0.6.*"

# Verify
/opt/backends/vllm/bin/python -c "import vllm; print(vllm.__version__)"
```

### Configuration for Subprocess Mode

Update `.env`:
```bash
# Disable Docker mode
USE_DOCKER=false

# Provide Python path
VLLM_PYTHON_PATH=/opt/backends/vllm/bin/python
```

### Shared Environment File

Create `/etc/sparkstation/env`:
```bash
# Backend Python paths
VLLM_PY=/opt/backends/vllm/bin/python

# HuggingFace cache (shared)
HF_HOME=/var/lib/models
HUGGINGFACE_HUB_CACHE=/var/lib/models
TOKENIZERS_PARALLELISM=false

# GPU configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Notes on Subprocess Mode

- More complex setup (manage conda environments)
- No automatic GPU isolation
- Requires careful version management
- Not recommended for production unless you have specific requirements

---

## Security Considerations

### Network Binding

- **Supervisor**: Binds to `127.0.0.1` (localhost only)
- **Gateway**: Binds to `127.0.0.1` (localhost only)
- **Model containers**: Expose ports only to host

### API Key Authentication

Generate strong API key:
```bash
openssl rand -hex 32
```

Add to `.env`:
```bash
API_KEY=<generated-key>
```

Use in requests:
```bash
curl -H "X-API-Key: <your-key>" http://localhost:9001/models/start ...
```

### Firewall Rules

```bash
# Allow only from localhost
sudo ufw default deny incoming
sudo ufw allow from 127.0.0.1 to any port 8000:9001
sudo ufw enable
```

---

## Performance Tuning

### Docker Container Limits

Containers automatically use:
- `--gpus all` for full GPU access
- `--shm-size 8g` for adequate shared memory
- `-v /var/lib/models:/root/.cache/huggingface` for model cache

### Model-Specific Settings

In `models.yaml` `extra_args`:

```yaml
extra_args:
  max_model_len: 8192  # Context length
  max_concurrent_requests: 32  # Batch size
  gpu_memory_utilization: 0.30  # GPU memory fraction (0.0-1.0)
  tensor_parallel_size: 1  # Multi-GPU support (future)
```

### DGX Spark Optimizations

- **Unified memory**: 128 GB shared CPU+GPU pool
- **Hard limit**: 110 GB (85% of total)
- **Max concurrent models**: 3 (configurable)
- **Thermal management**: Auto-suspend at 80°C sustained
- **Auto-suspend**: Frees GPU after 30 min idle

> **Cluster note:** the unified-memory budget and hard limit above apply to the **primary** host only. Remote workers report their own capacity to the target-host Docker daemon and are not currently rolled up into the primary's memory accounting; per-host memory tracking is a follow-up.

---

## Reference

- **Sparkstation GitHub**: https://github.com/kshetrajna12/sparkstation
- **vLLM Docs**: https://docs.vllm.ai/
- **LiteLLM Docs**: https://docs.litellm.ai/
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **DGX Spark Info**: NVIDIA documentation

---

**Last Updated**: 2025-11-02
**Version**: 0.1.0
**Maintainer**: Sparkstation Team
