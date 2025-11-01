# Sparkstation Production Deployment Guide

**Target Platform**: NVIDIA DGX Spark (Grace Blackwell)
**Architecture**: Multi-environment with separate backend isolation
**Version**: 0.1.0
**Last Updated**: 2025-10-31

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Layout](#directory-layout)
3. [Backend Environment Setup](#backend-environment-setup)
4. [Configuration Files](#configuration-files)
5. [Systemd Service Units](#systemd-service-units)
6. [Deployment Steps](#deployment-steps)
7. [GPU Partitioning & Performance](#gpu-partitioning--performance)
8. [Health Checks & Monitoring](#health-checks--monitoring)
9. [Logging & Retention](#logging--retention)
10. [Maintenance](#maintenance)

---

## Architecture Overview

### Design Decisions

**✅ YES: Separate Conda/Mamba Environments per Backend**
- SGLang in `/opt/backends/sglang` (conda env)
- vLLM in `/opt/backends/vllm` (conda env)
- Sparkstation + LiteLLM in uv environment (lightweight)

**✅ YES: Systemd Process Management**
- Each backend as independent systemd unit
- Sparkstation can still spawn/stop via subprocess
- Clean restart, logging, and watchdog support

**✅ YES: Shared Model Cache**
- Single `/var/lib/models` for all HuggingFace weights
- Reduces disk usage and download time

**❌ NO: Docker (for now)**
- Single-node setup doesn't need container isolation
- Direct GPU access is faster
- Saves overhead on unified memory system
- **Later**: Consider Podman if you need portability

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DGX Spark (128 GB Unified Memory)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sparkstation (uv env) - /opt/sparkstation                      │
│  ├── Supervisor (FastAPI) - Port 9001                           │
│  ├── LiteLLM Gateway - Port 8000                                │
│  └── Manages backends via subprocess + systemd                  │
│                                                                  │
│  Backend Environments (micromamba/conda)                        │
│  ├── /opt/backends/sglang - Ports 8011+                         │
│  │   ├── Vision models (Qwen2-VL, LLaVA, etc.)                 │
│  │   └── Text models (Qwen2.5, etc.)                            │
│  ├── /opt/backends/vllm - Ports 8021+                           │
│  │   ├── Text models (Llama, Mistral, etc.)                    │
│  │   └── Optimized for high throughput                          │
│  └── Future: /opt/backends/trt-llm                              │
│                                                                  │
│  Shared Resources                                               │
│  ├── /var/lib/models - HuggingFace cache (shared)              │
│  ├── /var/log/sparkstation - Supervisor logs                   │
│  └── /var/log/llm/{sglang,vllm} - Backend logs                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Layout

```bash
/opt/sparkstation/              # Sparkstation uv app
├── .venv/                      # uv virtual environment
├── supervisor/                 # Supervisor code
├── gateway/                    # Gateway code
├── scripts/                    # Utility scripts
└── data/                       # Runtime data (SQLite DB)

/opt/backends/                  # Backend environments
├── sglang/                     # SGLang conda/micromamba env
│   ├── bin/python             # Python interpreter
│   └── lib/python3.11/...
└── vllm/                       # vLLM conda/micromamba env
    ├── bin/python
    └── lib/python3.11/...

/var/lib/models/                # HuggingFace model cache (shared)
├── models--Qwen--Qwen2.5-7B-Instruct/
├── models--Qwen--Qwen2-VL-7B-Instruct/
└── ...

/var/log/sparkstation/          # Sparkstation logs
└── sparkstation.log            # Rotating log file

/var/log/llm/                   # Backend logs
├── sglang/
│   ├── stdout.log
│   └── stderr.log
└── vllm/
    ├── stdout.log
    └── stderr.log

/etc/sparkstation/              # Configuration
├── env                         # Central environment file
└── litellm.yaml                # LiteLLM routing config
```

---

## Backend Environment Setup

### Prerequisites

```bash
# Check CUDA version (should be 12.4 or 12.6 for DGX Spark)
nvidia-smi

# Install micromamba (lightweight conda alternative)
curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
sudo mv bin/micromamba /usr/local/bin/
```

### SGLang Environment

```bash
# Create directory
sudo mkdir -p /opt/backends/sglang
sudo chown $USER /opt/backends/sglang

# Create environment with Python 3.11
micromamba create -y -p /opt/backends/sglang -c conda-forge python=3.11

# Install PyTorch with CUDA 12.4 support
micromamba run -p /opt/backends/sglang pip install \
  "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu124

# Install SGLang with all dependencies
micromamba run -p /opt/backends/sglang pip install "sglang[all]==0.3.*"

# Optional: Install FlashInfer for faster inference (match CUDA version)
micromamba run -p /opt/backends/sglang pip install flashinfer

# Optional: Install Flash-Attention 2 (if needed)
micromamba run -p /opt/backends/sglang pip install "flash-attn==2.6.*"

# Verify installation
/opt/backends/sglang/bin/python -c "import sglang; print(sglang.__version__)"
```

### vLLM Environment

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

# Verify installation
/opt/backends/vllm/bin/python -c "import vllm; print(vllm.__version__)"
```

### Version Pinning Notes

**CRITICAL**: Match CUDA wheels to your driver version:
- DGX Spark typically runs CUDA 12.4 or 12.6
- Check with `nvidia-smi` → look for "CUDA Version: 12.x"
- Use corresponding PyTorch wheel: `cu124` for CUDA 12.4, `cu126` for CUDA 12.6
- FlashInfer and Flash-Attention must match the same CUDA version

**Don't mix CUDA versions** across PyTorch, FlashInfer, and Flash-Attention in the same env!

---

## Configuration Files

### 1. Central Environment File

`/etc/sparkstation/env`

```bash
# Backend Python paths (absolute)
SGLANG_PY=/opt/backends/sglang/bin/python
VLLM_PY=/opt/backends/vllm/bin/python

# HuggingFace cache (shared across all backends)
HF_HOME=/var/lib/models
HUGGINGFACE_HUB_CACHE=/var/lib/models
TOKENIZERS_PARALLELISM=false

# GPU configuration
CUDA_VISIBLE_DEVICES=0  # Single GPU on DGX Spark
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=0
NCCL_ASYNC_ERROR_HANDLING=1

# Port allocation
SGLANG_PORT=8011
VLLM_PORT=8021
LITELLM_PORT=8000
SUPERVISOR_PORT=9001

# Logging
LOG_DIR=/var/log/llm

# Performance tuning
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

### 2. Sparkstation Environment

Update `/opt/sparkstation/.env`:

```bash
# Server settings
HOST=127.0.0.1
PORT=9001
LOG_LEVEL=info

# Backend Python paths (for subprocess launching)
SGLANG_PYTHON_PATH=/opt/backends/sglang/bin/python
VLLM_PYTHON_PATH=/opt/backends/vllm/bin/python

# DGX Spark hardware constraints
TOTAL_UNIFIED_MEMORY_GB=128
MEMORY_HARD_LIMIT_GB=110
MEMORY_SOFT_LIMIT_GB=100
MAX_RESIDENT_MODELS=3

# Port allocation for model servers
MODEL_PORT_RANGE_START=8001
MODEL_PORT_RANGE_END=8100

# Auto-suspend settings
AUTO_SUSPEND_ENABLED=true
DEFAULT_IDLE_TIMEOUT_MINUTES=30
AUTO_SUSPEND_CHECK_INTERVAL_SECONDS=60

# Health checks
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL_SECONDS=300
HEALTH_CHECK_TIMEOUT_SECONDS=5
HEALTH_CHECK_MAX_FAILURES=3

# Model restart policy
AUTO_RESTART_ENABLED=true
AUTO_RESTART_MAX_ATTEMPTS=3
AUTO_RESTART_BACKOFF_MINUTES=1,5,15

# LiteLLM Gateway settings
LITELLM_ADMIN_URL=http://127.0.0.1:8000
LITELLM_MASTER_KEY=sk-sparkstation-admin
GATEWAY_SYNC_INTERVAL_SECONDS=60

# Database
DATABASE_URL=sqlite+aiosqlite:////opt/sparkstation/data/sparkstation.db

# Security
API_KEY=your-production-api-key-here

# Logging
LOG_TO_FILE=true
LOG_FILE_PATH=/var/log/sparkstation/sparkstation.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
```

### 3. LiteLLM Routing Configuration

`/etc/sparkstation/litellm.yaml`

```yaml
model_list:
  # SGLang models (vision + text)
  - model_name: qwen2-vl-7b
    litellm_params:
      model: openai/qwen2-vl-7b
      api_base: http://127.0.0.1:8011/v1
      api_key: "EMPTY"

  - model_name: qwen2.5-7b
    litellm_params:
      model: openai/qwen2.5-7b
      api_base: http://127.0.0.1:8012/v1
      api_key: "EMPTY"

  # vLLM models (text optimized)
  - model_name: llama3-8b
    litellm_params:
      model: openai/llama3-8b
      api_base: http://127.0.0.1:8021/v1
      api_key: "EMPTY"

router_settings:
  num_retries: 2
  timeout: 120
  enable_pre_call_checks: true

general_settings:
  master_key: sk-sparkstation-admin
  database_url: sqlite:////etc/sparkstation/litellm.db
```

---

## Systemd Service Units

### 1. SGLang Service (Vision Model Example)

`/etc/systemd/system/sglang-vision.service`

```ini
[Unit]
Description=SGLang Server - Vision Models
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/sparkstation/env
User=llm
Group=llm
WorkingDirectory=/var/lib/models

# Create log directory
ExecStartPre=/usr/bin/mkdir -p ${LOG_DIR}/sglang

# Launch SGLang with Qwen2-VL
ExecStart=/opt/backends/sglang/bin/python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8011 \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --trust-remote-code

# Restart policy
Restart=always
RestartSec=5
StartLimitBurst=3
StartLimitIntervalSec=300

# Logging
StandardOutput=append:/var/log/llm/sglang/stdout.log
StandardError=append:/var/log/llm/sglang/stderr.log

# Resource limits
LimitNOFILE=1048576
LimitNPROC=4096

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/lib/models /var/log/llm

[Install]
WantedBy=multi-user.target
```

### 2. vLLM Service (Text Model Example)

`/etc/systemd/system/vllm-text.service`

```ini
[Unit]
Description=vLLM Server - Text Models
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/sparkstation/env
User=llm
Group=llm
WorkingDirectory=/var/lib/models

# Create log directory
ExecStartPre=/usr/bin/mkdir -p ${LOG_DIR}/vllm

# Launch vLLM with Llama model
ExecStart=/opt/backends/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8021 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --quantization fp8

# Restart policy
Restart=always
RestartSec=5
StartLimitBurst=3
StartLimitIntervalSec=300

# Logging
StandardOutput=append:/var/log/llm/vllm/stdout.log
StandardError=append:/var/log/llm/vllm/stderr.log

# Resource limits
LimitNOFILE=1048576
LimitNPROC=4096

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/lib/models /var/log/llm

[Install]
WantedBy=multi-user.target
```

### 3. Sparkstation Supervisor Service

Use existing `/opt/sparkstation/scripts/systemd/sparkstation-supervisor.service`

Update paths if needed:
```ini
WorkingDirectory=/opt/sparkstation
ExecStart=/opt/sparkstation/.venv/bin/uvicorn supervisor.main:app \
  --host 127.0.0.1 \
  --port 9001
```

### 4. LiteLLM Gateway Service

`/etc/systemd/system/litellm-gateway.service`

```ini
[Unit]
Description=LiteLLM Gateway
After=network-online.target sglang-vision.service vllm-text.service
Wants=sglang-vision.service vllm-text.service

[Service]
EnvironmentFile=/etc/sparkstation/env
User=llm
Group=llm
WorkingDirectory=/etc/sparkstation

# Launch LiteLLM
ExecStart=/usr/local/bin/litellm \
  --port 8000 \
  --config /etc/sparkstation/litellm.yaml \
  --host 0.0.0.0

# Restart policy
Restart=always
RestartSec=3
StartLimitBurst=5
StartLimitIntervalSec=300

# Logging
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

---

## Deployment Steps

### Step 1: Create System Users

```bash
# Create llm user for backend services
sudo useradd -r -s /bin/false llm

# Create directories and set permissions
sudo mkdir -p /var/lib/models /var/log/llm/{sglang,vllm} /var/log/sparkstation
sudo chown -R llm:llm /var/lib/models /var/log/llm
sudo chown -R $USER:$USER /var/log/sparkstation
```

### Step 2: Install Backend Environments

Follow [Backend Environment Setup](#backend-environment-setup) section above.

### Step 3: Deploy Sparkstation

```bash
# Clone/copy Sparkstation to /opt
sudo mkdir -p /opt/sparkstation
sudo chown $USER:$USER /opt/sparkstation
cd /opt/sparkstation

# Install dependencies with uv
uv sync

# Create data directory
mkdir -p data

# Copy and configure environment
cp .env.example .env
# Edit .env with production values
```

### Step 4: Install Configuration Files

```bash
# Create config directory
sudo mkdir -p /etc/sparkstation

# Copy central env file
sudo cp /path/to/env /etc/sparkstation/env

# Copy LiteLLM config
sudo cp gateway/litellm.yaml /etc/sparkstation/litellm.yaml
```

### Step 5: Install Systemd Units

```bash
# Copy service files
sudo cp scripts/systemd/sparkstation-supervisor.service /etc/systemd/system/
sudo cp /path/to/sglang-vision.service /etc/systemd/system/
sudo cp /path/to/vllm-text.service /etc/systemd/system/
sudo cp /path/to/litellm-gateway.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload
```

### Step 6: Start Services

```bash
# Start backend services first
sudo systemctl enable sglang-vision vllm-text
sudo systemctl start sglang-vision vllm-text

# Wait for backends to be ready (check logs)
sudo journalctl -u sglang-vision -f &
sudo journalctl -u vllm-text -f &

# Start Sparkstation supervisor
sudo systemctl enable sparkstation-supervisor
sudo systemctl start sparkstation-supervisor

# Start LiteLLM gateway
sudo systemctl enable litellm-gateway
sudo systemctl start litellm-gateway

# Check status
sudo systemctl status sglang-vision vllm-text sparkstation-supervisor litellm-gateway
```

### Step 7: Verify Deployment

```bash
# Check Sparkstation supervisor
curl http://localhost:9001/health

# Check LiteLLM gateway
curl http://localhost:8000/health

# Check SGLang backend
curl http://localhost:8011/v1/models

# Check vLLM backend
curl http://localhost:8021/v1/models

# List models in Sparkstation
curl http://localhost:9001/models/detailed

# Test chat completion through gateway
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

---

## GPU Partitioning & Performance

### Single GPU Configuration (Default)

For DGX Spark with single Blackwell GPU:

```bash
# In /etc/sparkstation/env
CUDA_VISIBLE_DEVICES=0
```

All backends share the single GPU. Sparkstation's resource manager ensures:
- Max 3 concurrent models (configurable)
- Memory limit: 110 GB (85% of 128 GB unified memory)
- Auto-suspend idle models to free GPU

### Performance Tuning

**PyTorch CUDA Allocator**:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
- Reduces memory fragmentation
- Better for dynamic workloads

**Threading**:
```bash
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```
- Limit CPU threads to avoid oversubscription
- Adjust based on Grace CPU cores

**Model-Specific**:
- Use `fp8` quantization for large models (7B+)
- Set `--max-model-len` based on use case (32K is good default)
- Use `--gpu-memory-utilization 0.90-0.92` (leave headroom)
- Enable `--trust-remote-code` for community models

### Speculative Decoding (Future)

If you enable speculative decoding later:
- Do it inside vLLM: `--speculative-model <draft-model>`
- Or SGLang native speculative decoding
- Keep feature pinned per env to avoid dependency conflicts

---

## Health Checks & Monitoring

### HTTP Health Probes

**vLLM**:
```bash
# Models endpoint
curl http://localhost:8021/v1/models

# Health endpoint (if available)
curl http://localhost:8021/health
```

**SGLang**:
```bash
# Root endpoint
curl http://localhost:8011/

# Models endpoint
curl http://localhost:8011/v1/models
```

### Systemd Watchdog (Optional)

Add to service units:
```ini
[Service]
WatchdogSec=30
NotifyAccess=main
```

Create a simple health checker script that calls `sd_notify` if checks pass.

### Prometheus Metrics

Sparkstation exposes metrics at:
```bash
curl http://localhost:9001/metrics
```

Add to Prometheus config:
```yaml
scrape_configs:
  - job_name: 'sparkstation'
    static_configs:
      - targets: ['localhost:9001']
    scrape_interval: 15s
```

### Grafana Dashboard

Import the pre-built dashboard:
```bash
# Dashboard at monitoring/grafana-dashboard.json
# Import in Grafana UI → Dashboards → Import
```

Monitors:
- Unified memory usage
- GPU temperature and power
- Model status distribution
- Request rates and latency
- Auto-suspend events

---

## Logging & Retention

### Log Locations

**Sparkstation**:
- `/var/log/sparkstation/sparkstation.log` (rotating, 10 MB × 5 files)

**SGLang**:
- `/var/log/llm/sglang/stdout.log`
- `/var/log/llm/sglang/stderr.log`

**vLLM**:
- `/var/log/llm/vllm/stdout.log`
- `/var/log/llm/vllm/stderr.log`

**Systemd Journal**:
```bash
# View logs
sudo journalctl -u sglang-vision -f
sudo journalctl -u vllm-text -f
sudo journalctl -u sparkstation-supervisor -f
sudo journalctl -u litellm-gateway -f
```

### Log Rotation

Create `/etc/logrotate.d/llm`:

```
/var/log/llm/*/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 0640 llm llm
    sharedscripts
    postrotate
        systemctl reload sglang-vision vllm-text > /dev/null 2>&1 || true
    endscript
}

/var/log/sparkstation/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $USER $USER
}
```

Apply:
```bash
sudo logrotate -f /etc/logrotate.d/llm
```

---

## Maintenance

### Daily Maintenance

Sparkstation includes automated maintenance (runs at 3 AM):

```bash
# Enable maintenance timer
sudo systemctl enable sparkstation-maintenance.timer
sudo systemctl start sparkstation-maintenance.timer

# Check timer status
sudo systemctl list-timers sparkstation-maintenance.timer

# Manual run
python /opt/sparkstation/scripts/maintenance.py

# Dry run
python /opt/sparkstation/scripts/maintenance.py --dry-run
```

Maintenance tasks:
- Clean old log files (>30 days)
- Vacuum SQLite database
- Detect stale models
- Check for port leaks
- Generate resource usage snapshot

### Backend Updates

When updating backends:

```bash
# Stop services
sudo systemctl stop sglang-vision vllm-text

# Update environment
micromamba run -p /opt/backends/sglang pip install --upgrade "sglang[all]==0.3.5"
micromamba run -p /opt/backends/vllm pip install --upgrade "vllm==0.6.5"

# Test in development first!
# Then restart services
sudo systemctl start sglang-vision vllm-text
```

### Model Cache Cleanup

```bash
# Check cache size
du -sh /var/lib/models

# Remove unused models
cd /var/lib/models
# Manually remove model directories you no longer need
rm -rf models--old-model-name

# Or use HuggingFace CLI
pip install huggingface-hub
huggingface-cli scan-cache
huggingface-cli delete-cache  # interactive cleanup
```

### Monitoring Disk Usage

```bash
# Check key directories
df -h /opt/backends /var/lib/models /var/log

# Set up alerts in Grafana for:
# - /var/lib/models > 80% full
# - /var/log > 80% full
```

---

## Troubleshooting

### Backend Won't Start

```bash
# Check logs
sudo journalctl -u sglang-vision -n 100
tail -100 /var/log/llm/sglang/stderr.log

# Common issues:
# - CUDA version mismatch → reinstall with correct wheel
# - Missing model weights → check HF_HOME and download
# - Port already in use → check port allocation
# - OOM → reduce gpu-memory-utilization or model size
```

### Sparkstation Can't Connect to Backend

```bash
# Check backend is running
curl http://localhost:8011/v1/models

# Check firewall (if enabled)
sudo ufw status
sudo ufw allow 8011/tcp

# Check Sparkstation config
cat /opt/sparkstation/.env | grep PORT
```

### Model Loading Slow

```bash
# Pre-download models
export HF_HOME=/var/lib/models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Check network speed to HuggingFace
wget -O /dev/null https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/config.json
```

### High GPU Temperature

```bash
# Check temperature
nvidia-smi

# Sparkstation auto-suspends at 80°C (sustained)
# Verify auto-suspend is working
curl http://localhost:9001/models/detailed | jq '.models[] | {id, status, idle_seconds}'

# Reduce concurrent models
# Edit /opt/sparkstation/.env: MAX_RESIDENT_MODELS=2
```

---

## Security Considerations

### Network Binding

- **Backends**: Bind to `0.0.0.0` (accessible from localhost only via firewall)
- **Sparkstation**: Bind to `127.0.0.1` (localhost only)
- **LiteLLM**: Bind to `0.0.0.0` (or `127.0.0.1` if accessed locally)

### Firewall Rules

```bash
# Allow only from localhost
sudo ufw default deny incoming
sudo ufw allow from 127.0.0.1 to any port 8000:9001
sudo ufw enable
```

### API Key Authentication

Set strong API key in Sparkstation:
```bash
# Generate random key
openssl rand -hex 32

# Add to /opt/sparkstation/.env
API_KEY=<generated-key>
```

Use key in requests:
```bash
curl -H "X-API-Key: <your-key>" http://localhost:9001/models/start ...
```

---

## Next Steps After Deployment

1. **Test End-to-End**:
   - Start a model via Sparkstation API
   - Send requests through LiteLLM gateway
   - Verify auto-suspend/resume cycle

2. **Migrate Applications**:
   - Update Kavi to use `http://localhost:8000` (LiteLLM)
   - Update image_metadata_indexing to use Sparkstation

3. **Setup Monitoring**:
   - Configure Prometheus scraping
   - Import Grafana dashboard
   - Set up alerts for high temp, OOM, failures

4. **Load Testing**:
   - Use Locust or similar to test concurrent requests
   - Verify performance targets (<5s response time)
   - Test auto-suspend under load

5. **Documentation**:
   - Document your specific model choices
   - Create runbooks for common operations
   - Document application migration steps

---

## Reference

- **Sparkstation GitHub**: https://github.com/kshetrajna12/sparkstation
- **SGLang Docs**: https://sglang.readthedocs.io/
- **vLLM Docs**: https://docs.vllm.ai/
- **LiteLLM Docs**: https://docs.litellm.ai/
- **DGX Spark Info**: NVIDIA documentation

---

**Last Updated**: 2025-10-31
**Version**: 0.1.0
**Maintainer**: Sparkstation Team
