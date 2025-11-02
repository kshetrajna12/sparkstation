# Sparkstation

LLM orchestration and gateway service for DGX Spark — manages vLLM and SGLang backends with Docker for seamless model serving under an OpenAI-compatible API.

**Version**: 0.1.0 (Alpha)
**Platform**: NVIDIA DGX Spark (Grace Blackwell)
**Purpose**: Production-ready LLM gateway for Kavi and image_metadata_indexing projects

---

## Features

- **vLLM & SGLang backends**: NVIDIA-optimized backends with official Blackwell support via Docker
- **OpenAI-compatible API**: Drop-in replacement via LiteLLM gateway (chat + embeddings)
- **Embeddings support**: Text (bge-large) and image (CLIP) embeddings for RAG and search
- **Auto-suspend/resume**: Idle models auto-suspend to free GPU resources (~15s resume time)
- **Health monitoring**: Periodic 1-token probes detect unresponsive models
- **Auto-restart**: Failed models automatically restart with exponential backoff
- **DGX Spark optimized**: Unified memory tracking, thermal management, quantization
- **Dynamic model management**: Start/stop/suspend/resume models on-demand
- **Resource-aware**: Prevents OOM with hard limits and conservative allocation
- **Thermal protection**: Auto-suspend on sustained high temps with hysteresis
- **API key authentication**: Secure model management endpoints with X-API-Key headers
- **Prometheus metrics**: Track memory, temperature, power, and request metrics
- **File + stdout logging**: Rotating log files with configurable retention

---

## Architecture

```
Client Apps (Kavi, image_metadata_indexing)
           ↓
LiteLLM Gateway (Port 8000) ← OpenAI-compatible API
           ↓
Supervisor (Port 9001) ← Model lifecycle management
           ↓
Model Backends (vLLM, SGLang, TRT-LLM)
```

### Components

1. **Supervisor** (Port 9001): FastAPI service managing model lifecycle
2. **LiteLLM Gateway** (Port 8000): OpenAI-compatible routing layer
3. **Model Backends**: vLLM/SGLang/TRT-LLM servers on ports 8001-8100

---

## Quick Start

### For New Users (Automated Setup)

**One-command setup:**

```bash
# Clone repository
git clone https://github.com/kshetrajna12/sparkstation.git
cd sparkstation

# 1. Install Sparkstation dependencies
uv sync

# 2. Set up backend Docker images (SGLang)
./scripts/setup_backends.sh

# 3. Verify installation
./scripts/verify_backends.sh
```

**What this does:**
- Installs Sparkstation with uv (lightweight)
- Pulls vLLM and SGLang Docker images with full CUDA and Blackwell support
- Auto-detects ARM64 (DGX Spark) vs x86_64 architecture
- Configures `.env` for Docker mode
- Verifies CUDA is working in Docker containers

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.0+ driver
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [uv](https://github.com/astral-sh/uv) package manager

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install Docker & NVIDIA Container Toolkit:**
```bash
# Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Advanced: Subprocess Mode (Conda/Micromamba)

If you prefer isolated conda/micromamba environment instead of Docker:

```bash
# 1. Install Sparkstation
uv sync
cp .env.example .env
mkdir -p data

# 2. Set up vLLM backend (separate conda environment)
./scripts/setup_vllm_env.sh ./backends/vllm

# 3. Update .env for subprocess mode
echo "USE_DOCKER=false" >> .env
echo "VLLM_PYTHON_PATH=./backends/vllm/bin/python" >> .env

# 4. Verify
./scripts/verify_backends.sh
```

**Note**: Docker mode is strongly recommended for production, especially on Blackwell GPUs. Subprocess mode is only for specific use cases where Docker is not available.

### Start Services

```bash
# Terminal 1: Start Sparkstation supervisor
./scripts/start_supervisor.sh

# Terminal 2: Start LiteLLM gateway
./scripts/start_gateway.sh
```

Or directly with uv:
```bash
# Supervisor
uv run uvicorn supervisor.main:app --host 127.0.0.1 --port 9001

# Gateway
uv run litellm --config gateway/litellm.yaml --host 127.0.0.1 --port 8000
```

---

## CLI Usage

Sparkstation includes a unified CLI for easier management:

### Start/Stop Sparkstation

```bash
# Start supervisor in background
uv run python cli.py start -d

# Check status
uv run python cli.py status

# Stop supervisor
uv run python cli.py stop

# Restart supervisor
uv run python cli.py restart
```

### Manage Models

```bash
# List all models
uv run python cli.py models list

# Stop a model
uv run python cli.py models stop <model-id>

# View model logs (streaming)
uv run python cli.py models logs -f <model-id>
```

### Cleanup Database Issues

If you encounter database inconsistencies (stale entries, orphaned containers):

```bash
# Clean database and containers
uv run python cli.py cleanup --force
```

This will:
- Stop the supervisor
- Remove all stopped containers
- Delete the database (fresh start)
- Clean up orphaned entries

**Note**: Sparkstation now auto-reconciles database state on startup, so manual cleanup should rarely be needed.

---

## API Usage

### Start a Model

```bash
curl -X POST http://localhost:9001/models/start \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "model_name": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    "backend": "vllm",
    "model_alias": "qwen-vl-3b",
    "quantization": "awq",
    "idle_timeout_minutes": 30,
    "auto_suspend_enabled": true
  }'
```

**Note**: If `API_KEY` is set in `.env`, you must include the `X-API-Key` header. If not set, authentication is disabled (backwards compatible).

### List Running Models

```bash
# Simple format (for LiteLLM)
curl http://localhost:9001/models

# Detailed format (for monitoring)
curl http://localhost:9001/models/detailed
```

### Chat Completion via Gateway

```bash
# Using qwen-vl-3b model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-3b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'

# Using gpt-oss-20b model (reasoning model)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Explain why the sky is blue"}
    ],
    "max_tokens": 200
  }'
```

**Note**: The gateway requires an Authorization header. Use any dummy bearer token (e.g., `sk-1234`) for local development. For production, configure proper API keys in `gateway/litellm.yaml`.

### Embeddings

Sparkstation supports both text and image embeddings via OpenAI-compatible `/v1/embeddings` endpoint:

```bash
# Text embeddings (bge-large)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-large",
    "input": "The quick brown fox jumps over the lazy dog"
  }'

# Image embeddings (CLIP) - with URL
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clip-vit",
    "input": "https://example.com/image.jpg"
  }'

# Batch embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-large",
    "input": ["First document", "Second document", "Third document"]
  }'
```

**Supported embedding models:**
- `bge-large`: Text embeddings (1024 dimensions, vLLM) - for semantic search, RAG
- `clip-vit`: Image embeddings (768 dimensions, SGLang) - for image search, cross-modal retrieval

**Use cases:**
- Semantic search and similarity matching
- RAG (Retrieval Augmented Generation)
- Image search by text description
- Document classification

### Suspend/Resume Model

```bash
# Suspend (manual or automatic after idle timeout)
curl -X POST http://localhost:9001/models/{model_id}/suspend \
  -H "X-API-Key: your-api-key-here"

# Resume (automatic on incoming request or manual)
curl -X POST http://localhost:9001/models/{model_id}/resume \
  -H "X-API-Key: your-api-key-here"
```

### Monitor Resources

```bash
# Resource status
curl http://localhost:9001/resources

# Prometheus metrics
curl http://localhost:9001/metrics
```

---

## Configuration

### Environment Variables

Key settings in `.env`:

```bash
# Server
HOST=127.0.0.1  # Localhost only for security
PORT=9001

# DGX Spark Constraints
TOTAL_UNIFIED_MEMORY_GB=128
MEMORY_HARD_LIMIT_GB=110  # 85% of total
MAX_RESIDENT_MODELS=3

# Auto-suspend
AUTO_SUSPEND_ENABLED=true
DEFAULT_IDLE_TIMEOUT_MINUTES=30

# Health Checks (NEW)
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL_SECONDS=300  # 5 minutes
HEALTH_CHECK_MAX_FAILURES=3

# Auto-restart (NEW)
AUTO_RESTART_ENABLED=true
AUTO_RESTART_MAX_ATTEMPTS=3
AUTO_RESTART_BACKOFF_MINUTES=1,5,15  # Exponential backoff

# Thermal Management
THERMAL_SUSPEND_THRESHOLD_C=80  # Sustained high temp
THERMAL_RESUME_THRESHOLD_C=75   # Cooldown threshold

# Security (NEW)
API_KEY=your-secret-key-here  # Optional: Enable API key auth

# Logging (NEW)
LOG_TO_FILE=true
LOG_FILE_PATH=./data/sparkstation.log

# LiteLLM Gateway
LITELLM_ADMIN_URL=http://127.0.0.1:8000
LITELLM_MASTER_KEY=sk-sparkstation-admin  # Change in production!

# Database (uses SUPERVISOR_DATABASE_URL to avoid conflict with LiteLLM)
SUPERVISOR_DATABASE_URL=sqlite+aiosqlite:///./data/sparkstation.db
```

See `.env.example` for all options.

---

## DGX Spark Optimizations

### Unified Memory Tracking

DGX Spark has 128 GB unified CPU+GPU memory (not discrete VRAM). Sparkstation tracks **both** GPU and system memory to prevent OOM.

### Mandatory Quantization

All models must use fp8 or INT4 quantization to reduce memory footprint 2-4×.

### Health Monitoring & Auto-Restart

- **Periodic health checks**: 1-token chat completion probes every 5 minutes
- **Failure detection**: Marks models as FAILED after 3 consecutive failures
- **Auto-restart**: Failed models restart with exponential backoff (1 min → 5 min → 15 min)
- **Max 3 restart attempts** per model before permanent failure
- **Transparent recovery**: Models resume serving after successful restart

### Thermal Management

- Monitors GPU temperature every 60 seconds
- Auto-suspends least-used model if temp >80°C for 60s (sustained)
- Hysteresis prevents suspend/resume thrashing

### Resident Model Limits

- **Max 3 concurrent models** to prevent memory saturation
- Auto-suspend after 30 minutes idle (configurable)
- Models auto-resume on incoming requests (~15s startup)

---

## API Reference

### Supervisor Endpoints

#### `GET /health`
Health check

#### `GET /metrics`
Prometheus metrics

#### `GET /models`
List running models (LiteLLM format)

#### `GET /models/detailed`
Detailed model information

#### `POST /models/start` 🔒
Start new model server (requires API key if configured)

**Request**:
```json
{
  "model_name": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
  "backend": "vllm",
  "model_alias": "qwen-vl-3b",
  "quantization": "awq",
  "idle_timeout_minutes": 30,
  "auto_suspend_enabled": true
}
```

**Headers**: `X-API-Key: your-api-key` (if API_KEY set in .env)

#### `POST /models/{model_id}/stop` 🔒
Stop model server (requires API key if configured)

#### `POST /models/{model_id}/suspend` 🔒
Suspend model manually (requires API key if configured)

#### `POST /models/{model_id}/resume` 🔒
Resume suspended model (requires API key if configured)

#### `GET /models/{model_id}/status`
Detailed model status

#### `GET /resources`
System resource status

---

## Deployment

### Development (Subprocess)

Use the startup scripts for quick iteration:

```bash
./scripts/start_supervisor.sh
./scripts/start_gateway.sh
```

### Production (Systemd)

```bash
# Install systemd services
sudo cp scripts/systemd/sparkstation-supervisor.service /etc/systemd/system/
sudo cp scripts/systemd/sparkstation-gateway.service /etc/systemd/system/
sudo cp scripts/systemd/sparkstation-maintenance.service /etc/systemd/system/
sudo cp scripts/systemd/sparkstation-maintenance.timer /etc/systemd/system/

sudo systemctl daemon-reload

# Enable and start services
sudo systemctl enable sparkstation-supervisor
sudo systemctl start sparkstation-supervisor

sudo systemctl enable sparkstation-gateway
sudo systemctl start sparkstation-gateway

# Enable daily maintenance (runs at 3 AM)
sudo systemctl enable sparkstation-maintenance.timer
sudo systemctl start sparkstation-maintenance.timer

# Check status
sudo systemctl status sparkstation-supervisor
sudo systemctl status sparkstation-gateway
sudo systemctl list-timers sparkstation-maintenance.timer
sudo journalctl -u sparkstation-supervisor -f
```

### Maintenance

Automated daily maintenance runs at 3 AM via systemd timer:

```bash
# Manual run
python scripts/maintenance.py

# Dry run (show what would be done)
python scripts/maintenance.py --dry-run

# Verbose output
python scripts/maintenance.py --verbose

# Check last maintenance run
sudo journalctl -u sparkstation-maintenance -n 50
```

**Maintenance tasks**:
- Cleanup log files older than 30 days
- Vacuum SQLite database
- Detect stale/zombie models (STARTING >10 min)
- Check for port leaks
- Generate resource usage snapshot

---

## Monitoring

### Prometheus Metrics

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'sparkstation'
    static_configs:
      - targets: ['localhost:9001']
```

### Key Metrics

- `unified_memory_used_bytes` - Total memory usage
- `gpu_temperature_celsius` - GPU temperature
- `gpu_power_draw_watts` - Power consumption
- `model_status` - Model status (0-4)
- `resident_models_count` - Active models
- `model_requests_total` - Request counter

### Grafana Dashboard

Import the pre-built dashboard: `monitoring/grafana-dashboard.json`

**Includes**:
- Real-time memory, temperature, and power monitoring
- Model status distribution and per-model memory
- Request rate and latency metrics (p50, p95)
- Running vs suspended model counts
- Auto-refresh every 10 seconds

**Quick Import**:
1. Open Grafana → Dashboards → Import
2. Upload `monitoring/grafana-dashboard.json`
3. Select your Prometheus datasource

See [monitoring/README.md](monitoring/README.md) for detailed setup and alert configuration.

---

## Troubleshooting

### Model fails to start

Check logs:
```bash
# Systemd logs
journalctl -u sparkstation-supervisor -f

# Or file logs (if LOG_TO_FILE=true)
tail -f data/sparkstation.log
```

Common issues:
- **401 Unauthorized**: API key required - add `X-API-Key` header
- **Insufficient memory**: Check `/resources` endpoint
- **Missing quantized weights**: Use `quantization='fp16'` as fallback
- **Port already in use**: Check allocated ports in `/resources`

### Health checks failing

- Check model is actually responding: `curl http://localhost:8001/v1/models`
- Review health check logs for specific errors
- Verify `HEALTH_CHECK_TIMEOUT_SECONDS` is sufficient (default 5s)
- Check if model is overloaded with requests

### Model keeps restarting

- Review restart history in `/models/detailed` (check `restart_count`)
- Check logs for underlying failure cause
- Verify sufficient memory/resources available
- After 3 restart attempts, model marked permanently FAILED
- Manual restart: Stop model, fix issue, start fresh

### Auto-suspend not working

- Verify `AUTO_SUSPEND_ENABLED=true` in `.env`
- Check `idle_timeout_minutes` > 0 for model
- Review logs for auto-suspend manager activity
- Confirm no active requests keeping model busy

### Gateway returns 404

- Model may be suspended - first request triggers auto-resume (~15s)
- Check model status: `curl http://localhost:9001/models/detailed`
- Verify gateway sync is running
- Ensure LiteLLM gateway is started

---

## Development

### Run Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
uv run ruff check .
```

### Type Checking

```bash
uv run mypy supervisor/
```

---

## Known Issues & Technical Debt

### Gateway Database Features (Future Enhancement)

**Status**: Gateway is fully operational for model routing but lacks advanced features that require database support.

**Current State**:
- ✅ Gateway successfully routes requests to model backends
- ✅ OpenAI-compatible API working without database
- ✅ Model discovery and failover working
- ✅ Uses `SUPERVISOR_DATABASE_URL` to avoid conflicts with LiteLLM

**Missing Features** (optional future enhancements):
- Usage tracking and analytics
- API key management and authentication
- Rate limiting per user/key
- Request logging and audit trails
- Cost tracking per API key

**Future Options**:
1. Set up separate PostgreSQL database for gateway with full Prisma support
2. Use LiteLLM's built-in database features for advanced functionality
3. Alternative: Implement custom middleware for tracking/auth without database

---

## License

[Your License]

---

## Contributing

Contributions welcome! Please open an issue or PR.

---

## Support

- **Issues**: https://github.com/kshetrajna12/sparkstation/issues
- **Docs**: See `TECH_PLAN.md` for detailed architecture

---

**Built for NVIDIA DGX Spark (Grace Blackwell)**
