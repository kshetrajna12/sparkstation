# Sparkstation

Unified LLM orchestration and gateway service for DGX Spark — dynamically manages vLLM, SGLang, and TensorRT-LLM backends under a single OpenAI-compatible API.

**Version**: 0.1.0 (Alpha)
**Platform**: NVIDIA DGX Spark (Grace Blackwell)
**Purpose**: Production-ready LLM gateway for Kavi and image_metadata_indexing projects

---

## Features

- **Multi-backend support**: vLLM, SGLang, TensorRT-LLM
- **OpenAI-compatible API**: Drop-in replacement via LiteLLM gateway
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

### Prerequisites

- Python 3.11+
- NVIDIA DGX Spark (Grace Blackwell) or compatible GPU system
- CUDA 12.0+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone repository
git clone https://github.com/kshetrajna12/sparkstation.git
cd sparkstation

# Install dependencies with uv
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env to configure settings

# Ensure data directory exists
mkdir -p data
```

### Start Supervisor

```bash
# Using startup script
./scripts/start_supervisor.sh

# Or directly with uv
uv run uvicorn supervisor.main:app --host 127.0.0.1 --port 9001
```

### Start LiteLLM Gateway

```bash
# In a separate terminal
./scripts/start_gateway.sh

# Or directly with uv
uv run litellm --config gateway/litellm.yaml --host 127.0.0.1 --port 8000
```

---

## Usage

### Start a Model

```bash
curl -X POST http://localhost:9001/models/start \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",
    "model_alias": "qwen3-8b",
    "quantization": "fp8",
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
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

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
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "backend": "vllm",
  "model_alias": "qwen3-8b",
  "quantization": "fp8",
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
