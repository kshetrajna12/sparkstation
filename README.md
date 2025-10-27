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
- **DGX Spark optimized**: Unified memory tracking, thermal management, quantization
- **Dynamic model management**: Start/stop/suspend/resume models on-demand
- **Resource-aware**: Prevents OOM with hard limits and conservative allocation
- **Thermal protection**: Auto-suspend on sustained high temps with hysteresis
- **Prometheus metrics**: Track memory, temperature, power, and request metrics

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
git clone https://github.com/yourusername/sparkstation.git
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
  -d '{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",
    "model_alias": "qwen3-8b",
    "quantization": "fp8",
    "idle_timeout_minutes": 30,
    "auto_suspend_enabled": true
  }'
```

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
curl -X POST http://localhost:9001/models/{model_id}/suspend

# Resume (automatic on incoming request or manual)
curl -X POST http://localhost:9001/models/{model_id}/resume
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

# Thermal Management
THERMAL_SUSPEND_THRESHOLD_C=80  # Sustained high temp
THERMAL_RESUME_THRESHOLD_C=75   # Cooldown threshold

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

#### `POST /models/start`
Start new model server

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

#### `POST /models/{model_id}/stop`
Stop model server

#### `POST /models/{model_id}/suspend`
Suspend model (manual)

#### `POST /models/{model_id}/resume`
Resume suspended model

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
# Install systemd service
sudo cp scripts/systemd/sparkstation-supervisor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sparkstation-supervisor
sudo systemctl start sparkstation-supervisor

# Check status
sudo systemctl status sparkstation-supervisor
sudo journalctl -u sparkstation-supervisor -f
```

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

See `monitoring/grafana-dashboard.json` for pre-built dashboard.

---

## Troubleshooting

### Model fails to start

Check logs:
```bash
journalctl -u sparkstation-supervisor -f
```

Common issues:
- Insufficient memory (check `/resources`)
- Missing quantized weights
- Port already in use

### Auto-suspend not working

- Verify `AUTO_SUSPEND_ENABLED=true` in `.env`
- Check `idle_timeout_minutes` > 0 for model
- Review logs for auto-suspend manager activity

### Gateway returns 404

- Model may be suspended - first request triggers auto-resume
- Check model status: `curl http://localhost:9001/models/detailed`
- Verify gateway sync is running

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

- **Issues**: https://github.com/yourusername/sparkstation/issues
- **Docs**: See `TECH_PLAN.md` for detailed architecture

---

**Built for NVIDIA DGX Spark (Grace Blackwell)**
