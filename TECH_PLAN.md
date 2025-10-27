# Sparkstation Technical Plan

**Version**: 2.1 (Production-Ready DGX Spark Edition)
**Created**: 2025-10-26
**Last Updated**: 2025-10-26 (Critical production fixes applied)
**Status**: Planning → Implementation
**Target Platform**: NVIDIA DGX Spark (Grace Blackwell)
**Purpose**: Unified LLM Gateway for Kavi and image_metadata_indexing

**Version History**:
- v1.0: Initial plan with basic auto-suspend
- v1.1: Enhanced auto-suspend feature
- v2.0: **DGX Spark hardware-specific optimizations** (unified memory, thermal management, quantization)
- v2.1: **Critical production fixes** (LiteLLM push model, auto-resume middleware, quantization mapping, thermal hysteresis, unified memory tracking, localhost binding, version pinning)

---

## Critical Production Fixes (v2.1)

The following **MUST-FIX** issues have been addressed to prevent production failures:

### 1. ✅ LiteLLM Model Discovery (FIXED)
**Problem**: `fetch_from_url` is flaky and version-sensitive.
**Solution**: Supervisor **pushes** model list to LiteLLM admin API + fallback to YAML rewrite.
- Implemented `GatewaySync` class to push models every 60s
- Uses LiteLLM `/model/new` admin endpoint
- Fallback: Rewrites `litellm.yaml` and triggers `/config/reload`

### 2. ✅ Auto-Resume Trigger (FIXED)
**Problem**: LiteLLM doesn't know about suspended models, returns 404/ECONNREFUSED.
**Solution**: **FastAPI middleware** intercepts requests, checks model status, auto-resumes if suspended.
- `AutoResumeMiddleware` wraps LiteLLM
- Detects suspended models, POSTs `/models/{id}/resume` to Supervisor
- Waits (max 30s) for model to become ready
- Returns 503 if resume fails

### 3. ✅ Quantization Flags (FIXED)
**Problem**: `--quantization fp8` flags differ across vLLM vs SGLang.
**Solution**: Backend-specific **quantization mapping**.
- vLLM: `{"fp8": "fp8", "int4": "awq", "gptq": "gptq"}`
- SGLang: `{"fp8": "fp8", "int4": "int4", "awq": "awq"}`
- Fail fast if model doesn't have quantized weights
- Per-model `max_model_len` (not blanket 8192)

### 4. ✅ Health Probe Endpoint (FIXED)
**Problem**: Probing `/v1/completions` fails (most backends only support `/v1/chat/completions`).
**Solution**: Use `/v1/chat/completions` with minimal message.
- Probe: `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}`
- Timeout: 5 seconds
- More accurate than just `/v1/models` check

### 5. ✅ Unified Memory Tracking (FIXED)
**Problem**: `nvidia-smi` doesn't show full unified memory usage (CPU allocations invisible).
**Solution**: Track **both** GPU memory AND system memory.
- Reads GPU memory via `nvidia-smi`
- Reads system memory from `/proc/meminfo`
- Uses `MAX(gpu, system) + 16 GB buffer`
- Conservative allocator prevents OOM

### 6. ✅ Localhost Binding (FIXED)
**Problem**: Docker Compose maps ports on all interfaces (security risk).
**Solution**: Explicitly bind to `127.0.0.1` in all configs.
- Docker Compose: `ports: - "127.0.0.1:8000:8000"`
- vLLM/SGLang: `--host 127.0.0.1`
- LiteLLM: `--host 127.0.0.1`
- Systemd: `ExecStart=... --host 0.0.0.0 --port 9001` → should be localhost too

### 7. ✅ Thermal Hysteresis (FIXED)
**Problem**: Auto-suspend on 80°C causes suspend/resume thrashing.
**Solution**: **Hysteresis** with sustained high temp + cooldown.
- Suspend only if temp >80°C for **60 seconds** (sustained)
- Resume only after temp <75°C for **120 seconds** (cooldown)
- Prevents rapid cycling
- Environment variables: `THERMAL_SUSPEND_C`, `THERMAL_RESUME_C`, `THERMAL_SUSTAIN_MS`, `THERMAL_COOLDOWN_MS`

### 8. ✅ Version Pinning (ADDED)
**Problem**: vLLM/SGLang/LiteLLM update frequently, flags break.
**Solution**: `constraints.txt` with tested versions.
- Locks vLLM, SGLang, LiteLLM to known-good versions
- Quarterly review cycle
- Test updates in dev before production

### 9. ✅ Simplified /models Response (FIXED)
**Problem**: Nested JSON structure confuses LiteLLM.
**Solution**: Flat list format.
```json
[
  {"model_name": "qwen3-8b", "litellm_provider": "openai", "api_base": "http://127.0.0.1:8001/v1", "api_key": "EMPTY"}
]
```
- Separate `/models/detailed` endpoint for dashboard metadata

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Requirements Specification](#requirements-specification)
4. [Architecture Design](#architecture-design)
5. [Implementation Phases](#implementation-phases)
6. [API Specifications](#api-specifications)
7. [Deployment Strategy](#deployment-strategy)
8. [Migration Plan](#migration-plan)
9. [Testing Strategy](#testing-strategy)
10. [Performance Targets](#performance-targets)
11. [Security & Resource Management](#security--resource-management)

---

## Executive Summary

### Problem Statement
Two projects (Kavi and image_metadata_indexing) currently use or plan to use separate LLM services:
- **Kavi**: Uses Ollama (local) or Groq (cloud) for chat and vision
- **image_metadata_indexing**: Plans to use SGLang for vision models

This creates:
- Duplicate infrastructure
- Inefficient GPU utilization
- Complex configuration management
- No centralized resource control

### Solution
**Sparkstation** - A unified LLM gateway that:
1. Manages multiple model backends (vLLM, SGLang, TRT-LLM)
2. Exposes a single OpenAI-compatible API via LiteLLM
3. Dynamically starts/stops models on demand
4. Auto-suspends idle models to free GPU resources (~15s resume time)
5. Efficiently allocates GPU resources
6. Supports both text and vision models

### Success Criteria
- ✅ Both projects can migrate without code changes (OpenAI API compatibility)
- ✅ Single endpoint for all LLM operations
- ✅ GPU utilization improved (shared across models)
- ✅ Models can be started/stopped dynamically
- ✅ Auto-suspend idle models to free GPU resources
- ✅ <20 second resume time for suspended models
- ✅ <5 second response time for text models (when running)
- ✅ <5 second response time for vision models (when running)

---

## Hardware Environment: NVIDIA DGX Spark

### Platform Specifications

**Architecture**: Grace Blackwell (unified CPU+GPU design)

**Memory**:
- **128 GB LPDDR5x unified system memory** (shared by CPU + GPU)
- Bandwidth: ~273 GB/s
- **Critical**: Not discrete HBM3 - shared resource across all processes

**Compute**:
- 1× NVIDIA Blackwell GPU (GB10)
- High efficiency, lower memory bandwidth than discrete HBM3 GPUs
- Designed for sustained AI workloads

**Thermal Design**:
- Active DGX cooling system
- Designed for 24×7 operation
- Target ambient: <30°C

### Hardware Constraints & Design Impact

#### 1. Unified Memory Model
Unlike traditional discrete GPUs with dedicated VRAM, DGX Spark uses **unified memory** shared between CPU and GPU:
- **Total capacity**: 128 GB for **all** processes (OS, applications, models)
- **No separate GPU VRAM** to track
- **Memory pressure** affects both CPU and GPU performance
- **Paging risk**: Exceeding capacity causes severe slowdowns

**Design Impact**:
- Track **total unified memory**, not GPU memory
- Hard limit: **≤110 GB** (85% of 128 GB) for all models + overhead
- Refuse new model launches if memory threshold exceeded
- Aggressive memory monitoring and cleanup

#### 2. Thermal & Power Profile
- Each loaded 7B model: **40-80W continuous draw** (even when idle)
- Two 7B models: **~150-180W sustained**
- Sustained load > transient load for component longevity
- Must monitor temperatures continuously

**Design Impact**:
- Limit **resident models to 2-3 maximum**
- **Auto-suspend after 30 minutes** (not 15) to reduce thermal load
- Optional **auto-sleep**: Unload all models if system idle >1 hour
- Temperature monitoring integrated into health checks

#### 3. Memory Bandwidth
- Unified memory bandwidth (~273 GB/s) is lower than discrete HBM3
- Shared bandwidth across CPU + GPU operations

**Design Impact**:
- **Mandatory quantization** (fp8 or INT4) for all models
- Reduces memory footprint 2-4× and bandwidth pressure
- Smaller KV caches: `max_model_len=8192` (not unlimited)
- `gpu-memory-utilization=0.9` max

### Hardware-Optimized Policies

#### Resident Model Policy
| Rule | Rationale |
|------|-----------|
| **Max 2-3 resident models** | Prevent unified memory saturation |
| **Quantize all models (fp8/INT4)** | 2-4× memory reduction, lower bandwidth |
| **Auto-suspend after 30 min idle** | Free memory + reduce continuous power draw |
| **Pin 2-3 essential models** | Prevent fragmentation, guarantee readiness |
| **Daily model restart** | Clear allocator fragmentation |

#### Memory Management Rules
- Track **unified memory** via `nvidia-smi` and system tools
- **Hard limit**: 110 GB (85% of 128 GB total)
- **Soft limit**: 100 GB (78%) - warn if exceeded
- Refuse model launches beyond threshold
- Monitor fragmentation - restart servers daily/weekly

#### Thermal Management Rules
- Monitor GPU temperature every 30 seconds
- Alert if temp > 75°C
- Auto-suspend least-used model if temp > 80°C
- Ensure ambient < 30°C
- Track continuous power draw via `nvidia-smi dmon`

---

## Current State Analysis

### Project 1: Kavi (AI Assistant)

**Current Setup:**
- **Framework**: Pydantic-AI with OpenAI-compatible providers
- **Providers**: Ollama (local) + Groq (cloud)
- **Models Used**:
  - Text: `qwen3:8b` (via Ollama at `localhost:11434`)
  - Vision: `llama-4-maverick-17b-128e-instruct` (via Groq cloud)
- **Features**:
  - Conversational agent with 8 registered tools
  - Memory system (Memobase) with semantic search
  - Lightroom photo copilot (vision analysis)
  - Web search integration
- **API Format**: OpenAI-compatible `/v1/chat/completions`
- **Embedding Service**: Custom FastAPI service (port 8020)
  - Model: `all-MiniLM-L6-v2` (384 dimensions)
  - Endpoint: `/v1/embeddings`

**Configuration:**
```bash
# Current Kavi .env
LLM_PROVIDER=ollama  # or groq
OLLAMA_MODEL=qwen3:8b
GROQ_MODEL=qwen/qwen3-32b
GROQ_API_KEY=<key>
VISION_PROVIDER=groq
VISION_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
```

**Key Requirements:**
- OpenAI-compatible chat completions
- Vision model support (multimodal messages)
- Fast inference (prefers Groq speed over Ollama)
- Function/tool calling support
- 4K+ context window

### Project 2: image_metadata_indexing (Wildlife Photo Indexing)

**Current Setup:**
- **Status**: Planning phase (no implementation yet)
- **Planned Provider**: SGLang
- **Models Planned**:
  - Vision: `Qwen/Qwen2.5-VL-7B-Instruct` (via SGLang at `localhost:30000`)
- **Features**:
  - Analyze wildlife photographs
  - Generate detailed scene descriptions
  - Species and behavior detection
  - Burst photo optimization (shared descriptions)
- **API Format**: OpenAI-compatible `/v1/chat/completions`
- **Embedding Models**: Dual strategy
  - Fast: `all-MiniLM-L6-v2` (384 dim)
  - Quality: `multi-qa-mpnet-base-dot-v1` (768 dim)

**Configuration:**
```bash
# Planned image_metadata_indexing .env
LLM_PROVIDER=sglang
SGLANG_API_BASE=http://localhost:30000/v1
SGLANG_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
EMBEDDING_MODEL_FAST=all-MiniLM-L6-v2
EMBEDDING_MODEL_QUALITY=multi-qa-mpnet-base-dot-v1
```

**Key Requirements:**
- Multimodal vision-language model
- Process JPEG previews (500-2000px)
- ~3.5 second per-image latency target
- Batch processing support (10-20 images)
- Async concurrent requests
- 2000+ token context for detailed descriptions

---

## Requirements Specification

### Functional Requirements

#### FR-1: Model Backend Management
- **FR-1.1**: Start model servers dynamically (vLLM, SGLang, TRT-LLM)
- **FR-1.2**: Stop model servers and release resources
- **FR-1.3**: List all running models with status
- **FR-1.4**: Health check for each model server
- **FR-1.5**: Restart failed model servers
- **FR-1.6**: Auto-suspend idle models after configurable timeout
- **FR-1.7**: Auto-resume suspended models on incoming requests
- **FR-1.8**: Track last request time for each model

#### FR-2: OpenAI-Compatible API
- **FR-2.1**: `/v1/chat/completions` endpoint
- **FR-2.2**: Support for multimodal messages (text + images)
- **FR-2.3**: Streaming and non-streaming responses
- **FR-2.4**: Model selection via `model` parameter
- **FR-2.5**: Function/tool calling support
- **FR-2.6**: `/v1/models` endpoint (list available models)

#### FR-3: Resource Management (DGX Spark-Optimized)
- **FR-3.1**: **Unified memory tracking** (128 GB total, not GPU VRAM)
- **FR-3.2**: **Memory limits**: Hard 110 GB (85%), Soft 100 GB (78%)
- **FR-3.3**: **Resident model limit**: Maximum 2-3 models simultaneously
- **FR-3.4**: Port allocation and conflict prevention
- **FR-3.5**: **Thermal monitoring**: GPU temperature tracking
- **FR-3.6**: **Power monitoring**: Track continuous draw via nvidia-smi
- **FR-3.7**: Automatic resource cleanup on model stop
- **FR-3.8**: **Refuse launches** if memory threshold would be exceeded

#### FR-4: Model Types Support (DGX Spark-Optimized)
- **FR-4.1**: Text-only models (Qwen 3, Deepseek Coder) - **fp8 quantized**
- **FR-4.2**: Vision models (Qwen VL, Llama Vision) - **fp8 quantized**
- **FR-4.3**: **Mandatory quantization**: All models must use fp8 or INT4
- **FR-4.4**: **KV cache limits**: `max_model_len=8192`, `gpu-memory-utilization=0.9`
- **FR-4.5**: Embedding models (optional, separate service)

### Non-Functional Requirements

#### NFR-1: Performance
- **NFR-1.1**: Text model latency: <5 seconds (95th percentile)
- **NFR-1.2**: Vision model latency: <5 seconds (95th percentile)
- **NFR-1.3**: Support 10+ concurrent requests
- **NFR-1.4**: Model startup time: <60 seconds

#### NFR-2: Reliability
- **NFR-2.1**: 99% uptime for gateway
- **NFR-2.2**: Automatic health checks every 30 seconds
- **NFR-2.3**: Graceful degradation (failed models don't crash gateway)
- **NFR-2.4**: Retry logic for transient failures

#### NFR-3: Compatibility
- **NFR-3.1**: 100% OpenAI API compatibility for existing clients
- **NFR-3.2**: No code changes required in Kavi or image_metadata_indexing
- **NFR-3.3**: Support Python 3.11+

#### NFR-4: Observability (DGX Spark-Optimized)
- **NFR-4.1**: Structured logging (JSON format, journald or Docker)
- **NFR-4.2**: **Prometheus `/metrics` endpoint** exposing:
  - `model_memory_used_bytes{model_name}` - Per-model memory usage
  - `model_last_request_timestamp{model_name}` - Last request time
  - `unified_memory_used_bytes` - Total unified memory usage
  - `unified_memory_limit_bytes` - Memory hard limit (110 GB)
  - `gpu_temperature_celsius` - GPU temperature
  - `gpu_power_draw_watts` - Continuous power draw
  - `model_requests_total{model_name}` - Request counter
  - `model_requests_failed{model_name}` - Failed requests
  - `model_status{model_name, status}` - Model status gauge
- **NFR-4.3**: Request tracing (correlation IDs)
- **NFR-4.4**: **1-token health probes** every 5 minutes (verify responsiveness)
- **NFR-4.5**: Grafana dashboard for memory, temp, power, idle time

#### NFR-5: Security & Safety (DGX Spark-Optimized)
- **NFR-5.1**: **All ports bind to 127.0.0.1** (localhost only)
- **NFR-5.2**: **API key authentication** for Supervisor and Gateway
- **NFR-5.3**: **Non-root execution** for all processes
- **NFR-5.4**: Shared-secret header for Supervisor API
- **NFR-5.5**: TLS terminator (Caddy/Nginx) if remote access required
- **NFR-5.6**: No public exposure without explicit configuration

#### NFR-6: Maintenance & Longevity
- **NFR-6.1**: **Daily maintenance routine** (nightly cron):
  - Unload non-pinned models
  - Restart model servers (clear memory fragmentation)
  - Compact registry database
  - Log memory/thermal stats
- **NFR-6.2**: **Graceful degradation**: Component failures don't crash system
- **NFR-6.3**: **Auto-restart**: Systemd manages process lifecycle
- **NFR-6.4**: **Optional auto-sleep**: Unload all models if idle >1 hour

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Applications                    │
│         (Kavi, image_metadata_indexing)                 │
└─────────────────┬───────────────────────────────────────┘
                  │ OpenAI-compatible API
                  │ http://localhost:8000/v1/*
                  ▼
┌─────────────────────────────────────────────────────────┐
│               LiteLLM Gateway (Port 8000)               │
│  • Route requests by model name                          │
│  • Stream handling                                       │
│  • fetch_from_url: http://localhost:9001/models         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          Model Supervisor (FastAPI, Port 9001)          │
│  • Manage model lifecycle                               │
│  • Track GPU/port allocation                            │
│  • Health checks                                         │
│  • Auto-suspend idle models                             │
│  • Track last request time                              │
│  • Serve /models endpoint                               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ Launches/Manages
                  ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Backends                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  vLLM        │  │  SGLang      │  │  TRT-LLM     │  │
│  │  Port 8001   │  │  Port 8002   │  │  Port 8003   │  │
│  │  GPU 0       │  │  GPU 1       │  │  GPU 0       │  │
│  │              │  │              │  │              │  │
│  │ qwen3:8b     │  │ Qwen VL 7B   │  │ Deepseek     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Auto-Resume Middleware (Port 8000)
**Technology**: FastAPI middleware (sits in front of LiteLLM)
**Purpose**: Auto-resume suspended models on incoming requests

**Critical Fix**: LiteLLM doesn't know about suspended models. Need middleware to:
1. Intercept requests for suspended models
2. POST `/models/{id}/resume` to Supervisor
3. Wait (bounded 30s) for model to become ready
4. Forward request to LiteLLM

**Implementation** (`gateway/auto_resume_middleware.py`):
```python
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import asyncio

class AutoResumeMiddleware:
    """
    Middleware to auto-resume suspended models before forwarding to LiteLLM.
    CRITICAL: Without this, requests to suspended models will 404/ECONNREFUSED.
    """

    def __init__(self, app: FastAPI, supervisor_url: str, litellm_url: str):
        self.app = app
        self.supervisor_url = supervisor_url  # http://localhost:9001
        self.litellm_url = litellm_url  # http://localhost:8001
        self.client = httpx.AsyncClient(timeout=60.0)

    async def __call__(self, request: Request, call_next):
        # Only intercept /v1/* endpoints
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        # Extract model name from request body
        if request.method == "POST":
            body = await request.body()
            try:
                data = json.loads(body)
                model_name = data.get("model")

                if model_name:
                    # Check if model is suspended
                    status = await self._check_model_status(model_name)

                    if status == "suspended":
                        logger.info(f"Model {model_name} is suspended, auto-resuming...")

                        # Resume model
                        resumed = await self._resume_model(model_name)

                        if not resumed:
                            return Response(
                                content=json.dumps({"error": f"Failed to resume model {model_name}"}),
                                status_code=503,
                                media_type="application/json"
                            )

                        logger.info(f"Model {model_name} resumed successfully")

                # Reconstruct request with body
                request._body = body

            except Exception as e:
                logger.error(f"Auto-resume middleware error: {e}")

        # Forward to LiteLLM
        return await call_next(request)

    async def _check_model_status(self, model_name: str) -> str:
        """Check if model is suspended"""
        try:
            response = await self.client.get(f"{self.supervisor_url}/models")
            models = response.json().get("models", [])

            for model in models:
                if model.get("model_name") == model_name or model.get("alias") == model_name:
                    return model.get("status", "unknown")

            return "not_found"

        except Exception as e:
            logger.error(f"Failed to check model status: {e}")
            return "unknown"

    async def _resume_model(self, model_name: str) -> bool:
        """Resume a suspended model, wait for it to be ready"""
        try:
            # Find model ID
            response = await self.client.get(f"{self.supervisor_url}/models")
            models = response.json().get("models", [])

            model_id = None
            for model in models:
                if model.get("model_name") == model_name or model.get("alias") == model_name:
                    model_id = model.get("id")
                    break

            if not model_id:
                logger.error(f"Model {model_name} not found")
                return False

            # Resume model
            response = await self.client.post(
                f"{self.supervisor_url}/models/{model_id}/resume"
            )

            if response.status_code != 200:
                logger.error(f"Failed to resume model: {response.text}")
                return False

            # Wait for model to be ready (max 30 seconds)
            for attempt in range(30):
                await asyncio.sleep(1)

                status_response = await self.client.get(
                    f"{self.supervisor_url}/models/{model_id}/status"
                )
                status = status_response.json().get("status")

                if status == "running":
                    # Double-check with health probe
                    if await self._health_check(model_id):
                        return True

            logger.error(f"Model {model_name} did not become ready within 30s")
            return False

        except Exception as e:
            logger.error(f"Failed to resume model: {e}")
            return False

    async def _health_check(self, model_id: str) -> bool:
        """Quick health check to verify model is responsive"""
        try:
            status_response = await self.client.get(
                f"{self.supervisor_url}/models/{model_id}/status"
            )
            health = status_response.json().get("health_status")
            return health == "healthy"
        except:
            return False


# Gateway main.py
app = FastAPI()

# Add auto-resume middleware BEFORE LiteLLM
app.add_middleware(
    AutoResumeMiddleware,
    supervisor_url="http://127.0.0.1:9001",
    litellm_url="http://127.0.0.1:8001"
)

# Mount LiteLLM as sub-application
# (LiteLLM runs on internal port 8001, middleware proxies to it)
```

#### 2. LiteLLM Gateway (Port 8001, internal)
**Technology**: LiteLLM proxy
**Purpose**: Route requests to model backends

**Responsibilities**:
- Accept OpenAI-compatible requests (via middleware)
- Route to appropriate backend based on `model` parameter
- Handle streaming/non-streaming responses
- Receive model list from Supervisor via admin API

**Configuration** (`gateway/litellm.yaml`):
```yaml
# CRITICAL: fetch_from_url is flaky and version-sensitive
# Instead: Supervisor pushes model list via LiteLLM admin API
# This file serves as initial bootstrap only

model_list: []  # Populated dynamically by Supervisor via admin API

router_settings:
  routing_strategy: simple-shuffle  # Load balance if multiple backends
  allowed_fails: 3
  cooldown_time: 30

litellm_settings:
  drop_params: true  # Drop unsupported params
  set_verbose: true
  success_callback: ["prometheus"]
  failure_callback: ["prometheus"]

general_settings:
  master_key: "${LITELLM_MASTER_KEY}"  # API key for admin operations
```

**Dynamic Model Registration** (`supervisor/gateway_sync.py`):
```python
class GatewaySync:
    """Push model list to LiteLLM via admin API (more reliable than fetch_from_url)"""

    def __init__(self, litellm_admin_url: str, master_key: str):
        self.admin_url = litellm_admin_url  # http://localhost:8000
        self.master_key = master_key
        self.client = httpx.AsyncClient()

    async def sync_models(self, models: List[ModelInstance]):
        """Push current model list to LiteLLM"""
        model_list = []
        for model in models:
            if model.status == "running":  # Only include running models
                model_list.append({
                    "model_name": model.model_alias or model.model_name.split("/")[-1],
                    "litellm_params": {
                        "model": f"openai/{model.model_alias}",  # openai/ prefix for OpenAI-compatible
                        "api_base": f"{model.base_url}",
                        "api_key": "EMPTY"
                    }
                })

        # Push to LiteLLM admin API
        try:
            response = await self.client.post(
                f"{self.admin_url}/model/new",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={"models": model_list}
            )
            if response.status_code == 200:
                logger.info(f"Synced {len(model_list)} models to LiteLLM gateway")
            else:
                logger.error(f"Failed to sync models: {response.text}")
        except Exception as e:
            logger.error(f"Gateway sync error: {e}")
            # Fallback: rewrite litellm.yaml and reload
            await self._fallback_yaml_reload(model_list)

    async def _fallback_yaml_reload(self, model_list: List[dict]):
        """Fallback: rewrite litellm.yaml and trigger reload"""
        config_path = Path("gateway/litellm.yaml")
        config = yaml.safe_load(config_path.read_text())
        config["model_list"] = model_list
        config_path.write_text(yaml.dump(config))

        # Trigger reload via admin API
        await self.client.post(
            f"{self.admin_url}/config/reload",
            headers={"Authorization": f"Bearer {self.master_key}"}
        )
        logger.info("Fallback: rewrote litellm.yaml and triggered reload")

    async def background_sync(self):
        """Background task: sync every 60 seconds"""
        while True:
            await asyncio.sleep(60)
            models = self.registry.list_all()
            await self.sync_models(models)
```

#### 2. Model Supervisor (Port 9001)
**Technology**: FastAPI
**Purpose**: Manage model server lifecycle

**Responsibilities**:
- Start/stop model servers (vLLM, SGLang, TRT-LLM)
- Allocate GPUs and ports safely
- Track running models (in-memory + SQLite persistence)
- Health check model servers periodically
- Serve `/models` endpoint in LiteLLM format

**Data Model** (`supervisor/registry.py`):
```python
@dataclass
class ModelInstance:
    model_id: str              # Unique identifier
    model_name: str            # HF model name (e.g., "Qwen/Qwen2.5-VL-7B")
    backend: str               # vllm, sglang, trt-llm
    port: int                  # Assigned port
    gpu_ids: List[int]         # GPUs allocated [0, 1]
    status: str                # starting, running, suspended, stopped, failed
    base_url: str              # http://localhost:8001
    pid: Optional[int]         # Process ID
    started_at: datetime
    last_health_check: Optional[datetime]
    health_status: str         # healthy, unhealthy, unknown

    # Auto-suspend tracking
    last_request_time: Optional[datetime]  # Last time model served a request
    idle_timeout_minutes: int              # Minutes before auto-suspend (0 = disabled)
    auto_suspend_enabled: bool             # Can this model be auto-suspended?
    saved_config: Optional[dict]           # Config to restart suspended model
```

**API Endpoints**:
```python
GET  /models                      # List models (LiteLLM format)
POST /models/start                # Start a model server
POST /models/{model_id}/stop      # Stop a model server
POST /models/{model_id}/suspend   # Manually suspend a model
POST /models/{model_id}/resume    # Resume a suspended model
GET  /models/{model_id}/status    # Get model status
PATCH /models/{model_id}/config   # Update model config (e.g., idle timeout)
GET  /health                      # Supervisor health
GET  /resources                   # GPU/port availability
```

**Resource Tracking** (`supervisor/resources.py` - DGX Spark-Optimized):
```python
class ResourceManager:
    """DGX Spark unified memory resource manager"""

    def __init__(self):
        # DGX Spark: 1 GPU with unified 128 GB memory
        self.total_unified_memory_gb = 128
        self.hard_limit_gb = 110  # 85% of total
        self.soft_limit_gb = 100  # 78% of total
        self.max_resident_models = 3

        self.allocated_ports = {}              # model_id -> port
        self.port_range = (8001, 8100)

        self.model_memory_usage = {}           # model_id -> estimated_gb

    def get_unified_memory_usage(self) -> float:
        """
        Get current unified memory usage in GB.
        CRITICAL: DGX Spark has unified memory shared by CPU+GPU.
        nvidia-smi only shows GPU-visible portion, not total system usage.
        Must track both GPU memory AND system memory.
        """
        # Get GPU-reported memory usage
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        gpu_memory_mb = float(gpu_result.stdout.strip())

        # Get system memory usage from /proc/meminfo
        # MemTotal - MemAvailable = used memory
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                key, value = line.split(":")
                meminfo[key.strip()] = int(value.strip().split()[0])  # Value in kB

        mem_total_kb = meminfo.get("MemTotal", 0)
        mem_available_kb = meminfo.get("MemAvailable", 0)
        system_used_mb = (mem_total_kb - mem_available_kb) / 1024  # Convert kB to MB

        # Conservative estimate: use MAX of GPU-reported or system-reported
        # Plus headroom buffer (16 GB safety margin)
        estimated_used_mb = max(gpu_memory_mb, system_used_mb) + (16 * 1024)  # +16 GB buffer

        logger.debug(f"Memory usage: GPU={gpu_memory_mb/1024:.1f}GB, "
                    f"System={system_used_mb/1024:.1f}GB, "
                    f"Estimated={estimated_used_mb/1024:.1f}GB")

        return estimated_used_mb / 1024  # Convert to GB

    def get_gpu_temperature(self) -> float:
        """Get GPU temperature in Celsius"""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())

    def get_gpu_power_draw(self) -> float:
        """Get GPU power draw in Watts"""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())

    def can_allocate_model(self, estimated_memory_gb: float) -> bool:
        """Check if we can allocate a new model"""
        current_usage = self.get_unified_memory_usage()
        projected_usage = current_usage + estimated_memory_gb

        # Check memory limit
        if projected_usage > self.hard_limit_gb:
            logger.warning(f"Cannot allocate model: would exceed hard limit "
                          f"({projected_usage:.1f} GB > {self.hard_limit_gb} GB)")
            return False

        # Check resident model count
        if len(self.model_memory_usage) >= self.max_resident_models:
            logger.warning(f"Cannot allocate model: max resident models "
                          f"({self.max_resident_models}) reached")
            return False

        # Warn on soft limit
        if projected_usage > self.soft_limit_gb:
            logger.warning(f"Approaching memory soft limit: "
                          f"{projected_usage:.1f} GB / {self.soft_limit_gb} GB")

        return True

    def allocate_model(self, model_id: str, estimated_memory_gb: float):
        """Allocate resources for a model"""
        if not self.can_allocate_model(estimated_memory_gb):
            raise ResourceError(f"Cannot allocate model {model_id}: resource limits exceeded")

        self.model_memory_usage[model_id] = estimated_memory_gb
        port = self.allocate_port()
        return port

    def allocate_port(self) -> int:
        """Find and allocate available port"""
        for port in range(*self.port_range):
            if port not in self.allocated_ports.values():
                return port
        raise ResourceError("No available ports")

    def release(self, model_id: str):
        """Release all resources for a model"""
        if model_id in self.model_memory_usage:
            del self.model_memory_usage[model_id]
        # Keep port reserved for suspended models
        # Only fully release on stop

    def estimate_model_memory(self, model_name: str, quantization: str) -> float:
        """Estimate model memory usage in GB"""
        # Rough estimates for quantized 7B models on DGX Spark
        if "7B" in model_name or "8B" in model_name:
            if quantization == "fp8":
                return 8  # ~8 GB for fp8 quantized 7B
            elif quantization == "int4":
                return 5  # ~5 GB for int4 quantized 7B
            else:
                return 14  # ~14 GB for fp16 (not recommended)
        elif "3B" in model_name:
            return 4 if quantization == "fp8" else 7
        else:
            # Default conservative estimate
            return 10
```

**Auto-Suspend Manager** (`supervisor/auto_suspend.py`):
```python
class AutoSuspendManager:
    """Manages automatic suspension of idle models (DGX Spark-optimized with thermal hysteresis)"""

    def __init__(self, registry: ModelRegistry, launcher_factory):
        self.registry = registry
        self.launcher_factory = launcher_factory
        self.check_interval_seconds = 60  # Check every minute
        self.resource_manager = ResourceManager()  # For thermal monitoring

        # Thermal hysteresis configuration (prevent thrashing)
        self.thermal_suspend_threshold_c = int(os.getenv("THERMAL_SUSPEND_C", "80"))
        self.thermal_resume_threshold_c = int(os.getenv("THERMAL_RESUME_C", "75"))
        self.thermal_sustain_ms = int(os.getenv("THERMAL_SUSTAIN_MS", "60000"))  # 60s
        self.thermal_cooldown_ms = int(os.getenv("THERMAL_COOLDOWN_MS", "120000"))  # 120s

        # Thermal state tracking
        self.high_temp_start_time = None  # When temp first exceeded threshold
        self.last_thermal_suspend_time = None  # Last time we thermal-suspended

    async def start_monitoring(self):
        """Background task that checks for idle models"""
        while True:
            await asyncio.sleep(self.check_interval_seconds)
            await self._check_idle_models()

    async def _check_idle_models(self):
        """
        Check all running models for idle timeout (DGX Spark-optimized with thermal hysteresis).
        CRITICAL: Add hysteresis to prevent suspend/resume thrashing.
        """
        now = datetime.now()

        # DGX Spark: Check thermal status with hysteresis
        gpu_temp = self.resource_manager.get_gpu_temperature()

        # Hysteresis logic: only suspend if temp sustained high, only resume if sustained low
        if gpu_temp > self.thermal_suspend_threshold_c:
            # Temperature is high
            if self.high_temp_start_time is None:
                # First time above threshold, start timer
                self.high_temp_start_time = now
                logger.info(f"GPU temp {gpu_temp}°C > {self.thermal_suspend_threshold_c}°C, "
                          f"monitoring for {self.thermal_sustain_ms/1000}s")

            else:
                # Check if sustained for required duration
                high_temp_duration = (now - self.high_temp_start_time).total_seconds() * 1000
                if high_temp_duration >= self.thermal_sustain_ms:
                    # Temperature sustained high, check cooldown
                    if self.last_thermal_suspend_time:
                        cooldown_elapsed = (now - self.last_thermal_suspend_time).total_seconds() * 1000
                        if cooldown_elapsed < self.thermal_cooldown_ms:
                            logger.warning(f"Thermal cooldown active, skipping suspend "
                                         f"({cooldown_elapsed/1000:.0f}s / {self.thermal_cooldown_ms/1000:.0f}s)")
                            return

                    # Suspend least-used model
                    logger.warning(f"GPU temp {gpu_temp}°C sustained for {high_temp_duration/1000:.0f}s, "
                                 f"thermal-suspending least-used model")
                    await self._suspend_least_used_model()
                    self.last_thermal_suspend_time = now
                    self.high_temp_start_time = None  # Reset timer
                    return

        elif gpu_temp <= self.thermal_resume_threshold_c:
            # Temperature is normal, reset high temp timer
            if self.high_temp_start_time is not None:
                logger.info(f"GPU temp {gpu_temp}°C dropped below {self.thermal_resume_threshold_c}°C, "
                          f"resetting thermal monitoring")
                self.high_temp_start_time = None

        for model in self.registry.list_running():
            if not model.auto_suspend_enabled:
                continue

            if model.idle_timeout_minutes == 0:
                continue  # Auto-suspend disabled for this model

            # Check if model has been idle too long
            if model.last_request_time is None:
                model.last_request_time = model.started_at

            idle_duration = now - model.last_request_time
            if idle_duration.total_seconds() / 60 >= model.idle_timeout_minutes:
                logger.info(f"Auto-suspending idle model {model.model_id} "
                           f"(idle for {idle_duration.total_seconds()//60} minutes)")
                await self.suspend_model(model.model_id)

    async def _suspend_least_used_model(self):
        """Emergency suspend of least recently used model (thermal protection)"""
        models = sorted(
            self.registry.list_running(),
            key=lambda m: m.last_request_time or m.started_at
        )
        if models:
            least_used = models[0]
            logger.warning(f"Emergency thermal suspend: {least_used.model_id}")
            await self.suspend_model(least_used.model_id)

    async def suspend_model(self, model_id: str):
        """Suspend a model and release GPU resources"""
        model = self.registry.get(model_id)
        if model is None or model.status != "running":
            return

        # Save config for resume
        model.saved_config = {
            "model_name": model.model_name,
            "backend": model.backend,
            "gpu_ids": model.gpu_ids,
            "port": model.port,
            # ... other settings
        }

        # Stop the model process
        launcher = self.launcher_factory.get_launcher(model.backend)
        await launcher.stop(model.model_id)

        # Update status but keep port reserved
        model.status = "suspended"
        model.pid = None
        self.registry.update(model)

        logger.info(f"Model {model.model_id} suspended, GPU freed")

    async def resume_model(self, model_id: str):
        """Resume a suspended model"""
        model = self.registry.get(model_id)
        if model is None or model.status != "suspended":
            raise ValueError(f"Model {model_id} is not suspended")

        if model.saved_config is None:
            raise ValueError(f"No saved config for {model_id}")

        logger.info(f"Resuming model {model.model_id}...")

        # Restart the model with saved config
        launcher = self.launcher_factory.get_launcher(model.backend)
        config = ModelConfig(**model.saved_config)
        new_instance = await launcher.launch(config)

        # Update registry
        model.status = "running"
        model.pid = new_instance.pid
        model.last_request_time = datetime.now()
        self.registry.update(model)

        logger.info(f"Model {model.model_id} resumed and ready")

    def record_request(self, model_id: str):
        """Update last request time for a model"""
        model = self.registry.get(model_id)
        if model:
            model.last_request_time = datetime.now()
            self.registry.update(model)
```

#### 3. Model Launchers (`supervisor/launchers/`)

**Base Interface** (`supervisor/launchers/base.py`):
```python
class ModelLauncher(ABC):
    @abstractmethod
    def launch(self, config: ModelConfig) -> ModelInstance:
        """Launch a model server"""

    @abstractmethod
    def stop(self, model_id: str) -> bool:
        """Stop a model server"""

    @abstractmethod
    def health_check(self, model: ModelInstance) -> bool:
        """Check if model is healthy"""
```

**vLLM Launcher** (`supervisor/launchers/vllm_launcher.py` - DGX Spark):
```python
class VLLMLauncher(ModelLauncher):
    """DGX Spark-optimized vLLM launcher with mandatory quantization"""

    # Backend-specific quantization flag mapping
    QUANTIZATION_MAP = {
        "fp8": "fp8",           # vLLM native fp8
        "int4": "awq",          # vLLM uses AWQ for int4
        "awq": "awq",           # Direct AWQ
        "gptq": "gptq",         # GPTQ quantization
    }

    def launch(self, config: ModelConfig) -> ModelInstance:
        # DGX Spark: Mandatory quantization
        if not config.quantization:
            config.quantization = "fp8"  # Default to fp8
            logger.warning(f"No quantization specified, defaulting to fp8")

        # Map logical quantization to vLLM-specific flag
        vllm_quant = self.QUANTIZATION_MAP.get(config.quantization.lower())
        if not vllm_quant:
            raise ValueError(f"Unsupported quantization for vLLM: {config.quantization}. "
                           f"Supported: {list(self.QUANTIZATION_MAP.keys())}")

        # Verify model has quantized weights
        if not self._has_quantized_weights(config.model_name, vllm_quant):
            raise ValueError(f"Model {config.model_name} does not have {vllm_quant} quantized weights. "
                           f"Download quantized model first.")

        # Per-model max_model_len (not blanket 8192)
        max_len = config.extra_args.get("max_model_len", 8192)  # Default 8192

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model_name,
            "--host", "127.0.0.1",  # CRITICAL: Localhost only
            "--port", str(config.port),
            "--quantization", vllm_quant,  # Backend-specific flag
            "--max-model-len", str(max_len),  # Per-model KV cache limit
            "--gpu-memory-utilization", "0.9",  # Max 90%
            "--disable-log-requests",  # Reduce overhead
            "--max-num-seqs", str(config.extra_args.get("max_concurrent_requests", 32)),  # Concurrency cap
        ]

        # DGX Spark: Single GPU, no tensor parallelism needed
        # CUDA_VISIBLE_DEVICES not needed (only 1 GPU)
        env = os.environ.copy()

        # Option 1: Subprocess (for development)
        if config.use_subprocess:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return ModelInstance(pid=process.pid, ...)

        # Option 2: Systemd service (recommended for production)
        else:
            service_name = f"sparkstation-{config.model_id}"
            self._create_systemd_service(service_name, cmd)
            subprocess.run(["systemctl", "--user", "start", service_name])
            return ModelInstance(systemd_service=service_name, ...)

    def _create_systemd_service(self, service_name: str, cmd: List[str]):
        """Create systemd user service for model"""
        service_content = f"""
[Unit]
Description=Sparkstation Model: {service_name}
After=network.target

[Service]
Type=simple
ExecStart={' '.join(cmd)}
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""
        service_path = Path.home() / ".config/systemd/user" / f"{service_name}.service"
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"])

    def _has_quantized_weights(self, model_name: str, quant_type: str) -> bool:
        """
        Verify model has quantized weights available.
        CRITICAL: Fail fast if quantized model not downloaded.
        """
        # Check for quantized model variants in HuggingFace cache or local path
        # For production: implement proper check against model config.json
        # For now: assume models with "fp8", "awq", "gptq" in name have weights
        model_lower = model_name.lower()

        if quant_type == "fp8" and "fp8" in model_lower:
            return True
        if quant_type == "awq" and "awq" in model_lower:
            return True
        if quant_type == "gptq" and "gptq" in model_lower:
            return True

        # For non-explicit names, check model directory for quantization config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            quant_config = getattr(config, "quantization_config", None)
            if quant_config and quant_config.get("quant_method") == quant_type:
                return True
        except Exception as e:
            logger.warning(f"Could not verify quantization for {model_name}: {e}")

        # Warn but don't fail - let vLLM error if weights are actually missing
        logger.warning(f"Could not verify {quant_type} weights for {model_name}, proceeding anyway")
        return True  # Assume okay, let backend fail if wrong
```

**SGLang Launcher** (`supervisor/launchers/sglang_launcher.py` - DGX Spark):
```python
class SGLangLauncher(ModelLauncher):
    """DGX Spark-optimized SGLang launcher for vision models"""

    # Backend-specific quantization flag mapping (SGLang uses different flags than vLLM)
    QUANTIZATION_MAP = {
        "fp8": "fp8",           # SGLang native fp8
        "int4": "int4",         # SGLang int4 (different from vLLM)
        "awq": "awq",           # AWQ support
        "gptq": "gptq",         # GPTQ support
    }

    def launch(self, config: ModelConfig) -> ModelInstance:
        # DGX Spark: Mandatory quantization for vision models
        if not config.quantization:
            config.quantization = "fp8"
            logger.warning(f"No quantization specified, defaulting to fp8")

        # Map logical quantization to SGLang-specific flag
        sglang_quant = self.QUANTIZATION_MAP.get(config.quantization.lower())
        if not sglang_quant:
            raise ValueError(f"Unsupported quantization for SGLang: {config.quantization}. "
                           f"Supported: {list(self.QUANTIZATION_MAP.keys())}")

        # Per-model context length (not blanket 8192)
        # Vision models may need longer context
        context_len = config.extra_args.get("max_model_len", 8192)

        # Option 1: Subprocess (development)
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", config.model_name,
            "--host", "127.0.0.1",  # CRITICAL: Localhost only
            "--port", str(config.port),
            "--quantization", sglang_quant,  # Backend-specific flag
            "--context-length", str(context_len),  # Per-model context
            "--mem-fraction-static", "0.9",  # Max 90% memory
            "--max-running-requests", str(config.extra_args.get("max_concurrent_requests", 16)),  # Lower for vision
        ]

        if config.use_subprocess:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return ModelInstance(pid=process.pid, ...)

        # Option 2: Docker (preferred for SGLang)
        # DGX Spark: Single GPU, shared memory for vision models
        docker_cmd = [
            "docker", "run", "-d",
            "--name", f"sparkstation-{config.model_id}",
            "--gpus", "all",  # DGX Spark: Only 1 GPU anyway
            "-p", f"127.0.0.1:{config.port}:8000",  # Bind to localhost
            "--shm-size", "32g",  # Shared memory for vision processing
            "--restart", "unless-stopped",
            "lmsysorg/sglang:latest",
            "--model-path", config.model_name,
            "--quantization", config.quantization,
            "--context-length", "8192",
            "--mem-fraction-static", "0.9",
        ]

        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        container_id = result.stdout.strip()
        return ModelInstance(container_id=container_id, ...)
```

### Data Flow

#### Scenario 1: Chat Completion (Text Model)

```
1. Client → LiteLLM Gateway
   POST /v1/chat/completions
   {
     "model": "qwen3-8b",
     "messages": [{"role": "user", "content": "Hello"}]
   }

2. LiteLLM → Model Registry (via fetch_from_url)
   GET http://localhost:9001/models
   Response: List of active models with base URLs

3. LiteLLM → vLLM Backend
   POST http://localhost:8001/v1/chat/completions
   (Forward original request)

4. vLLM → Client (via LiteLLM)
   Response: Completion
```

#### Scenario 2: Vision Analysis

```
1. Client → LiteLLM Gateway
   POST /v1/chat/completions
   {
     "model": "qwen-vl-7b",
     "messages": [{
       "role": "user",
       "content": [
         {"type": "text", "text": "Analyze this image"},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
       ]
     }]
   }

2. LiteLLM → SGLang Backend
   POST http://localhost:8002/v1/chat/completions
   (Forward multimodal request)

3. SGLang → Client (via LiteLLM)
   Response: Image analysis
```

#### Scenario 3: Start New Model

```
1. Admin → Supervisor API
   POST /models/start
   {
     "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
     "backend": "sglang",
     "num_gpus": 1
   }

2. Supervisor → ResourceManager
   Allocate GPU and port

3. Supervisor → SGLangLauncher
   Launch model server

4. Supervisor → Health Check Loop
   Monitor until healthy

5. Supervisor → Response
   {
     "model_id": "qwen-vl-7b-xyz",
     "status": "running",
     "base_url": "http://localhost:8002"
   }

6. LiteLLM (next fetch_from_url refresh)
   Automatically discovers new model
```

#### Scenario 4: Auto-Suspend Idle Model

```
Background:
- Model "qwen3-8b" has been running for 20 minutes
- idle_timeout_minutes = 15
- Last request was 15 minutes ago

1. AutoSuspendManager (background task, every 60s)
   Check all running models for idle timeout

2. AutoSuspendManager detects idle model
   qwen3-8b idle for 15+ minutes

3. AutoSuspendManager → suspend_model("qwen3-8b")
   - Save model config to model.saved_config
   - Stop model process via launcher
   - Update status to "suspended"
   - Release GPU (port kept reserved)

4. GPU Resource
   Now available for other models

Result: GPU freed, model can be quickly resumed
```

#### Scenario 5: Auto-Resume Suspended Model

```
Background:
- Model "qwen3-8b" is suspended
- Client sends request for this model

1. Client → LiteLLM Gateway
   POST /v1/chat/completions
   {"model": "qwen3-8b", ...}

2. LiteLLM → Supervisor /models
   Sees "qwen3-8b" with status="suspended"

3. LiteLLM → Supervisor /models/{model_id}/resume
   Trigger resume (automatic or via middleware)

4. Supervisor → AutoSuspendManager.resume_model()
   - Get saved config
   - Allocate GPU (same or different)
   - Launch model process (~15 seconds)
   - Wait for health check
   - Update status to "running"

5. Supervisor → Response
   Model ready (took ~15s)

6. LiteLLM → Retry original request
   POST http://localhost:8001/v1/chat/completions

7. Model → Response
   Client gets result (with ~15s delay for first request)

Note: Subsequent requests are instant until model idles again
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-2)

**Goal**: Basic supervisor with subprocess management

**Tasks**:
- ✅ Set up project structure
- ✅ Create Supervisor FastAPI app (`supervisor/main.py`)
- ✅ Implement ResourceManager (GPU/port tracking)
- ✅ Implement ModelRegistry (in-memory storage)
- ✅ Create base ModelLauncher interface
- ✅ Implement VLLMLauncher (subprocess-based)
- ✅ Basic `/models` endpoint
- ✅ Basic `/models/start` endpoint (vLLM only)
- ✅ Basic `/models/stop` endpoint

**Deliverable**: Can start/stop a single vLLM model

**Testing**:
```bash
# Start supervisor
cd supervisor && uvicorn main:app --host 0.0.0.0 --port 9001

# Start a model
curl -X POST http://localhost:9001/models/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "vllm",
    "num_gpus": 1
  }'

# List models
curl http://localhost:9001/models

# Test model directly
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Phase 2: Health Checks & Persistence (Days 3-4)

**Goal**: Reliable model management with persistence

**Tasks**:
- ✅ Implement health check system
  - Periodic checks (every 30s)
  - `/v1/models` endpoint verification
  - Update model status
- ✅ Add SQLite persistence for model registry
- ✅ Implement graceful shutdown
- ✅ Add restart capability for failed models
- ✅ Implement SGLangLauncher
- ✅ Implement AutoSuspendManager
  - Background task for idle checking
  - Track last_request_time per model
  - suspend_model() and resume_model() methods
  - Configurable idle_timeout_minutes
- ✅ Add suspend/resume API endpoints
- ✅ Better error handling and logging

**Deliverable**: Supervisor survives restarts, models tracked persistently, auto-suspend working

**Testing**:
```bash
# Kill a model process manually
kill <pid>

# Supervisor should detect failure
curl http://localhost:9001/models
# Should show model as "failed"

# Restart failed model
curl -X POST http://localhost:9001/models/{model_id}/restart

# Test auto-suspend
# 1. Start a model with short idle timeout
curl -X POST http://localhost:9001/models/start \
  -d '{"model_name": "...", "backend": "vllm", "idle_timeout_minutes": 2}'

# 2. Wait 3 minutes without sending requests
sleep 180

# 3. Check status - should be "suspended"
curl http://localhost:9001/models/{model_id}/status
# Should show status="suspended"

# 4. Send a request - should auto-resume
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "...", "messages": [...]}'
# First request takes ~15s (resume time), subsequent requests instant

# 5. Manual suspend/resume
curl -X POST http://localhost:9001/models/{model_id}/suspend
curl -X POST http://localhost:9001/models/{model_id}/resume
```

### Phase 3: LiteLLM Integration (Days 5-6)

**Goal**: Unified gateway routing to backends

**Tasks**:
- ✅ Install and configure LiteLLM
- ✅ Create `gateway/litellm.yaml` with `fetch_from_url`
- ✅ Test routing to vLLM backend
- ✅ Test routing to SGLang backend
- ✅ Implement model name mapping
- ✅ Test streaming responses
- ✅ Test multimodal (vision) requests

**Deliverable**: Single endpoint (`localhost:8000`) routes to all backends

**Testing**:
```bash
# Start LiteLLM gateway
litellm --config gateway/litellm.yaml --port 8000

# Test text model via gateway
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Test"}]
  }'

# Test vision model via gateway
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-7b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe"},
        {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}}
      ]
    }]
  }'
```

### Phase 4: Production Readiness & Security (Days 7-8)

**Goal**: Production-ready deployment with DGX Spark optimizations

**Tasks**:
- ✅ **Security defaults**:
  - Bind all services to 127.0.0.1 (localhost only)
  - Add API key authentication for Supervisor & Gateway
  - Implement shared-secret header for Supervisor API
  - Non-root execution for all processes
- ✅ **Systemd services** (preferred over subprocess for DGX Spark):
  - Supervisor service
  - Gateway service
  - Model launcher systemd templates
- ✅ **Prometheus metrics endpoint**:
  - Unified memory tracking
  - GPU temperature & power monitoring
  - Per-model metrics
  - Auto-suspend events
- ✅ **Structured logging** (journald integration)
- ✅ **Daily maintenance cron**:
  - Unload non-pinned models
  - Restart pinned models (clear fragmentation)
  - Compact registry DB
  - Log stats
- ✅ **Grafana dashboard** for DGX Spark monitoring
- ✅ **1-token health probes** (every 5 minutes)
- ✅ Optional: Create docker-compose.yml as alternative
- ✅ Write deployment docs

**Deliverable**: Production-ready, secure, DGX Spark-optimized deployment

### Phase 5: Testing & Migration (Days 9-10)

**Goal**: Validate with real workloads

**Tasks**:
- ✅ Load testing with Locust
- ✅ Integration tests with mock clients
- ✅ Migrate Kavi to use Sparkstation
  - Update `LLM_PROVIDER` config
  - Test all 8 agent tools
  - Test vision photo copilot
- ✅ Migrate image_metadata_indexing (when ready)
  - Update `SGLANG_API_BASE` to point to Sparkstation
  - Test batch processing
- ✅ Performance validation
- ✅ Write migration guides

**Deliverable**: Both projects running on Sparkstation

---

## API Specifications

### Supervisor API (`http://localhost:9001`)

#### `GET /models`
**Purpose**: List all running models in simple LiteLLM-compatible format

**CRITICAL**: Keep it simple for LiteLLM. Flat list, no nested structures.

**Response**:
```json
[
  {
    "model_name": "qwen3-8b",
    "litellm_provider": "openai",
    "api_base": "http://127.0.0.1:8001/v1",
    "api_key": "EMPTY"
  },
  {
    "model_name": "qwen-vl-7b",
    "litellm_provider": "openai",
    "api_base": "http://127.0.0.1:8002/v1",
    "api_key": "EMPTY"
  }
]
```

**Notes**:
- Only includes models with `status="running"` (not suspended or stopped)
- `litellm_provider` is always "openai" for OpenAI-compatible backends
- `api_base` uses 127.0.0.1 (localhost only)
- Suspended models are excluded (handled by auto-resume middleware)

#### `GET /models/detailed`
**Purpose**: Get detailed model metadata (for dashboard, not LiteLLM)

**Response**:
```json
{
  "models": [
    {
      "id": "qwen3-8b-abc123",
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "alias": "qwen3-8b",
      "backend": "vllm",
      "status": "running",
      "health_status": "healthy",
      "port": 8001,
      "memory_gb": 8.2,
      "last_request_time": "2025-10-26T12:00:00Z",
      "idle_seconds": 120,
      "auto_suspend_enabled": true,
      "idle_timeout_minutes": 30
    }
  ]
}
```

#### `POST /models/start`
**Purpose**: Start a new model server

**Request**:
```json
{
  "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
  "backend": "sglang",
  "num_gpus": 1,
  "model_alias": "qwen-vl-7b",
  "idle_timeout_minutes": 15,         // Auto-suspend after 15 min idle (0 = disabled)
  "auto_suspend_enabled": true,       // Allow auto-suspend (default: true)
  "extra_args": {
    "quantization": "fp8",
    "max_total_tokens": 4096
  }
}
```

**Response**:
```json
{
  "model_id": "qwen-vl-7b-xyz789",
  "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
  "backend": "sglang",
  "status": "starting",
  "port": 8002,
  "gpu_ids": [1],
  "base_url": "http://localhost:8002",
  "started_at": "2025-10-26T10:00:00Z",
  "idle_timeout_minutes": 15,
  "auto_suspend_enabled": true
}
```

#### `POST /models/{model_id}/stop`
**Purpose**: Stop a running model

**Response**:
```json
{
  "model_id": "qwen-vl-7b-xyz789",
  "status": "stopped",
  "stopped_at": "2025-10-26T11:00:00Z"
}
```

#### `POST /models/{model_id}/suspend`
**Purpose**: Manually suspend a model (stop process but keep config for quick resume)

**Response**:
```json
{
  "model_id": "qwen-vl-7b-xyz789",
  "status": "suspended",
  "suspended_at": "2025-10-26T11:00:00Z",
  "gpu_released": [1],
  "port_reserved": 8002
}
```

#### `POST /models/{model_id}/resume`
**Purpose**: Resume a suspended model

**Response**:
```json
{
  "model_id": "qwen-vl-7b-xyz789",
  "status": "running",
  "resumed_at": "2025-10-26T11:01:00Z",
  "startup_time_seconds": 14.5,
  "base_url": "http://localhost:8002"
}
```

#### `PATCH /models/{model_id}/config`
**Purpose**: Update model configuration (e.g., change idle timeout)

**Request**:
```json
{
  "idle_timeout_minutes": 30,
  "auto_suspend_enabled": false
}
```

**Response**:
```json
{
  "model_id": "qwen-vl-7b-xyz789",
  "updated_fields": ["idle_timeout_minutes", "auto_suspend_enabled"],
  "idle_timeout_minutes": 30,
  "auto_suspend_enabled": false
}
```

#### `GET /models/{model_id}/status`
**Purpose**: Get detailed model status

**Response**:
```json
{
  "model_id": "qwen-vl-7b-xyz789",
  "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
  "status": "running",
  "health_status": "healthy",
  "uptime_seconds": 3600,
  "last_health_check": "2025-10-26T11:05:00Z",
  "last_request_time": "2025-10-26T11:03:00Z",
  "idle_seconds": 120,
  "idle_timeout_minutes": 15,
  "auto_suspend_enabled": true,
  "seconds_until_suspend": 780,
  "metrics": {
    "requests_total": 1523,
    "requests_failed": 2,
    "avg_latency_ms": 3200
  }
}
```

#### `GET /resources`
**Purpose**: View available resources

**Response**:
```json
{
  "gpus": {
    "total": 2,
    "available": 0,
    "allocated": [
      {"gpu_id": 0, "model_id": "qwen3-8b-abc123"},
      {"gpu_id": 1, "model_id": "qwen-vl-7b-xyz789"}
    ]
  },
  "ports": {
    "range": [8001, 8100],
    "available": 98,
    "allocated": [
      {"port": 8001, "model_id": "qwen3-8b-abc123"},
      {"port": 8002, "model_id": "qwen-vl-7b-xyz789"}
    ]
  }
}
```

### Gateway API (`http://localhost:8000`)

Standard OpenAI-compatible endpoints:

#### `POST /v1/chat/completions`
OpenAI-compatible chat completions

#### `GET /v1/models`
List available models (proxied from Supervisor)

#### `GET /health`
LiteLLM health check

---

## Deployment Strategy

### Development Deployment (Subprocess-based)

**Advantages**:
- Simpler debugging
- Faster iteration
- Direct process control

**Start Supervisor**:
```bash
cd /home/kshetrajna/src/github.com/sparkstation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd supervisor
uvicorn main:app --host 0.0.0.0 --port 9001 --reload
```

**Start LiteLLM Gateway**:
```bash
cd gateway
litellm --config litellm.yaml --port 8000
```

### Production Deployment (Docker Compose)

**Advantages**:
- Isolation
- Reproducibility
- Easy scaling

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  supervisor:
    build: ./supervisor
    ports:
      - "127.0.0.1:9001:9001"  # CRITICAL: Bind to localhost only
    volumes:
      - ./data:/app/data
      - /var/run/docker.sock:/var/run/docker.sock  # For launching model containers
    environment:
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  gateway:
    image: ghcr.io/berriai/litellm:latest
    ports:
      - "127.0.0.1:8000:8000"  # CRITICAL: Bind to localhost only
    volumes:
      - ./gateway/litellm.yaml:/app/litellm.yaml
    command: ["--config", "/app/litellm.yaml", "--port", "8000", "--host", "127.0.0.1"]
    depends_on:
      - supervisor

  # Example: Pre-launch a vLLM model
  vllm-qwen3:
    image: vllm/vllm-openai:latest
    ports:
      - "127.0.0.1:8001:8000"  # CRITICAL: Bind to localhost only
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: [
      "--model", "Qwen/Qwen2.5-7B-Instruct",
      "--port", "8000",
      "--gpu-memory-utilization", "0.9"
    ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```

### Systemd Services (Alternative)

**sparkstation-supervisor.service**:
```ini
[Unit]
Description=Sparkstation Supervisor
After=network.target

[Service]
Type=simple
User=kshetrajna
WorkingDirectory=/home/kshetrajna/src/github.com/sparkstation/supervisor
ExecStart=/home/kshetrajna/src/github.com/sparkstation/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 9001
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

---

## Dependency Management & Version Pinning

### Critical: Lock Versions

**Why**: vLLM, SGLang, and LiteLLM move fast. Flags break between versions.

**Solution**: Pin to known-good versions via `constraints.txt`:

```txt
# constraints.txt - DGX Spark tested versions

# LLM Backends
vllm==0.4.2  # or latest stable
sglang==0.2.5  # or latest stable

# Gateway
litellm==1.35.8  # or latest stable

# Core dependencies
fastapi==0.110.0
uvicorn==0.29.0
httpx==0.27.0
pydantic==2.6.4
prometheus-client==0.20.0

# NVIDIA/CUDA
nvidia-ml-py3==7.352.0  # For nvidia-smi parsing
torch==2.2.2  # Match with vLLM/SGLang requirements

# Database
sqlalchemy==2.0.29
aiosqlite==0.20.0

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.1
```

**Install**:
```bash
pip install -r requirements.txt -c constraints.txt
```

**Update Strategy**:
1. Test new versions in dev environment first
2. Update one backend at a time
3. Verify all flags still work
4. Update constraints.txt
5. Deploy to production

**Quarterly Review**: Check for security updates and new features

---

## Monitoring & Maintenance (DGX Spark)

### Prometheus Metrics Endpoint

**Location**: `GET /metrics` on Supervisor (port 9001)

**Exposed Metrics**:
```python
# supervisor/metrics.py
from prometheus_client import Gauge, Counter, Histogram

# Memory metrics (DGX Spark unified memory)
unified_memory_used_bytes = Gauge(
    'unified_memory_used_bytes',
    'Total unified memory usage in bytes (DGX Spark)'
)
unified_memory_limit_bytes = Gauge(
    'unified_memory_limit_bytes',
    'Unified memory hard limit in bytes (110 GB)'
)
model_memory_used_bytes = Gauge(
    'model_memory_used_bytes',
    'Estimated memory usage per model in bytes',
    ['model_name']
)

# GPU metrics
gpu_temperature_celsius = Gauge(
    'gpu_temperature_celsius',
    'GPU temperature in Celsius'
)
gpu_power_draw_watts = Gauge(
    'gpu_power_draw_watts',
    'GPU power draw in Watts'
)

# Model status
model_status = Gauge(
    'model_status',
    'Model status (0=stopped, 1=starting, 2=running, 3=suspended, 4=failed)',
    ['model_name', 'model_id']
)
model_last_request_timestamp = Gauge(
    'model_last_request_timestamp',
    'Unix timestamp of last request served by model',
    ['model_name']
)

# Request metrics
model_requests_total = Counter(
    'model_requests_total',
    'Total requests served by model',
    ['model_name']
)
model_requests_failed = Counter(
    'model_requests_failed',
    'Failed requests by model',
    ['model_name']
)
model_request_latency_seconds = Histogram(
    'model_request_latency_seconds',
    'Request latency in seconds',
    ['model_name']
)

# System metrics
resident_models_count = Gauge(
    'resident_models_count',
    'Number of currently resident (running) models'
)
suspended_models_count = Gauge(
    'suspended_models_count',
    'Number of suspended models'
)
```

### Health Check System (1-Token Probes)

**Enhanced Health Checks** (`supervisor/health.py`):
```python
class HealthChecker:
    """DGX Spark-optimized health checker with 1-token probes"""

    async def health_check(self, model: ModelInstance) -> bool:
        """
        Perform 1-token chat completion to verify model responsiveness.
        CRITICAL: Use /v1/chat/completions not /v1/completions (most backends reject /completions).
        """
        try:
            # 1-token chat completion probe (2-5s timeout)
            response = await self.client.post(
                f"{model.base_url}/v1/chat/completions",  # CRITICAL: chat not completions
                json={
                    "model": model.model_name,
                    "messages": [{"role": "user", "content": "hi"}],  # Minimal chat message
                    "max_tokens": 1,
                    "temperature": 0
                },
                timeout=5.0
            )

            if response.status_code == 200:
                # Update metrics
                model.health_status = "healthy"
                model.last_health_check = datetime.now()
                return True
            else:
                logger.warning(f"Model {model.model_id} unhealthy: {response.status_code}")
                model.health_status = "unhealthy"
                return False

        except Exception as e:
            logger.error(f"Health check failed for {model.model_id}: {e}")
            model.health_status = "unhealthy"
            return False

    async def continuous_health_monitoring(self):
        """Background task: check all models every 5 minutes"""
        while True:
            await asyncio.sleep(300)  # 5 minutes
            for model in self.registry.list_running():
                await self.health_check(model)
```

### Daily Maintenance Routine

**Cron Job** (`/etc/cron.d/sparkstation-maintenance`):
```bash
# Run maintenance at 3 AM daily
0 3 * * * kshetrajna /home/kshetrajna/src/github.com/sparkstation/scripts/daily_maintenance.sh
```

**Maintenance Script** (`scripts/daily_maintenance.sh`):
```bash
#!/bin/bash
# DGX Spark Daily Maintenance

set -euo pipefail

SUPERVISOR_API="http://localhost:9001"
LOG_FILE="/var/log/sparkstation/maintenance.log"

echo "=== Sparkstation Maintenance $(date) ===" | tee -a "$LOG_FILE"

# 1. Collect stats before maintenance
echo "Pre-maintenance stats:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv | tee -a "$LOG_FILE"

# 2. Unload all non-pinned models
echo "Unloading non-pinned models..." | tee -a "$LOG_FILE"
curl -s "$SUPERVISOR_API/maintenance/unload-unpinned" | tee -a "$LOG_FILE"

# 3. Wait for models to stop
sleep 30

# 4. Restart any pinned model servers (clear fragmentation)
echo "Restarting pinned models..." | tee -a "$LOG_FILE"
for model_id in $(curl -s "$SUPERVISOR_API/models?pinned=true" | jq -r '.models[].id'); do
    echo "Restarting $model_id" | tee -a "$LOG_FILE"
    curl -X POST "$SUPERVISOR_API/models/$model_id/restart" | tee -a "$LOG_FILE"
done

# 5. Compact registry database
echo "Compacting registry..." | tee -a "$LOG_FILE"
curl -X POST "$SUPERVISOR_API/maintenance/compact-db" | tee -a "$LOG_FILE"

# 6. Collect stats after maintenance
echo "Post-maintenance stats:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv | tee -a "$LOG_FILE"

# 7. Log model usage statistics
curl -s "$SUPERVISOR_API/metrics/usage-report" | tee -a "$LOG_FILE"

echo "=== Maintenance complete ===" | tee -a "$LOG_FILE"
```

### Grafana Dashboard

**DGX Spark Monitoring Dashboard** (`monitoring/grafana-dashboard.json`):

**Panels**:
1. **Unified Memory Usage** (time series)
   - Current usage vs 110 GB hard limit
   - Per-model memory breakdown

2. **GPU Temperature** (gauge + time series)
   - Current temp
   - Thermal threshold (80°C) marker
   - Historical trend

3. **GPU Power Draw** (time series)
   - Current wattage
   - Continuous vs transient load pattern

4. **Model Status** (table)
   - Model name, status, idle time, memory usage
   - Last request timestamp

5. **Request Metrics** (time series)
   - Requests/minute per model
   - Failed requests
   - P50/P95/P99 latencies

6. **Auto-Suspend Events** (event log)
   - Suspend/resume timeline
   - Reason (idle timeout, thermal emergency, manual)

7. **Resident Model Count** (gauge)
   - Current vs max (3)

**Alerts**:
- Memory usage > 100 GB (soft limit)
- Memory usage > 105 GB (approaching hard limit)
- GPU temperature > 75°C (warning)
- GPU temperature > 80°C (critical - auto-suspend triggered)
- Health check failures
- Model startup failures

### Optional Auto-Sleep (Idle System)

**System-Wide Idle Detection** (`supervisor/auto_sleep.py`):
```python
class AutoSleepManager:
    """Unload all models if entire system idle > 1 hour"""

    def __init__(self, idle_threshold_minutes=60):
        self.idle_threshold_minutes = idle_threshold_minutes
        self.enabled = os.getenv("AUTO_SLEEP_ENABLED", "false").lower() == "true"

    async def monitor_system_idle(self):
        """Background task: check if all models idle"""
        if not self.enabled:
            return

        while True:
            await asyncio.sleep(600)  # Check every 10 minutes

            all_models = self.registry.list_all()
            if not all_models:
                continue

            # Check if ALL models are idle
            max_last_request = max(
                (m.last_request_time or m.started_at for m in all_models),
                default=datetime.now()
            )

            idle_duration = datetime.now() - max_last_request
            if idle_duration.total_seconds() / 60 > self.idle_threshold_minutes:
                logger.info(f"System idle for {idle_duration}, auto-sleeping all models")
                await self._sleep_all_models()

    async def _sleep_all_models(self):
        """Suspend all running models"""
        for model in self.registry.list_running():
            logger.info(f"Auto-sleep: suspending {model.model_id}")
            await self.suspend_manager.suspend_model(model.model_id)
```

**Configuration**:
```bash
# .env
AUTO_SLEEP_ENABLED=true  # Optional: auto-sleep if idle > 1 hour
AUTO_SLEEP_IDLE_MINUTES=60
```

---

## Migration Plan

### Phase 1: Parallel Deployment (Week 1)

**Goal**: Run Sparkstation alongside existing services

**Kavi**:
1. Keep Ollama running on port 11434
2. Start Sparkstation with vLLM + Qwen on port 8000
3. Add new env var: `SPARKSTATION_API_BASE=http://localhost:8000/v1`
4. Test both providers work

**image_metadata_indexing**:
1. Keep planned SGLang setup
2. Document Sparkstation integration for future

### Phase 2: Kavi Migration (Week 2)

**Steps**:
1. Update Kavi `llm_config.py` to add "sparkstation" provider:
   ```python
   if provider == "sparkstation":
       return OpenAIModel(
           model_name=os.getenv("SPARKSTATION_MODEL", "qwen3-8b"),
           base_url=os.getenv("SPARKSTATION_API_BASE"),
           api_key="EMPTY"
       )
   ```

2. Update `.env`:
   ```bash
   LLM_PROVIDER=sparkstation
   SPARKSTATION_API_BASE=http://localhost:8000/v1
   SPARKSTATION_MODEL=qwen3-8b
   ```

3. Test all features:
   - ✅ Basic chat
   - ✅ Memory search
   - ✅ Web search tool
   - ✅ Photo copilot (vision model)

4. Monitor for 1 week

5. Deprecate Ollama setup

### Phase 3: image_metadata_indexing Integration (Week 3)

**Steps**:
1. When implementation starts, use Sparkstation from day 1
2. Update config:
   ```bash
   LLM_PROVIDER=sparkstation
   SPARKSTATION_API_BASE=http://localhost:8000/v1
   SPARKSTATION_VISION_MODEL=qwen-vl-7b
   ```

3. No need for separate SGLang setup

### Rollback Plan

If issues arise:
1. Switch `LLM_PROVIDER` back to `ollama` or `groq`
2. Keep Sparkstation running for debugging
3. Compare responses between old and new setup
4. Fix Sparkstation issues
5. Re-attempt migration

---

## Testing Strategy

### Unit Tests

**Supervisor Components**:
```python
# tests/test_resource_manager.py
def test_gpu_allocation():
    rm = ResourceManager(total_gpus=2)
    gpus = rm.allocate_gpu(num_gpus=1)
    assert len(gpus) == 1
    assert gpus[0] in [0, 1]

# tests/test_registry.py
def test_register_model():
    registry = ModelRegistry()
    model = ModelInstance(...)
    registry.register(model)
    assert registry.get(model.model_id) == model
```

**Launchers**:
```python
# tests/test_vllm_launcher.py
@pytest.mark.integration
def test_vllm_launch():
    launcher = VLLMLauncher()
    config = ModelConfig(
        model_name="facebook/opt-125m",  # Tiny model for testing
        backend="vllm",
        num_gpus=1
    )
    model = launcher.launch(config)
    assert model.status == "running"
    # Wait and health check
    time.sleep(30)
    assert launcher.health_check(model) is True
    # Cleanup
    launcher.stop(model.model_id)
```

### Integration Tests

**End-to-End Flow**:
```python
# tests/test_e2e.py
@pytest.mark.slow
def test_full_workflow():
    # 1. Start supervisor
    # 2. Start model via API
    response = requests.post(
        "http://localhost:9001/models/start",
        json={"model_name": "...", "backend": "vllm"}
    )
    model_id = response.json()["model_id"]

    # 3. Wait for healthy
    # 4. Query via LiteLLM gateway
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "...", "messages": [...]}
    )
    assert response.status_code == 200

    # 5. Stop model
    requests.post(f"http://localhost:9001/models/{model_id}/stop")
```

**Auto-Suspend Flow**:
```python
# tests/test_auto_suspend.py
@pytest.mark.slow
def test_auto_suspend_resume():
    # 1. Start model with short idle timeout (2 minutes)
    response = requests.post(
        "http://localhost:9001/models/start",
        json={
            "model_name": "facebook/opt-125m",
            "backend": "vllm",
            "idle_timeout_minutes": 2,
            "auto_suspend_enabled": True
        }
    )
    model_id = response.json()["model_id"]

    # 2. Send initial request to ensure model is running
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "opt-125m", "messages": [{"role": "user", "content": "test"}]}
    )
    assert response.status_code == 200

    # 3. Wait for auto-suspend (3 minutes to be safe)
    time.sleep(180)

    # 4. Check model status - should be suspended
    status = requests.get(f"http://localhost:9001/models/{model_id}/status")
    assert status.json()["status"] == "suspended"

    # 5. Send request - should auto-resume
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "opt-125m", "messages": [{"role": "user", "content": "test"}]}
    )
    resume_time = time.time() - start_time

    assert response.status_code == 200
    assert 10 < resume_time < 30  # Should take ~15s to resume

    # 6. Subsequent request should be fast
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "opt-125m", "messages": [{"role": "user", "content": "test"}]}
    )
    fast_time = time.time() - start_time

    assert response.status_code == 200
    assert fast_time < 5  # Should be instant (model already running)

def test_manual_suspend_resume():
    """Test manual suspend/resume operations"""
    # Start model
    response = requests.post(
        "http://localhost:9001/models/start",
        json={"model_name": "facebook/opt-125m", "backend": "vllm"}
    )
    model_id = response.json()["model_id"]

    # Wait for running
    time.sleep(30)

    # Manually suspend
    response = requests.post(f"http://localhost:9001/models/{model_id}/suspend")
    assert response.json()["status"] == "suspended"

    # Verify status
    status = requests.get(f"http://localhost:9001/models/{model_id}/status")
    assert status.json()["status"] == "suspended"

    # Manually resume
    response = requests.post(f"http://localhost:9001/models/{model_id}/resume")
    assert response.json()["status"] == "running"

    # Verify running
    time.sleep(15)
    status = requests.get(f"http://localhost:9001/models/{model_id}/status")
    assert status.json()["status"] == "running"
```

### Load Tests

**Locust Test**:
```python
# tests/locustfile.py
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat_completion(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
```

**Run**:
```bash
locust -f tests/locustfile.py --host http://localhost:8000 --users 10 --spawn-rate 2
```

### Migration Tests

**Kavi Compatibility**:
```python
# tests/test_kavi_compat.py
def test_kavi_pydantic_ai_integration():
    """Test Sparkstation works with Pydantic-AI"""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel

    model = OpenAIModel(
        model_name="qwen3-8b",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    agent = Agent(model=model)
    result = agent.run_sync("Hello")
    assert result.data
```

---

## Performance Targets

### Latency Targets

| Operation | Target (P50) | Target (P95) | Target (P99) |
|-----------|--------------|--------------|--------------|
| Text completion (qwen3-8b) | 2s | 5s | 10s |
| Vision analysis (qwen-vl-7b) | 3s | 5s | 8s |
| Model startup | 30s | 60s | 90s |
| Health check | 100ms | 500ms | 1s |

### Throughput Targets

- **Concurrent requests**: 10-20 per model
- **Requests per minute**: 100+ per model
- **Daily requests**: 10,000+ across all models

### Resource Utilization

- **GPU memory**: 80-90% utilization per model
- **CPU**: <20% per model
- **RAM**: <8GB per model process

---

## Security & Resource Management

### GPU Safety

**Prevent Conflicts**:
- Track GPU allocation in ResourceManager
- Use `CUDA_VISIBLE_DEVICES` environment variable
- Verify GPU availability before allocation
- Release GPUs on model stop

**Example**:
```python
# Before launching vLLM on GPU 1
env = {
    "CUDA_VISIBLE_DEVICES": "1"  # Only GPU 1 visible to this process
}
```

### Port Safety

**Prevent Conflicts**:
- Maintain allocated port registry
- Check port availability with socket binding
- Use port range 8001-8100
- Release ports on model stop

### Process Management

**Cleanup on Exit**:
```python
import atexit
import signal

def cleanup_all_models():
    """Stop all models on supervisor exit"""
    for model in registry.list_all():
        launcher.stop(model.model_id)

atexit.register(cleanup_all_models)
signal.signal(signal.SIGTERM, lambda: cleanup_all_models())
```

### API Security (Future)

- **API Keys**: Optional authentication for gateway
- **Rate Limiting**: Per-client or per-model limits
- **Quotas**: Token-based usage tracking
- **Audit Logging**: Request logs with client IDs

---

## Open Questions & Decisions

### Q1: Docker vs Subprocess for Model Backends?

**Options**:
- **Subprocess**: Simpler, direct control, easier debugging
- **Docker**: Better isolation, reproducible, easier cleanup

**Decision**: Start with subprocess (Phase 1-3), add Docker option (Phase 4)

### Q2: Persistent Storage - SQLite vs PostgreSQL?

**Options**:
- **SQLite**: Simple, no setup, file-based
- **PostgreSQL**: More robust, better for production

**Decision**: SQLite initially (good enough for single-machine deployment)

### Q3: How to handle model updates/versions?

**Options**:
- Keep old model running while starting new
- Stop old, then start new (brief downtime)
- Blue-green deployment (two instances)

**Decision**: Stop-then-start initially (simpler), add blue-green later if needed

### Q4: Embedding service integration?

**Options**:
- Kavi already has custom embedding service
- Could integrate into Sparkstation
- Or keep separate

**Decision**: Keep separate (embeddings are different workload, Kavi already works)

### Q5: Multi-machine support?

**Scope**: Not in v1, but design should allow for future:
- Supervisor manages models across multiple machines
- Resource manager tracks per-machine GPUs
- LiteLLM routes to remote backends

**Decision**: Single-machine for v1, document extension path

### Q6: Auto-suspend default timeout?

**Options**:
- 5 minutes (very aggressive, frequent suspends)
- 15 minutes (balanced, ~15s resume acceptable)
- 30 minutes (conservative, better for thermals)
- 0 / disabled (opt-in per model)

**Decision**: Default to **30 minutes** with auto_suspend_enabled=true (DGX Spark-optimized)
- Rationale:
  - **Thermal consideration**: Reduces start/stop cycling, better for component longevity
  - Sustained load > transient load for DGX Spark hardware health
  - Still provides >50% GPU hour savings for sporadic use patterns
  - ~15s resume time acceptable for 30-minute idle gap
  - Balances resource efficiency with hardware stress
- Can be overridden per model or globally disabled
- "Always-on" models can set idle_timeout_minutes=0
- Emergency thermal suspend (>80°C) overrides timeout

**Configuration** (DGX Spark):
```bash
# supervisor/.env

# Auto-suspend settings
DEFAULT_IDLE_TIMEOUT_MINUTES=30  # DGX Spark: Thermal-optimized
AUTO_SUSPEND_ENABLED_BY_DEFAULT=true
AUTO_SUSPEND_CHECK_INTERVAL_SECONDS=60

# Thermal management with hysteresis (CRITICAL: prevents thrashing)
THERMAL_SUSPEND_C=80           # Suspend threshold
THERMAL_RESUME_C=75            # Resume threshold (5°C hysteresis)
THERMAL_SUSTAIN_MS=60000       # Sustain high temp for 60s before suspending
THERMAL_COOLDOWN_MS=120000     # Wait 120s after suspend before next suspend

# Memory limits (DGX Spark unified memory)
MEMORY_HARD_LIMIT_GB=110       # 85% of 128 GB
MEMORY_SOFT_LIMIT_GB=100       # 78% warning threshold
MAX_RESIDENT_MODELS=3          # Max simultaneous models

# Optional auto-sleep (all models if idle >1h)
AUTO_SLEEP_ENABLED=false       # Disabled by default
AUTO_SLEEP_IDLE_MINUTES=60

# Security
LITELLM_MASTER_KEY=<generate-secure-key>  # For admin API
SUPERVISOR_API_KEY=<generate-secure-key>  # For supervisor API
```

---

## Success Metrics

### Technical Metrics
- ✅ Supervisor uptime: >99%
- ✅ Model startup success rate: >95%
- ✅ Health check success rate: >99%
- ✅ API compatibility: 100% (no code changes in clients)
- ✅ P95 latency: <5s for both text and vision
- ✅ Auto-suspend success rate: >95%
- ✅ Auto-resume time: <20s (P95)
- ✅ False suspends: <5% (models suspended while still needed)

### Business Metrics
- ✅ Both Kavi and image_metadata_indexing running on Sparkstation
- ✅ GPU utilization: >70% when models running (vs idle GPUs with separate setups)
- ✅ GPU hours saved: >50% via auto-suspend (models idle most of the time)
- ✅ Developer time saved: No need to manage separate LLM setups
- ✅ Cost reduction: Shared infrastructure, efficient GPU use, auto-suspend idle models

---

## Next Actions

1. **Immediate** (Today):
   - ✅ Review this technical plan
   - ⏳ Set up project structure (Phase 1)
   - ⏳ Implement ResourceManager
   - ⏳ Implement ModelRegistry

2. **This Week**:
   - Complete Phase 1 (Foundation)
   - Complete Phase 2 (Health checks)
   - Start Phase 3 (LiteLLM integration)

3. **Next Week**:
   - Complete Phase 3
   - Start Phase 4 (Docker)
   - Begin Kavi migration testing

4. **Week 3**:
   - Complete Phase 5 (Testing & migration)
   - Deploy to production
   - Monitor and iterate

---

## Document Summary

### Key DGX Spark Optimizations (v2.0)

This plan has been specifically optimized for NVIDIA DGX Spark hardware:

1. **Unified Memory Management**:
   - Track 128 GB shared memory (not discrete GPU VRAM)
   - Hard limit: 110 GB (85%), Soft limit: 100 GB (78%)
   - Max 2-3 resident models simultaneously

2. **Mandatory Quantization**:
   - All models must use fp8 or INT4 quantization
   - 2-4× memory reduction + lower bandwidth pressure
   - KV cache limited to 8192 tokens max

3. **Thermal Management**:
   - 30-minute idle timeout (thermal-optimized, not 15)
   - Emergency suspend at >80°C GPU temperature
   - Continuous temperature and power monitoring
   - Daily model restarts to clear fragmentation

4. **Enhanced Observability**:
   - Prometheus metrics for unified memory, temp, power
   - 1-token health probes (verify responsiveness)
   - Grafana dashboard with DGX Spark-specific panels
   - Auto-suspend event tracking

5. **Security & Reliability**:
   - All ports bind to 127.0.0.1 (localhost only)
   - API key authentication
   - Systemd process management (not subprocess)
   - Daily maintenance routine
   - Optional auto-sleep (idle >1 hour)

6. **Resident Model Policy**:
   - Max 3 models running simultaneously
   - Pin 2-3 essential models (GP, VLM, Coder)
   - Auto-suspend non-pinned models after 30 min idle
   - Nightly restart for pinned models

---

**Document Status**: Living document - update as implementation progresses

**Last Updated**: 2025-10-26 (v2.1 - Production-Ready with Critical Fixes)
**Next Review**: After Phase 1 completion

**Changelog**:
- 2025-10-26 v2.1: **Critical production fixes** (9 must-fix items)
  - ✅ LiteLLM push model sync (not fetch_from_url)
  - ✅ Auto-resume middleware for suspended models
  - ✅ Backend-specific quantization mapping (vLLM ≠ SGLang)
  - ✅ Health probe uses /v1/chat/completions not /completions
  - ✅ Unified memory tracking (nvidia-smi + /proc/meminfo + 16GB buffer)
  - ✅ Localhost-only binding (127.0.0.1) in all configs
  - ✅ Thermal hysteresis (60s sustain, 120s cooldown, prevents thrashing)
  - ✅ Version pinning (constraints.txt for vLLM/SGLang/LiteLLM)
  - ✅ Simplified /models response (flat list for LiteLLM)

- 2025-10-26 v2.0: DGX Spark hardware-specific optimizations
  - Unified memory tracking (128 GB shared)
  - Mandatory quantization (fp8/INT4)
  - Thermal management (30-min timeout, emergency suspend)
  - Enhanced observability (Prometheus, 1-token probes)
  - Security defaults (localhost bind, API keys, systemd)
  - Daily maintenance routines
  - Max 2-3 resident models policy
