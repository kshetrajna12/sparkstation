# Sparkstation Technical Plan

**Version**: 1.1
**Created**: 2025-10-26
**Last Updated**: 2025-10-26 (Added auto-suspend feature)
**Status**: Planning → Implementation
**Purpose**: Unified LLM Gateway for Kavi and image_metadata_indexing

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

#### FR-3: Resource Management
- **FR-3.1**: GPU allocation and tracking
- **FR-3.2**: Port allocation and conflict prevention
- **FR-3.3**: Memory monitoring
- **FR-3.4**: Automatic resource cleanup on model stop

#### FR-4: Model Types Support
- **FR-4.1**: Text-only models (Qwen 3, Deepseek Coder)
- **FR-4.2**: Vision models (Qwen VL, Llama Vision)
- **FR-4.3**: Embedding models (optional)

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

#### NFR-4: Observability
- **NFR-4.1**: Structured logging (JSON format)
- **NFR-4.2**: Prometheus metrics endpoint
- **NFR-4.3**: Request tracing (correlation IDs)
- **NFR-4.4**: GPU usage metrics

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

#### 1. LiteLLM Gateway (Port 8000)
**Technology**: LiteLLM proxy
**Purpose**: Single entry point for all LLM requests

**Responsibilities**:
- Accept OpenAI-compatible requests
- Route to appropriate backend based on `model` parameter
- Handle streaming/non-streaming responses
- Fetch active models from Supervisor dynamically

**Configuration** (`gateway/litellm.yaml`):
```yaml
model_list:
  - fetch_from_url: http://localhost:9001/models  # Dynamic model list

router_settings:
  routing_strategy: simple-shuffle  # Load balance if multiple backends
  allowed_fails: 3
  cooldown_time: 30

litellm_settings:
  drop_params: true  # Drop unsupported params
  set_verbose: true
  success_callback: ["prometheus"]
  failure_callback: ["prometheus"]
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

**Resource Tracking** (`supervisor/resources.py`):
```python
class ResourceManager:
    def __init__(self):
        self.total_gpus = self._detect_gpus()  # via nvidia-smi
        self.allocated_gpus = {}               # model_id -> [gpu_ids]
        self.allocated_ports = {}              # model_id -> port
        self.port_range = (8001, 8100)

    def allocate_gpu(self, num_gpus=1) -> List[int]:
        """Find and allocate available GPUs"""

    def allocate_port(self) -> int:
        """Find and allocate available port"""

    def release(self, model_id: str):
        """Release all resources for a model"""
```

**Auto-Suspend Manager** (`supervisor/auto_suspend.py`):
```python
class AutoSuspendManager:
    """Manages automatic suspension of idle models"""

    def __init__(self, registry: ModelRegistry, launcher_factory):
        self.registry = registry
        self.launcher_factory = launcher_factory
        self.check_interval_seconds = 60  # Check every minute

    async def start_monitoring(self):
        """Background task that checks for idle models"""
        while True:
            await asyncio.sleep(self.check_interval_seconds)
            await self._check_idle_models()

    async def _check_idle_models(self):
        """Check all running models for idle timeout"""
        now = datetime.now()
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

**vLLM Launcher** (`supervisor/launchers/vllm_launcher.py`):
```python
class VLLMLauncher(ModelLauncher):
    def launch(self, config: ModelConfig) -> ModelInstance:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model_name,
            "--port", str(config.port),
            "--tensor-parallel-size", str(len(config.gpu_ids)),
            "--gpu-memory-utilization", "0.9",
        ]
        env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, config.gpu_ids))}
        process = subprocess.Popen(cmd, env=env)
        return ModelInstance(...)
```

**SGLang Launcher** (`supervisor/launchers/sglang_launcher.py`):
```python
class SGLangLauncher(ModelLauncher):
    def launch(self, config: ModelConfig) -> ModelInstance:
        # Option 1: Subprocess
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", config.model_name,
            "--port", str(config.port),
            "--tp-size", str(len(config.gpu_ids)),
        ]

        # Option 2: Docker (preferred)
        docker_cmd = [
            "docker", "run", "-d",
            "--gpus", f"device={','.join(map(str, config.gpu_ids))}",
            "-p", f"{config.port}:8000",
            "--shm-size", "32g",
            "lmsysorg/sglang:latest",
            "--model-path", config.model_name,
        ]
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

### Phase 4: Docker & Production Readiness (Days 7-8)

**Goal**: Containerized deployment

**Tasks**:
- ✅ Create Dockerfile for Supervisor
- ✅ Create docker-compose.yml
  - Supervisor service
  - LiteLLM service
  - Example vLLM service
  - Example SGLang service
- ✅ Add Prometheus metrics
- ✅ Add structured logging
- ✅ Create systemd service files (optional)
- ✅ Write deployment docs

**Deliverable**: Production-ready deployment

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
**Purpose**: List all models (LiteLLM-compatible format)

**Response**:
```json
{
  "data": [
    {
      "litellm_params": {
        "model": "openai/qwen3-8b",
        "api_base": "http://localhost:8001/v1",
        "api_key": "EMPTY"
      },
      "model_info": {
        "id": "qwen3-8b-abc123",
        "db_model": false
      },
      "model_name": "qwen3-8b"
    },
    {
      "litellm_params": {
        "model": "openai/qwen-vl-7b",
        "api_base": "http://localhost:8002/v1",
        "api_key": "EMPTY"
      },
      "model_info": {
        "id": "qwen-vl-7b-def456",
        "db_model": false
      },
      "model_name": "qwen-vl-7b"
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
      - "9001:9001"
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
      - "8000:8000"
    volumes:
      - ./gateway/litellm.yaml:/app/litellm.yaml
    command: ["--config", "/app/litellm.yaml", "--port", "8000"]
    depends_on:
      - supervisor

  # Example: Pre-launch a vLLM model
  vllm-qwen3:
    image: vllm/vllm-openai:latest
    ports:
      - "8001:8000"
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
- 30 minutes (conservative, less GPU savings)
- 0 / disabled (opt-in per model)

**Decision**: Default to 15 minutes with auto_suspend_enabled=true
- Rationale: With ~15s resume time, users won't notice much delay
- GPU freed after 15 min idle = good resource utilization
- Can be overridden per model or globally disabled
- "Always-on" models can set idle_timeout_minutes=0

**Configuration**:
```bash
# supervisor/.env
DEFAULT_IDLE_TIMEOUT_MINUTES=15
AUTO_SUSPEND_ENABLED_BY_DEFAULT=true
AUTO_SUSPEND_CHECK_INTERVAL_SECONDS=60
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

**Document Status**: Living document - update as implementation progresses

**Last Updated**: 2025-10-26
**Next Review**: After Phase 1 completion
