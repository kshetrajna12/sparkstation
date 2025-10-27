# 🔧 Project Plan: Unified LLM Gateway with Supervisor

**Version**: 0.1
**Last Updated**: 2025-10-26
**Status**: Draft

---

## 🎯 Goal

Build a unified, production-ready service that:

* Launches and manages multiple LLM backends (e.g. vLLM, SGLang, TRT-LLM)
* Exposes a standard OpenAI-compatible API via **LiteLLM**
* Supports dynamic model start/stop
* Enables multi-tenant application use of models deployed on DGX Spark

---

## 🧱 Components

### 1. **Model Supervisor Service** (FastAPI-based)

Responsible for managing the lifecycle of model servers.

**Responsibilities:**

* Launch vLLM/SGLang/TRT-LLM processes (or containers)
* Assign GPUs and ports safely
* Track running models (name → port, GPU, backend)
* Health-check running servers (`/v1/models`)
* Serve `/models` endpoint in LiteLLM format
* Optionally expose metrics/logs

**API:**

```http
GET /models                # List active models
POST /start_model          # Start a new model server
POST /stop_model           # Stop and clean up a model server
```

**Storage:** In-memory tracking or SQLite state store.

### 2. **LiteLLM Gateway**

Frontend router that speaks OpenAI-style API and dispatches to active models.

**Responsibilities:**

* Route `/v1/chat/completions` to correct backend
* Stream support
* Uses `fetch_from_url` to pull live model list from Supervisor
* Optional: API key auth, rate limits, analytics

**Config Snippet:**

```yaml
model_list:
  - fetch_from_url: http://localhost:9001/models
```

---

## 🧠 Model Backends

Each model runs on a dedicated port with its own process or container.

### ✅ Examples:

* `vLLM` — for GPT-OSS or Deepseek
* `SGLang` — for Qwen 3 VL or similar
* `TensorRT-LLM` — for quantized fast models
* Custom embedding servers

---

## ⚙️ Deployment Strategy

### Option 1: Native (subprocess)

```bash
python -m vllm.entrypoints.openai.api_server --model ... --port ...
```

### Option 2: Docker (preferred for isolation)

```bash
docker run --gpus device=0 -p 8001:8000 vllm-server:latest
```

Supervisor abstracts both options under a simple config/command interface.

---

## 🗂️ Folder Structure

```bash
sparkstation/
├── supervisor/
│   ├── main.py              # FastAPI app
│   ├── launcher.py          # Starts/stops subprocesses or containers
│   ├── registry.py          # Tracks running models
│   └── config.py            # GPU/port limits
├── scripts/
│   ├── start_vllm.sh        # Helper script
│   └── start_sglang.sh
├── gateway/
│   └── litellm.yaml         # Config for LiteLLM
├── docker/
│   └── Dockerfile           # Optional container build
└── README.md
```

---

## ✅ Benefits

* Minimal external dependencies
* Centralized resource and lifecycle control
* Dynamic registration of models
* OpenAI-compatible API for all clients
* Extensible to support auto-scaling, quotas, etc.

---

## 🚀 Next Steps

1. Scaffold Supervisor FastAPI service
2. Add support for launching/stopping subprocesses with GPU/port tracking
3. Integrate model registry with `/models` API
4. Configure and test LiteLLM with `fetch_from_url`
5. Test with 3 models: GPT-OSS (vLLM), Qwen 3 VL (SGLang), Deepseek Coder (vLLM)

---

Let me know when you're ready to scaffold the codebase.
