# Embeddings Support Implementation Plan

**Status**: Planning Phase
**Last Updated**: November 2, 2025
**Goal**: Add OpenAI-compatible embeddings endpoints to Sparkstation

---

## 🎯 Overview

Enable Sparkstation to serve embedding models alongside chat models, providing a unified interface for both text generation and embedding generation through OpenAI-compatible APIs.

### Current State
- ✅ LLM inference models (vLLM) running in Docker
- ✅ LiteLLM gateway for `/v1/chat/completions`
- ✅ Auto-suspend, health checks, model lifecycle management
- ❌ No embedding model support

### Target State
- ✅ Support both **chat** and **embedding** models
- ✅ Route `/v1/embeddings` requests via LiteLLM
- ✅ Same lifecycle management (auto-suspend, health checks, restart)
- ✅ Unified CLI and API experience

---

## ⚠️ CRITICAL: Engine Selection (vLLM V0 vs V1 vs SGLang)

### The V0/V1 Issue

**Current Situation:**
- Our NVIDIA image (`nvcr.io/nvidia/vllm:25.10-py3`) uses **vLLM v0.10.2** ✅
- vLLM **V0** supports embeddings (via `--runner pooling` or auto-detection)
- vLLM **V1** (released Jan 2025, v1.0.0+) does **NOT** support embeddings yet
- vLLM is gradually migrating from V0 to V1

**Implications:**
1. ✅ **We're currently safe** - v0.10.2 is V0 and supports embeddings
2. ⚠️ **Future risk** - NVIDIA may update to V1 in future images
3. ⚠️ **Performance** - V0 embeddings have acknowledged performance limitations
4. 📈 **V1 migration** - Eventually we'll need to migrate to V1 for chat models

**Risk Assessment:**

| Risk | Severity | Timeline | Mitigation |
|------|----------|----------|------------|
| NVIDIA image updates to V1 | High | 3-6 months | Pin to specific v0.x tag |
| V0 performance limitations | Medium | Current | Acceptable for DGX Spark workloads |
| V0 deprecation | Medium | 6-12 months | Migrate to SGLang for embeddings |

### Alternative: SGLang

**SGLang Advantages:**
- ✅ Full embedding model support (no V0/V1 issues)
- ✅ Better embedding performance than vLLM V0
- ✅ OpenAI-compatible `/v1/embeddings` API
- ✅ Supports multimodal embeddings
- ✅ Active development and optimization

**SGLang Disadvantages:**
- ❌ Additional dependency (new Docker image)
- ❌ Different launcher code needed
- ❌ More complexity in codebase

### 🎯 Recommended Approach: Hybrid Strategy

**Phase 1 (Immediate - This Plan):**
- Implement embeddings with **vLLM V0** (current image)
- Pin Docker image to `nvcr.io/nvidia/vllm:25.10-py3` or specific v0.x tag
- Get embeddings working quickly with minimal changes

**Phase 2 (Future - Separate Plan):**
- Add SGLang as alternative backend alongside vLLM
- Migrate chat models to vLLM V1 when ready
- Use SGLang exclusively for embeddings
- Support both engines: `backend: vllm` or `backend: sglang`

**Rationale:**
1. Start simple with existing infrastructure
2. Don't block embeddings on multi-engine refactor
3. Minimize risk by using proven v0.10.2
4. Plan for future without over-engineering now

---

## 📋 Implementation Tasks

### ✅ Phase 0: Research & Planning
- [x] Research vLLM embeddings capabilities
- [x] Research LiteLLM embeddings routing
- [x] Verify NVIDIA image version (v0.10.2 ✅)
- [x] Evaluate alternatives (SGLang)
- [x] Create implementation plan

---

### ✅ Phase 1: Core Data Model Changes **COMPLETE**

**Goal**: Add `model_type` field to distinguish chat vs embedding models

#### Files Modified:
- `supervisor/models.py` ✅
- `supervisor/models_config.py` ✅
- `models.yaml` ✅

#### Tasks Completed:
- [x] Add `ModelType` enum: `CHAT = "chat"`, `EMBEDDING = "embedding"`
- [x] Add `model_type: str` to `ModelInstanceDB` (default: "chat")
- [x] Add `model_type: ModelType` to:
  - `ModelConfig` ✅
  - `ModelInstance` ✅
  - `ModelStartRequest` ✅
  - `ModelStartResponse` ✅
  - `ModelStatusResponse` ✅
- [x] Add `model_type` to `ModelConfigYAML` for auto-loading
- [x] Update `models.yaml` with example embedding model config (BAAI/bge-small-en-v1.5)

**Example Config:**
```yaml
- name: "BAAI/bge-small-en-v1.5"
  alias: "bge-small"
  backend: "vllm"
  model_type: "embedding"
  quantization: "none"
  extra_args:
    gpu_memory_utilization: 0.2
```

**Testing:**
- [ ] Run supervisor with new model schema
- [ ] Verify existing models still load as type='chat'
- [ ] Add test embedding model to models.yaml

---

### ✅ Phase 2: vLLM Launcher Modifications **COMPLETE**

**Goal**: Launch embedding models in pooling mode

#### Files Modified:
- `supervisor/launchers/vllm_launcher.py` ✅

#### Tasks Completed:
- [x] Import `ModelType` enum
- [x] Detect `model_type` from `ModelConfig` (check at start of launch method)
- [x] For embedding models, add `--task embedding` flag
- [x] Skip chat-specific args for embeddings:
  - Skip: `--max-model-len`, `--max-num-seqs` for embedding models
  - Keep: `--trust-remote-code`, `--gpu-memory-utilization` for all models
- [x] Update Docker command builder with conditional logic
- [x] Update quantization logic (embedding models default to no quantization)
- [x] Pass `model_type` to ModelInstance creation (both Docker and subprocess)

**Code Structure:**
```python
# In launch() method
if config.model_type == ModelType.EMBEDDING:
    # Add embedding-specific flags
    docker_cmd.extend(["--task", "embedding"])  # or --runner pooling
    # Skip max-model-len, max-num-seqs
else:
    # Existing chat model logic
    docker_cmd.extend(["--max-model-len", str(max_len)])
    # ... rest of chat args
```

**Testing:**
- [ ] Launch test embedding model manually
- [ ] Verify container starts with pooling mode
- [ ] Check logs for correct vLLM flags
- [ ] Test `/v1/embeddings` endpoint directly on model port

---

### ✅ Phase 3: Health Check Updates **COMPLETE**

**Goal**: Different health checks for chat vs embedding models

#### Files Modified:
- `supervisor/launchers/vllm_launcher.py` (health_check method) ✅

#### Tasks Completed:
- [x] Update `VLLMLauncher.health_check()`:
  - For chat: `POST /v1/chat/completions` (existing)
  - For embedding: `POST /v1/embeddings` with test payload
- [x] Implement conditional logic based on `instance.model_type`
- [x] Embedding test payload: `{"input": "test", "model": "<model_name>"}`
- [x] Chat test payload: 1-token completion as before
- [x] Update docstring to reflect dual-mode operation

**Testing:**
- [ ] Health check passes for chat models (regression)
- [ ] Health check passes for embedding models
- [ ] Health check fails correctly for stopped models

---

### ⏭️ Phase 4: Gateway Sync Updates **PENDING**

**Goal**: Route embedding models to LiteLLM correctly

#### Files to Modify:
- `supervisor/gateway_sync.py`

#### Tasks:
- [ ] Verify current gateway sync works with embedding models
- [ ] Test LiteLLM routing for embeddings:
  - Uses same `openai/` prefix as chat models
  - LiteLLM auto-detects `/v1/embeddings` endpoint
- [ ] Test config format:
  ```yaml
  model_list:
    - model_name: "bge-small"
      litellm_params:
        model: "openai/BAAI/bge-small-en-v1.5"
        api_base: "http://127.0.0.1:8001/v1"
        api_key: "EMPTY"
  ```
- [ ] Update `get_models_for_litellm()` to include embedding models

**Testing:**
- [ ] LiteLLM sees embedding models in config
- [ ] `/v1/embeddings` request routes correctly
- [ ] Both chat and embedding requests work simultaneously

---

### ✅ Phase 5: Registry Extensions **COMPLETE**

**Goal**: Query models by type

#### Files Modified:
- `supervisor/registry.py` ✅

#### Tasks Completed:
- [x] Add `list_by_type(model_type: str)` method
- [x] Update `_to_pydantic()` to handle model_type field (with fallback to CHAT)
- [x] Update `create()` to save model_type to database
- [x] Reconcile_state handles both types (no changes needed)

**API Examples:**
```python
# Get all embedding models
embedding_models = await registry.list_by_type(ModelType.EMBEDDING)

# Get all chat models
chat_models = await registry.list_by_type(ModelType.CHAT)
```

**Testing:**
- [ ] List models by type via registry
- [ ] Verify mixed model lists work
- [ ] Database reconciliation handles both types

---

### 💻 Phase 6: CLI Enhancements

**Goal**: User-friendly CLI for embeddings

#### Files to Modify:
- `cli.py`

#### Tasks:
- [ ] Add `--type` filter to `models list`:
  ```bash
  sparkstation models list --type chat
  sparkstation models list --type embedding
  sparkstation models list --type all
  ```
- [ ] Update status output to show model type
- [ ] Add `--type` to `models start` (optional, auto-detect from model name)
- [ ] Add examples to help text

**Output Example:**
```
Model ID              Status    Type       Alias       Port
-----------------------------------------------------------
qwen-vl-3b-abc123     running   chat       qwen-vl     8001
bge-small-def456      running   embedding  bge-small   8002
```

**Testing:**
- [ ] CLI lists models with types
- [ ] Filtering by type works
- [ ] Start embedding model via CLI

---

### 📚 Phase 7: Documentation

**Goal**: Document embeddings usage for users

#### Files to Modify:
- `README.md`
- `DEPLOYMENT_PRODUCTION.md`
- `models.yaml` (comments)

#### Tasks:
- [ ] Add "Embeddings Support" section to README
- [ ] Add example embedding model configs
- [ ] Add curl examples for `/v1/embeddings`
- [ ] Document model_type field in models.yaml
- [ ] Add troubleshooting section for embeddings

**Example Usage in README:**
```bash
# Auto-load embedding model (add to models.yaml)
sparkstation start

# Start embedding model manually
sparkstation models start BAAI/bge-small-en-v1.5 --alias bge --type embedding

# Test embeddings endpoint
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox",
    "model": "bge-small"
  }'
```

**Testing:**
- [ ] Follow documentation as new user
- [ ] Verify all examples work
- [ ] Check clarity and completeness

---

## 🧪 Testing Strategy

### Test Models

**Small Embedding Models (for testing):**
1. `BAAI/bge-small-en-v1.5` (~130MB, fast download)
2. `sentence-transformers/all-MiniLM-L6-v2` (~90MB)
3. `intfloat/e5-small-v2` (~130MB)

**Production Embedding Models:**
1. `BAAI/bge-large-en-v1.5` (~1.3GB, best quality)
2. `intfloat/e5-mistral-7b-instruct` (~7B params, high performance)
3. `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (~1.5B params, 32K context)

### Test Checklist

**Unit Tests:**
- [ ] Model type enum serialization
- [ ] Database schema with model_type field
- [ ] Config validation with model_type

**Integration Tests:**
- [ ] Launch embedding model in Docker
- [ ] Health check for embedding model
- [ ] Gateway routing to embedding endpoint
- [ ] Auto-suspend/resume for embedding models
- [ ] Mixed workload (chat + embedding)

**End-to-End Tests:**
- [ ] Auto-load embedding model from models.yaml
- [ ] Query `/v1/embeddings` via LiteLLM gateway
- [ ] Verify correct embeddings returned
- [ ] Multiple concurrent embedding requests
- [ ] Model restart after failure
- [ ] Resource limits respected (3 max models)

**Performance Tests:**
- [ ] Embedding latency (target: <100ms for small batches)
- [ ] Throughput (requests/second)
- [ ] GPU memory usage vs chat models
- [ ] Idle timeout and resume speed

---

## 🚧 Known Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| vLLM V0 deprecation | High | Medium (6mo) | Pin image version, plan SGLang migration |
| LiteLLM routing issues | Medium | Low | Test thoroughly, fallback to direct routing |
| Performance insufficient | Medium | Low | Acceptable for DGX Spark, optimize later |
| Docker image size | Low | Low | Embeddings use same vLLM image |
| Mixed model resource limits | Medium | Medium | Test 3-model limit with mixed types |

---

## 🔮 Future Enhancements (Out of Scope)

### Phase 2: Multi-Engine Support (SGLang)

**Goals:**
- Add `backend: sglang` option to models.yaml
- Create `SGLangLauncher` class
- Use SGLang exclusively for embeddings (better performance)
- Migrate chat models to vLLM V1

**Files to Create:**
- `supervisor/launchers/sglang_launcher.py`
- `supervisor/launchers/sglang_config.py`

**Timeline:** 3-6 months after embeddings v1 complete

### Advanced Features
- Batch embedding optimization
- Embedding caching layer
- Multi-vector embeddings
- Reranker model support

---

## 📊 Success Metrics

- [ ] Embedding models auto-load on startup
- [ ] `/v1/embeddings` endpoint accessible via LiteLLM
- [ ] Latency: <100ms for single embedding (256 tokens)
- [ ] Can run 1 chat + 2 embedding models simultaneously
- [ ] Auto-suspend works for embedding models
- [ ] Health checks succeed for both model types
- [ ] Documentation allows new users to deploy embeddings in <10 minutes

---

## 🗓️ Timeline Estimate

- **Phase 1** (Data Models): 1-2 hours
- **Phase 2** (Launcher): 2-3 hours
- **Phase 3** (Health Checks): 1 hour
- **Phase 4** (Gateway): 1-2 hours
- **Phase 5** (Registry): 1 hour
- **Phase 6** (CLI): 1-2 hours
- **Phase 7** (Docs): 1-2 hours
- **Testing**: 2-3 hours

**Total Estimate:** 10-16 hours of development + testing

---

## 🔄 Update Log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2025-11-02 | Planning | ✅ Complete | Initial plan created, V0/V1 issue researched |
| 2025-11-02 | Phase 1 | ✅ Complete | Added ModelType enum, model_type fields to all classes |
| 2025-11-02 | Phase 2 | ✅ Complete | Updated vLLM launcher with --task embedding support |
| 2025-11-02 | Phase 3 | ✅ Complete | Updated health checks for embedding models |
| 2025-11-02 | Phase 5 | ✅ Complete | Added list_by_type() to registry |
| 2025-11-02 | Main.py | ✅ Complete | Updated auto-load and API endpoints |
| 2025-11-02 | Phase 4-7 | ⏳ Pending | Gateway sync, CLI, docs, testing remain |
