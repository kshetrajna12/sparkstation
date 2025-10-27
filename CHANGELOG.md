# Changelog

All notable changes to Sparkstation will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Health check system with continuous monitoring
- TensorRT-LLM launcher implementation
- Docker Compose deployment option
- Grafana dashboard for monitoring
- Integration tests with mock backends
- Load testing suite

---

## [0.1.0] - 2025-10-26

### Added
- **Initial project scaffolding** with uv dependency management
- **Supervisor FastAPI service** (port 9001) for model lifecycle management
  - Model registry with SQLite persistence (SQLAlchemy + async)
  - Resource manager with DGX Spark unified memory tracking (128 GB)
  - Auto-suspend/resume system with thermal hysteresis
  - Gateway sync for pushing models to LiteLLM admin API
  - Prometheus metrics endpoint (`/metrics`)
  - Model launchers for vLLM and SGLang with quantization support

- **API Endpoints**:
  - `POST /models/start` - Launch models with auto-suspend configuration
  - `POST /models/{id}/stop` - Stop running models
  - `POST /models/{id}/suspend` - Manually suspend models
  - `POST /models/{id}/resume` - Resume suspended models
  - `GET /models` - List running models (LiteLLM-compatible format)
  - `GET /models/detailed` - Detailed model metadata with status
  - `GET /models/{id}/status` - Individual model status with idle tracking
  - `GET /resources` - System resource status (memory, temperature, power)
  - `GET /metrics` - Prometheus metrics
  - `GET /health` - Health check

- **DGX Spark Optimizations**:
  - Unified CPU+GPU memory tracking (128 GB shared)
  - Hard/soft memory limits (110 GB / 100 GB)
  - Thermal management with sustained temperature monitoring
  - Hysteresis to prevent suspend/resume thrashing
  - Mandatory quantization support (fp8, int4, awq, gptq)
  - Backend-specific quantization flag mapping (vLLM vs SGLang)

- **Gateway Components**:
  - LiteLLM configuration with dynamic model discovery
  - Auto-resume middleware for suspended models
  - 30-second timeout for model resume operations

- **Infrastructure**:
  - Project structure (supervisor/, gateway/, scripts/, data/)
  - Version pinning via `constraints.txt`
  - Environment configuration (`.env.example`)
  - Startup scripts for supervisor and gateway
  - Systemd service template for production deployment
  - Comprehensive README with quick start guide
  - Detailed TECH_PLAN.md (2786 lines)

- **Model Launchers**:
  - vLLM launcher with subprocess management
  - SGLang launcher with subprocess management
  - Factory pattern for launcher selection
  - Quantization support with backend-specific mapping

- **Monitoring**:
  - Prometheus metrics for memory, temperature, power
  - Per-model request tracking
  - Model status gauges
  - Resource utilization metrics

### Developer Experience
- uv for fast dependency management
- Black + Ruff for code formatting
- MyPy for type checking
- Pytest for testing
- Comprehensive documentation

### Known Limitations
- Subprocess-based launchers only (systemd/Docker coming soon)
- No health check background task yet (probes implemented but not scheduled)
- TensorRT-LLM launcher not yet implemented
- No integration tests yet (requires vLLM/SGLang installed)

---

## Release Notes

### v0.1.0 - Foundation Release

This is the initial scaffolding release implementing the core architecture from TECH_PLAN.md v2.1.

**What Works**:
- ✅ Supervisor starts and serves API endpoints
- ✅ Model registry with SQLite persistence
- ✅ Resource tracking (memory, temperature, power)
- ✅ Auto-suspend/resume logic implemented
- ✅ Gateway sync for LiteLLM integration
- ✅ Prometheus metrics endpoint
- ✅ Model launcher framework (vLLM, SGLang)

**What's Next**:
- Integration testing with actual vLLM/SGLang backends
- Health check background task activation
- Docker/systemd launcher implementations
- Grafana dashboard deployment
- Production testing on DGX Spark hardware

**Target Platform**: NVIDIA DGX Spark (Grace Blackwell)

**Python**: 3.11+

---

[Unreleased]: https://github.com/kshetrajna12/sparkstation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kshetrajna12/sparkstation/releases/tag/v0.1.0
