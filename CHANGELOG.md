# Changelog

All notable changes to Sparkstation will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- **Gateway sync race condition**: Resolved critical startup timing issue where gateway would start before models were ready
  - Supervisor now waits for all autoload models to complete startup before activating gateway sync
  - Health endpoint (`/health`) returns `503` during model loading, `200` only when ready
  - CLI properly waits for supervisor startup completion (600s timeout)
  - Prevents incomplete model list in gateway API
- **Startup reliability**: Two-phase startup ensures deterministic initialization order

### Changed
- **Upgraded to NVIDIA containers 25.11**
  - vLLM: 0.10.2 → 0.11.0 (flashinfer 0.5.0, transformers 4.57.1)
  - SGLang: 0.5.3rc1 → 0.5.4.post1 (stable release)
  - PyTorch 2.10.0 with CUDA 13.0.2

### Added
- **Structured logging**: Supervisor and gateway logs now saved to `~/.sparkstation/logs/`
  - `supervisor.log` - Supervisor startup, model loading, and runtime logs
  - `gateway.log` - LiteLLM gateway logs
  - CLI displays log file paths during startup for easy debugging

---

## [0.2.0] - 2025-11-02

### Added

#### Embeddings Support
- **Text embeddings** with BAAI/bge-large-en-v1.5 model
  - 384-dimensional embeddings for semantic search and RAG
  - OpenAI-compatible `/v1/embeddings` API endpoint
  - Memory-efficient: ~2.5GB including vLLM overhead
- **Image embeddings** with OpenAI CLIP (clip-vit-large-patch14)
  - Visual embeddings for image search and classification
  - SGLang backend for CLIP support
  - Memory usage: ~3.5GB including overhead
- Hybrid vLLM/SGLang architecture supporting both text and image embeddings
- Auto-load configuration in models.yaml for embedding models

#### Docker Backend Support
- **Full Docker containerization** for vLLM and SGLang backends
  - NVIDIA official images with Blackwell support
  - ARM64 (DGX Spark) architecture support
  - Automatic platform detection
  - HuggingFace cache volume mounting
- Backend setup and verification scripts
  - `scripts/setup_backends.sh` - Pull and configure Docker images
  - `scripts/verify_backends.sh` - CUDA verification
- Subprocess mode still supported for development

#### CLI Enhancements
- **sparkstation init** command to create CLAUDE.md project documentation
- **Improved lifecycle management** with real-time status updates
- **Database reconciliation** at startup for orphaned models
- **Auto-load models** from models.yaml configuration
- Better error messages and user feedback

#### Multi-Model Management
- Support for multiple concurrent models (chat + embeddings)
- Named model profiles (dev, prod, inference)
- Per-model configuration with extra_args support
- Speculative decoding support (num_speculative_tokens parameter)

### Fixed
- **Gateway sync race condition** - Renamed DATABASE_URL to SUPERVISOR_DATABASE_URL to avoid LiteLLM auto-detection
- **Model registration issues** - Improved startup reliability
- **Cleanup code** - Removed dead code from AutoResumeMiddleware
- Gateway startup issues with database conflicts

### Changed
- **Upgraded to vLLM 25.10** with full Blackwell support
- **Docker as recommended deployment** (USE_DOCKER=true by default)
- Improved documentation for public release
- Generic path examples in deployment documentation

### Documentation
- Cleaned up and updated all documentation
- Added embeddings usage guide (EMBEDDINGS_PLAN.md)
- Removed personal paths for public release
- Updated README with embeddings examples
- Enhanced deployment documentation

### Verified (October 31, 2025 - Bug Verification)

All critical bugs identified in BUG_REPORT.md have been **verified as fixed**:
- ✅ BUG #1 (MEDIUM): `last_request_time` persistence - Fixed in auto_suspend.py:139
- ✅ BUG #2 (MEDIUM): Process termination verification - Fixed in vllm_launcher.py:165-186
- ✅ BUG #3 (HIGH): `saved_config` on model start - Fixed in main.py:256-268
- ✅ BUG #4 (MEDIUM): SGLang process termination - Fixed in sglang_launcher.py:165-190

All fixes were implemented in commit ed7fd0b and earlier commits. All 24 unit tests pass.

**Status**: Production-ready with all known bugs resolved.

### Added (October 27, 2025 - Production Hardening)

#### Health Check & Auto-Restart System
- **Health check manager** with 1-token chat completion probes
  - Periodic checks every 5 minutes (configurable)
  - Failure tracking with configurable threshold (default 3 failures)
  - Marks models as FAILED after threshold
  - Fully integrated background task
- **Auto-restart manager** with exponential backoff
  - Automatic restart of failed models (1 min → 5 min → 15 min)
  - Max 3 restart attempts before permanent failure
  - Resource-aware restart logic
  - Persistent state saving for reliable restart
- Database schema additions: `restart_count` and `last_restart_time` columns

#### API Key Authentication
- X-API-Key header validation for all model management endpoints
- Backwards compatible (auth disabled if no API_KEY configured)
- Exempt paths: `/health`, `/metrics`, `/docs`, `/openapi.json`, `/redoc`
- Clear 401 Unauthorized responses with actionable messages
- Protects: `POST /models/start`, `/models/{id}/stop`, `/models/{id}/suspend`, `/models/{id}/resume`

#### Comprehensive Error Handling
- Custom exception hierarchy inheriting from `SparkstationError`
- Structured error responses with `error`, `detail`, and `suggestion` fields
- Proper HTTP status codes: 404, 409, 507, 500
- Exception classes:
  - `ModelNotFoundError` (404) - Model ID not found
  - `ModelAlreadyExistsError` (409) - Duplicate alias
  - `InsufficientResourcesError` (507) - Not enough memory/resources
  - `ModelLaunchError` (500) - Backend startup failure
  - `ModelNotRunningError` (409) - Operation requires RUNNING state
  - `ModelNotSuspendedError` (409) - Operation requires SUSPENDED state
- All endpoints enhanced with validation and error handling

#### Unit Test Suite
- 24 tests across 5 test files, all passing
- No backend dependencies - runs on any machine
- Test coverage:
  - `test_config.py` (7 tests) - Settings validation, thermal hysteresis
  - `test_registry.py` (5 tests) - ID generation, uniqueness
  - `test_resources.py` (7 tests) - Memory estimation, port allocation
  - `test_auth.py` (2 tests) - API key authentication
  - `test_errors.py` (6 tests) - Exception formatting
- pytest configuration with asyncio support
- Execution time: <1 second

#### Production Deployment
- **Gateway systemd service**: `scripts/systemd/sparkstation-gateway.service`
  - Depends on supervisor service
  - Same security hardening (NoNewPrivileges, PrivateTmp, ProtectSystem)
- **Daily maintenance script**: `scripts/maintenance.py` (371 lines)
  - Log cleanup (removes files >30 days old)
  - Database vacuum (SQLite optimization)
  - Stale model detection (STARTING >10 min, FAILED models)
  - Port leak detection
  - Resource usage snapshots
  - Dry-run and verbose modes
- **Maintenance systemd timer**: Runs daily at 3 AM
  - `scripts/systemd/sparkstation-maintenance.service` (oneshot)
  - `scripts/systemd/sparkstation-maintenance.timer` (daily schedule)
  - Persistent execution if missed

#### Enhanced Logging
- Dual output: stdout + rotating file logs
- Configurable rotation: 10 MB per file, 5 backup files
- Log file location: `./data/sparkstation.log` (configurable)
- Settings: `LOG_TO_FILE`, `LOG_FILE_PATH`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`

#### Configuration Additions
Added 13 new settings in `supervisor/config.py`:
- Health checks: `HEALTH_CHECK_ENABLED`, `HEALTH_CHECK_INTERVAL_SECONDS`, `HEALTH_CHECK_TIMEOUT_SECONDS`, `HEALTH_CHECK_MAX_FAILURES`
- Auto-restart: `AUTO_RESTART_ENABLED`, `AUTO_RESTART_MAX_ATTEMPTS`, `AUTO_RESTART_BACKOFF_MINUTES`
- Security: `API_KEY`
- Logging: `LOG_TO_FILE`, `LOG_FILE_PATH`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`

### Fixed

#### Critical Production Bugs (October 27, 2025 - Code Review Fixes)
- **ResourceManager split-brain bug** (CRITICAL)
  - AutoSuspendManager was creating its own ResourceManager instance instead of using shared instance
  - Caused memory deadlock after a few suspend/resume cycles
  - System would falsely reject new models claiming insufficient resources while GPUs idle
  - Fixed by passing shared ResourceManager instance from main.py to AutoSuspendManager
  - Impact: Prevents production deadlock within hours of operation

- **Subprocess pipe buffer deadlock** (CRITICAL)
  - vLLM and SGLang launchers piped stdout/stderr to subprocess.PIPE but never consumed output
  - When logs filled 64KB pipe buffer, child process blocked on write() causing hangs
  - Models appeared "hung" with health checks failing and requests timing out
  - Fixed by redirecting subprocess output to log files at `data/model_logs/{model_id}.log`
  - Impact: Prevents random model hangs, provides debugging logs as bonus

- **httpx.AsyncClient file descriptor leak** (RESOURCE LEAK)
  - GatewaySync and HealthCheckManager never called aclose() on their httpx clients
  - Each AsyncClient holds TCP sockets, connection pools, SSL contexts
  - Over days/weeks would exhaust file descriptors causing "too many open files" errors
  - Fixed by adding cleanup to stop() methods: `await self.client.aclose()`
  - Note: AutoResumeMiddleware client is intentionally long-lived (single instance for process lifetime)
  - Impact: Prevents file descriptor exhaustion during long production runs

- SQLAlchemy deprecation warning: Use `declarative_base` from `sqlalchemy.orm`
- Pydantic deprecation warning: Use `ConfigDict` instead of `class Config`
- Test suite now runs with 0 warnings

### Dependencies
- Added `greenlet>=3.0.0` for async SQLAlchemy support

### Documentation
- Updated README.md:
  - Added new features section
  - Updated deployment instructions with all 3 systemd services
  - Added maintenance section with usage examples
  - Enhanced troubleshooting guide
- Updated PROGRESS.md with complete implementation status

#### Grafana Dashboard & Monitoring (October 27, 2025)
- **Production Grafana dashboard** with 11 comprehensive panels
  - Real-time gauges: Memory, temperature, power, model count
  - Time series: Memory usage, GPU temperature trends
  - Analytics: Model status distribution, per-model memory, request rates
  - Performance: Request latency (p50, p95 percentiles)
  - Auto-refresh every 10 seconds
  - Color-coded thresholds (green/yellow/red)
- **Monitoring documentation** (`monitoring/README.md`)
  - Complete Prometheus + Grafana setup guide
  - Recommended alert rules (memory, temperature, model health, capacity)
  - Useful PromQL queries (error rate, idle detection, etc.)
  - Troubleshooting guide for common issues
  - Performance impact metrics

### Planned
- TensorRT-LLM launcher implementation
- Docker Compose deployment option
- Integration tests with real backends
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

[Unreleased]: https://github.com/kshetrajna12/sparkstation/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kshetrajna12/sparkstation/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kshetrajna12/sparkstation/releases/tag/v0.1.0
