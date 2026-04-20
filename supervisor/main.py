"""
Sparkstation Supervisor - Main FastAPI application.

DGX Spark-optimized LLM model lifecycle management.
"""
import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

from supervisor.config import settings
from supervisor.models import (
    Backend,
    ModelStartRequest,
    ModelStartResponse,
    ModelStatusResponse,
    ModelInstance,
    ModelStatus,
    ResourceStatus,
    LiteLLMModelFormat,
    ModelConfig,
)
from supervisor.registry import ModelRegistry
from supervisor.resources import ResourceManager, ResourceError
from supervisor.launchers.factory import LauncherFactory
from supervisor.auto_suspend import AutoSuspendManager
from supervisor.gateway_sync import GatewaySync
from supervisor.health_check import HealthCheckManager
from supervisor.restart_manager import RestartManager
from supervisor.auth import require_api_key
from supervisor.errors import (
    ModelNotFoundError,
    ModelAlreadyExistsError,
    InsufficientResourcesError,
    ModelLaunchError,
    ModelNotRunningError,
    ModelNotSuspendedError,
    handle_exception,
)
from supervisor import metrics

# Configure logging (stdout + file)
handlers = [logging.StreamHandler()]

if settings.log_to_file:
    # Ensure log directory exists
    log_path = Path(settings.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Add rotating file handler
    file_handler = RotatingFileHandler(
        settings.log_file_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
    )
    handlers.append(file_handler)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

# Global instances
registry: Optional[ModelRegistry] = None
resource_manager: Optional[ResourceManager] = None
launcher_factory: Optional[LauncherFactory] = None
auto_suspend_manager: Optional[AutoSuspendManager] = None
gateway_sync: Optional[GatewaySync] = None
health_check_manager: Optional[HealthCheckManager] = None
restart_manager: Optional[RestartManager] = None

# Startup state tracking
startup_complete: bool = False
default_model_alias: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global registry, resource_manager, launcher_factory, auto_suspend_manager, gateway_sync, health_check_manager, restart_manager, default_model_alias

    logger.info("Initializing Sparkstation Supervisor...")

    # Initialize components
    registry = ModelRegistry()
    await registry.initialize()

    # Reconcile database state with reality (fix stale/orphaned entries)
    reconcile_summary = await registry.reconcile_state()
    if reconcile_summary["fixed_stopped"] or reconcile_summary["fixed_running"] or reconcile_summary["removed_orphaned"]:
        logger.warning(f"Database reconciliation found inconsistencies: {reconcile_summary}")

    # Purge stale DB entries (stopped/failed/starting, or "running" with no live container)
    # This prevents confusion when switching between profiles or after unclean shutdown
    import subprocess as _sp
    all_models = await registry.list_all()
    for model in all_models:
        if model.status not in [ModelStatus.RUNNING]:
            logger.info(f"Purging stale DB entry: {model.model_alias or model.model_name} (status: {model.status})")
            await registry.delete(model.id)
        elif model.container_id:
            # Verify the container is actually running
            result = _sp.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", model.container_id],
                capture_output=True, text=True
            )
            if result.returncode != 0 or result.stdout.strip() != "true":
                logger.info(f"Purging stale DB entry: {model.model_alias or model.model_name} (container {model.container_id} not running)")
                await registry.delete(model.id)

    resource_manager = ResourceManager()
    launcher_factory = LauncherFactory()
    auto_suspend_manager = AutoSuspendManager(registry, launcher_factory, resource_manager)
    from supervisor.models_config import get_default_model_alias
    default_model_alias = get_default_model_alias(settings.startup_profile)
    gateway_sync = GatewaySync(registry, default_model_alias=default_model_alias)
    restart_manager = RestartManager(registry, launcher_factory, resource_manager)
    health_check_manager = HealthCheckManager(registry, restart_manager)

    # Start background tasks
    if settings.auto_suspend_enabled:
        await auto_suspend_manager.start()

    if settings.health_check_enabled:
        await health_check_manager.start()
        logger.info("Health check manager activated")

    # Auto-load models from config (or from named profile)
    from supervisor.models_config import get_autoload_models, get_profile_models
    from supervisor.models import ModelStartRequest

    if settings.startup_profile:
        autoload_models = get_profile_models(settings.startup_profile)
        logger.info(f"Loading profile: {settings.startup_profile} ({len(autoload_models)} models)")
    else:
        autoload_models = get_autoload_models()
    if autoload_models:
        # Separate models by backend for ordered loading:
        # 1. vLLM/SGLang first - they calculate gpu_memory_utilization as percentage of TOTAL memory
        # 2. CLIP next - Docker container that grabs memory immediately
        # 3. FLUX last - largest Docker container, must load after vLLM has reserved its memory
        vllm_sglang_models = [m for m in autoload_models if m.backend in ("vllm", "sglang")]
        clip_models = [m for m in autoload_models if m.backend == "clip"]
        flux_models = [m for m in autoload_models if m.backend == "flux"]
        species_models = [m for m in autoload_models if m.backend == "species"]
        face_models = [m for m in autoload_models if m.backend == "face"]

        logger.info(f"Auto-loading models: {len(vllm_sglang_models)} vLLM/SGLang, {len(clip_models)} CLIP, {len(species_models)} Species, {len(face_models)} Face, {len(flux_models)} FLUX")
        launched_model_ids = []  # Track models we actually launched

        # Helper function to wait for models to be ready
        async def wait_for_models(model_ids: list, next_phase: str):
            if not model_ids:
                logger.info(f"No models to wait for, proceeding to {next_phase}")
                return
            logger.info(f"Waiting for {len(model_ids)} model(s) to be ready before {next_phase}...")
            while True:
                models = await registry.list_all()
                models_by_id = {m.id: m for m in models}
                pending_models = []
                for model_id in model_ids:
                    model = models_by_id.get(model_id)
                    # Model is pending if: not in registry yet, OR still in STARTING state
                    if model is None or model.status == ModelStatus.STARTING:
                        pending_models.append(model_id)
                if not pending_models:
                    running_count = sum(1 for mid in model_ids
                                       if models_by_id.get(mid) and models_by_id[mid].status == ModelStatus.RUNNING)
                    failed_count = sum(1 for mid in model_ids
                                      if models_by_id.get(mid) and models_by_id[mid].status == ModelStatus.FAILED)
                    logger.info(f"All models ready ({running_count} running, {failed_count} failed), proceeding to {next_phase}")
                    break
                # Get names for logging
                pending_names = []
                for model_id in pending_models:
                    model = models_by_id.get(model_id) if isinstance(model_id, str) else None
                    if model:
                        pending_names.append(model.model_alias or model.model_name)
                    else:
                        pending_names.append(model_id if isinstance(model_id, str) else "unknown")
                logger.info(f"Still waiting for {len(pending_models)} model(s): {pending_names}")
                await asyncio.sleep(10)

        # Phase 1: Load vLLM/SGLang models first (sequential - wait for each to be ready)
        logger.info("=== PHASE 1: Loading vLLM/SGLang models ===")
        for i, model_config in enumerate(vllm_sglang_models):
            try:
                # Check if model already exists in database
                existing = await registry.get_by_alias(model_config.alias or model_config.name)
                if existing:
                    if existing.status in [ModelStatus.RUNNING, ModelStatus.STARTING]:
                        logger.info(f"Model {model_config.alias or model_config.name} already running, skipping")
                        continue
                    else:
                        # Remove old stopped/failed entry to avoid duplicates
                        logger.info(f"Removing old entry for {model_config.alias or model_config.name} (status: {existing.status})")
                        await registry.delete(existing.id)

                # Generate model ID
                model_id = registry.generate_id(model_config.name)

                # Use explicit memory if provided, otherwise estimate
                if model_config.memory_gb is not None:
                    memory_estimate = model_config.memory_gb
                    logger.info(f"Using explicit memory allocation: {memory_estimate}GB for {model_config.name}")
                else:
                    memory_estimate = resource_manager.estimate_model_memory(
                        model_config.name, model_config.quantization
                    )

                # Allocate port
                port = resource_manager.allocate_model(model_id, memory_estimate)

                # Create model config
                from supervisor.models import ModelConfig, Backend, ModelType
                config = ModelConfig(
                    model_name=model_config.name,
                    backend=Backend(model_config.backend),
                    model_type=ModelType(model_config.model_type),
                    model_alias=model_config.alias,
                    num_gpus=1,
                    quantization=model_config.quantization,
                    idle_timeout_minutes=model_config.idle_timeout_minutes,
                    auto_suspend_enabled=model_config.auto_suspend_enabled,
                    speculative_model=model_config.speculative_model,
                    num_speculative_tokens=model_config.num_speculative_tokens,
                    speculative_method=model_config.speculative_method,
                    extra_args=model_config.extra_args,
                    docker_image=model_config.docker_image,
                    env_vars=model_config.env_vars,
                    volumes=model_config.volumes,
                )

                # Launch model
                launcher = launcher_factory.get_launcher(config.backend)
                instance = await launcher.launch(config, model_id, port, memory_gb=memory_estimate)
                instance.memory_gb = memory_estimate

                # Save config for auto-restart
                instance.saved_config = {
                    "model_name": config.model_name,
                    "backend": config.backend.value,
                    "model_type": config.model_type.value,
                    "model_alias": config.model_alias,
                    "gpu_ids": instance.gpu_ids,
                    "port": port,
                    "quantization": config.quantization,
                    "auto_suspend_enabled": config.auto_suspend_enabled,
                    "idle_timeout_minutes": config.idle_timeout_minutes,
                    "speculative_model": config.speculative_model,
                    "num_speculative_tokens": config.num_speculative_tokens,
                    "speculative_method": config.speculative_method,
                    "extra_args": config.extra_args,
                    "docker_image": config.docker_image,
                    "env_vars": config.env_vars,
                    "volumes": config.volumes,
                }

                # Save to registry
                await registry.create(instance)
                launched_model_ids.append(instance.id)
                logger.info(f"Auto-loaded model: {model_config.alias or model_config.name} (id: {instance.id})")

                # Wait for this model to be RUNNING before starting the next one.
                # This prevents memory race conditions where later models grab memory
                # while earlier models are still in torch.compile/KV cache allocation.
                if i < len(vllm_sglang_models) - 1:  # Don't wait after the last model
                    logger.info(f"Waiting for {model_config.alias or model_config.name} to be ready before next model...")
                    await wait_for_models([instance.id], f"launch model {i+2}/{len(vllm_sglang_models)}")

            except Exception as e:
                logger.error(f"Failed to auto-load model {model_config.name}: {e}")
                # Continue with other models

        logger.info(f"=== PHASE 1 COMPLETE: Launched {len(launched_model_ids)} vLLM/SGLang models ===")

        # Wait for vLLM/SGLang models before loading CLIP
        logger.info("=== Waiting for vLLM/SGLang models to be RUNNING ===")
        await wait_for_models(launched_model_ids, "load CLIP")

        # Phase 2: Load CLIP models
        logger.info("=== PHASE 2: Loading CLIP models ===")
        clip_model_ids = []
        for model_config in clip_models:
            try:
                existing = await registry.get_by_alias(model_config.alias or model_config.name)
                if existing:
                    if existing.status in [ModelStatus.RUNNING, ModelStatus.STARTING]:
                        logger.info(f"Model {model_config.alias or model_config.name} already running, skipping")
                        continue
                    else:
                        await registry.delete(existing.id)

                model_id = registry.generate_id(model_config.name)
                memory_estimate = model_config.memory_gb if model_config.memory_gb is not None else 5.0

                port = resource_manager.allocate_model(model_id, memory_estimate)

                from supervisor.models import ModelConfig, Backend, ModelType
                config = ModelConfig(
                    model_name=model_config.name,
                    backend=Backend(model_config.backend),
                    model_type=ModelType(model_config.model_type),
                    model_alias=model_config.alias,
                    num_gpus=1,
                    quantization=model_config.quantization,
                    idle_timeout_minutes=model_config.idle_timeout_minutes,
                    auto_suspend_enabled=model_config.auto_suspend_enabled,
                    extra_args=model_config.extra_args,
                    docker_image=model_config.docker_image,
                    env_vars=model_config.env_vars,
                    volumes=model_config.volumes,
                )

                launcher = launcher_factory.get_launcher(config.backend)
                instance = await launcher.launch(config, model_id, port, memory_gb=memory_estimate)
                instance.memory_gb = memory_estimate
                instance.saved_config = {
                    "model_name": config.model_name,
                    "backend": config.backend.value,
                    "model_type": config.model_type.value,
                    "model_alias": config.model_alias,
                    "gpu_ids": instance.gpu_ids,
                    "port": port,
                    "quantization": config.quantization,
                    "auto_suspend_enabled": config.auto_suspend_enabled,
                    "idle_timeout_minutes": config.idle_timeout_minutes,
                    "extra_args": config.extra_args,
                    "docker_image": config.docker_image,
                    "env_vars": config.env_vars,
                    "volumes": config.volumes,
                }

                await registry.create(instance)
                clip_model_ids.append(instance.id)
                logger.info(f"Auto-loaded CLIP model: {model_config.alias or model_config.name} (id: {instance.id})")

            except Exception as e:
                logger.error(f"Failed to auto-load CLIP model {model_config.name}: {e}")

        logger.info(f"=== PHASE 2 COMPLETE: Launched {len(clip_model_ids)} CLIP models ===")

        # Wait for CLIP models before loading Species/FLUX
        logger.info("=== Waiting for CLIP models to be RUNNING ===")
        await wait_for_models(clip_model_ids, "load Species/FLUX")

        # Phase 2.5: Load Species detection models
        logger.info("=== PHASE 2.5: Loading Species detection models ===")
        species_model_ids = []
        for model_config in species_models:
            try:
                existing = await registry.get_by_alias(model_config.alias or model_config.name)
                if existing:
                    if existing.status in [ModelStatus.RUNNING, ModelStatus.STARTING]:
                        logger.info(f"Model {model_config.alias or model_config.name} already running, skipping")
                        continue
                    else:
                        await registry.delete(existing.id)

                model_id = registry.generate_id(model_config.name)
                memory_estimate = model_config.memory_gb if model_config.memory_gb is not None else 5.0

                port = resource_manager.allocate_model(model_id, memory_estimate)

                from supervisor.models import ModelConfig, Backend, ModelType
                config = ModelConfig(
                    model_name=model_config.name,
                    backend=Backend(model_config.backend),
                    model_type=ModelType(model_config.model_type),
                    model_alias=model_config.alias,
                    num_gpus=1,
                    quantization=model_config.quantization,
                    idle_timeout_minutes=model_config.idle_timeout_minutes,
                    auto_suspend_enabled=model_config.auto_suspend_enabled,
                    extra_args=model_config.extra_args,
                    docker_image=model_config.docker_image,
                    env_vars=model_config.env_vars,
                    volumes=model_config.volumes,
                )

                launcher = launcher_factory.get_launcher(config.backend)
                instance = await launcher.launch(config, model_id, port, memory_gb=memory_estimate)
                instance.memory_gb = memory_estimate
                instance.saved_config = {
                    "model_name": config.model_name,
                    "backend": config.backend.value,
                    "model_type": config.model_type.value,
                    "model_alias": config.model_alias,
                    "gpu_ids": instance.gpu_ids,
                    "port": port,
                    "quantization": config.quantization,
                    "auto_suspend_enabled": config.auto_suspend_enabled,
                    "idle_timeout_minutes": config.idle_timeout_minutes,
                    "extra_args": config.extra_args,
                    "docker_image": config.docker_image,
                    "env_vars": config.env_vars,
                    "volumes": config.volumes,
                }

                await registry.create(instance)
                species_model_ids.append(model_id)
                logger.info(f"Auto-loaded Species model: {model_config.alias or model_config.name}")

            except Exception as e:
                logger.error(f"Failed to auto-load Species model {model_config.name}: {e}")

        logger.info(f"=== PHASE 2.5 COMPLETE: Launched {len(species_model_ids)} Species models ===")

        # Phase 2.6: Load Face recognition models
        logger.info("=== PHASE 2.6: Loading Face recognition models ===")
        face_model_ids = []
        for model_config in face_models:
            try:
                existing = await registry.get_by_alias(model_config.alias or model_config.name)
                if existing:
                    if existing.status in [ModelStatus.RUNNING, ModelStatus.STARTING]:
                        logger.info(f"Model {model_config.alias or model_config.name} already running, skipping")
                        continue
                    else:
                        await registry.delete(existing.id)

                model_id = registry.generate_id(model_config.name)
                memory_estimate = model_config.memory_gb if model_config.memory_gb is not None else 1.0

                port = resource_manager.allocate_model(model_id, memory_estimate)

                from supervisor.models import ModelConfig, Backend, ModelType
                config = ModelConfig(
                    model_name=model_config.name,
                    backend=Backend(model_config.backend),
                    model_type=ModelType(model_config.model_type),
                    model_alias=model_config.alias,
                    num_gpus=1,
                    quantization=model_config.quantization,
                    idle_timeout_minutes=model_config.idle_timeout_minutes,
                    auto_suspend_enabled=model_config.auto_suspend_enabled,
                    extra_args=model_config.extra_args,
                    docker_image=model_config.docker_image,
                    env_vars=model_config.env_vars,
                    volumes=model_config.volumes,
                )

                launcher = launcher_factory.get_launcher(config.backend)
                instance = await launcher.launch(config, model_id, port, memory_gb=memory_estimate)
                instance.memory_gb = memory_estimate
                instance.saved_config = {
                    "model_name": config.model_name,
                    "backend": config.backend.value,
                    "model_type": config.model_type.value,
                    "model_alias": config.model_alias,
                    "gpu_ids": instance.gpu_ids,
                    "port": port,
                    "quantization": config.quantization,
                    "auto_suspend_enabled": config.auto_suspend_enabled,
                    "idle_timeout_minutes": config.idle_timeout_minutes,
                    "extra_args": config.extra_args,
                    "docker_image": config.docker_image,
                    "env_vars": config.env_vars,
                    "volumes": config.volumes,
                }

                await registry.create(instance)
                face_model_ids.append(model_id)
                logger.info(f"Auto-loaded Face model: {model_config.alias or model_config.name}")

            except Exception as e:
                logger.error(f"Failed to auto-load Face model {model_config.name}: {e}")

        logger.info(f"=== PHASE 2.6 COMPLETE: Launched {len(face_model_ids)} Face models ===")

        # Phase 3: Load FLUX models (last due to large memory footprint)
        logger.info("=== PHASE 3: Loading FLUX models ===")
        if flux_models:
            logger.info(f"Loading {len(flux_models)} FLUX model(s) after other models are ready...")
            for model_config in flux_models:
                try:
                    existing = await registry.get_by_alias(model_config.alias or model_config.name)
                    if existing:
                        if existing.status in [ModelStatus.RUNNING, ModelStatus.STARTING]:
                            logger.info(f"Model {model_config.alias or model_config.name} already running, skipping")
                            continue
                        else:
                            await registry.delete(existing.id)

                    model_id = registry.generate_id(model_config.name)
                    memory_estimate = model_config.memory_gb if model_config.memory_gb is not None else 35.0

                    port = resource_manager.allocate_model(model_id, memory_estimate)

                    from supervisor.models import ModelConfig, Backend, ModelType
                    config = ModelConfig(
                        model_name=model_config.name,
                        backend=Backend(model_config.backend),
                        model_type=ModelType(model_config.model_type),
                        model_alias=model_config.alias,
                        num_gpus=1,
                        quantization=model_config.quantization,
                        idle_timeout_minutes=model_config.idle_timeout_minutes,
                        auto_suspend_enabled=model_config.auto_suspend_enabled,
                        extra_args=model_config.extra_args,
                        docker_image=model_config.docker_image,
                        env_vars=model_config.env_vars,
                        volumes=model_config.volumes,
                    )

                    launcher = launcher_factory.get_launcher(config.backend)
                    instance = await launcher.launch(config, model_id, port, memory_gb=memory_estimate)
                    instance.memory_gb = memory_estimate
                    instance.saved_config = {
                        "model_name": config.model_name,
                        "backend": config.backend.value,
                        "model_type": config.model_type.value,
                        "model_alias": config.model_alias,
                        "gpu_ids": instance.gpu_ids,
                        "port": port,
                        "quantization": config.quantization,
                        "auto_suspend_enabled": config.auto_suspend_enabled,
                        "idle_timeout_minutes": config.idle_timeout_minutes,
                        "extra_args": config.extra_args,
                        "docker_image": config.docker_image,
                        "env_vars": config.env_vars,
                        "volumes": config.volumes,
                    }

                    await registry.create(instance)
                    logger.info(f"Auto-loaded FLUX model: {model_config.alias or model_config.name}")

                except Exception as e:
                    logger.error(f"Failed to auto-load FLUX model {model_config.name}: {e}")

    # Start gateway sync after models are loaded and running
    await gateway_sync.start()
    logger.info("Gateway sync activated")

    # Mark startup as complete - health endpoint will now return 200
    global startup_complete
    startup_complete = True
    logger.info(f"Supervisor started on {settings.host}:{settings.port} - startup complete")

    yield

    # Cleanup
    logger.info("Shutting down Supervisor...")
    if health_check_manager:
        await health_check_manager.stop()
    await auto_suspend_manager.stop()
    await gateway_sync.stop()


# Create FastAPI app
app = FastAPI(
    title="Sparkstation Supervisor",
    description="DGX Spark LLM model lifecycle management",
    version="0.1.0",
    lifespan=lifespan,
)


# Health check
@app.get("/health")
async def health_check():
    """
    Supervisor health check.

    Returns 503 during startup (while models are loading).
    Returns 200 only after startup is complete and gateway sync is ready.
    """
    if not startup_complete:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "message": "Supervisor is loading models, not ready yet",
                "timestamp": datetime.now().isoformat()
            }
        )

    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Prometheus metrics
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    # Update metrics before returning
    if resource_manager:
        status = resource_manager.get_resource_status()
        metrics.unified_memory_used_bytes.set(status["unified_memory_used_gb"] * 1024**3)
        metrics.unified_memory_limit_bytes.set(status["unified_memory_limit_gb"] * 1024**3)
        metrics.gpu_temperature_celsius.set(status["gpu_temperature_c"])
        metrics.gpu_power_draw_watts.set(status["gpu_power_draw_w"])
        metrics.resident_models_count.set(status["resident_models_count"])

    if registry:
        suspended = await registry.list_suspended()
        metrics.suspended_models_count.set(len(suspended))

        # Update per-model metrics
        all_models = await registry.list_all()
        # Status mapping: string -> numeric for Prometheus
        status_map = {
            "stopped": 0,
            "starting": 1,
            "running": 2,
            "suspended": 3,
            "failed": 4,
        }
        for model in all_models:
            # Set model status
            # Handle both enum and string status values
            if hasattr(model.status, 'value'):
                status_str = model.status.value
            else:
                status_str = str(model.status)
            status_value = status_map.get(status_str.lower(), 0)
            metrics.model_status.labels(
                model_name=model.model_alias or model.model_name,
                model_id=model.id
            ).set(status_value)

            # Set model memory usage
            if model.memory_gb:
                metrics.model_memory_used_bytes.labels(
                    model_name=model.model_alias or model.model_name
                ).set(model.memory_gb * 1024**3)

            # Set last request timestamp
            if model.last_request_time:
                timestamp = model.last_request_time.timestamp()
                metrics.model_last_request_timestamp.labels(
                    model_name=model.model_alias or model.model_name
                ).set(timestamp)

    return metrics.metrics_response()


# Models endpoints
@app.get("/models", response_model=List[LiteLLMModelFormat])
async def list_models():
    """
    List running models in LiteLLM-compatible format.

    CRITICAL: Only includes RUNNING models (not suspended).
    Suspended models are handled by auto-resume middleware.
    """
    if gateway_sync is None:
        raise HTTPException(status_code=503, detail="Gateway sync not initialized")

    models = await gateway_sync.get_models_for_litellm()
    return models


@app.get("/models/detailed")
async def list_models_detailed():
    """Get detailed model information (for dashboard)."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    all_models = await registry.list_all()
    detailed = []

    for model in all_models:
        idle_seconds = None
        if model.last_request_time:
            idle_seconds = (datetime.now() - model.last_request_time).total_seconds()

        detailed.append(
            {
                "id": model.id,
                "model_name": model.model_name,
                "alias": model.model_alias,
                "backend": model.backend,
                "model_type": model.model_type,
                "status": model.status,
                "health_status": model.health_status,
                "port": model.port,
                "memory_gb": model.memory_gb,
                "is_default": (model.model_alias or model.model_name) == default_model_alias,
                "last_request_time": model.last_request_time.isoformat()
                if model.last_request_time
                else None,
                "idle_seconds": idle_seconds,
                "auto_suspend_enabled": model.auto_suspend_enabled,
                "idle_timeout_minutes": model.idle_timeout_minutes,
            }
        )

    return {"models": detailed}


@app.post("/models/start", response_model=ModelStartResponse, dependencies=[Depends(require_api_key)])
async def start_model(request: ModelStartRequest):
    """Start a new model server."""
    if not all([registry, resource_manager, launcher_factory]):
        raise HTTPException(status_code=503, detail="Supervisor not fully initialized")

    logger.info(f"Starting model: {request.model_name} ({request.backend})")

    model_id = None
    try:
        # Check if alias already exists
        if request.model_alias:
            existing = await registry.get_by_alias(request.model_alias)
            if existing:
                raise ModelAlreadyExistsError(request.model_alias)

        # Generate model ID
        model_id = ModelRegistry.generate_id(request.model_name)

        # Estimate memory
        memory_estimate = resource_manager.estimate_model_memory(
            request.model_name, request.quantization
        )

        # Allocate resources
        try:
            port = resource_manager.allocate_model(model_id, memory_estimate)
        except ResourceError as e:
            current = resource_manager.get_unified_memory_usage()
            raise InsufficientResourcesError(str(e), current, resource_manager.hard_limit_gb)

        # Create config
        config = ModelConfig(
            model_name=request.model_name,
            backend=request.backend,
            model_type=request.model_type,
            model_alias=request.model_alias,
            num_gpus=request.num_gpus,
            quantization=request.quantization,
            idle_timeout_minutes=request.idle_timeout_minutes,
            auto_suspend_enabled=request.auto_suspend_enabled,
            speculative_model=request.speculative_model,
            num_speculative_tokens=request.num_speculative_tokens,
            speculative_method=request.speculative_method,
            extra_args=request.extra_args,
        )

        # Launch model
        try:
            launcher = launcher_factory.get_launcher(request.backend)
            instance = await launcher.launch(config, model_id, port, memory_gb=memory_estimate)
            instance.memory_gb = memory_estimate

            # CRITICAL: Save config for auto-restart and resume
            # This MUST be set here so restart_manager can recover failed models
            instance.saved_config = {
                "model_name": config.model_name,
                "backend": config.backend.value,
                "model_type": config.model_type.value,
                "model_alias": config.model_alias,
                "gpu_ids": instance.gpu_ids,
                "port": port,
                "quantization": config.quantization,
                "auto_suspend_enabled": config.auto_suspend_enabled,
                "idle_timeout_minutes": config.idle_timeout_minutes,
                "speculative_model": config.speculative_model,
                "num_speculative_tokens": config.num_speculative_tokens,
                "speculative_method": config.speculative_method,
                "extra_args": config.extra_args,
            }
        except Exception as e:
            raise ModelLaunchError(request.backend.value, str(e))

        # Save to registry
        try:
            await registry.create(instance)
        except Exception as e:
            logger.error(f"Failed to save to registry: {e}")
            # Try to stop the launched model
            await launcher.stop(instance)
            raise

        # Trigger gateway sync
        try:
            await gateway_sync.sync_models()
        except Exception as e:
            logger.warning(f"Gateway sync failed: {e}")
            # Don't fail the whole operation if sync fails

        logger.info(f"Model {model_id} started on port {port}")

        return ModelStartResponse(
            model_id=instance.id,
            model_name=instance.model_name,
            backend=instance.backend.value if hasattr(instance.backend, 'value') else instance.backend,
            model_type=instance.model_type.value if hasattr(instance.model_type, 'value') else instance.model_type,
            status=instance.status.value if hasattr(instance.status, 'value') else instance.status,
            port=instance.port,
            gpu_ids=instance.gpu_ids,
            base_url=instance.base_url,
            started_at=instance.started_at,
            idle_timeout_minutes=instance.idle_timeout_minutes,
            auto_suspend_enabled=instance.auto_suspend_enabled,
        )

    except (ModelAlreadyExistsError, InsufficientResourcesError, ModelLaunchError) as e:
        logger.error(f"Failed to start model: {e}")
        # Cleanup on failure
        if model_id and resource_manager:
            resource_manager.release_model(model_id, full_release=True)
        raise e.to_http_exception()

    except Exception as e:
        logger.error(f"Unexpected error starting model: {e}", exc_info=True)
        # Cleanup on failure
        if model_id and resource_manager:
            resource_manager.release_model(model_id, full_release=True)
        raise handle_exception(e)


@app.post("/models/{model_id}/stop", dependencies=[Depends(require_api_key)])
async def stop_model(model_id: str):
    """Stop a running model."""
    if not all([registry, resource_manager, launcher_factory]):
        raise HTTPException(status_code=503, detail="Supervisor not initialized")

    model = await registry.get(model_id)
    if model is None:
        raise ModelNotFoundError(model_id).to_http_exception()

    logger.info(f"Stopping model: {model_id}")

    try:
        launcher = launcher_factory.get_launcher(model.backend)
        stopped = await launcher.stop(model)

        if stopped:
            model.status = ModelStatus.STOPPED
            model.stopped_at = datetime.now()
            model.pid = None
            await registry.update(model)

            # Release all resources
            resource_manager.release_model(model_id, full_release=True)

            # Trigger gateway sync
            try:
                await gateway_sync.sync_models()
            except Exception as e:
                logger.warning(f"Gateway sync failed: {e}")

            return {"model_id": model_id, "status": "stopped", "stopped_at": model.stopped_at.isoformat()}
        else:
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to stop model", "detail": "Backend stop command failed"}
            )

    except ModelNotFoundError as e:
        raise e.to_http_exception()
    except Exception as e:
        logger.error(f"Failed to stop model {model_id}: {e}", exc_info=True)
        raise handle_exception(e)


@app.post("/models/{model_id}/suspend", dependencies=[Depends(require_api_key)])
async def suspend_model(model_id: str):
    """Manually suspend a model."""
    if auto_suspend_manager is None:
        raise HTTPException(status_code=503, detail="Auto-suspend not initialized")

    try:
        model = await registry.get(model_id)
        if model is None:
            raise ModelNotFoundError(model_id)

        if model.status != ModelStatus.RUNNING:
            raise ModelNotRunningError(model_id, model.status.value)

        await auto_suspend_manager.suspend_model(model_id)

        model = await registry.get(model_id)  # Refresh
        return {
            "model_id": model_id,
            "status": model.status.value,
            "suspended_at": model.stopped_at.isoformat() if model.stopped_at else None,
            "port_reserved": model.port,
        }

    except (ModelNotFoundError, ModelNotRunningError) as e:
        raise e.to_http_exception()
    except Exception as e:
        logger.error(f"Failed to suspend model {model_id}: {e}", exc_info=True)
        raise handle_exception(e)


@app.post("/models/{model_id}/resume", dependencies=[Depends(require_api_key)])
async def resume_model(model_id: str):
    """Resume a suspended model."""
    if auto_suspend_manager is None:
        raise HTTPException(status_code=503, detail="Auto-suspend not initialized")

    try:
        model = await registry.get(model_id)
        if model is None:
            raise ModelNotFoundError(model_id)

        if model.status != ModelStatus.SUSPENDED:
            raise ModelNotSuspendedError(model_id, model.status.value)

        resumed = await auto_suspend_manager.resume_model(model_id)

        if not resumed:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to resume model",
                    "detail": "Resume operation returned false",
                    "suggestion": "Check logs for details, verify sufficient resources available"
                }
            )

        model = await registry.get(model_id)  # Refresh
        startup_time = (datetime.now() - model.started_at).total_seconds() if model.started_at else None

        return {
            "model_id": model_id,
            "status": model.status.value,
            "resumed_at": model.started_at.isoformat() if model.started_at else None,
            "startup_time_seconds": startup_time,
            "base_url": model.base_url,
        }

    except (ModelNotFoundError, ModelNotSuspendedError) as e:
        raise e.to_http_exception()
    except Exception as e:
        logger.error(f"Failed to resume model {model_id}: {e}", exc_info=True)
        raise handle_exception(e)


@app.get("/models/{model_id}/status", response_model=ModelStatusResponse)
async def get_model_status(model_id: str):
    """Get detailed model status."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    model = await registry.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    uptime_seconds = None
    idle_seconds = None
    seconds_until_suspend = None

    if model.started_at and model.status == ModelStatus.RUNNING:
        uptime_seconds = (datetime.now() - model.started_at).total_seconds()

    if model.last_request_time:
        idle_seconds = (datetime.now() - model.last_request_time).total_seconds()
        if model.auto_suspend_enabled and model.idle_timeout_minutes > 0:
            seconds_until_suspend = max(
                0, (model.idle_timeout_minutes * 60) - idle_seconds
            )

    return ModelStatusResponse(
        model_id=model.id,
        model_name=model.model_name,
        status=model.status.value,
        health_status=model.health_status.value,
        uptime_seconds=uptime_seconds,
        last_health_check=model.last_health_check,
        last_request_time=model.last_request_time,
        idle_seconds=idle_seconds,
        idle_timeout_minutes=model.idle_timeout_minutes,
        auto_suspend_enabled=model.auto_suspend_enabled,
        seconds_until_suspend=seconds_until_suspend,
    )


@app.get("/resources", response_model=ResourceStatus)
async def get_resources():
    """Get system resource status."""
    if resource_manager is None:
        raise HTTPException(status_code=503, detail="Resource manager not initialized")

    status = resource_manager.get_resource_status()
    return ResourceStatus(**status)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "supervisor.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
