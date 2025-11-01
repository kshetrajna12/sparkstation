"""
Sparkstation Supervisor - Main FastAPI application.

DGX Spark-optimized LLM model lifecycle management.
"""
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

from supervisor.config import settings
from supervisor.models import (
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global registry, resource_manager, launcher_factory, auto_suspend_manager, gateway_sync, health_check_manager, restart_manager

    logger.info("Initializing Sparkstation Supervisor...")

    # Initialize components
    registry = ModelRegistry()
    await registry.initialize()

    resource_manager = ResourceManager()
    launcher_factory = LauncherFactory()
    auto_suspend_manager = AutoSuspendManager(registry, launcher_factory, resource_manager)
    gateway_sync = GatewaySync(registry)
    restart_manager = RestartManager(registry, launcher_factory, resource_manager)
    health_check_manager = HealthCheckManager(registry, restart_manager)

    # Start background tasks
    if settings.auto_suspend_enabled:
        await auto_suspend_manager.start()

    await gateway_sync.start()

    if settings.health_check_enabled:
        await health_check_manager.start()
        logger.info("Health check manager activated")

    logger.info(f"Supervisor started on {settings.host}:{settings.port}")

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
    """Supervisor health check."""
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
                "status": model.status,
                "health_status": model.health_status,
                "port": model.port,
                "memory_gb": model.memory_gb,
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
            model_alias=request.model_alias,
            num_gpus=request.num_gpus,
            quantization=request.quantization,
            idle_timeout_minutes=request.idle_timeout_minutes,
            auto_suspend_enabled=request.auto_suspend_enabled,
            extra_args=request.extra_args,
        )

        # Launch model
        try:
            launcher = launcher_factory.get_launcher(request.backend)
            instance = await launcher.launch(config, model_id, port)
            instance.memory_gb = memory_estimate

            # CRITICAL: Save config for auto-restart and resume
            # This MUST be set here so restart_manager can recover failed models
            instance.saved_config = {
                "model_name": config.model_name,
                "backend": config.backend.value,
                "model_alias": config.model_alias,
                "gpu_ids": instance.gpu_ids,
                "port": port,
                "quantization": config.quantization,
                "auto_suspend_enabled": config.auto_suspend_enabled,
                "idle_timeout_minutes": config.idle_timeout_minutes,
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
