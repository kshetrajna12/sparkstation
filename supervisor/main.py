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
    build_saved_config,
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
vision_model_alias: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global registry, resource_manager, launcher_factory, auto_suspend_manager, gateway_sync, health_check_manager, restart_manager, default_model_alias, vision_model_alias

    logger.info("Initializing Sparkstation Supervisor...")

    # Initialize components
    registry = ModelRegistry()
    await registry.initialize()

    # Reconcile database state with reality (fix stale/orphaned entries)
    reconcile_summary = await registry.reconcile_state()
    if reconcile_summary["fixed_stopped"] or reconcile_summary["fixed_running"] or reconcile_summary["removed_orphaned"]:
        logger.warning(f"Database reconciliation found inconsistencies: {reconcile_summary}")

    # Purge stale DB entries. Rule: keep iff (status is RUNNING or STARTING) AND
    # the container is actually alive. Everything else is stale (SUSPENDED, STOPPED,
    # FAILED, or status-says-up-but-container-dead) and is removed so the autoload
    # / API paths can recreate cleanly.
    #
    # Previously this purged ALL non-RUNNING entries, including STARTING — which
    # killed any model still mid-cold-load on a supervisor restart, even though
    # its container was healthily booting. The follow-up autoload then collided
    # on the orphaned container's port. Keep STARTING + live container; the
    # health-check loop transitions it to RUNNING within ~10s.
    import subprocess as _sp
    from supervisor.cluster_helpers import merged_env as _merged_env
    all_models = await registry.list_all()
    for model in all_models:
        container_alive = False
        if model.container_id:
            # Talk to the daemon on THIS model's host, not always the local one.
            # Without this, chat on worker1 gets "container_alive: False"
            # (its container_id doesn't exist locally), the row gets purged,
            # and autoload double-launches it on the wrong host with port collisions.
            result = _sp.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", model.container_id],
                capture_output=True, text=True,
                env=_merged_env(model.host or "primary"),
                timeout=15,  # SSH transport is slower than the local socket
            )
            container_alive = result.returncode == 0 and result.stdout.strip() == "true"
        if model.status in [ModelStatus.RUNNING, ModelStatus.STARTING] and container_alive:
            continue
        logger.info(
            f"Purging stale DB entry: {model.model_alias or model.model_name} "
            f"(host={model.host or 'primary'}, status: {model.status}, container_alive: {container_alive})"
        )
        await registry.delete(model.id)

    resource_manager = ResourceManager()
    # Re-register any RUNNING or STARTING models surviving from the previous
    # supervisor process so resource_manager knows which ports/memory are
    # already in use. STARTING is included because mid-cold-load models still
    # hold a port and memory reservation — without this, /models/start (and
    # autoload of new aliases) can allocate a port that an existing container
    # is already bound to, causing the next launch to collide on Docker bind.
    for _surviving in await registry.list_all():
        if _surviving.status in [ModelStatus.RUNNING, ModelStatus.STARTING]:
            if _surviving.port:
                resource_manager.allocated_ports[_surviving.id] = _surviving.port
            # Memory only counts against the primary Spark's budget (same rule
            # as allocate_model) — counting worker-host models here would
            # double-book the primary and cause spurious InsufficientResources.
            if _surviving.memory_gb and (_surviving.host or "primary") == "primary":
                resource_manager.model_memory_usage[_surviving.id] = _surviving.memory_gb
    launcher_factory = LauncherFactory()
    auto_suspend_manager = AutoSuspendManager(registry, launcher_factory, resource_manager)
    from supervisor.models_config import get_default_model_alias, get_vision_model_alias
    default_model_alias = get_default_model_alias(settings.startup_profile)
    vision_model_alias = get_vision_model_alias(settings.startup_profile)
    gateway_sync = GatewaySync(
        registry,
        default_model_alias=default_model_alias,
        vision_model_alias=vision_model_alias,
    )
    restart_manager = RestartManager(registry, launcher_factory, resource_manager)
    health_check_manager = HealthCheckManager(registry, restart_manager, gateway_sync=gateway_sync)

    # Start background tasks
    if settings.auto_suspend_enabled:
        await auto_suspend_manager.start()

    if settings.health_check_enabled:
        await health_check_manager.start()
        logger.info("Health check manager activated")

    # RestartManager runs its own watcher independent of health checks, so
    # crashes marked FAILED via reconcile / container-exit / any other path
    # get recovered too — not just the health-check-counter path.
    await restart_manager.start()

    # Auto-load models from config (or from named profile)
    from supervisor.models_config import get_autoload_models, get_profile_models
    from supervisor.models import ModelStartRequest

    # Log WHERE the resolved profile came from — two days were lost to a
    # systemd unit silently overriding the profile; make the source explicit.
    if settings.startup_profile:
        autoload_models = get_profile_models(settings.startup_profile)
        logger.info(
            f"Loading profile: {settings.startup_profile} "
            f"(source: STARTUP_PROFILE env / --profile flag; {len(autoload_models)} models)"
        )
    else:
        from supervisor.models_config import load_models_config
        _default_profile = load_models_config().default_profile
        autoload_models = get_autoload_models()
        logger.info(
            f"Loading profile: {_default_profile or '(none — autoload list)'} "
            f"(source: models.yaml default_profile; {len(autoload_models)} models)"
        )
    async def _autoload():
        """Autoload configured models. Runs as a BACKGROUND task so the
        supervisor binds its HTTP port immediately — `sparkstation start`
        and the gateway no longer blind-wait behind multi-minute (DSV4:
        potentially ~45 min) model loads. Progress is observable via
        /models/detailed; gateway_sync publishes models as they come up."""
        if autoload_models:
            # Separate models by backend for ordered loading:
            # 1. vLLM/SGLang first - they calculate gpu_memory_utilization as percentage of TOTAL memory
            # 2. CLIP next - Docker container that grabs memory immediately
            # 3. FLUX last - largest Docker container, must load after vLLM has reserved its memory
            # DSpark (2-node script stack: DSV4, GLM-5.3) joins phase 1 and must
            # finish before any other model GPU-profiles ON ITS HOSTS — but
            # models on OTHER hosts (e.g. primary aux while the stack owns the
            # workers) launch first so a ~20 min stack boot doesn't hold up
            # 1-minute loads elsewhere (2026-08-28: deep profile moved all
            # non-chat models to primary).
            # voicechat joins phase 1 LAST: its Nano engine sizes off free
            # memory at init (0.42 of total), so every other model on its host
            # must have reserved first.
            vllm_sglang_models = [m for m in autoload_models if m.backend in ("dspark", "vllm", "sglang", "voicechat")]
            _dspark_hosts = {m.host for m in vllm_sglang_models if m.backend == "dspark"}

            # voicechat's Nano engine sizes off FREE memory at init, so models
            # on ITS host must reserve first — but only its host: when nothing
            # else runs there (e.g. voice profile: chat on worker1, voicechat
            # alone on worker2) it launches immediately instead of waiting out
            # a ~12 min chat-stack boot (2026-08-29, same host-scoping as the
            # dspark rule above). CAVEAT: dspark stacks may implicitly span
            # hosts beyond their configured `host` (e.g. GLM TP2's second rank
            # on worker2) — if a profile ever pairs such a stack with
            # voicechat, this check under-counts; declare the extra host on
            # the spec or keep voicechat out of that profile.
            _voice_hosts = {m.host for m in vllm_sglang_models if m.backend == "voicechat"}
            _shared_voice_host = any(
                m.host in _voice_hosts
                for m in vllm_sglang_models
                if m.backend != "voicechat"
            )

            def _phase1_key(m):
                if m.backend == "voicechat":
                    return 3 if _shared_voice_host else 0
                if m.backend == "dspark":
                    return 1
                return 2 if m.host in _dspark_hosts else 0

            vllm_sglang_models.sort(key=_phase1_key)
            clip_models = [m for m in autoload_models if m.backend == "clip"]
            flux_models = [m for m in autoload_models if m.backend == "flux"]
            species_models = [m for m in autoload_models if m.backend == "species"]
            face_models = [m for m in autoload_models if m.backend == "face"]

            # A backend missing from every bucket above would be dropped without
            # a trace (how voicechat's first autoload was lost, 2026-08-25).
            _bucketed = {id(m) for bucket in (vllm_sglang_models, clip_models, flux_models, species_models, face_models) for m in bucket}
            for m in autoload_models:
                if id(m) not in _bucketed:
                    logger.error(f"Autoload has no phase for backend {m.backend!r} — {m.alias or m.name} will NOT be loaded")

            logger.info(f"Auto-loading models: {len(vllm_sglang_models)} vLLM/SGLang/DSpark/Voicechat, {len(clip_models)} CLIP, {len(species_models)} Species, {len(face_models)} Face, {len(flux_models)} FLUX")
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
                    port = resource_manager.allocate_model(model_id, memory_estimate, host=model_config.host)

                    # Create model config
                    from supervisor.models import ModelConfig, Backend, ModelType
                    config = ModelConfig(
                        model_name=model_config.name,
                        backend=Backend(model_config.backend),
                        model_type=ModelType(model_config.model_type),
                        model_alias=model_config.alias,
                        host=model_config.host,
                        num_gpus=1,
                        quantization=model_config.quantization,
                        idle_timeout_minutes=model_config.idle_timeout_minutes,
                        auto_suspend_enabled=model_config.auto_suspend_enabled,
                        speculative_model=model_config.speculative_model,
                        num_speculative_tokens=model_config.num_speculative_tokens,
                        speculative_method=model_config.speculative_method,
                        speculative_extra=model_config.speculative_extra,
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
                    instance.saved_config = build_saved_config(config, instance.gpu_ids, port, memory_estimate)

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

                    port = resource_manager.allocate_model(model_id, memory_estimate, host=model_config.host)

                    from supervisor.models import ModelConfig, Backend, ModelType
                    config = ModelConfig(
                        model_name=model_config.name,
                        backend=Backend(model_config.backend),
                        model_type=ModelType(model_config.model_type),
                        model_alias=model_config.alias,
                        host=model_config.host,
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
                    instance.saved_config = build_saved_config(config, instance.gpu_ids, port, memory_estimate)

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

                    port = resource_manager.allocate_model(model_id, memory_estimate, host=model_config.host)

                    from supervisor.models import ModelConfig, Backend, ModelType
                    config = ModelConfig(
                        model_name=model_config.name,
                        backend=Backend(model_config.backend),
                        model_type=ModelType(model_config.model_type),
                        model_alias=model_config.alias,
                        host=model_config.host,
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
                    instance.saved_config = build_saved_config(config, instance.gpu_ids, port, memory_estimate)

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

                    port = resource_manager.allocate_model(model_id, memory_estimate, host=model_config.host)

                    from supervisor.models import ModelConfig, Backend, ModelType
                    config = ModelConfig(
                        model_name=model_config.name,
                        backend=Backend(model_config.backend),
                        model_type=ModelType(model_config.model_type),
                        model_alias=model_config.alias,
                        host=model_config.host,
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
                    instance.saved_config = build_saved_config(config, instance.gpu_ids, port, memory_estimate)

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

                        port = resource_manager.allocate_model(model_id, memory_estimate, host=model_config.host)

                        from supervisor.models import ModelConfig, Backend, ModelType
                        config = ModelConfig(
                            model_name=model_config.name,
                            backend=Backend(model_config.backend),
                            model_type=ModelType(model_config.model_type),
                            model_alias=model_config.alias,
                            host=model_config.host,
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
                        instance.saved_config = build_saved_config(config, instance.gpu_ids, port, memory_estimate)

                        await registry.create(instance)
                        logger.info(f"Auto-loaded FLUX model: {model_config.alias or model_config.name}")

                    except Exception as e:
                        logger.error(f"Failed to auto-load FLUX model {model_config.name}: {e}")


    autoload_task = asyncio.create_task(_autoload())
    app.state.autoload_task = autoload_task

    # Gateway sync starts alongside autoload (not after): each sync pass
    # publishes whatever is RUNNING, so models appear in the gateway
    # incrementally as they come up.
    await gateway_sync.start()
    logger.info("Gateway sync activated")

    # Mark startup as complete - health endpoint will now return 200
    global startup_complete
    startup_complete = True
    logger.info(f"Supervisor started on {settings.host}:{settings.port} - startup complete")

    yield

    # Cleanup
    logger.info("Shutting down Supervisor...")
    if not autoload_task.done():
        autoload_task.cancel()
        try:
            await autoload_task
        except (asyncio.CancelledError, Exception):
            pass
    if health_check_manager:
        await health_check_manager.stop()
    if restart_manager:
        await restart_manager.stop()
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
    # Update metrics before returning. System-level metrics (unified memory,
    # GPU temp / power) are pinned to host="primary" for now — the supervisor
    # only scrapes its own local nvidia-smi / /proc/meminfo. A worker-side
    # exporter would populate host="worker1" etc. for the same metric names.
    if resource_manager:
        # to_thread: shells out to nvidia-smi (up to 3×5s) — don't stall the loop
        status = await asyncio.to_thread(resource_manager.get_resource_status)
        # These are the supervisor's ALLOCATION view (sum of declared model
        # memory budgets) — NOT actual OS memory pressure. node_exporter +
        # the sparkstation_rules.yml recording rule expose the real /proc/
        # meminfo value under the name `unified_memory_used_bytes`. Two
        # distinct metrics with distinct meanings.
        metrics.sparkstation_allocated_memory_bytes.labels(host="primary").set(status["unified_memory_used_gb"] * 1024**3)
        metrics.sparkstation_memory_budget_bytes.labels(host="primary").set(status["unified_memory_limit_gb"] * 1024**3)
        # GPU temp/power are intentionally NOT exported here anymore —
        # nvidia_gpu_exporter runs on every host (incl. primary) and the
        # sparkstation_rules.yml translation owns gpu_temperature_celsius /
        # gpu_power_draw_watts. Emitting them here too double-counted primary.
        # NOTE: resident_models_count is set below from the registry, NOT from
        # resource_manager — the latter only tracks primary-host memory, so
        # worker-host models (e.g. chat on worker1) were missing ("4 of 5").

    if registry:
        suspended = await registry.list_suspended()
        metrics.suspended_models_count.set(len(suspended))

        running = await registry.list_running()
        metrics.resident_models_count.set(len(running))

        # Update per-model metrics. Clear first: generate_id mints a fresh
        # model_id per start, so without this every start/stop/swap cycle
        # leaks a stale label set that keeps exporting forever.
        metrics.model_status.clear()
        metrics.model_memory_used_bytes.clear()
        metrics.model_last_request_timestamp.clear()
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
            model_host = model.host or "primary"
            display_name = model.model_alias or model.model_name

            # Set model status
            # Handle both enum and string status values
            if hasattr(model.status, 'value'):
                status_str = model.status.value
            else:
                status_str = str(model.status)
            status_value = status_map.get(status_str.lower(), 0)
            metrics.model_status.labels(
                host=model_host,
                model_name=display_name,
                model_id=model.id
            ).set(status_value)

            # Set model memory usage
            if model.memory_gb:
                metrics.model_memory_used_bytes.labels(
                    host=model_host,
                    model_name=display_name,
                ).set(model.memory_gb * 1024**3)

            # Set last request timestamp
            if model.last_request_time:
                timestamp = model.last_request_time.timestamp()
                metrics.model_last_request_timestamp.labels(
                    host=model_host,
                    model_name=display_name,
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
                "host": model.host or "primary",
                "status": model.status,
                "health_status": model.health_status,
                "port": model.port,
                "base_url": model.base_url,
                "memory_gb": model.memory_gb,
                "is_default": (model.model_alias or model.model_name) == default_model_alias,
                "is_vision": bool(vision_model_alias)
                and (model.model_alias or model.model_name) == vision_model_alias,
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
        # Check if alias already exists. If the existing entry is in a non-live
        # state (stopped / suspended / failed), recycle it — that's exactly the
        # "alias is free, the registry just hasn't been cleaned" case the swap
        # flow runs into. Only RUNNING/STARTING entries should block a new start.
        if request.model_alias:
            existing = await registry.get_by_alias(request.model_alias)
            if existing:
                if existing.status in [ModelStatus.RUNNING, ModelStatus.STARTING]:
                    raise ModelAlreadyExistsError(request.model_alias)
                logger.info(
                    f"Recycling stale registry entry for alias '{request.model_alias}' "
                    f"(id={existing.id}, status={existing.status})"
                )
                # Release any leftover resource reservation, then drop the row
                try:
                    resource_manager.release_model(existing.id, full_release=True)
                except Exception as e:
                    logger.debug(f"Resource release for {existing.id} during recycle: {e}")
                await registry.delete(existing.id)

        # Generate model ID
        model_id = ModelRegistry.generate_id(request.model_name)

        # Estimate memory — honor explicit request value if provided,
        # otherwise fall back to the heuristic.
        if request.memory_gb is not None:
            memory_estimate = request.memory_gb
        else:
            memory_estimate = resource_manager.estimate_model_memory(
                request.model_name, request.quantization
            )

        # Allocate resources (memory limit is only enforced for the primary
        # host; remote hosts have their own capacity managed on the target).
        try:
            # to_thread: can_allocate_model inside shells out to nvidia-smi
            port = await asyncio.to_thread(
                resource_manager.allocate_model, model_id, memory_estimate, host=request.host
            )
        except ResourceError as e:
            current = resource_manager.get_unified_memory_usage()
            raise InsufficientResourcesError(str(e), current, resource_manager.hard_limit_gb)

        # Create config
        config = ModelConfig(
            model_name=request.model_name,
            backend=request.backend,
            model_type=request.model_type,
            model_alias=request.model_alias,
            host=request.host,
            num_gpus=request.num_gpus,
            quantization=request.quantization,
            idle_timeout_minutes=request.idle_timeout_minutes,
            auto_suspend_enabled=request.auto_suspend_enabled,
            speculative_model=request.speculative_model,
            num_speculative_tokens=request.num_speculative_tokens,
            speculative_method=request.speculative_method,
            speculative_extra=request.speculative_extra,
            extra_args=request.extra_args,
            docker_image=request.docker_image,
            env_vars=request.env_vars,
            volumes=request.volumes,
        )

        # Launch model
        try:
            launcher = launcher_factory.get_launcher(request.backend)
            instance = await launcher.launch(config, model_id, port, memory_gb=memory_estimate)
            instance.memory_gb = memory_estimate

            # CRITICAL: Save config for auto-restart and resume
            # This MUST be set here so restart_manager can recover failed models
            instance.saved_config = build_saved_config(config, instance.gpu_ids, port, memory_estimate)
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
            # model.status is a string due to ModelInstance ConfigDict(use_enum_values=True);
            # str-enum comparison above works, but .value would crash here. Use directly.
            raise ModelNotRunningError(model_id, model.status)

        await auto_suspend_manager.suspend_model(model_id)

        model = await registry.get(model_id)  # Refresh
        return {
            "model_id": model_id,
            "status": model.status,
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
            # See suspend handler note on use_enum_values=True.
            raise ModelNotSuspendedError(model_id, model.status)

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
            "status": model.status,
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
        # model_type is required by ModelStatusResponse — was missing here, which
        # made this endpoint always 500 with a Pydantic ValidationError. Anyone
        # polling /models/{id}/status (e.g. our new swap CLI) never made progress.
        model_type=model.model_type,
        status=model.status,
        health_status=model.health_status,
        uptime_seconds=uptime_seconds,
        last_health_check=model.last_health_check,
        last_request_time=model.last_request_time,
        idle_seconds=idle_seconds,
        idle_timeout_minutes=model.idle_timeout_minutes,
        auto_suspend_enabled=model.auto_suspend_enabled,
        seconds_until_suspend=seconds_until_suspend,
    )


@app.get("/prometheus/targets")
async def prometheus_targets():
    """Prometheus http_sd endpoint: scrape targets for running engine backends.

    Point a Prometheus job at this URL (http_sd_configs) and every running
    vLLM/SGLang backend is scraped automatically — targets follow model
    swaps/restarts/host moves with no prometheus.yml edits. Only engines
    that natively expose /metrics are listed; the bespoke CLIP/species/face
    servers don't, so they're excluded to avoid permanent 'down' targets.
    """
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    # dspark = the Anemll DSV4 stack, vLLM underneath — its head serves the
    # full vllm /metrics family on the API port.
    ENGINES_WITH_METRICS = {"vllm", "sglang", "trt-llm", "dspark"}
    targets = []
    for model in await registry.list_running():
        backend = model.backend.value if hasattr(model.backend, "value") else str(model.backend)
        if backend == "voicechat":
            # The Pipecat bot exports session/latency/tool metrics plus the
            # model-server frame traces on its own port (extra_args.metrics_port,
            # default 7861) — not on the runner port.
            metrics_port = int((model.extra_args or {}).get("metrics_port", 7861))
            host_ip = model.base_url.split("://", 1)[-1].rsplit(":", 1)[0]
            targets.append(
                {
                    "targets": [f"{host_ip}:{metrics_port}"],
                    "labels": {
                        "host": model.host or "primary",
                        "alias": model.model_alias or model.model_name,
                        "engine": backend,
                    },
                }
            )
            continue
        if backend not in ENGINES_WITH_METRICS:
            continue
        # base_url is routable from the supervisor host (QSFP IP for workers),
        # which is also where Prometheus runs.
        hostport = model.base_url.split("://", 1)[-1]
        targets.append(
            {
                "targets": [hostport],
                "labels": {
                    "host": model.host or "primary",
                    "alias": model.model_alias or model.model_name,
                    "engine": backend,
                },
            }
        )
    return targets


@app.get("/resources", response_model=ResourceStatus)
async def get_resources():
    """Get system resource status."""
    if resource_manager is None:
        raise HTTPException(status_code=503, detail="Resource manager not initialized")

    status = await asyncio.to_thread(resource_manager.get_resource_status)
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
