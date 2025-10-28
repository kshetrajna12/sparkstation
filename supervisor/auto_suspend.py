"""
Auto-suspend manager for idle models with thermal hysteresis (DGX Spark optimized).
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from supervisor.registry import ModelRegistry
from supervisor.resources import ResourceManager
from supervisor.models import ModelStatus, ModelInstance
from supervisor.launchers.factory import LauncherFactory
from supervisor.config import settings

logger = logging.getLogger(__name__)


class AutoSuspendManager:
    """
    Manages automatic suspension of idle models (DGX Spark-optimized with thermal hysteresis).

    CRITICAL: Implements hysteresis to prevent suspend/resume thrashing.
    """

    def __init__(self, registry: ModelRegistry, launcher_factory: LauncherFactory, resource_manager: ResourceManager):
        self.registry = registry
        self.launcher_factory = launcher_factory
        self.resource_manager = resource_manager

        # Check interval
        self.check_interval_seconds = settings.auto_suspend_check_interval_seconds

        # Thermal hysteresis configuration (prevent thrashing)
        self.thermal_suspend_threshold_c = settings.thermal_suspend_threshold_c
        self.thermal_resume_threshold_c = settings.thermal_resume_threshold_c
        self.thermal_sustain_ms = settings.thermal_sustain_ms
        self.thermal_cooldown_ms = settings.thermal_cooldown_ms

        # Thermal state tracking
        self.high_temp_start_time: Optional[datetime] = None
        self.last_thermal_suspend_time: Optional[datetime] = None

        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background monitoring task."""
        if self._task is None:
            self._task = asyncio.create_task(self._monitoring_loop())
            logger.info("Auto-suspend manager started")

    async def stop(self):
        """Stop background monitoring task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Auto-suspend manager stopped")

    async def _monitoring_loop(self):
        """Background task that checks for idle models and thermal issues."""
        while True:
            try:
                await asyncio.sleep(self.check_interval_seconds)
                await self._check_idle_and_thermal()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-suspend monitoring: {e}", exc_info=True)

    async def _check_idle_and_thermal(self):
        """
        Check all running models for idle timeout and thermal issues.
        CRITICAL: Add hysteresis to prevent suspend/resume thrashing.
        """
        now = datetime.now()

        # DGX Spark: Check thermal status with hysteresis
        gpu_temp = self.resource_manager.get_gpu_temperature()

        # Hysteresis logic: only suspend if temp sustained high
        if gpu_temp > self.thermal_suspend_threshold_c:
            # Temperature is high
            if self.high_temp_start_time is None:
                # First time above threshold, start timer
                self.high_temp_start_time = now
                logger.info(
                    f"GPU temp {gpu_temp}°C > {self.thermal_suspend_threshold_c}°C, "
                    f"monitoring for {self.thermal_sustain_ms/1000}s"
                )
            else:
                # Check if sustained for required duration
                high_temp_duration = (now - self.high_temp_start_time).total_seconds() * 1000
                if high_temp_duration >= self.thermal_sustain_ms:
                    # Temperature sustained high, check cooldown
                    if self.last_thermal_suspend_time:
                        cooldown_elapsed = (
                            now - self.last_thermal_suspend_time
                        ).total_seconds() * 1000
                        if cooldown_elapsed < self.thermal_cooldown_ms:
                            logger.warning(
                                f"Thermal cooldown active, skipping suspend "
                                f"({cooldown_elapsed/1000:.0f}s / {self.thermal_cooldown_ms/1000:.0f}s)"
                            )
                            return

                    # Suspend least-used model
                    logger.warning(
                        f"GPU temp {gpu_temp}°C sustained for {high_temp_duration/1000:.0f}s, "
                        f"thermal-suspending least-used model"
                    )
                    await self._suspend_least_used_model()
                    self.last_thermal_suspend_time = now
                    self.high_temp_start_time = None  # Reset timer
                    return

        elif gpu_temp <= self.thermal_resume_threshold_c:
            # Temperature is normal, reset high temp timer
            if self.high_temp_start_time is not None:
                logger.info(
                    f"GPU temp {gpu_temp}°C dropped below {self.thermal_resume_threshold_c}°C, "
                    f"resetting thermal monitoring"
                )
                self.high_temp_start_time = None

        # Check idle timeouts for all running models
        running_models = await self.registry.list_running()
        for model in running_models:
            if not model.auto_suspend_enabled:
                continue

            if model.idle_timeout_minutes == 0:
                continue  # Auto-suspend disabled for this model

            # Check if model has been idle too long
            if model.last_request_time is None:
                model.last_request_time = model.started_at
                await self.registry.update(model)  # Persist to database

            idle_duration = now - model.last_request_time
            if idle_duration.total_seconds() / 60 >= model.idle_timeout_minutes:
                logger.info(
                    f"Auto-suspending idle model {model.id} "
                    f"(idle for {idle_duration.total_seconds()//60} minutes)"
                )
                await self.suspend_model(model.id)

    async def _suspend_least_used_model(self):
        """Emergency suspend of least recently used model (thermal protection)."""
        running_models = await self.registry.list_running()
        if not running_models:
            return

        # Sort by last request time (oldest first)
        models = sorted(running_models, key=lambda m: m.last_request_time or m.started_at)
        least_used = models[0]
        logger.warning(f"Emergency thermal suspend: {least_used.id}")
        await self.suspend_model(least_used.id)

    async def suspend_model(self, model_id: str):
        """
        Suspend a model and release GPU resources.

        Args:
            model_id: Model ID to suspend
        """
        model = await self.registry.get(model_id)
        if model is None or model.status != ModelStatus.RUNNING:
            logger.warning(f"Cannot suspend model {model_id}: not running")
            return

        # Save config for resume
        model.saved_config = {
            "model_name": model.model_name,
            "backend": model.backend.value,
            "model_alias": model.model_alias,
            "gpu_ids": model.gpu_ids,
            "port": model.port,
            "quantization": model.extra_args.get("quantization"),
            "auto_suspend_enabled": model.auto_suspend_enabled,
            "idle_timeout_minutes": model.idle_timeout_minutes,
            "extra_args": model.extra_args,
        }

        # Stop the model process
        launcher = self.launcher_factory.get_launcher(model.backend)
        stopped = await launcher.stop(model)

        if stopped:
            # Update status but keep port reserved
            model.status = ModelStatus.SUSPENDED
            model.pid = None
            model.stopped_at = datetime.now()
            await self.registry.update(model)

            # Release memory but keep port
            self.resource_manager.release_model(model_id, full_release=False)

            logger.info(f"Model {model_id} suspended, GPU freed")
        else:
            logger.error(f"Failed to suspend model {model_id}")

    async def resume_model(self, model_id: str) -> bool:
        """
        Resume a suspended model.

        Args:
            model_id: Model ID to resume

        Returns:
            True if resumed successfully
        """
        model = await self.registry.get(model_id)
        if model is None:
            logger.error(f"Model {model_id} not found")
            return False

        if model.status != ModelStatus.SUSPENDED:
            logger.warning(f"Model {model_id} is not suspended (status: {model.status})")
            return False

        if model.saved_config is None:
            logger.error(f"No saved config for {model_id}")
            return False

        logger.info(f"Resuming model {model_id}...")

        try:
            # Check if we can allocate resources
            memory_estimate = self.resource_manager.estimate_model_memory(
                model.model_name, model.saved_config.get("quantization", "fp8")
            )

            if not self.resource_manager.can_allocate_model(memory_estimate):
                logger.error(f"Cannot resume {model_id}: insufficient resources")
                return False

            # Allocate resources (reuse port)
            self.resource_manager.model_memory_usage[model_id] = memory_estimate

            # Restart the model with saved config
            from supervisor.models import ModelConfig, Backend

            config = ModelConfig(
                model_name=model.saved_config["model_name"],
                backend=Backend(model.saved_config["backend"]),
                model_alias=model.saved_config.get("model_alias"),
                num_gpus=len(model.saved_config.get("gpu_ids", [0])),
                quantization=model.saved_config.get("quantization"),
                idle_timeout_minutes=model.saved_config["idle_timeout_minutes"],
                auto_suspend_enabled=model.saved_config["auto_suspend_enabled"],
                extra_args=model.saved_config.get("extra_args", {}),
            )

            launcher = self.launcher_factory.get_launcher(config.backend)
            new_instance = await launcher.launch(config, model_id, model.port)

            # Update registry
            model.status = ModelStatus.RUNNING
            model.pid = new_instance.pid
            model.last_request_time = datetime.now()
            model.started_at = new_instance.started_at
            await self.registry.update(model)

            logger.info(f"Model {model_id} resumed and ready")
            return True

        except Exception as e:
            logger.error(f"Failed to resume model {model_id}: {e}", exc_info=True)
            # Release resources on failure
            self.resource_manager.release_model(model_id, full_release=False)
            return False
