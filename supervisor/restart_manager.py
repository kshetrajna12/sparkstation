"""
Model restart manager for automatic recovery from failures.

Implements exponential backoff and restart attempt tracking.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from supervisor.registry import ModelRegistry
from supervisor.resources import ResourceManager
from supervisor.models import ModelStatus, ModelInstance, ModelConfig, Backend
from supervisor.launchers.factory import LauncherFactory
from supervisor.config import settings

logger = logging.getLogger(__name__)


class RestartManager:
    """
    Manages automatic restart of failed models with exponential backoff.

    Features:
    - Exponential backoff (1 min → 5 min → 15 min)
    - Tracks restart attempts per model
    - Max 3 restart attempts (configurable)
    - Marks as permanently FAILED after max attempts
    - Cleans up resources on failure
    """

    def __init__(
        self,
        registry: ModelRegistry,
        launcher_factory: LauncherFactory,
        resource_manager: ResourceManager,
    ):
        self.registry = registry
        self.launcher_factory = launcher_factory
        self.resource_manager = resource_manager

        # Configuration
        self.enabled = settings.auto_restart_enabled
        self.max_attempts = settings.auto_restart_max_attempts

        # Parse backoff timings from comma-separated string
        try:
            self.backoff_minutes = [
                int(x.strip()) for x in settings.auto_restart_backoff_minutes.split(",")
            ]
        except Exception as e:
            logger.error(f"Failed to parse backoff minutes: {e}, using defaults")
            self.backoff_minutes = [1, 5, 15]

        # Ensure we have enough backoff values
        while len(self.backoff_minutes) < self.max_attempts:
            # Repeat last value if not enough specified
            self.backoff_minutes.append(self.backoff_minutes[-1])

    async def handle_failed_model(self, model_id: str):
        """
        Handle a model that has been marked as FAILED by health checks.

        Args:
            model_id: ID of failed model
        """
        if not self.enabled:
            logger.info(f"Auto-restart disabled, skipping {model_id}")
            return

        model = await self.registry.get(model_id)
        if not model:
            logger.error(f"Model {model_id} not found in registry")
            return

        if model.status != ModelStatus.FAILED:
            logger.warning(f"Model {model_id} is not FAILED (status: {model.status})")
            return

        # Check restart attempts
        if model.restart_count >= self.max_attempts:
            logger.error(
                f"Model {model_id} has reached max restart attempts ({self.max_attempts}), "
                "marking as permanently FAILED"
            )
            await self._mark_permanently_failed(model)
            return

        # Schedule restart with backoff
        backoff_index = min(model.restart_count, len(self.backoff_minutes) - 1)
        backoff_minutes = self.backoff_minutes[backoff_index]

        logger.info(
            f"Scheduling restart for {model_id} (attempt {model.restart_count + 1}/{self.max_attempts}) "
            f"after {backoff_minutes} minute(s)"
        )

        # Wait for backoff period
        await asyncio.sleep(backoff_minutes * 60)

        # Attempt restart
        await self._restart_model(model)

    async def _restart_model(self, model: ModelInstance):
        """
        Attempt to restart a failed model.

        Args:
            model: Failed model instance
        """
        model_id = model.id
        display_name = model.model_alias or model.model_name

        logger.info(
            f"Attempting restart of {display_name} (attempt {model.restart_count + 1})"
        )

        try:
            # Check if saved config exists
            if not model.saved_config:
                logger.error(f"No saved config for {model_id}, cannot restart")
                await self._mark_permanently_failed(model)
                return

            # Force stop any existing process (in case it's hung)
            launcher = self.launcher_factory.get_launcher(model.backend)
            try:
                await launcher.stop(model)
                logger.debug(f"Stopped existing process for {model_id}")
            except Exception as e:
                logger.warning(f"Failed to stop existing process: {e}")

            # Check resources
            memory_estimate = self.resource_manager.estimate_model_memory(
                model.model_name, model.saved_config.get("quantization", "fp8")
            )

            if not self.resource_manager.can_allocate_model(memory_estimate):
                logger.error(
                    f"Cannot restart {model_id}: insufficient resources, will retry later"
                )
                # Don't increment restart count if resource issue (not model's fault)
                return

            # Allocate resources (reuse port if available)
            self.resource_manager.model_memory_usage[model_id] = memory_estimate

            # Recreate model config from saved config
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

            # Launch model
            new_instance = await launcher.launch(config, model_id, model.port)

            # Update registry
            model.status = ModelStatus.RUNNING
            model.pid = new_instance.pid
            model.started_at = new_instance.started_at
            model.restart_count += 1
            model.last_restart_time = datetime.now()
            model.last_request_time = datetime.now()  # Reset idle timer
            await self.registry.update(model)

            logger.info(
                f"Successfully restarted {display_name} (total restarts: {model.restart_count})"
            )

        except Exception as e:
            logger.error(f"Failed to restart {model_id}: {e}", exc_info=True)

            # Increment restart count
            model.restart_count += 1
            model.last_restart_time = datetime.now()

            if model.restart_count >= self.max_attempts:
                logger.error(
                    f"Model {model_id} reached max restart attempts, marking permanently FAILED"
                )
                await self._mark_permanently_failed(model)
            else:
                # Keep as FAILED, will retry with next backoff
                model.status = ModelStatus.FAILED
                await self.registry.update(model)

            # Cleanup resources
            self.resource_manager.release_model(model_id, full_release=False)

    async def _mark_permanently_failed(self, model: ModelInstance):
        """
        Mark model as permanently failed (no more restart attempts).

        Args:
            model: Failed model instance
        """
        model.status = ModelStatus.FAILED
        model.stopped_at = datetime.now()
        model.pid = None
        await self.registry.update(model)

        # Release all resources
        self.resource_manager.release_model(model.id, full_release=True)

        logger.error(
            f"Model {model.id} marked as permanently FAILED after {model.restart_count} restart attempts"
        )

    async def reset_restart_count(self, model_id: str):
        """
        Reset restart count for a model (manual intervention).

        Useful after fixing underlying issues (e.g., model weights, config).

        Args:
            model_id: Model ID to reset
        """
        model = await self.registry.get(model_id)
        if model:
            old_count = model.restart_count
            model.restart_count = 0
            model.last_restart_time = None
            await self.registry.update(model)
            logger.info(f"Reset restart count for {model_id} (was {old_count})")
