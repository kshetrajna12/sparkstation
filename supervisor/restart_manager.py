"""
Model restart manager for automatic recovery from failures.

Implements exponential backoff and restart attempt tracking.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Set

from supervisor.registry import ModelRegistry
from supervisor.resources import ResourceManager
from supervisor.models import ModelStatus, HealthStatus, ModelInstance, ModelConfig
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

        # Background watcher that catches FAILED transitions from ANY path
        # (reconcile, container-exit, direct API, race conditions), not just
        # the health-check → max-consecutive-failures path. Prior to this
        # loop, a container that crashed mid-run and got marked FAILED via
        # some other route would sit dead forever: list_running() excludes
        # FAILED, so the health-check counter never incremented, and
        # handle_failed_model() was never called.
        self.watch_interval_seconds = settings.auto_restart_watch_interval_seconds
        self._task: Optional[asyncio.Task] = None
        # Model ids currently in a restart attempt (backoff sleep OR active
        # relaunch). Prevents the watcher and the health-check callback from
        # both firing on the same model.
        self._in_flight: Set[str] = set()

    async def start(self):
        """Start the FAILED-model watcher loop."""
        if self._task is None and self.enabled:
            self._task = asyncio.create_task(self._watch_failed_loop())
            logger.info(
                f"RestartManager watcher started (interval: {self.watch_interval_seconds}s, "
                f"max_attempts: {self.max_attempts}, backoff: {self.backoff_minutes} min)"
            )

    async def stop(self):
        """Stop the watcher loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _watch_failed_loop(self):
        """Poll for FAILED models and trigger restart for any not already in flight."""
        while True:
            try:
                await asyncio.sleep(self.watch_interval_seconds)
                all_models = await self.registry.list_all()
                for model in all_models:
                    if model.status != ModelStatus.FAILED:
                        continue
                    if model.id in self._in_flight:
                        continue
                    if model.restart_count >= self.max_attempts:
                        # Already permanently failed — don't spam retries.
                        continue
                    logger.warning(
                        f"Watcher found FAILED model not in restart flight: "
                        f"{model.model_alias or model.model_name} (id={model.id}) — triggering handle_failed_model"
                    )
                    self._in_flight.add(model.id)
                    asyncio.create_task(self._handle_and_untrack(model.id))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Restart watcher loop error: {e}", exc_info=True)

    async def _handle_and_untrack(self, model_id: str):
        """Wrap handle_failed_model with in_flight cleanup on any exit."""
        try:
            await self.handle_failed_model(model_id, _already_tracked=True)
        finally:
            self._in_flight.discard(model_id)

    async def handle_failed_model(self, model_id: str, _already_tracked: bool = False):
        """
        Handle a model that has been marked as FAILED.

        Args:
            model_id: ID of failed model
            _already_tracked: Internal — True when called by the watcher
                (in_flight already has model_id). External callers (e.g. the
                health-check callback) leave this False; we add to in_flight
                here so a subsequent watcher pass won't double-fire.
        """
        if not self.enabled:
            logger.info(f"Auto-restart disabled, skipping {model_id}")
            return

        if not _already_tracked:
            if model_id in self._in_flight:
                logger.debug(f"handle_failed_model: {model_id} already in flight, skipping")
                return
            self._in_flight.add(model_id)

        try:
            await self._handle_failed_model_impl(model_id)
        finally:
            if not _already_tracked:
                self._in_flight.discard(model_id)

    async def _handle_failed_model_impl(self, model_id: str):
        """The original body of handle_failed_model, minus the in_flight bookkeeping."""
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

        # Re-fetch after the backoff: the world may have changed while we
        # slept — an operator stopped/deleted the instance, or another
        # recovery path already handled it. Restarting from the stale object
        # blindly relaunches over whatever is running now (it clobbered a
        # manual benchmark deployment on 2026-08-28). Same re-fetch-guard
        # class as the 2026-07-02 zombie-ModelInstance fix.
        current = await self.registry.get(model_id)
        if not current or current.status != ModelStatus.FAILED:
            logger.info(
                f"Skipping scheduled restart of {model_id}: "
                f"{'instance gone' if not current else f'status now {current.status}'} "
                f"after backoff"
            )
            return

        # Attempt restart
        await self._restart_model(current)

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

            # Rebuild the exact original launch config (host, docker image,
            # model type, env vars, volumes, speculative config). A partial
            # reconstruction here used to relaunch worker-pinned models on the
            # primary with the default image.
            config = ModelConfig.from_saved_config(model.saved_config)

            # Memory: prefer the explicitly configured allocation over the
            # name-based heuristic (which mis-sizes MoE names like "35B-A3B").
            memory_estimate = model.saved_config.get("memory_gb") or model.memory_gb
            if memory_estimate is None:
                memory_estimate = self.resource_manager.estimate_model_memory(
                    model.model_name, model.saved_config.get("quantization", "fp8")
                )

            # Memory limits are only tracked for the primary Spark (same rule
            # as ResourceManager.allocate_model).
            if config.host == "primary":
                can_allocate = await asyncio.to_thread(
                    self.resource_manager.can_allocate_model, memory_estimate
                )
                if not can_allocate:
                    logger.error(
                        f"Cannot restart {model_id}: insufficient resources, will retry later"
                    )
                    # Don't increment restart count if resource issue (not model's fault)
                    return
                # Allocate resources (reuse port if available)
                self.resource_manager.model_memory_usage[model_id] = memory_estimate

            # Launch model. Pass memory_gb so gpu_memory_utilization is derived
            # from the model's real budget instead of the 0.9 default.
            new_instance = await launcher.launch(config, model_id, model.port, memory_gb=memory_estimate)

            # Update registry. STARTING (not RUNNING): the replacement backend
            # still has to cold-load weights; the startup monitor promotes it
            # to RUNNING once /health responds, and only then does gateway
            # sync route traffic to it. Persisting the NEW container_id is
            # load-bearing — with the old id in the DB, the next
            # reconcile_state() classified the live replacement container as
            # an orphan and docker rm -f'd it.
            model.status = ModelStatus.STARTING
            model.health_status = HealthStatus.UNKNOWN
            model.pid = new_instance.pid
            model.container_id = new_instance.container_id
            model.started_at = new_instance.started_at
            model.stopped_at = None
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
