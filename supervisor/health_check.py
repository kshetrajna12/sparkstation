"""
Health check manager for continuous model monitoring.

Performs periodic 1-token chat completion probes to verify model responsiveness.
Tracks consecutive failures and marks models as FAILED after threshold exceeded.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict
import httpx

from supervisor.registry import ModelRegistry
from supervisor.models import Backend, ModelStatus, HealthStatus, ModelInstance
from supervisor.config import settings


def _health_path(model: ModelInstance) -> str:
    """Liveness probe path for a model's server.

    Backends speaking the OpenAI-ish HTTP surface all serve /health. The
    voicecascade backend is a Pipecat runner: no /health route (404, which kept
    it stuck in STARTING forever, 2026-08-25) — its playground page /client/
    is the lightweight always-200 route.
    """
    return "/client/" if model.backend == Backend.VOICECASCADE else "/health"

logger = logging.getLogger(__name__)


class HealthCheckManager:
    """
    Manages periodic health checks for running models.

    Features:
    - 1-token chat completion probes every 5 minutes (configurable)
    - Tracks consecutive failures per model
    - Marks model as FAILED after N consecutive failures
    - Updates health_status in registry
    - Logs all health check results
    """

    def __init__(self, registry: ModelRegistry, restart_manager=None, gateway_sync=None):
        self.registry = registry
        self.restart_manager = restart_manager  # Optional: for triggering restarts
        # Optional: publish a model to the gateway the instant it turns RUNNING,
        # instead of waiting up to gateway_sync_interval (60s) for the periodic
        # pass. Without this, a model (and its default/vision aliases) is ready
        # but unroutable through :8000 for up to a minute after startup.
        self.gateway_sync = gateway_sync
        self.client = httpx.AsyncClient(timeout=settings.health_check_timeout_seconds)

        # Configuration
        self.check_interval = settings.health_check_interval_seconds
        self.max_failures = settings.health_check_max_failures

        # Failure tracking: model_id -> consecutive_failure_count
        self.failure_counts: Dict[str, int] = {}

        self._task: Optional[asyncio.Task] = None
        self._startup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background health check task."""
        if self._task is None:
            self._task = asyncio.create_task(self._monitoring_loop())
            # Also start a task to monitor models in "starting" state
            self._startup_task = asyncio.create_task(self._startup_monitoring_loop())
            logger.info(
                f"Health check manager started (interval: {self.check_interval}s, "
                f"max_failures: {self.max_failures})"
            )

    async def stop(self):
        """Stop background health check task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._startup_task:
            self._startup_task.cancel()
            try:
                await self._startup_task
            except asyncio.CancelledError:
                pass
            self._startup_task = None

        # CRITICAL: Close httpx client to prevent file descriptor leak
        await self.client.aclose()
        logger.info("Health check manager stopped")

    async def _monitoring_loop(self):
        """Background task: check all running models periodically."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_all_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check monitoring: {e}", exc_info=True)

    async def _startup_monitoring_loop(self):
        """Background task: check models in STARTING state every 10 seconds."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._check_starting_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in startup monitoring: {e}", exc_info=True)

    async def _check_starting_models(self):
        """Check models in STARTING state and transition them to RUNNING if healthy."""
        # Get all models in STARTING state
        all_models = await self.registry.list_all()
        starting_models = [m for m in all_models if m.status == ModelStatus.STARTING]

        if not starting_models:
            return

        logger.debug(f"Checking {len(starting_models)} model(s) in STARTING state")

        for model in starting_models:
            try:
                display_name = model.model_alias or model.model_name

                # Give up on models stuck in STARTING past the timeout — a
                # relaunch whose container died would otherwise sit in
                # STARTING forever (never health-counted, never restarted).
                started_at = model.started_at
                # A row with no container_id is a script-stack (dspark)
                # STARTING placeholder: the launcher is still blocked in the
                # start script (its own timeout, up to 45 min for GLM TP2)
                # and removes the row itself. Never time it out here — a
                # FAILED flip would make RestartManager stop the stack mid-boot.
                if not model.container_id:
                    continue
                if started_at and (datetime.now() - started_at).total_seconds() > settings.starting_timeout_minutes * 60:
                    logger.error(
                        f"Model {display_name} stuck in STARTING for over "
                        f"{settings.starting_timeout_minutes} min — marking FAILED"
                    )
                    fresh = await self.registry.get(model.id)
                    if fresh and fresh.status == ModelStatus.STARTING:
                        fresh.status = ModelStatus.FAILED
                        fresh.health_status = HealthStatus.UNHEALTHY
                        await self.registry.update(fresh)
                    continue

                # Try to health check the model (without counting failures)
                try:
                    # Use /health endpoint for all models (liveness check)
                    response = await self.client.get(f"{model.base_url}{_health_path(model)}")

                    if response.status_code == 200:
                        # Re-fetch before promoting: the row may have been
                        # deleted (purge/recycle) or transitioned elsewhere
                        # (stop/suspend) while the probe was in flight —
                        # writing the stale object back would resurrect it
                        # as RUNNING.
                        fresh = await self.registry.get(model.id)
                        if fresh is None or fresh.status != ModelStatus.STARTING:
                            logger.info(
                                f"Model {display_name} no longer STARTING "
                                f"({'deleted' if fresh is None else fresh.status}), skipping promotion"
                            )
                            continue

                        # Model is healthy! Transition to RUNNING
                        fresh.status = ModelStatus.RUNNING
                        fresh.health_status = HealthStatus.HEALTHY
                        fresh.last_health_check = datetime.now()
                        # Reset any failure counts from before
                        if fresh.id in self.failure_counts:
                            del self.failure_counts[fresh.id]
                        await self.registry.update(fresh)

                        logger.info(f"✓ Model {display_name} is now RUNNING (transitioned from STARTING)")

                        # Publish to the gateway immediately so this model and
                        # its default/vision aliases become routable the moment
                        # it's ready — not up to 60s later on the periodic sync.
                        if self.gateway_sync is not None:
                            try:
                                await self.gateway_sync.sync_models()
                            except Exception as e:
                                logger.warning(f"Gateway sync after promoting {display_name} failed: {e}")
                    else:
                        # Not ready yet, keep waiting
                        logger.debug(f"Model {display_name} not ready yet (HTTP {response.status_code}), will retry...")

                except (httpx.ConnectError, httpx.TimeoutException):
                    # Connection failed or timeout - model still starting up
                    logger.debug(f"Model {display_name} not ready yet (connection/timeout), will retry...")

            except Exception as e:
                logger.error(f"Error checking starting model {model.id}: {e}", exc_info=True)

    async def _check_all_models(self):
        """Check health of all running models."""
        running_models = await self.registry.list_running()

        # Also list FAILED models so we can log them each cycle. list_running()
        # excludes FAILED, which used to hide dead models: a crash-then-mark-
        # failed path (reconcile, container-exit) would show up as "4 healthy"
        # forever with no warning that a supposedly-registered model was gone.
        # RestartManager's watcher handles the actual recovery; this log is
        # for operator visibility.
        all_models = await self.registry.list_all()
        failed_models = [m for m in all_models if m.status == ModelStatus.FAILED]

        if not running_models and not failed_models:
            logger.debug("No running models to health check")
            return

        if failed_models:
            failed_names = ", ".join(m.model_alias or m.model_name for m in failed_models)
            logger.warning(
                f"{len(failed_models)} model(s) in FAILED state — "
                f"RestartManager should be handling recovery: {failed_names}"
            )

        if not running_models:
            return

        logger.info(f"Running health checks on {len(running_models)} models")

        # Check all models concurrently
        tasks = [self._check_model_health(model) for model in running_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log summary
        healthy = sum(1 for r in results if r is True)
        unhealthy = sum(1 for r in results if r is False)
        errors = sum(1 for r in results if isinstance(r, Exception))

        logger.info(
            f"Health check complete: {healthy} healthy, {unhealthy} unhealthy, {errors} errors"
        )

    async def _check_model_health(self, model: ModelInstance) -> bool:
        """
        Perform liveness check to verify model server is alive.

        Uses /health endpoint for all model types - this is a lightweight check
        that doesn't queue behind inference requests. This prevents false positives
        when models are busy processing long-running requests (e.g., VLMs with images).

        Args:
            model: Model instance to check

        Returns:
            True if healthy, False if unhealthy
        """
        model_id = model.id
        display_name = model.model_alias or model.model_name

        try:
            # Use /health endpoint for all models (liveness check)
            # This is lightweight and doesn't queue behind inference requests
            response = await self.client.get(f"{model.base_url}{_health_path(model)}")

            if response.status_code == 200:
                # Health check passed
                await self._handle_healthy(model_id, display_name)
                return True
            else:
                # Health check failed (non-200 response)
                logger.warning(
                    f"Health check failed for {display_name}: HTTP {response.status_code}"
                )
                await self._handle_unhealthy(model_id, display_name, f"HTTP {response.status_code}")
                return False

        except httpx.TimeoutException:
            logger.warning(f"Health check timeout for {display_name}")
            await self._handle_unhealthy(model_id, display_name, "timeout")
            return False

        except httpx.ConnectError:
            logger.warning(f"Health check connection failed for {display_name}")
            await self._handle_unhealthy(model_id, display_name, "connection refused")
            return False

        except Exception as e:
            logger.error(f"Health check error for {display_name}: {e}", exc_info=True)
            await self._handle_unhealthy(model_id, display_name, str(e))
            return False

    async def _handle_healthy(self, model_id: str, display_name: str):
        """
        Handle successful health check.

        Resets failure count and updates health status to HEALTHY.
        """
        # Reset failure count
        if model_id in self.failure_counts:
            previous_failures = self.failure_counts[model_id]
            del self.failure_counts[model_id]
            if previous_failures > 0:
                logger.info(
                    f"Model {display_name} recovered (was {previous_failures} failures)"
                )

        # Update health status in registry
        model = await self.registry.get(model_id)
        if model:
            model.health_status = HealthStatus.HEALTHY
            model.last_health_check = datetime.now()

            # Decay restart_count after sustained healthy uptime so only
            # consecutive rapid failures count toward max_attempts —
            # otherwise occasional recovered crashes accumulate over days
            # and the model eventually goes permanently FAILED.
            if (
                model.restart_count > 0
                and model.last_restart_time
                and (datetime.now() - model.last_restart_time).total_seconds()
                > settings.restart_count_reset_minutes * 60
            ):
                logger.info(
                    f"Model {display_name} healthy for over "
                    f"{settings.restart_count_reset_minutes} min since last restart — "
                    f"resetting restart_count (was {model.restart_count})"
                )
                model.restart_count = 0

            await self.registry.update(model)

        logger.debug(f"Health check passed: {display_name}")

    async def _handle_unhealthy(self, model_id: str, display_name: str, reason: str):
        """
        Handle failed health check.

        Increments failure count and marks model as FAILED if threshold exceeded.

        Args:
            model_id: Model ID
            display_name: Human-readable model name
            reason: Failure reason
        """
        # Increment failure count
        self.failure_counts[model_id] = self.failure_counts.get(model_id, 0) + 1
        failure_count = self.failure_counts[model_id]

        logger.warning(
            f"Health check failed for {display_name}: {reason} "
            f"(failures: {failure_count}/{self.max_failures})"
        )

        # Update health status in registry
        model = await self.registry.get(model_id)
        if not model:
            return

        model.last_health_check = datetime.now()

        if failure_count >= self.max_failures:
            # Mark as FAILED after consecutive failures
            logger.error(
                f"Model {display_name} marked as FAILED after {failure_count} consecutive failures"
            )
            model.status = ModelStatus.FAILED
            model.health_status = HealthStatus.UNHEALTHY

            # Remove from failure tracking (will be handled by restart logic)
            del self.failure_counts[model_id]

            # Persist the FAILED status BEFORE triggering restart. The old
            # ordering (schedule task → await registry.update) was a TOCTOU
            # race: create_task yields, the task's own await self.registry.get
            # reads the still-running row, and handle_failed_model's
            # `if model.status != FAILED: return` swallows the trigger. Real
            # incident 2026-07-01 20:17 EDT — chat was marked FAILED but the
            # restart guard bailed 83ms later reading status="running", and
            # the model sat dead for 3h until manual intervention. The
            # RestartManager watcher (auto_restart_watch_interval_seconds)
            # is the belt-and-suspenders safety net.
            await self.registry.update(model)

            if self.restart_manager:
                logger.info(f"Triggering restart for {display_name}")
                asyncio.create_task(self.restart_manager.handle_failed_model(model_id))
        else:
            # Still RUNNING but UNHEALTHY
            model.health_status = HealthStatus.UNHEALTHY
            await self.registry.update(model)

    async def check_model_now(self, model_id: str) -> bool:
        """
        Manually trigger health check for a specific model (for testing/debugging).

        Args:
            model_id: Model ID to check

        Returns:
            True if healthy, False otherwise
        """
        model = await self.registry.get(model_id)
        if not model or model.status != ModelStatus.RUNNING:
            logger.warning(f"Cannot health check model {model_id}: not running")
            return False

        return await self._check_model_health(model)

    def get_failure_count(self, model_id: str) -> int:
        """Get current consecutive failure count for a model."""
        return self.failure_counts.get(model_id, 0)
