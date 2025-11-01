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
from supervisor.models import ModelStatus, HealthStatus, ModelInstance
from supervisor.config import settings

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

    def __init__(self, registry: ModelRegistry, restart_manager=None):
        self.registry = registry
        self.restart_manager = restart_manager  # Optional: for triggering restarts
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

                # Try to health check the model (without counting failures)
                try:
                    response = await self.client.post(
                        f"{model.base_url}/v1/chat/completions",
                        json={
                            "model": model.model_name,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                            "temperature": 0,
                        },
                    )

                    if response.status_code == 200:
                        # Model is healthy! Transition to RUNNING
                        model.status = ModelStatus.RUNNING
                        model.health_status = HealthStatus.HEALTHY
                        model.last_health_check = datetime.now()
                        # Reset any failure counts from before
                        if model.id in self.failure_counts:
                            del self.failure_counts[model.id]
                        await self.registry.update(model)

                        logger.info(f"✓ Model {display_name} is now RUNNING (transitioned from STARTING)")
                    else:
                        # Not ready yet, keep waiting
                        logger.debug(f"Model {display_name} not ready yet (HTTP {response.status_code}), will retry...")

                except (httpx.ConnectError, httpx.TimeoutException):
                    # Connection failed or timeout - model still starting up
                    logger.debug(f"Model {display_name} not ready yet (connection/timeout), will retry...")

            except Exception as e:
                logger.error(f"Error checking starting model {model.id}: {e}")

    async def _check_all_models(self):
        """Check health of all running models."""
        running_models = await self.registry.list_running()

        if not running_models:
            logger.debug("No running models to health check")
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
        Perform 1-token chat completion probe to verify model responsiveness.

        Args:
            model: Model instance to check

        Returns:
            True if healthy, False if unhealthy
        """
        model_id = model.id
        display_name = model.model_alias or model.model_name

        try:
            # 1-token chat completion probe
            # CRITICAL: Use /v1/chat/completions (most backends don't support /v1/completions)
            response = await self.client.post(
                f"{model.base_url}/v1/chat/completions",
                json={
                    "model": model.model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )

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

            # Trigger restart if manager available
            if self.restart_manager:
                logger.info(f"Triggering restart for {display_name}")
                # Schedule restart in background (don't block health checks)
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
