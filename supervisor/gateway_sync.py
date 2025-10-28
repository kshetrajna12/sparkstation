"""
Gateway sync: Push model list to LiteLLM admin API.

CRITICAL FIX: fetch_from_url is flaky and version-sensitive.
Instead, Supervisor pushes model list to LiteLLM via admin API.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
import httpx
import yaml
from supervisor.registry import ModelRegistry
from supervisor.models import ModelStatus, ModelInstance
from supervisor.config import settings

logger = logging.getLogger(__name__)


class GatewaySync:
    """
    Push model list to LiteLLM via admin API (more reliable than fetch_from_url).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        litellm_admin_url: Optional[str] = None,
        master_key: Optional[str] = None,
    ):
        self.registry = registry
        self.admin_url = litellm_admin_url or settings.litellm_admin_url
        self.master_key = master_key or settings.litellm_master_key
        self.sync_interval = settings.gateway_sync_interval_seconds
        self.client = httpx.AsyncClient(timeout=30.0)
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background sync task."""
        if self._task is None:
            self._task = asyncio.create_task(self._sync_loop())
            logger.info("Gateway sync started")

    async def stop(self):
        """Stop background sync task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # CRITICAL: Close httpx client to prevent file descriptor leak
        await self.client.aclose()
        logger.info("Gateway sync stopped")

    async def _sync_loop(self):
        """Background task: sync every N seconds."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self.sync_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in gateway sync: {e}", exc_info=True)

    async def sync_models(self):
        """
        Push current model list to LiteLLM.

        Only includes RUNNING models (not suspended).
        Suspended models are excluded; auto-resume middleware handles them.
        """
        # Get running models only
        all_models = await self.registry.list_running()

        # Format for LiteLLM
        model_list = []
        for model in all_models:
            # Use alias if available, otherwise use last part of model name
            display_name = model.model_alias or model.model_name.split("/")[-1]

            model_list.append(
                {
                    "model_name": display_name,
                    "litellm_params": {
                        "model": f"openai/{display_name}",  # openai/ prefix for OpenAI-compatible
                        "api_base": f"{model.base_url}/v1",
                        "api_key": "EMPTY",
                    },
                }
            )

        logger.debug(f"Syncing {len(model_list)} running models to LiteLLM gateway")

        # Try pushing to LiteLLM admin API
        try:
            if self.master_key:
                response = await self.client.post(
                    f"{self.admin_url}/model/new",
                    headers={"Authorization": f"Bearer {self.master_key}"},
                    json={"models": model_list},
                )

                if response.status_code == 200:
                    logger.info(f"Synced {len(model_list)} models to LiteLLM gateway")
                    return
                else:
                    logger.warning(
                        f"Failed to sync models via admin API: {response.status_code} {response.text}"
                    )
            else:
                logger.debug("No master_key configured, skipping admin API push")

        except Exception as e:
            logger.error(f"Gateway sync error: {e}")

        # Fallback: rewrite litellm.yaml and trigger reload
        await self._fallback_yaml_reload(model_list)

    async def _fallback_yaml_reload(self, model_list: List[dict]):
        """
        Fallback: rewrite litellm.yaml and trigger reload.

        Args:
            model_list: List of model configurations
        """
        try:
            config_path = Path("gateway/litellm.yaml")

            # Read existing config
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}

            # Update model_list
            config["model_list"] = model_list

            # Write updated config
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(f"Fallback: rewrote litellm.yaml with {len(model_list)} models")

            # Trigger reload via admin API
            if self.master_key:
                try:
                    reload_response = await self.client.post(
                        f"{self.admin_url}/config/reload",
                        headers={"Authorization": f"Bearer {self.master_key}"},
                    )
                    if reload_response.status_code == 200:
                        logger.info("Triggered LiteLLM config reload")
                    else:
                        logger.warning(f"Config reload failed: {reload_response.status_code}")
                except Exception as e:
                    logger.error(f"Failed to trigger config reload: {e}")

        except Exception as e:
            logger.error(f"Fallback YAML reload failed: {e}", exc_info=True)

    async def get_models_for_litellm(self) -> List[dict]:
        """
        Get models in LiteLLM-compatible format (for /models endpoint).

        Returns:
            List of models in simplified LiteLLM format
        """
        running_models = await self.registry.list_running()

        models = []
        for model in running_models:
            display_name = model.model_alias or model.model_name.split("/")[-1]
            models.append(
                {
                    "model_name": display_name,
                    "litellm_provider": "openai",
                    "api_base": f"{model.base_url}/v1",
                    "api_key": "EMPTY",
                }
            )

        return models
