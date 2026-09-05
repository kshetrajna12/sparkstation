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


def _litellm_params(model_name: str, base_url: str, model_type) -> dict:
    """Build litellm_params, forwarding reasoning controls for chat models."""
    mt = model_type.value if hasattr(model_type, "value") else (model_type or "chat")
    params = {
        "model": f"openai/{model_name}",
        "api_base": f"{base_url}/v1",
        "api_key": "EMPTY",
    }
    if mt == "chat":
        # Forward (don't drop) reasoning-control params like chat_template_kwargs
        # so a client's per-request thinking reaches the backend. drop_params:false
        # routes unknown fields into the downstream extra_body (request body).
        # NOTE: do NOT list them in allowed_openai_params — that promotes them to
        # "known" params which LiteLLM passes as kwargs to the OpenAI SDK, which
        # rejects chat_template_kwargs ("unexpected keyword argument").
        params["drop_params"] = False
    else:
        # Embeddings / detection: keep dropping unsupported params.
        params["drop_params"] = True
    return params


def _model_info(is_vision: bool) -> dict | None:
    """LiteLLM `model_info` block so clients can DISCOVER capabilities.

    models.yaml already knows which model is the profile's VLM (`vision_default`),
    but until 2026-09-05 that only shaped the `vision` alias — `/v1/models` is the
    bare OpenAI shape (id/object/created/owned_by) and `/model/info` reported
    `supports_vision: None`. Every client therefore had to hardcode "this alias is
    multimodal", which silently rotted on each model swap (openclaw kept routing
    images to a retired Qwen3-VL sidecar). Emitting supports_vision makes the
    capability readable over the API instead of guessed.
    """
    return {"supports_vision": True} if is_vision else None


class GatewaySync:
    """
    Push model list to LiteLLM via admin API (more reliable than fetch_from_url).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        litellm_admin_url: Optional[str] = None,
        master_key: Optional[str] = None,
        default_model_alias: Optional[str] = None,
        vision_model_alias: Optional[str] = None,
        vision_capable_aliases: Optional[set] = None,
    ):
        self.registry = registry
        self.admin_url = litellm_admin_url or settings.litellm_admin_url
        self.master_key = master_key or settings.litellm_master_key
        self.sync_interval = settings.gateway_sync_interval_seconds
        self.default_model_alias = default_model_alias
        self.vision_model_alias = vision_model_alias
        # Every alias that accepts images, not just the "vision" alias target.
        self.vision_capable_aliases = set(vision_capable_aliases or ())
        if vision_model_alias:
            self.vision_capable_aliases.add(vision_model_alias)
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
        # Sync immediately on startup, then loop
        try:
            await self.sync_models()
        except Exception as e:
            logger.error(f"Initial gateway sync failed: {e}", exc_info=True)

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
            # Voice models speak WebSocket/WebRTC audio, not the OpenAI API —
            # a LiteLLM route would just 502. Clients connect to them directly.
            mt = getattr(model.model_type, "value", model.model_type)
            if mt == "voice":
                continue
            # Use alias for display name, but actual model_name for the backend
            display_name = model.model_alias or model.model_name.split("/")[-1]

            entry = {
                "model_name": display_name,
                "litellm_params": _litellm_params(
                    model.model_name, model.base_url, model.model_type
                ),
            }
            info = _model_info(display_name in self.vision_capable_aliases)
            if info:
                entry["model_info"] = info
            model_list.append(entry)

        # Add "default" alias pointing to the default model
        if self.default_model_alias:
            for model in all_models:
                display_name = model.model_alias or model.model_name.split("/")[-1]
                if display_name == self.default_model_alias:
                    entry = {
                        "model_name": "default",
                        "litellm_params": _litellm_params(
                            model.model_name, model.base_url, model.model_type
                        ),
                    }
                    # The daily driver is usually natively multimodal and is its
                    # own vision_default, so "default" must advertise vision too
                    # — clients pinned to "default" (openclaw) otherwise think
                    # they're on a text-only model.
                    info = _model_info(display_name in self.vision_capable_aliases)
                    if info:
                        entry["model_info"] = info
                    model_list.append(entry)
                    break

        # Add "vision" alias pointing to the profile's vision model — same
        # mechanism as "default": consumers (ds4f-vision MCP etc.) request
        # model "vision" and it follows profile switches instead of
        # hardcoding an alias that may not exist in every profile.
        if self.vision_model_alias:
            for model in all_models:
                display_name = model.model_alias or model.model_name.split("/")[-1]
                if display_name == self.vision_model_alias:
                    model_list.append(
                        {
                            "model_name": "vision",
                            "litellm_params": _litellm_params(
                                model.model_name, model.base_url, model.model_type
                            ),
                            "model_info": _model_info(True),
                        }
                    )
                    break

        logger.debug(f"Syncing {len(model_list)} running models to LiteLLM gateway")

        # Write model list to litellm.yaml (gateway reads this at startup).
        # NOTE: LiteLLM admin API (/model/new) requires a database which we don't use.
        # The YAML-based approach is simpler and more reliable.
        await self._fallback_yaml_reload(model_list)

    async def _fallback_yaml_reload(self, model_list: List[dict]):
        """
        Fallback: rewrite litellm.yaml so the gateway picks up changes on next restart.

        NOTE: LiteLLM 1.79+ removed /config/reload and /model/new requires a DB.
        The YAML rewrite is sufficient — the gateway reads it at startup, and the
        deploy script restarts the gateway after model changes.

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

            # Only write on a REAL change. The CLI runs LiteLLM under a
            # watcher that restarts it when this file's mtime moves, so an
            # unconditional rewrite every sync pass would bounce the gateway
            # every N seconds.
            if config.get("model_list") == model_list:
                logger.debug(f"litellm.yaml already current ({len(model_list)} models); skipping write")
                return

            # Update model_list
            config["model_list"] = model_list

            # Write updated config ATOMICALLY (tmp + rename): the CLI's
            # watcher md5-polls this file every 5s, and a plain open("w")
            # lets it read a half-written config — one logical change then
            # looks like two, double-bouncing LiteLLM.
            tmp_path = config_path.with_suffix(".yaml.tmp")
            with open(tmp_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            tmp_path.replace(config_path)

            logger.info(f"Wrote litellm.yaml with {len(model_list)} models (gateway watcher will restart LiteLLM)")

        except Exception as e:
            logger.error(f"YAML rewrite failed: {e}", exc_info=True)

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

            # Add "default" alias for the default model
            if self.default_model_alias and display_name == self.default_model_alias:
                models.append(
                    {
                        "model_name": "default",
                        "litellm_provider": "openai",
                        "api_base": f"{model.base_url}/v1",
                        "api_key": "EMPTY",
                    }
                )

            # Add "vision" alias for the profile's vision model
            if self.vision_model_alias and display_name == self.vision_model_alias:
                models.append(
                    {
                        "model_name": "vision",
                        "litellm_provider": "openai",
                        "api_base": f"{model.base_url}/v1",
                        "api_key": "EMPTY",
                    }
                )

        return models
