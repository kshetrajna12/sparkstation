"""
Auto-Resume Middleware for LiteLLM.

CRITICAL FIX: LiteLLM doesn't know about suspended models.
This middleware intercepts requests, checks model status, and auto-resumes if suspended.

TODO: This middleware is NOT currently wired into the gateway. LiteLLM is started as a
standalone process (`uv run litellm --config gateway/litellm.yaml`) with no hook to add
custom ASGI middleware. Need to either:
  1. Wrap LiteLLM's app in a custom FastAPI/Starlette app that adds this middleware, or
  2. Use LiteLLM's custom callback/hook mechanism to achieve the same effect, or
  3. Run a reverse proxy in front of LiteLLM that handles auto-resume.
Until this is fixed, auto-suspend should be disabled for all models.
"""
import asyncio
import json
import logging
from typing import Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AutoResumeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to auto-resume suspended models before forwarding to LiteLLM.

    CRITICAL: Without this, requests to suspended models will 404/ECONNREFUSED.
    """

    def __init__(self, app, supervisor_url: str = "http://127.0.0.1:9001"):
        super().__init__(app)
        self.supervisor_url = supervisor_url
        # NOTE: This client lives for the entire gateway process lifetime.
        # It's intentionally long-lived (not a leak) since middleware is instantiated once.
        # Connection pooling benefits from reusing the same client across all requests.
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Auto-resume middleware initialized, supervisor: {supervisor_url}")

    async def dispatch(self, request: Request, call_next):
        """
        Intercept requests and auto-resume suspended models.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Only intercept /v1/* endpoints (LiteLLM API)
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        # Extract model name from request body
        model_name = None
        if request.method == "POST":
            try:
                # Read body
                body = await request.body()
                data = json.loads(body) if body else {}
                model_name = data.get("model")

                if model_name:
                    logger.debug(f"Request for model: {model_name}")

                    # Check if model is suspended
                    status = await self._check_model_status(model_name)

                    if status == "suspended":
                        logger.info(f"Model {model_name} is suspended, auto-resuming...")

                        # Resume model
                        resumed = await self._resume_model(model_name)

                        if not resumed:
                            return JSONResponse(
                                status_code=503,
                                content={
                                    "error": f"Failed to resume model {model_name}",
                                    "type": "model_suspended",
                                    "message": "Model is suspended and could not be resumed. Please try again.",
                                },
                            )

                        logger.info(f"Model {model_name} resumed successfully")

                # Reconstruct request with body
                # (FastAPI consumes body, need to restore it)
                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive

            except json.JSONDecodeError:
                # Invalid JSON, let it pass through
                pass
            except Exception as e:
                logger.error(f"Auto-resume middleware error: {e}", exc_info=True)

        # Forward to LiteLLM
        return await call_next(request)

    async def _check_model_status(self, model_name: str) -> str:
        """
        Check if model is suspended.

        Args:
            model_name: Model name or alias

        Returns:
            Status: "running", "suspended", "not_found", "unknown"
        """
        try:
            response = await self.client.get(f"{self.supervisor_url}/models/detailed")
            data = response.json()
            models = data.get("models", [])

            for model in models:
                # Match by alias, model name, or "default" keyword
                if model.get("alias") == model_name or model.get("model_name") == model_name:
                    return model.get("status", "unknown")
                if model_name == "default" and model.get("is_default"):
                    return model.get("status", "unknown")

            logger.warning(f"Model {model_name} not found in supervisor")
            return "not_found"

        except Exception as e:
            logger.error(f"Failed to check model status: {e}")
            return "unknown"

    async def _resume_model(self, model_name: str) -> bool:
        """
        Resume a suspended model and wait for it to be ready.

        Args:
            model_name: Model name or alias

        Returns:
            True if resumed successfully
        """
        try:
            # Find model ID from supervisor
            response = await self.client.get(f"{self.supervisor_url}/models/detailed")
            data = response.json()
            models = data.get("models", [])

            model_id = None
            for model in models:
                if model.get("alias") == model_name or model.get("model_name") == model_name:
                    model_id = model.get("id")
                    break
                if model_name == "default" and model.get("is_default"):
                    model_id = model.get("id")
                    break

            if not model_id:
                logger.error(f"Model {model_name} not found")
                return False

            # Resume model
            logger.info(f"Resuming model {model_id}...")
            resume_response = await self.client.post(
                f"{self.supervisor_url}/models/{model_id}/resume"
            )

            if resume_response.status_code != 200:
                logger.error(f"Failed to resume model: {resume_response.text}")
                return False

            # Wait for model to be ready (max 30 seconds)
            for attempt in range(30):
                await asyncio.sleep(1)

                status_response = await self.client.get(
                    f"{self.supervisor_url}/models/{model_id}/status"
                )
                status_data = status_response.json()
                status = status_data.get("status")

                logger.debug(f"Resume attempt {attempt + 1}/30, status: {status}")

                if status == "running":
                    # Double-check with health probe
                    health = status_data.get("health_status")
                    if health == "healthy" or health == "unknown":
                        # Unknown is okay - might just be starting
                        logger.info(f"Model {model_name} resumed and ready")
                        return True

            logger.error(f"Model {model_name} did not become ready within 30s")
            return False

        except Exception as e:
            logger.error(f"Failed to resume model: {e}", exc_info=True)
            return False
