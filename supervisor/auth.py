"""
API key authentication for Supervisor endpoints.

Implements X-API-Key header validation with configurable enforcement.
"""
import hmac
import logging
from typing import Optional

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader

from supervisor.config import settings

logger = logging.getLogger(__name__)

# API key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Exempt endpoints (no authentication required)
EXEMPT_PATHS = {
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def is_path_exempt(path: str) -> bool:
    """Check if request path is exempt from authentication."""
    return path in EXEMPT_PATHS or path.startswith("/docs") or path.startswith("/redoc")


async def require_api_key(request: Request) -> None:
    """
    FastAPI dependency to enforce API key authentication.

    Validates X-API-Key header against settings.api_key.

    Behavior:
    - If settings.api_key is None: Authentication disabled (backwards compatible)
    - If settings.api_key is set: Requires valid X-API-Key header
    - Exempt paths: /health, /metrics, /docs (for monitoring)

    Raises:
        HTTPException: 401 Unauthorized if API key invalid or missing

    Usage:
        @app.post("/models/start", dependencies=[Depends(require_api_key)])
        async def start_model(...):
            ...
    """
    # Check if path is exempt
    if is_path_exempt(request.url.path):
        return

    # If no API key configured, allow all requests (backwards compatible)
    if settings.api_key is None:
        return

    # Extract API key from header
    provided_key = request.headers.get("X-API-Key")

    if not provided_key:
        logger.warning(
            f"Unauthorized request to {request.url.path}: Missing X-API-Key header"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header. Authentication required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key (constant-time compare — plain != leaks timing)
    if not hmac.compare_digest(provided_key, settings.api_key):
        logger.warning(
            f"Unauthorized request to {request.url.path}: Invalid X-API-Key"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-API-Key. Authentication failed.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Authentication successful
    logger.debug(f"Authenticated request to {request.url.path}")


class APIKeyAuth:
    """
    Optional: Class-based API key authentication.

    Provides more flexibility for advanced use cases (multiple keys, rate limiting, etc.).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.api_key

    async def __call__(self, request: Request) -> None:
        """Callable to use as FastAPI dependency."""
        # Check if path is exempt
        if is_path_exempt(request.url.path):
            return

        # If no API key configured, allow all
        if self.api_key is None:
            return

        # Extract and validate
        provided_key = request.headers.get("X-API-Key")

        if not provided_key or not hmac.compare_digest(provided_key, self.api_key):
            logger.warning(f"Unauthorized request to {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
