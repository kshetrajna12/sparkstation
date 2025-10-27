"""
Centralized error handling for Sparkstation Supervisor.

Provides custom exceptions and error response formatting.
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class SparkstationError(Exception):
    """Base exception for Sparkstation errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.suggestion = suggestion
        super().__init__(message)

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        content: Dict[str, Any] = {"error": self.message}

        if self.detail:
            content["detail"] = self.detail

        if self.suggestion:
            content["suggestion"] = self.suggestion

        return HTTPException(status_code=self.status_code, detail=content)


class ModelNotFoundError(SparkstationError):
    """Model instance not found in registry."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model {model_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model with ID '{model_id}' exists in the registry",
            suggestion="Check /models/detailed for available models",
        )


class ModelAlreadyExistsError(SparkstationError):
    """Model with same ID or alias already exists."""

    def __init__(self, identifier: str):
        super().__init__(
            message=f"Model {identifier} already exists",
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A model with ID or alias '{identifier}' is already running",
            suggestion="Use a different model_alias or stop the existing model first",
        )


class InsufficientResourcesError(SparkstationError):
    """Not enough resources to start model."""

    def __init__(self, reason: str, current_usage: Optional[float] = None, limit: Optional[float] = None):
        detail = reason
        if current_usage and limit:
            detail += f" (current: {current_usage:.1f}GB, limit: {limit:.1f}GB)"

        super().__init__(
            message="Insufficient resources to start model",
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=detail,
            suggestion="Stop or suspend other models, or wait for idle models to auto-suspend",
        )


class ModelLaunchError(SparkstationError):
    """Failed to launch model backend."""

    def __init__(self, backend: str, reason: str):
        super().__init__(
            message=f"Failed to launch {backend} model",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=reason,
            suggestion="Check logs for details, verify model name and quantization settings",
        )


class InvalidConfigError(SparkstationError):
    """Invalid model configuration."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Invalid configuration: {field}",
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=reason,
            suggestion="Check API documentation for valid configuration options",
        )


class ModelNotRunningError(SparkstationError):
    """Operation requires model to be running."""

    def __init__(self, model_id: str, current_status: str):
        super().__init__(
            message=f"Model {model_id} is not running",
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model status is '{current_status}', expected 'running'",
            suggestion="Start the model first, or check if it's suspended/failed",
        )


class ModelNotSuspendedError(SparkstationError):
    """Operation requires model to be suspended."""

    def __init__(self, model_id: str, current_status: str):
        super().__init__(
            message=f"Model {model_id} is not suspended",
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model status is '{current_status}', expected 'suspended'",
            suggestion="Suspend the model first, or check current status",
        )


class DatabaseError(SparkstationError):
    """Database operation failed."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Database {operation} failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=reason,
            suggestion="Check database file permissions and disk space",
        )


class HealthCheckError(SparkstationError):
    """Health check failed."""

    def __init__(self, model_id: str, reason: str):
        super().__init__(
            message=f"Health check failed for {model_id}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=reason,
            suggestion="Check model logs and resource usage",
        )


class QuantizationError(SparkstationError):
    """Quantization-related error."""

    def __init__(self, model_name: str, quantization: str, reason: str):
        super().__init__(
            message=f"Quantization error for {model_name}",
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot use '{quantization}' quantization: {reason}",
            suggestion="Try 'fp16' as fallback, or download quantized weights manually",
        )


def handle_exception(e: Exception) -> HTTPException:
    """
    Convert any exception to HTTPException with proper formatting.

    Args:
        e: Exception to handle

    Returns:
        HTTPException with formatted error response
    """
    if isinstance(e, SparkstationError):
        return e.to_http_exception()

    # Generic exception handling
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "Internal server error",
            "detail": str(e),
            "suggestion": "Check server logs for more information",
        },
    )
