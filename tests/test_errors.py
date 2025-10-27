"""
Tests for error handling.
"""
import pytest
from fastapi import status
from supervisor.errors import (
    ModelNotFoundError,
    ModelAlreadyExistsError,
    InsufficientResourcesError,
    ModelLaunchError,
    ModelNotRunningError,
    ModelNotSuspendedError,
)


def test_model_not_found_error():
    """Test ModelNotFoundError formatting."""
    error = ModelNotFoundError("test-model-123")

    assert error.status_code == status.HTTP_404_NOT_FOUND
    assert "test-model-123" in error.message
    assert error.suggestion is not None

    http_exc = error.to_http_exception()
    assert http_exc.status_code == 404


def test_model_already_exists_error():
    """Test ModelAlreadyExistsError formatting."""
    error = ModelAlreadyExistsError("qwen-7b")

    assert error.status_code == status.HTTP_409_CONFLICT
    assert "qwen-7b" in error.message
    assert error.suggestion is not None


def test_insufficient_resources_error():
    """Test InsufficientResourcesError formatting."""
    error = InsufficientResourcesError("Memory limit exceeded", 105.0, 110.0)

    assert error.status_code == status.HTTP_507_INSUFFICIENT_STORAGE
    assert "105.0" in error.detail
    assert "110.0" in error.detail
    assert error.suggestion is not None


def test_model_launch_error():
    """Test ModelLaunchError formatting."""
    error = ModelLaunchError("vllm", "Process crashed on startup")

    assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "vllm" in error.message
    assert "Process crashed" in error.detail


def test_model_not_running_error():
    """Test ModelNotRunningError formatting."""
    error = ModelNotRunningError("model-123", "suspended")

    assert error.status_code == status.HTTP_409_CONFLICT
    assert "model-123" in error.message
    assert "suspended" in error.detail
    assert "running" in error.detail  # Expected status


def test_model_not_suspended_error():
    """Test ModelNotSuspendedError formatting."""
    error = ModelNotSuspendedError("model-456", "running")

    assert error.status_code == status.HTTP_409_CONFLICT
    assert "model-456" in error.message
    assert "running" in error.detail
    assert "suspended" in error.detail  # Expected status
