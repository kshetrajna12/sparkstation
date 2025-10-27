"""
Tests for API key authentication.
"""
import pytest
from fastapi import Request, HTTPException
from supervisor.auth import require_api_key, is_path_exempt
from supervisor.config import settings


def test_exempt_paths():
    """Test that monitoring paths are exempt from auth."""
    assert is_path_exempt("/health") is True
    assert is_path_exempt("/metrics") is True
    assert is_path_exempt("/docs") is True
    assert is_path_exempt("/openapi.json") is True
    assert is_path_exempt("/redoc") is True


def test_protected_paths():
    """Test that model management paths require auth."""
    assert is_path_exempt("/models/start") is False
    assert is_path_exempt("/models/123/stop") is False
    assert is_path_exempt("/models/123/suspend") is False
    assert is_path_exempt("/models/123/resume") is False


@pytest.mark.asyncio
async def test_require_api_key_no_key_configured():
    """Test authentication bypassed when no API key configured."""
    # Save original
    original_key = settings.api_key
    settings.api_key = None

    # Create mock request
    class MockRequest:
        url = type('obj', (object,), {'path': '/models/start'})()
        headers = {}

    request = MockRequest()

    # Should not raise
    await require_api_key(request)

    # Restore
    settings.api_key = original_key


@pytest.mark.asyncio
async def test_require_api_key_exempt_path():
    """Test exempt paths bypass authentication."""
    # Set API key
    original_key = settings.api_key
    settings.api_key = "test-key"

    class MockRequest:
        url = type('obj', (object,), {'path': '/health'})()
        headers = {}

    request = MockRequest()

    # Should not raise even without key header
    await require_api_key(request)

    # Restore
    settings.api_key = original_key
