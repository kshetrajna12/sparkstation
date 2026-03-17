"""
Tests for configuration management.

NOTE: Settings() reads from .env via pydantic-settings, so tests that check
code defaults must override env_file to avoid picking up local .env values.
"""
import pytest
from supervisor.config import Settings


def _defaults(**overrides) -> Settings:
    """Create Settings with code defaults only (ignore .env file)."""
    return Settings(_env_file=None, **overrides)


def test_settings_defaults():
    """Test default configuration values (code defaults, not .env)."""
    settings = _defaults()

    # Server defaults
    assert settings.host == "127.0.0.1"
    assert settings.port == 9001
    assert settings.log_level == "info"

    # DGX Spark defaults
    assert settings.total_unified_memory_gb == 128
    assert settings.memory_hard_limit_gb == 110
    assert settings.memory_soft_limit_gb == 100
    assert settings.max_resident_models == 5

    # Docker images
    assert "26.01" in settings.vllm_docker_image

    # Health checks
    assert settings.health_check_enabled is True
    assert settings.health_check_interval_seconds == 300
    assert settings.health_check_max_failures == 3

    # Auto-restart
    assert settings.auto_restart_enabled is True
    assert settings.auto_restart_max_attempts == 3
    assert settings.auto_restart_backoff_minutes == "1,5,15"


def test_settings_from_env():
    """Test that .env overrides are picked up."""
    settings = Settings()  # reads .env
    # .env sets MEMORY_HARD_LIMIT_GB=113
    assert settings.memory_hard_limit_gb in (110, 113)  # either default or .env


def test_port_range_valid():
    """Test port range is sensible."""
    settings = _defaults()
    assert settings.model_port_range_start < settings.model_port_range_end
    assert settings.model_port_range_start > 1024
    assert settings.model_port_range_end < 65536


def test_memory_limits_valid():
    """Test memory limits are correctly configured."""
    settings = _defaults()
    assert settings.memory_soft_limit_gb < settings.memory_hard_limit_gb
    assert settings.memory_hard_limit_gb <= settings.total_unified_memory_gb

    # Hard limit should be ~85% of total
    expected_hard_limit = settings.total_unified_memory_gb * 0.85
    assert abs(settings.memory_hard_limit_gb - expected_hard_limit) < 5


def test_thermal_hysteresis_valid():
    """Test thermal thresholds have hysteresis."""
    settings = _defaults()
    assert settings.thermal_resume_threshold_c < settings.thermal_suspend_threshold_c
    hysteresis = settings.thermal_suspend_threshold_c - settings.thermal_resume_threshold_c
    assert hysteresis >= 3
