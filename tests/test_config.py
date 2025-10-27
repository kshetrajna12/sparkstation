"""
Tests for configuration management.
"""
import pytest
from supervisor.config import Settings


def test_settings_defaults():
    """Test default configuration values."""
    settings = Settings()

    # Server defaults
    assert settings.host == "127.0.0.1"
    assert settings.port == 9001
    assert settings.log_level == "info"

    # DGX Spark defaults
    assert settings.total_unified_memory_gb == 128
    assert settings.memory_hard_limit_gb == 110
    assert settings.memory_soft_limit_gb == 100
    assert settings.max_resident_models == 3

    # Health checks
    assert settings.health_check_enabled is True
    assert settings.health_check_interval_seconds == 300
    assert settings.health_check_max_failures == 3

    # Auto-restart
    assert settings.auto_restart_enabled is True
    assert settings.auto_restart_max_attempts == 3
    assert settings.auto_restart_backoff_minutes == "1,5,15"


def test_port_range_valid():
    """Test port range is sensible."""
    settings = Settings()
    assert settings.model_port_range_start < settings.model_port_range_end
    assert settings.model_port_range_start > 1024  # Above privileged ports
    assert settings.model_port_range_end < 65536  # Below max port


def test_memory_limits_valid():
    """Test memory limits are correctly configured."""
    settings = Settings()
    assert settings.memory_soft_limit_gb < settings.memory_hard_limit_gb
    assert settings.memory_hard_limit_gb <= settings.total_unified_memory_gb

    # Hard limit should be ~85% of total
    expected_hard_limit = settings.total_unified_memory_gb * 0.85
    assert abs(settings.memory_hard_limit_gb - expected_hard_limit) < 5


def test_thermal_hysteresis_valid():
    """Test thermal thresholds have hysteresis."""
    settings = Settings()
    assert settings.thermal_resume_threshold_c < settings.thermal_suspend_threshold_c

    # Should have at least 3°C hysteresis
    hysteresis = settings.thermal_suspend_threshold_c - settings.thermal_resume_threshold_c
    assert hysteresis >= 3
