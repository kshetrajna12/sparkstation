"""
Supervisor configuration management for DGX Spark.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Global configuration for Sparkstation Supervisor."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server settings
    host: str = "127.0.0.1"  # CRITICAL: Localhost only for security
    port: int = 9001
    log_level: str = "info"

    # DGX Spark hardware constraints
    total_unified_memory_gb: int = 128  # Total unified CPU+GPU memory
    memory_hard_limit_gb: int = 110  # 85% of total
    memory_soft_limit_gb: int = 100  # 78% of total
    max_resident_models: int = 3  # Maximum concurrent models

    # Port allocation
    model_port_range_start: int = 8001
    model_port_range_end: int = 8100

    # Auto-suspend settings
    auto_suspend_enabled: bool = True
    default_idle_timeout_minutes: int = 30  # Default idle timeout
    auto_suspend_check_interval_seconds: int = 60  # Check every minute

    # Thermal management (DGX Spark specific)
    thermal_suspend_threshold_c: int = 80  # Suspend if above this temp
    thermal_resume_threshold_c: int = 75  # Resume only if below this
    thermal_sustain_ms: int = 60000  # Must sustain high temp for 60s
    thermal_cooldown_ms: int = 120000  # Cooldown period after thermal suspend

    # Health checks
    health_check_enabled: bool = True  # Enable health check background task
    health_check_interval_seconds: int = 300  # 5 minutes
    health_check_timeout_seconds: int = 5
    health_check_max_failures: int = 3  # Mark as FAILED after N consecutive failures

    # Model restart policy
    auto_restart_enabled: bool = True  # Auto-restart failed models
    auto_restart_max_attempts: int = 3  # Max restart attempts per model
    auto_restart_backoff_minutes: str = "1,5,15"  # Exponential backoff (comma-separated)

    # LiteLLM Gateway settings
    litellm_admin_url: str = "http://127.0.0.1:8000"
    litellm_master_key: Optional[str] = None  # For admin API
    gateway_sync_interval_seconds: int = 60  # Sync models every 60s

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/sparkstation.db"

    # Security
    api_key: Optional[str] = None  # Shared secret for Supervisor API

    # Logging
    log_to_file: bool = True  # Also log to file (in addition to stdout)
    log_file_path: str = "./data/sparkstation.log"  # Log file location
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB per file
    log_backup_count: int = 5  # Keep 5 backup files

    # Optional auto-sleep (unload all if system idle > 1 hour)
    auto_sleep_enabled: bool = False
    auto_sleep_idle_minutes: int = 60


# Global settings instance
settings = Settings()
