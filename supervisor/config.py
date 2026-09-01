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

    # Backend configuration
    # Docker mode: Use Docker containers for vLLM backend (recommended for production)
    # Subprocess mode: Use direct Python execution from conda/micromamba environments
    use_docker: bool = True  # Default to Docker for better isolation and easier setup
    vllm_docker_image: str = "nvcr.io/nvidia/vllm:26.01-py3"  # NVIDIA official image with Blackwell support
    sglang_docker_image: str = "nvcr.io/nvidia/sglang:26.01-py3"  # NVIDIA SGLang image with Blackwell support

    # HuggingFace token for gated models (FLUX.1-dev, etc.)
    hf_token: Optional[str] = None

    # Backend Python path (for subprocess mode only)
    # If use_docker=False, this path points to conda/micromamba environment
    vllm_python_path: Optional[str] = None

    # DGX Spark hardware constraints
    total_unified_memory_gb: int = 128  # Total unified CPU+GPU memory
    memory_hard_limit_gb: int = 110  # 85% of total
    memory_soft_limit_gb: int = 100  # 78% of total
    max_resident_models: int = 5  # Maximum concurrent models (2 chat + 2 embedding + headroom)
    # Coexistence-aware allocation floor (2026-08-15): the primary admission
    # check is now "would MemAvailable stay above this floor" — ownership-
    # agnostic, so externally-managed models (e.g. the DSV4 2x-Spark stack)
    # occupying memory don't produce phantom rejections the way the old
    # max(gpu, system)+16GB estimate did.
    memory_safety_floor_gb: int = 8

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
    # A model stuck in STARTING longer than this is marked FAILED (and picked
    # up by the RestartManager). Generous on purpose: a 35B cold load with
    # torch.compile can take 10-20 minutes on a Spark.
    starting_timeout_minutes: int = 30

    # Model restart policy
    auto_restart_enabled: bool = True  # Auto-restart failed models
    auto_restart_max_attempts: int = 3  # Max restart attempts per model
    auto_restart_backoff_minutes: str = "1,5,15"  # Exponential backoff (comma-separated)
    # Reset restart_count once a model has stayed healthy this long after its
    # last restart. Without decay, an intermittent-but-recoverable crash (e.g.
    # the marlin cudaErrorUnknown every ~5h) accumulates to max_attempts over
    # days and the model goes permanently FAILED despite every restart working.
    restart_count_reset_minutes: int = 60
    auto_restart_watch_interval_seconds: int = 30  # How often the RestartManager
    # polls for FAILED models that no code path has yet triggered a restart for.
    # 30s means a crash detected via reconcile / container-exit is recovered
    # within ~30s + backoff (default 1 min).

    # LiteLLM Gateway settings
    litellm_admin_url: str = "http://127.0.0.1:8000"
    litellm_master_key: Optional[str] = None  # For admin API
    gateway_sync_interval_seconds: int = 60  # Sync models every 60s

    # Database (using SUPERVISOR_DATABASE_URL to avoid conflict with LiteLLM's auto-detection)
    supervisor_database_url: str = "sqlite+aiosqlite:///./data/sparkstation.db"

    # Security
    api_key: Optional[str] = None  # Shared secret for Supervisor API

    # Sparkstation Console (static SPA at /console/, Voice Studio API at /voice/*)
    console_enabled: bool = True
    # API key the playground proxy presents to the gateway (clients.yaml
    # 'console' entry) so console traffic is attributed under its own client.
    gateway_console_key: str = "console"
    # Optional link-out for the Metrics section (Grafana is not rebuilt in the
    # console). Leave unset to hide the link. Set via env: CONSOLE_GRAFANA_URL.
    console_grafana_url: Optional[str] = None

    # Logging
    log_to_file: bool = True  # Also log to file (in addition to stdout)
    log_file_path: str = "./data/sparkstation.log"  # Log file location
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB per file
    log_backup_count: int = 5  # Keep 5 backup files

    # Startup profile (load named profile instead of autoload models)
    startup_profile: Optional[str] = None

    # Optional auto-sleep (unload all if system idle > 1 hour)
    auto_sleep_enabled: bool = False
    auto_sleep_idle_minutes: int = 60


# Global settings instance
settings = Settings()
