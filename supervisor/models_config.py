"""
Load and parse models.yaml configuration.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelConfigYAML(BaseModel):
    """Model configuration from YAML."""
    name: str
    alias: Optional[str] = None
    backend: str
    model_type: str = "chat"  # Default to chat for backward compatibility
    quantization: Optional[str] = "fp8"
    memory_gb: Optional[float] = None  # Explicit memory allocation (overrides estimation)
    idle_timeout_minutes: int = 30
    auto_suspend_enabled: bool = False
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5
    speculative_method: Optional[str] = None
    default: bool = False  # Mark as default model for the profile (one per profile)
    extra_args: Dict[str, Any] = Field(default_factory=dict)
    docker_image: Optional[str] = None  # Per-model Docker image override
    env_vars: Dict[str, str] = Field(default_factory=dict)  # Extra container env vars
    volumes: List[str] = Field(default_factory=list)  # Extra volume mounts (host:container)


class ModelsConfiguration(BaseModel):
    """Full models.yaml configuration."""
    autoload: Dict[str, Any] = Field(default_factory=dict)
    profiles: Dict[str, List[ModelConfigYAML]] = Field(default_factory=dict)


def load_models_config(config_path: Optional[Path] = None) -> ModelsConfiguration:
    """
    Load models configuration from YAML file.

    Args:
        config_path: Path to models.yaml, defaults to ./models.yaml

    Returns:
        Parsed configuration
    """
    if config_path is None:
        config_path = Path("models.yaml")

    if not config_path.exists():
        logger.warning(f"Models config not found: {config_path}, using defaults")
        return ModelsConfiguration()

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Parse autoload models
        autoload_config = data.get("autoload", {})
        if autoload_config.get("enabled") and "models" in autoload_config:
            autoload_config["models"] = [
                ModelConfigYAML(**model) for model in autoload_config["models"]
            ]

        # Parse profiles
        profiles = {}
        for profile_name, profile_models in data.get("profiles", {}).items():
            profiles[profile_name] = [
                ModelConfigYAML(**model) for model in profile_models
            ]

        return ModelsConfiguration(
            autoload=autoload_config,
            profiles=profiles
        )

    except Exception as e:
        logger.error(f"Failed to load models config: {e}")
        return ModelsConfiguration()


def get_autoload_models() -> List[ModelConfigYAML]:
    """Get models to auto-load on startup."""
    config = load_models_config()

    if not config.autoload.get("enabled"):
        return []

    return config.autoload.get("models", [])


def get_profile_models(profile_name: str) -> List[ModelConfigYAML]:
    """Get models for a named profile."""
    config = load_models_config()

    if profile_name not in config.profiles:
        logger.error(f"Profile not found: {profile_name}")
        return []

    return config.profiles[profile_name]


def get_default_model_alias(profile_name: Optional[str] = None) -> Optional[str]:
    """Get the alias of the default model for a profile (or autoload).

    Returns the alias of the model marked with `default: true`, or None if
    no model is marked as default.
    """
    if profile_name:
        models = get_profile_models(profile_name)
    else:
        models = get_autoload_models()

    for model in models:
        if model.default:
            return model.alias or model.name
    return None


def list_profiles() -> List[str]:
    """List available model profiles."""
    config = load_models_config()
    return list(config.profiles.keys())
