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
    quantization: Optional[str] = "fp8"
    idle_timeout_minutes: int = 30
    auto_suspend_enabled: bool = True
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5
    speculative_method: Optional[str] = None
    extra_args: Dict[str, Any] = Field(default_factory=dict)


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


def list_profiles() -> List[str]:
    """List available model profiles."""
    config = load_models_config()
    return list(config.profiles.keys())
