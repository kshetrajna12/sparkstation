"""
Base interface for model launchers.
"""
from abc import ABC, abstractmethod
from typing import Optional
from supervisor.models import ModelConfig, ModelInstance


class ModelLauncher(ABC):
    """Abstract base class for model launchers."""

    @abstractmethod
    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float | None = None) -> ModelInstance:
        """
        Launch a model server.

        Args:
            config: Model configuration
            model_id: Unique model ID
            port: Allocated port
            memory_gb: Allocated memory in GB (drives gpu_memory_utilization /
                mem-fraction sizing and the pre-launch host headroom check)

        Returns:
            Model instance with runtime information

        Raises:
            LaunchError: If launch fails
        """
        pass

    @abstractmethod
    async def stop(self, instance: ModelInstance) -> bool:
        """
        Stop a model server.

        Args:
            instance: Model instance to stop

        Returns:
            True if stopped successfully
        """
        pass

    @abstractmethod
    async def health_check(self, instance: ModelInstance) -> bool:
        """
        Check if model is healthy.

        Args:
            instance: Model instance

        Returns:
            True if healthy
        """
        pass


class LaunchError(Exception):
    """Raised when model launch fails."""

    pass
