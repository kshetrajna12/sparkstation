"""
FLUX launcher for image generation with FLUX.1-dev.
"""
import asyncio
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx
from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend, ModelType
from supervisor.config import settings

logger = logging.getLogger(__name__)


class FluxLauncher(ModelLauncher):
    """FLUX image generation launcher."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """Cleanup resources (close httpx client)."""
        await self.client.aclose()

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        """
        Launch FLUX image generation server.

        Args:
            config: Model configuration
            model_id: Unique model ID
            port: Allocated port
            memory_gb: Allocated memory in GB

        Returns:
            Model instance

        Raises:
            LaunchError: If launch fails
        """
        logger.info(f"Launching FLUX image model: {config.model_name} on port {port}")

        # Get HuggingFace token from settings
        if not settings.hf_token:
            raise LaunchError("HF_TOKEN not configured in .env file. Required for FLUX.1-dev.")

        try:
            # Launch using Docker
            if settings.use_docker:
                # Check if flux-server image exists
                check_image = subprocess.run(
                    ["docker", "images", "-q", "flux-server:latest"],
                    capture_output=True,
                    text=True,
                )

                if not check_image.stdout.strip():
                    raise LaunchError(
                        "FLUX Docker image not found. Please build it first:\n"
                        "  cd docker/flux\n"
                        "  docker build --platform linux/arm64 -t flux-server:latest ."
                    )
                # Build docker run command for FLUX
                docker_cmd = [
                    "docker",
                    "run",
                    "-d",  # Detached mode
                    "--platform", "linux/arm64",  # Explicit ARM64 for DGX Spark
                    "--gpus", "all",  # GPU passthrough
                    "--shm-size", "32g",  # Shared memory for model loading
                    "--ipc=host",  # IPC mode host
                    "-p", f"{port}:8000",  # Port mapping (FLUX internal port is 8000)
                    "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",  # HuggingFace cache
                    "-e", f"HF_TOKEN={settings.hf_token}",  # HuggingFace token for gated model access
                    "-e", "HOST=0.0.0.0",  # Bind to all interfaces
                    "-e", "PORT=8000",  # Internal port
                    "--name", f"sparkstation-{model_id}",  # Container name
                    "flux-server:latest",
                ]

                logger.debug(f"Docker command: {' '.join(docker_cmd)}")

                # Launch Docker container
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                container_id = result.stdout.strip()
                logger.info(f"Docker container started: {container_id[:12]}, model_id={model_id}")

                # Wait a moment for container to start
                await asyncio.sleep(3)

                # Check if container is still running
                check_cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_id]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)

                if check_result.stdout.strip() != "true":
                    # Get container logs for debugging
                    logs_cmd = ["docker", "logs", container_id]
                    logs_result = subprocess.run(logs_cmd, capture_output=True, text=True)
                    error_context = logs_result.stdout + logs_result.stderr
                    raise LaunchError(f"Docker container failed to start. Logs:\n{error_context[:1000]}")

                # Create instance
                instance = ModelInstance(
                    id=model_id,
                    model_name=config.model_name,
                    model_alias=config.model_alias,
                    backend=Backend.FLUX,
                    model_type=ModelType.IMAGE,
                    status=ModelStatus.STARTING,
                    health_status=HealthStatus.UNKNOWN,
                    port=port,
                    gpu_ids=[0],  # DGX Spark: single GPU
                    base_url=f"http://127.0.0.1:{port}",
                    container_id=container_id,
                    started_at=datetime.now(),
                    auto_suspend_enabled=config.auto_suspend_enabled,
                    idle_timeout_minutes=config.idle_timeout_minutes,
                    extra_args=config.extra_args,
                )

                return instance

            else:
                raise LaunchError("FLUX subprocess mode not yet implemented. Please use Docker mode.")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to launch FLUX: {e.stderr}"
            logger.error(error_msg)
            raise LaunchError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error launching FLUX: {e}")
            raise LaunchError(str(e))

    async def stop(self, instance: ModelInstance) -> bool:
        """
        Stop FLUX model server.

        Args:
            instance: Model instance

        Returns:
            True if stopped successfully
        """
        try:
            if instance.container_id:
                # Stop Docker container
                result = subprocess.run(
                    ["docker", "stop", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    logger.info(f"Stopped FLUX container: {instance.container_id[:12]}")

                    # Remove container
                    subprocess.run(
                        ["docker", "rm", instance.container_id],
                        capture_output=True,
                        text=True,
                    )
                    logger.debug(f"Removed container: {instance.container_id[:12]}")
                    return True
                else:
                    logger.error(f"Failed to stop container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error stopping FLUX instance: {e}")
            return False

    async def suspend(self, instance: ModelInstance) -> bool:
        """
        Suspend FLUX model (pause Docker container).

        Args:
            instance: Model instance

        Returns:
            True if suspended successfully
        """
        try:
            if instance.container_id:
                result = subprocess.run(
                    ["docker", "pause", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    logger.info(f"Paused FLUX container: {instance.container_id[:12]}")
                    return True
                else:
                    logger.error(f"Failed to pause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error suspending FLUX instance: {e}")
            return False

    async def resume(self, instance: ModelInstance) -> bool:
        """
        Resume FLUX model (unpause Docker container).

        Args:
            instance: Model instance

        Returns:
            True if resumed successfully
        """
        try:
            if instance.container_id:
                result = subprocess.run(
                    ["docker", "unpause", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    logger.info(f"Unpaused FLUX container: {instance.container_id[:12]}")
                    return True
                else:
                    logger.error(f"Failed to unpause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error resuming FLUX instance: {e}")
            return False

    async def health_check(self, instance: ModelInstance) -> bool:
        """
        Perform health check for FLUX image generation server.

        Args:
            instance: Model instance

        Returns:
            True if healthy
        """
        try:
            response = await self.client.get(
                f"{instance.base_url}/health",
                timeout=settings.health_check_timeout_seconds,
            )
            return response.status_code == 200

        except Exception as e:
            logger.debug(f"Health check failed for {instance.id}: {e}")
            return False
