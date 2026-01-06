"""
CLIP launcher for image and text embeddings.
"""
import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import httpx
from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend, ModelType
from supervisor.config import settings

logger = logging.getLogger(__name__)


class CLIPLauncher(ModelLauncher):
    """CLIP embedding launcher using custom transformers-based server."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """Cleanup resources (close httpx client)."""
        await self.client.aclose()

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        """
        Launch CLIP embedding server.

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
        logger.info(f"Launching CLIP embedding model: {config.model_name} on port {port}")

        try:
            if settings.use_docker:
                # Check if clip-server image exists
                check_image = subprocess.run(
                    ["docker", "images", "-q", "clip-server:latest"],
                    capture_output=True,
                    text=True,
                )

                if not check_image.stdout.strip():
                    raise LaunchError(
                        "CLIP Docker image not found. Please build it first:\n"
                        "  cd docker/clip\n"
                        "  docker build --platform linux/arm64 -t clip-server:latest ."
                    )

                # Build docker run command
                docker_cmd = [
                    "docker",
                    "run",
                    "-d",  # Detached mode
                    "--platform", "linux/arm64",  # Explicit ARM64 for DGX Spark
                    "--gpus", "all",  # GPU passthrough
                    "--shm-size", "16g",  # Shared memory
                    "--ipc=host",  # IPC mode host
                    "-p", f"{port}:8000",  # Port mapping
                    "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",  # HuggingFace cache
                    "-e", "HOST=0.0.0.0",
                    "-e", "PORT=8000",
                    "-e", f"MODEL_PATH={config.model_name}",
                    "--name", f"sparkstation-{model_id}",
                    "clip-server:latest",
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

                # Wait for container to start
                await asyncio.sleep(3)

                # Check if container is still running
                check_cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_id]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)

                if check_result.stdout.strip() != "true":
                    logs_cmd = ["docker", "logs", container_id]
                    logs_result = subprocess.run(logs_cmd, capture_output=True, text=True)
                    error_context = logs_result.stdout + logs_result.stderr
                    raise LaunchError(f"Docker container failed to start. Logs:\n{error_context[:1000]}")

                # Create instance
                instance = ModelInstance(
                    id=model_id,
                    model_name=config.model_name,
                    model_alias=config.model_alias,
                    backend=Backend.CLIP,
                    model_type=ModelType.EMBEDDING,
                    status=ModelStatus.STARTING,
                    health_status=HealthStatus.UNKNOWN,
                    port=port,
                    gpu_ids=[0],
                    base_url=f"http://127.0.0.1:{port}",
                    container_id=container_id,
                    started_at=datetime.now(),
                    auto_suspend_enabled=config.auto_suspend_enabled,
                    idle_timeout_minutes=config.idle_timeout_minutes,
                    extra_args=config.extra_args,
                )

                return instance

            else:
                raise LaunchError("CLIP subprocess mode not implemented. Please use Docker mode.")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to launch CLIP: {e.stderr}"
            logger.error(error_msg)
            raise LaunchError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error launching CLIP: {e}")
            raise LaunchError(str(e))

    async def stop(self, instance: ModelInstance) -> bool:
        """Stop CLIP server."""
        try:
            if instance.container_id:
                result = subprocess.run(
                    ["docker", "stop", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    logger.info(f"Stopped CLIP container: {instance.container_id[:12]}")
                    subprocess.run(
                        ["docker", "rm", instance.container_id],
                        capture_output=True,
                        text=True,
                    )
                    return True
                else:
                    logger.error(f"Failed to stop container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error stopping CLIP instance: {e}")
            return False

    async def suspend(self, instance: ModelInstance) -> bool:
        """Suspend CLIP model (pause Docker container)."""
        try:
            if instance.container_id:
                result = subprocess.run(
                    ["docker", "pause", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    logger.info(f"Paused CLIP container: {instance.container_id[:12]}")
                    return True
                else:
                    logger.error(f"Failed to pause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error suspending CLIP instance: {e}")
            return False

    async def resume(self, instance: ModelInstance) -> bool:
        """Resume CLIP model (unpause Docker container)."""
        try:
            if instance.container_id:
                result = subprocess.run(
                    ["docker", "unpause", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    logger.info(f"Unpaused CLIP container: {instance.container_id[:12]}")
                    return True
                else:
                    logger.error(f"Failed to unpause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error resuming CLIP instance: {e}")
            return False

    async def health_check(self, instance: ModelInstance) -> bool:
        """Perform health check for CLIP server."""
        try:
            response = await self.client.get(
                f"{instance.base_url}/health",
                timeout=settings.health_check_timeout_seconds,
            )
            return response.status_code == 200

        except Exception as e:
            logger.debug(f"Health check failed for {instance.id}: {e}")
            return False
