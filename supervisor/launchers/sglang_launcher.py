"""
SGLang launcher for DGX Spark with embeddings support.
"""
import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx
from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend, ModelType
from supervisor.config import settings

logger = logging.getLogger(__name__)


class SGLangLauncher(ModelLauncher):
    """DGX Spark-optimized SGLang launcher with embeddings support."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """Cleanup resources (close httpx client)."""
        await self.client.aclose()

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        """
        Launch SGLang model server.

        Args:
            config: Model configuration
            model_id: Unique model ID
            port: Allocated port
            memory_gb: Allocated memory in GB (used to calculate mem_fraction_static)

        Returns:
            Model instance

        Raises:
            LaunchError: If launch fails
        """
        # Check if this is an embedding model
        is_embedding = config.model_type == ModelType.EMBEDDING

        logger.info(f"Launching SGLang {config.model_type.value} model: {config.model_name} on port {port}")

        try:
            # Launch using Docker (recommended)
            if settings.use_docker:
                # Build docker run command for SGLang
                docker_cmd = [
                    "docker",
                    "run",
                    "-d",  # Detached mode
                    "--platform", "linux/arm64",  # Explicit ARM64 for DGX Spark
                    "--gpus", "all",  # GPU passthrough
                    "--shm-size", "32g",  # Shared memory for model loading
                    "--ipc=host",  # IPC mode host
                    "--ulimit", "memlock=-1",  # Unlimited memory locking
                    "--ulimit", "stack=67108864",  # 64MB stack
                    "-p", f"{port}:8000",  # Port mapping (SGLang internal port is 8000)
                    "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",  # HuggingFace cache
                    "--name", f"sparkstation-{model_id}",  # Container name
                    settings.sglang_docker_image,
                    "python3", "-m", "sglang.launch_server",
                    "--model-path", config.model_name,
                    "--host", "0.0.0.0",  # Bind to all interfaces
                    "--port", "8000",  # Internal port (mapped to external port)
                    "--trust-remote-code",  # Required for many models
                ]

                # Calculate mem_fraction_static from allocated memory_gb
                # DGX Spark: 119 GB usable memory (128 GB total - system overhead)
                if memory_gb is not None:
                    mem_fraction = memory_gb / 119.0
                    logger.info(f"Using mem_fraction_static={mem_fraction:.3f} (from {memory_gb}GB allocation)")
                    docker_cmd.extend(["--mem-fraction-static", str(mem_fraction)])
                else:
                    # SGLang default is 0.85
                    docker_cmd.extend(["--mem-fraction-static", "0.85"])

                # Add model-specific settings
                if not is_embedding:
                    # For chat models, add context length if specified
                    max_len = config.extra_args.get("max_model_len", 8192)
                    docker_cmd.extend(["--context-length", str(max_len)])

                # Add quantization if specified
                if config.quantization and config.quantization.lower() != "none":
                    docker_cmd.extend(["--quantization", config.quantization])

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
                    backend=Backend.SGLANG,
                    model_type=config.model_type,
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
                raise LaunchError("SGLang subprocess mode not yet implemented. Please use Docker mode.")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to launch SGLang: {e.stderr}"
            logger.error(error_msg)
            raise LaunchError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error launching SGLang: {e}")
            raise LaunchError(str(e))

    async def stop(self, instance: ModelInstance) -> bool:
        """
        Stop SGLang model server.

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
                    logger.info(f"Stopped SGLang container: {instance.container_id[:12]}")

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
            logger.error(f"Error stopping SGLang instance: {e}")
            return False

    async def suspend(self, instance: ModelInstance) -> bool:
        """
        Suspend SGLang model (pause Docker container).

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
                    logger.info(f"Paused SGLang container: {instance.container_id[:12]}")
                    return True
                else:
                    logger.error(f"Failed to pause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error suspending SGLang instance: {e}")
            return False

    async def resume(self, instance: ModelInstance) -> bool:
        """
        Resume SGLang model (unpause Docker container).

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
                    logger.info(f"Unpaused SGLang container: {instance.container_id[:12]}")
                    return True
                else:
                    logger.error(f"Failed to unpause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error resuming SGLang instance: {e}")
            return False

    async def health_check(self, instance: ModelInstance) -> bool:
        """
        Perform health check based on model type.
        - Chat models: 1-token chat completion
        - Embedding models: Simple embedding request

        Args:
            instance: Model instance

        Returns:
            True if healthy
        """
        try:
            if instance.model_type == ModelType.EMBEDDING:
                # Embedding model health check
                response = await self.client.post(
                    f"{instance.base_url}/v1/embeddings",
                    json={
                        "input": "test",
                        "model": instance.model_name,
                    },
                    timeout=settings.health_check_timeout_seconds,
                )
            else:
                # Chat model health check (1-token completion)
                response = await self.client.post(
                    f"{instance.base_url}/v1/chat/completions",
                    json={
                        "model": instance.model_name,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                    timeout=settings.health_check_timeout_seconds,
                )

            return response.status_code == 200

        except Exception as e:
            logger.debug(f"Health check failed for {instance.id}: {e}")
            return False
