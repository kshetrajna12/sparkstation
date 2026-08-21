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
from supervisor.cluster_helpers import merged_env, base_url_for_host

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
                # Cluster-aware env: local for host=primary, DOCKER_HOST=ssh://...
                # for remote roles.
                subprocess_env = merged_env(config.host)

                # Build docker run command for SGLang. Docker-level flags first
                # (up to and including the image name), then sglang args.
                docker_image = config.docker_image or settings.sglang_docker_image
                docker_cmd = [
                    "docker",
                    "run",
                    "-d",  # Detached mode; sparkstation's restart_manager owns
                           # relaunch, so NO docker --restart policy here.
                    "--platform", "linux/arm64",  # Explicit ARM64 for DGX Spark
                    "--gpus", "all",  # GPU passthrough
                    "--shm-size", "32g",  # Shared memory for model loading
                    "--ipc=host",  # IPC mode host
                    "--ulimit", "memlock=-1",  # Unlimited memory locking
                    "--ulimit", "stack=67108864",  # 64MB stack
                    "-p", f"{port}:8000",  # Port mapping (SGLang internal port is 8000)
                    "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",  # HuggingFace cache
                ]

                # Optional docker memory cgroup cap — a hard safety backstop.
                # On GB10 unified memory an over-large --mem-fraction-static can
                # starve the OS during load and HANG the whole node (0.88 wedged
                # worker1, 2026-08-18, only recoverable by unplugging). A cgroup
                # cap makes the container OOM-killable BEFORE it takes the box.
                dm = config.extra_args.get("docker_memory_gb")
                if dm:
                    docker_cmd.extend(["--memory", f"{dm}g", "--memory-swap", f"{dm}g"])

                # Optional docker-level flags some recipes require. The DFlash2
                # daily driver runs --privileged and pins to the Cortex-X5 cores
                # (--cpuset-cpus) per its qualified recipe; both are docker args
                # (not sglang args) so they can't ride in sglang_flags.
                if config.extra_args.get("docker_privileged"):
                    docker_cmd.append("--privileged")
                cpuset = config.extra_args.get("docker_cpuset")
                if cpuset:
                    docker_cmd.extend(["--cpuset-cpus", str(cpuset)])

                # Per-model env vars (e.g. TORCHINDUCTOR_CACHE_DIR for a
                # persistent torch.compile cache → fast subsequent boots).
                for env_key, env_val in (config.env_vars or {}).items():
                    docker_cmd.extend(["-e", f"{env_key}={env_val}"])

                # Extra volume mounts (host:container) on the TARGET host — e.g.
                # a patched chat template dir. Relative host paths resolve under
                # the project dir; absolute pass through.
                project_dir = Path.cwd()
                for vol in (config.volumes or []):
                    parts = vol.split(":", 1)
                    if len(parts) == 2 and not Path(parts[0]).is_absolute():
                        vol = f"{project_dir / parts[0]}:{parts[1]}"
                    docker_cmd.extend(["-v", vol])

                docker_cmd += [
                    "--name", f"sparkstation-{model_id}",  # Container name
                    docker_image,
                    "python3", "-m", "sglang.launch_server",
                    "--model-path", config.model_name,
                    "--host", "0.0.0.0",  # Bind to all interfaces
                    "--port", "8000",  # Internal port (mapped to external port)
                    "--trust-remote-code",  # Required for many models
                ]

                # Calculate mem_fraction_static from allocated memory_gb, with a
                # SAFETY CLAMP: never exceed 0.82 on unified memory (see the
                # wedge note above; 0.88 hung the node, 0.78 booted clean).
                if memory_gb is not None:
                    mem_fraction = min(memory_gb / 119.0, 0.82)
                    logger.info(f"Using mem_fraction_static={mem_fraction:.3f} (from {memory_gb}GB, clamped ≤0.82)")
                    docker_cmd.extend(["--mem-fraction-static", str(mem_fraction)])
                else:
                    docker_cmd.extend(["--mem-fraction-static", "0.80"])

                # Add model-specific settings
                if is_embedding:
                    # For embedding models, add --is-embedding flag (required by SGLang)
                    docker_cmd.append("--is-embedding")
                else:
                    # For chat models, add context length UNLESS the model asks
                    # for its native default. Passing --context-length explicitly
                    # changes the torch.compile graph shape vs SGLang's default —
                    # which cost a cache MISS and a Triton codegen crash on the
                    # daily driver (2026-08-18). "native" (or an unset value) →
                    # omit the flag so SGLang uses the model's own max, matching
                    # the hand-rolled recipe and reusing its compiled kernels.
                    max_len = config.extra_args.get("max_model_len")
                    if max_len not in (None, "native", "auto", 0, "0"):
                        docker_cmd.extend(["--context-length", str(max_len)])
                    else:
                        logger.info("Omitting --context-length (native): SGLang will use the model default")

                # Add quantization if specified
                if config.quantization and config.quantization.lower() != "none":
                    docker_cmd.extend(["--quantization", config.quantization])

                # Raw SGLang flag passthrough — a list of already-split args
                # appended verbatim. This is how the daily driver specifies the
                # DSpark spec-decode, fp8 KV cache, float32 SSM, torch-compile,
                # mamba, and chat-template flags without the launcher needing to
                # know each one. e.g. sglang_flags: ["--kv-cache-dtype",
                # "fp8_e4m3", "--speculative-algorithm", "DSPARK", ...].
                for flag in config.extra_args.get("sglang_flags", []):
                    docker_cmd.append(str(flag))

                logger.debug(f"Docker command: {' '.join(docker_cmd)}")

                # Launch Docker container on the assigned host
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=subprocess_env,
                )

                container_id = result.stdout.strip()
                logger.info(f"Docker container started on host={config.host}: {container_id[:12]}, model_id={model_id}")

                # Wait a moment for container to start
                await asyncio.sleep(3)

                # Check if container is still running (same host)
                check_cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_id]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True, env=subprocess_env)

                if check_result.stdout.strip() != "true":
                    # Get container logs for debugging
                    logs_cmd = ["docker", "logs", container_id]
                    logs_result = subprocess.run(logs_cmd, capture_output=True, text=True, env=subprocess_env)
                    error_context = logs_result.stdout + logs_result.stderr
                    raise LaunchError(f"Docker container failed to start. Logs:\n{error_context[:1000]}")

                # Create instance. base_url resolves to the target host's IP so
                # the gateway + health-check can reach it over QSFP.
                instance = ModelInstance(
                    id=model_id,
                    model_name=config.model_name,
                    model_alias=config.model_alias,
                    backend=Backend.SGLANG,
                    model_type=config.model_type,
                    host=config.host,
                    status=ModelStatus.STARTING,
                    health_status=HealthStatus.UNKNOWN,
                    port=port,
                    gpu_ids=[0],  # DGX Spark: single GPU
                    base_url=base_url_for_host(config.host, port),
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
                subprocess_env = merged_env(instance.host or "primary")
                # Stop Docker container
                result = subprocess.run(
                    ["docker", "stop", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=subprocess_env,
                )

                if result.returncode == 0:
                    logger.info(f"Stopped SGLang container: {instance.container_id[:12]} on host={instance.host or 'primary'}")

                    # Remove container
                    subprocess.run(
                        ["docker", "rm", instance.container_id],
                        capture_output=True,
                        text=True,
                        env=subprocess_env,
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
                subprocess_env = merged_env(instance.host or "primary")
                result = subprocess.run(
                    ["docker", "pause", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=subprocess_env,
                )

                if result.returncode == 0:
                    logger.info(f"Paused SGLang container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
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
                subprocess_env = merged_env(instance.host or "primary")
                result = subprocess.run(
                    ["docker", "unpause", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=subprocess_env,
                )

                if result.returncode == 0:
                    logger.info(f"Unpaused SGLang container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
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
