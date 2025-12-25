"""
vLLM launcher for DGX Spark with quantization support.
"""
import asyncio
import logging
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import httpx
from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend, ModelType
from supervisor.config import settings

logger = logging.getLogger(__name__)


class VLLMLauncher(ModelLauncher):
    """DGX Spark-optimized vLLM launcher with mandatory quantization."""

    # Backend-specific quantization flag mapping
    QUANTIZATION_MAP = {
        "fp8": "fp8",  # vLLM native fp8
        "int4": "awq",  # vLLM uses AWQ for int4
        "awq": "awq",  # Direct AWQ
        "gptq": "gptq",  # GPTQ quantization
        "none": None,  # No quantization (for models with built-in quantization)
    }

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """Cleanup resources (close httpx client)."""
        await self.client.aclose()

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        """
        Launch vLLM model server.

        Args:
            config: Model configuration
            model_id: Unique model ID
            port: Allocated port
            memory_gb: Allocated memory in GB (used to calculate gpu_memory_utilization)

        Returns:
            Model instance

        Raises:
            LaunchError: If launch fails
        """
        # Check if this is an embedding model
        is_embedding = config.model_type == ModelType.EMBEDDING

        # DGX Spark: Quantization (some models have built-in quantization)
        # Default to fp8 for chat models, None for embedding models
        vllm_quant = None

        if config.quantization:
            quantization = config.quantization.lower()
            if quantization not in self.QUANTIZATION_MAP:
                raise LaunchError(
                    f"Unsupported quantization for vLLM: {config.quantization}. "
                    f"Supported: {list(self.QUANTIZATION_MAP.keys())}"
                )
            vllm_quant = self.QUANTIZATION_MAP[quantization]
        else:
            # Default to fp8 for chat models (DGX Spark memory efficiency)
            # Embedding models typically don't need quantization
            if not is_embedding:
                vllm_quant = "fp8"

        # Per-model max_model_len (not blanket 8192) - only for chat models
        max_len = config.extra_args.get("max_model_len", 8192)
        max_concurrent = config.extra_args.get("max_concurrent_requests", 32)

        logger.info(f"Launching vLLM {config.model_type.value} model: {config.model_name} on port {port}")

        try:
            # Launch using Docker (recommended)
            if settings.use_docker:
                # Build docker run command for vLLM
                # Build vLLM command - skip quantization flag for AWQ (auto-detected)
                # Check if this is a vision-language model (has conv3d for video processing)
                is_vision_model = "VL" in config.model_name.upper() or "vision" in config.model_name.lower()

                docker_cmd = [
                    "docker",
                    "run",
                    "-d",  # Detached mode
                    "--platform", "linux/arm64",  # Explicit ARM64 for DGX Spark
                    "--gpus", "all",  # GPU passthrough
                    "--shm-size", "32g",  # Shared memory for model loading
                    "--ipc=host",  # IPC mode host
                    "-p", f"{port}:{port}",  # Port mapping
                    "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",  # HuggingFace cache
                ]

                # Vision models need cuDNN V8 API disabled for conv3d on Blackwell GPUs
                if is_vision_model:
                    docker_cmd.extend(["-e", "TORCH_CUDNN_V8_API_DISABLED=1"])

                docker_cmd.extend([
                    "--name", f"sparkstation-{model_id}",  # Container name
                    settings.vllm_docker_image,
                    "vllm", "serve",
                    config.model_name,
                    "--host", "0.0.0.0",  # Bind to all interfaces
                    "--port", str(port),
                    "--trust-remote-code",  # Required for many models
                ])

                # Calculate gpu_memory_utilization from allocated memory_gb
                # DGX Spark: 119 GB usable memory (128 GB total - system overhead)
                if memory_gb is not None:
                    gpu_mem_util = memory_gb / 119.0
                    logger.info(f"Using gpu_memory_utilization={gpu_mem_util:.3f} (from {memory_gb}GB allocation)")
                else:
                    # Fallback to config or default
                    gpu_mem_util = config.extra_args.get("gpu_memory_utilization", 0.9)

                docker_cmd.extend(["--gpu-memory-utilization", str(gpu_mem_util)])

                # Add embedding-specific or chat-specific flags
                if is_embedding:
                    # For embedding models, use --task embedding flag (vLLM v0.10+)
                    docker_cmd.extend(["--task", "embedding"])
                    logger.info(f"Using embedding mode for {config.model_name}")
                else:
                    # For chat models, add max-model-len and max-num-seqs
                    docker_cmd.extend([
                        "--max-model-len", str(max_len),
                        "--max-num-seqs", str(max_concurrent),
                    ])

                    # Enable tool calling if tool_call_parser is specified in config
                    # Supported parsers: hermes, llama3_json, mistral, granite, etc.
                    tool_parser = config.extra_args.get("tool_call_parser")
                    if tool_parser:
                        docker_cmd.extend([
                            "--enable-auto-tool-choice",
                            "--tool-call-parser", tool_parser,
                        ])
                        logger.info(f"Enabled tool calling with parser: {tool_parser}")

                # Only add quantization flag if specified and not auto-detected
                # AWQ is auto-detected from model config, None means skip quantization
                if vllm_quant and vllm_quant.lower() != "awq":
                    docker_cmd.extend(["--quantization", vllm_quant])

                # Add speculative decoding if specified
                if config.speculative_model:
                    import json
                    spec_config = {
                        "model": config.speculative_model,
                        "num_speculative_tokens": config.num_speculative_tokens,
                    }
                    # Add method if explicitly specified
                    if config.speculative_method:
                        spec_config["method"] = config.speculative_method

                    docker_cmd.extend(["--speculative-config", json.dumps(spec_config)])
                    logger.info(f"Enabling speculative decoding with draft model: {config.speculative_model}")

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
                    backend=Backend.VLLM,
                    model_type=config.model_type,
                    status=ModelStatus.STARTING,
                    health_status=HealthStatus.UNKNOWN,
                    port=port,
                    gpu_ids=[0],  # DGX Spark: single GPU
                    base_url=f"http://{settings.host}:{port}",
                    container_id=container_id,
                    started_at=datetime.now(),
                    auto_suspend_enabled=config.auto_suspend_enabled,
                    idle_timeout_minutes=config.idle_timeout_minutes,
                    extra_args=config.extra_args,
                )

                logger.info(f"vLLM Docker container ready: container_id={container_id[:12]}, model_id={model_id}")
                return instance

            # Launch as subprocess (legacy mode for conda/micromamba environments)
            elif config.use_subprocess:
                # CRITICAL: Open log file to prevent pipe buffer deadlock
                # Without this, backend logs fill the pipe buffer and the process hangs
                log_dir = Path("data/model_logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = open(log_dir / f"{model_id}.log", "a")

                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    env=None,  # Inherit environment
                )

                # Wait a moment for process to start
                await asyncio.sleep(2)

                # Check if process started successfully
                if process.poll() is not None:
                    # Read last 100 lines from log file for error context
                    log_file.flush()
                    try:
                        with open(log_dir / f"{model_id}.log", "r") as lf:
                            log_lines = lf.readlines()
                            error_context = "".join(log_lines[-100:])
                    except:
                        error_context = "(could not read log file)"
                    raise LaunchError(f"vLLM process failed to start. Check logs at {log_dir}/{model_id}.log\nLast output: {error_context[:500]}")

                # Create instance
                instance = ModelInstance(
                    id=model_id,
                    model_name=config.model_name,
                    model_alias=config.model_alias,
                    backend=Backend.VLLM,
                    model_type=config.model_type,
                    status=ModelStatus.STARTING,
                    health_status=HealthStatus.UNKNOWN,
                    port=port,
                    gpu_ids=[0],  # DGX Spark: single GPU
                    base_url=f"http://{settings.host}:{port}",
                    pid=process.pid,
                    started_at=datetime.now(),
                    auto_suspend_enabled=config.auto_suspend_enabled,
                    idle_timeout_minutes=config.idle_timeout_minutes,
                    extra_args=config.extra_args,
                )

                logger.info(
                    f"vLLM process started: PID={process.pid}, model_id={model_id}"
                )
                return instance

            else:
                # TODO: Implement systemd service launching
                raise LaunchError("Systemd launching not yet implemented")

        except Exception as e:
            logger.error(f"Failed to launch vLLM model: {e}")
            raise LaunchError(f"vLLM launch failed: {e}")

    async def stop(self, instance: ModelInstance) -> bool:
        """
        Stop vLLM model server.

        Args:
            instance: Model instance

        Returns:
            True if stopped successfully
        """
        try:
            if instance.pid:
                # Send SIGTERM for graceful shutdown
                try:
                    os.kill(instance.pid, signal.SIGTERM)
                    logger.debug(f"Sent SIGTERM to vLLM process PID={instance.pid}")

                    # Wait up to 10 seconds for process to terminate
                    for _ in range(100):
                        try:
                            # Check if process still exists (sends signal 0)
                            os.kill(instance.pid, 0)
                            await asyncio.sleep(0.1)
                        except ProcessLookupError:
                            # Process is dead
                            logger.info(f"Stopped vLLM process PID={instance.pid}")
                            return True

                    # Process still alive after 10s, force kill
                    logger.warning(f"vLLM process PID={instance.pid} didn't respond to SIGTERM, sending SIGKILL")
                    os.kill(instance.pid, signal.SIGKILL)
                    logger.warning(f"Force killed vLLM process PID={instance.pid}")
                    return True

                except ProcessLookupError:
                    # Process already dead
                    logger.info(f"vLLM process PID={instance.pid} already stopped")
                    return True

            elif instance.container_id:
                # Stop Docker container
                try:
                    # Stop container with timeout
                    subprocess.run(
                        ["docker", "stop", "-t", "10", instance.container_id],
                        capture_output=True,
                        text=True,
                        timeout=15,  # Give extra time
                    )
                    logger.info(f"Stopped Docker container {instance.container_id[:12]}")

                    # Remove container
                    subprocess.run(
                        ["docker", "rm", instance.container_id],
                        capture_output=True,
                        text=True,
                    )
                    logger.info(f"Removed Docker container {instance.container_id[:12]}")
                    return True

                except subprocess.TimeoutExpired:
                    # Force kill if timeout
                    logger.warning(f"Docker container {instance.container_id[:12]} didn't stop gracefully, forcing kill")
                    subprocess.run(
                        ["docker", "kill", instance.container_id],
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["docker", "rm", instance.container_id],
                        capture_output=True,
                        text=True,
                    )
                    logger.warning(f"Force killed and removed Docker container {instance.container_id[:12]}")
                    return True

                except Exception as e:
                    logger.error(f"Failed to stop Docker container: {e}")
                    return False

            elif instance.systemd_service:
                # TODO: Stop systemd service
                pass

            return False

        except Exception as e:
            logger.error(f"Failed to stop vLLM model: {e}")
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

            if response.status_code == 200:
                logger.debug(f"Health check passed for {instance.id} ({instance.model_type.value})")
                return True
            else:
                logger.warning(
                    f"Health check failed for {instance.id}: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Health check failed for {instance.id}: {e}")
            return False
