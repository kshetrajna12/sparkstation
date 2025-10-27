"""
vLLM launcher for DGX Spark with quantization support.
"""
import asyncio
import logging
import os
import signal
import subprocess
from datetime import datetime
from typing import Dict, Optional
import httpx
from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend
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
    }

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def launch(self, config: ModelConfig, model_id: str, port: int) -> ModelInstance:
        """
        Launch vLLM model server.

        Args:
            config: Model configuration
            model_id: Unique model ID
            port: Allocated port

        Returns:
            Model instance

        Raises:
            LaunchError: If launch fails
        """
        # DGX Spark: Mandatory quantization
        quantization = config.quantization or "fp8"
        if quantization.lower() not in self.QUANTIZATION_MAP:
            raise LaunchError(
                f"Unsupported quantization for vLLM: {quantization}. "
                f"Supported: {list(self.QUANTIZATION_MAP.keys())}"
            )

        vllm_quant = self.QUANTIZATION_MAP[quantization.lower()]

        # Per-model max_model_len (not blanket 8192)
        max_len = config.extra_args.get("max_model_len", 8192)
        max_concurrent = config.extra_args.get("max_concurrent_requests", 32)

        # Build vLLM command
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            config.model_name,
            "--host",
            settings.host,  # CRITICAL: Localhost only
            "--port",
            str(port),
            "--quantization",
            vllm_quant,
            "--max-model-len",
            str(max_len),
            "--gpu-memory-utilization",
            "0.9",  # Max 90%
            "--disable-log-requests",  # Reduce overhead
            "--max-num-seqs",
            str(max_concurrent),
        ]

        logger.info(f"Launching vLLM model: {config.model_name} on port {port}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            # Launch as subprocess
            if config.use_subprocess:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=None,  # Inherit environment
                )

                # Wait a moment for process to start
                await asyncio.sleep(2)

                # Check if process started successfully
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    raise LaunchError(f"vLLM process failed to start: {stderr}")

                # Create instance
                instance = ModelInstance(
                    id=model_id,
                    model_name=config.model_name,
                    model_alias=config.model_alias,
                    backend=Backend.VLLM,
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

            elif instance.systemd_service:
                # TODO: Stop systemd service
                pass

            return False

        except Exception as e:
            logger.error(f"Failed to stop vLLM model: {e}")
            return False

    async def health_check(self, instance: ModelInstance) -> bool:
        """
        Perform 1-token chat completion to verify model responsiveness.

        CRITICAL: Use /v1/chat/completions not /v1/completions
        (most backends reject /completions).

        Args:
            instance: Model instance

        Returns:
            True if healthy
        """
        try:
            # 1-token chat completion probe
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
                logger.debug(f"Health check passed for {instance.id}")
                return True
            else:
                logger.warning(
                    f"Health check failed for {instance.id}: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Health check failed for {instance.id}: {e}")
            return False
