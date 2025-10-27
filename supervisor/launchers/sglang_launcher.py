"""
SGLang launcher for DGX Spark vision models with quantization support.
"""
import asyncio
import logging
import subprocess
from datetime import datetime
from typing import Dict, Optional
import httpx
from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend
from supervisor.config import settings

logger = logging.getLogger(__name__)


class SGLangLauncher(ModelLauncher):
    """DGX Spark-optimized SGLang launcher for vision models."""

    # Backend-specific quantization flag mapping (SGLang uses different flags than vLLM)
    QUANTIZATION_MAP = {
        "fp8": "fp8",  # SGLang native fp8
        "int4": "int4",  # SGLang int4 (different from vLLM)
        "awq": "awq",  # AWQ support
        "gptq": "gptq",  # GPTQ support
    }

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def launch(self, config: ModelConfig, model_id: str, port: int) -> ModelInstance:
        """
        Launch SGLang model server.

        Args:
            config: Model configuration
            model_id: Unique model ID
            port: Allocated port

        Returns:
            Model instance

        Raises:
            LaunchError: If launch fails
        """
        # DGX Spark: Mandatory quantization for vision models
        quantization = config.quantization or "fp8"
        if quantization.lower() not in self.QUANTIZATION_MAP:
            raise LaunchError(
                f"Unsupported quantization for SGLang: {quantization}. "
                f"Supported: {list(self.QUANTIZATION_MAP.keys())}"
            )

        sglang_quant = self.QUANTIZATION_MAP[quantization.lower()]

        # Per-model context length (vision models may need longer context)
        context_len = config.extra_args.get("max_model_len", 8192)
        max_concurrent = config.extra_args.get("max_concurrent_requests", 16)  # Lower for vision

        # Build SGLang command
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            config.model_name,
            "--host",
            settings.host,  # CRITICAL: Localhost only
            "--port",
            str(port),
            "--quantization",
            sglang_quant,
            "--context-length",
            str(context_len),
            "--mem-fraction-static",
            "0.9",  # Max 90% memory
            "--max-running-requests",
            str(max_concurrent),
        ]

        logger.info(f"Launching SGLang model: {config.model_name} on port {port}")
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
                    raise LaunchError(f"SGLang process failed to start: {stderr}")

                # Create instance
                instance = ModelInstance(
                    id=model_id,
                    model_name=config.model_name,
                    model_alias=config.model_alias,
                    backend=Backend.SGLANG,
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
                    f"SGLang process started: PID={process.pid}, model_id={model_id}"
                )
                return instance

            else:
                # TODO: Implement Docker launching
                raise LaunchError("Docker launching not yet implemented")

        except Exception as e:
            logger.error(f"Failed to launch SGLang model: {e}")
            raise LaunchError(f"SGLang launch failed: {e}")

    async def stop(self, instance: ModelInstance) -> bool:
        """
        Stop SGLang model server.

        Args:
            instance: Model instance

        Returns:
            True if stopped successfully
        """
        try:
            if instance.pid:
                # Kill subprocess
                try:
                    process = subprocess.Popen(["kill", str(instance.pid)])
                    process.wait(timeout=10)
                    logger.info(f"Stopped SGLang process PID={instance.pid}")
                    return True
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    subprocess.run(["kill", "-9", str(instance.pid)])
                    logger.warning(f"Force killed SGLang process PID={instance.pid}")
                    return True
            elif instance.container_id:
                # TODO: Stop Docker container
                pass

            return False

        except Exception as e:
            logger.error(f"Failed to stop SGLang model: {e}")
            return False

    async def health_check(self, instance: ModelInstance) -> bool:
        """
        Perform 1-token chat completion to verify model responsiveness.

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
