"""
SGLang launcher for DGX Spark vision models with quantization support.
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

    async def cleanup(self):
        """Cleanup resources (close httpx client)."""
        await self.client.aclose()

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
                    raise LaunchError(f"SGLang process failed to start. Check logs at {log_dir}/{model_id}.log\nLast output: {error_context[:500]}")

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
                # Send SIGTERM for graceful shutdown
                try:
                    os.kill(instance.pid, signal.SIGTERM)
                    logger.debug(f"Sent SIGTERM to SGLang process PID={instance.pid}")

                    # Wait up to 10 seconds for process to terminate
                    for _ in range(100):
                        try:
                            # Check if process still exists (sends signal 0)
                            os.kill(instance.pid, 0)
                            await asyncio.sleep(0.1)
                        except ProcessLookupError:
                            # Process is dead
                            logger.info(f"Stopped SGLang process PID={instance.pid}")
                            return True

                    # Process still alive after 10s, force kill
                    logger.warning(f"SGLang process PID={instance.pid} didn't respond to SIGTERM, sending SIGKILL")
                    os.kill(instance.pid, signal.SIGKILL)
                    logger.warning(f"Force killed SGLang process PID={instance.pid}")
                    return True

                except ProcessLookupError:
                    # Process already dead
                    logger.info(f"SGLang process PID={instance.pid} already stopped")
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
