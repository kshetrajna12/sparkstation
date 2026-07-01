"""
Face recognition launcher using InsightFace (SCRFD + ArcFace).
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
from supervisor.cluster_helpers import merged_env, base_url_for_host

logger = logging.getLogger(__name__)


class FaceLauncher(ModelLauncher):
    """Face recognition launcher using custom InsightFace server."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """Cleanup resources (close httpx client)."""
        await self.client.aclose()

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        """Launch face recognition server."""
        logger.info(f"Launching Face Recognition server on port {port}")

        try:
            if settings.use_docker:
                subprocess_env = merged_env(config.host)

                check_image = subprocess.run(
                    ["docker", "images", "-q", "face-server:latest"],
                    capture_output=True,
                    text=True,
                    env=subprocess_env,
                )

                if not check_image.stdout.strip():
                    raise LaunchError(
                        f"Face Recognition Docker image not found on host={config.host}. Please build it first:\n"
                        "  cd docker/face\n"
                        "  docker build --platform linux/arm64 -t face-server:latest ."
                    )

                # Gallery path — check extra_args first, then default location
                gallery_host_path = config.extra_args.get("gallery_path", "")
                if not gallery_host_path or not Path(gallery_host_path).exists():
                    default_gallery = Path.home() / ".openclaw" / "workspace-whatsapp-schoolfriends-group" / "gallery.json"
                    if default_gallery.exists():
                        gallery_host_path = str(default_gallery)
                    else:
                        logger.warning("No gallery.json found — recognition will be disabled until gallery is built")
                        gallery_host_path = None

                docker_cmd = [
                    "docker",
                    "run",
                    "-d",
                    "--platform", "linux/arm64",
                    "--gpus", "all",
                    "--shm-size", "16g",
                    "--ipc=host",
                    "-p", f"{port}:8000",
                    "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
                    "-v", f"{Path.home()}/.insightface:/root/.insightface",
                    "-e", "HOST=0.0.0.0",
                    "-e", "PORT=8000",
                    "--name", f"sparkstation-{model_id}",
                ]

                # Mount gallery if available
                if gallery_host_path:
                    gallery_dir = str(Path(gallery_host_path).parent)
                    docker_cmd.extend([
                        "-v", f"{gallery_dir}:/gallery:ro",
                        "-e", f"GALLERY_PATH=/gallery/{Path(gallery_host_path).name}",
                    ])

                docker_cmd.append("face-server:latest")

                logger.debug(f"Docker command: {' '.join(docker_cmd)}")

                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=subprocess_env,
                )

                container_id = result.stdout.strip()
                logger.info(f"Docker container started on host={config.host}: {container_id[:12]}, model_id={model_id}")

                await asyncio.sleep(3)

                check_cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container_id]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True, env=subprocess_env)

                if check_result.stdout.strip() != "true":
                    logs_cmd = ["docker", "logs", container_id]
                    logs_result = subprocess.run(logs_cmd, capture_output=True, text=True, env=subprocess_env)
                    error_context = logs_result.stdout + logs_result.stderr
                    raise LaunchError(f"Docker container failed to start. Logs:\n{error_context[:1000]}")

                instance = ModelInstance(
                    id=model_id,
                    model_name=config.model_name,
                    model_alias=config.model_alias,
                    backend=Backend.FACE,
                    model_type=ModelType.DETECTION,
                    host=config.host,
                    status=ModelStatus.STARTING,
                    health_status=HealthStatus.UNKNOWN,
                    port=port,
                    gpu_ids=[0],
                    base_url=base_url_for_host(config.host, port),
                    container_id=container_id,
                    started_at=datetime.now(),
                    auto_suspend_enabled=config.auto_suspend_enabled,
                    idle_timeout_minutes=config.idle_timeout_minutes,
                    extra_args=config.extra_args,
                )

                return instance

            else:
                raise LaunchError("Face Recognition subprocess mode not implemented. Please use Docker mode.")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to launch Face Recognition: {e.stderr}"
            logger.error(error_msg)
            raise LaunchError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error launching Face Recognition: {e}")
            raise LaunchError(str(e))

    async def stop(self, instance: ModelInstance) -> bool:
        """Stop face recognition server."""
        try:
            if instance.container_id:
                subprocess_env = merged_env(instance.host or "primary")
                result = subprocess.run(
                    ["docker", "stop", instance.container_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=subprocess_env,
                )

                if result.returncode == 0:
                    logger.info(f"Stopped face container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
                    subprocess.run(
                        ["docker", "rm", instance.container_id],
                        capture_output=True,
                        text=True,
                        env=subprocess_env,
                    )
                    return True
                else:
                    logger.error(f"Failed to stop container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error stopping face instance: {e}")
            return False

    async def suspend(self, instance: ModelInstance) -> bool:
        """Suspend face recognition (pause Docker container)."""
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
                    logger.info(f"Paused face container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
                    return True
                else:
                    logger.error(f"Failed to pause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error suspending face instance: {e}")
            return False

    async def resume(self, instance: ModelInstance) -> bool:
        """Resume face recognition (unpause Docker container)."""
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
                    logger.info(f"Unpaused face container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
                    return True
                else:
                    logger.error(f"Failed to unpause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error resuming face instance: {e}")
            return False

    async def health_check(self, instance: ModelInstance) -> bool:
        """Perform health check for face recognition server."""
        try:
            response = await self.client.get(
                f"{instance.base_url}/health",
                timeout=settings.health_check_timeout_seconds,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("models", {}).get("insightface", False)
            return False

        except Exception as e:
            logger.debug(f"Health check failed for {instance.id}: {e}")
            return False
