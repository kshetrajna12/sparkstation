"""
Species detection launcher for wildlife species identification.
Ensemble: MegaDetector v5a + SpeciesNet + iNat21 ViT-L/14.
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


class SpeciesLauncher(ModelLauncher):
    """Species detection launcher using custom server with MegaDetector + SpeciesNet + iNat21."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """Cleanup resources (close httpx client)."""
        await self.client.aclose()

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        """Launch species detection server."""
        logger.info(f"Launching Species Detection server on port {port}")

        try:
            if settings.use_docker:
                # Cluster-aware env: local for host=primary, DOCKER_HOST=ssh://...
                # for remote roles. Custom images (clip-server, face-server,
                # species-server) must exist on the target host — see
                # `sparkstation cluster sync-images` (TODO) or build manually.
                subprocess_env = merged_env(config.host)

                # Check if species-server image exists on the target host
                check_image = subprocess.run(
                    ["docker", "images", "-q", "species-server:latest"],
                    capture_output=True,
                    text=True,
                    env=subprocess_env,
                )

                if not check_image.stdout.strip():
                    raise LaunchError(
                        f"Species Detection Docker image not found on host={config.host}. Please build it there:\n"
                        "  cd docker/species\n"
                        "  docker build --platform linux/arm64 -t species-server:latest ."
                    )

                # Find MegaDetector weights - check common locations
                md_search_paths = [
                    config.extra_args.get("megadetector_path", ""),
                    str(Path.home() / "src/github.com/image_metadata_indexing/models/md_v5a.0.0.pt"),
                    str(Path.home() / "models/md_v5a.0.0.pt"),
                ]
                md_host_path = None
                for p in md_search_paths:
                    if p and Path(p).exists():
                        md_host_path = p
                        break

                if md_host_path:
                    logger.info(f"Found MegaDetector weights at: {md_host_path}")
                else:
                    logger.warning("MegaDetector weights not found locally. SpeciesNet will download its own copy.")

                # Build docker run command
                docker_cmd = [
                    "docker",
                    "run",
                    "-d",  # Detached mode
                    "--platform", "linux/arm64",
                    "--gpus", "all",
                    "--shm-size", "16g",
                    "--ipc=host",
                    "-p", f"{port}:8000",
                    # Mount caches for model weights
                    "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
                    "-v", f"{Path.home()}/.cache/kagglehub:/root/.cache/kagglehub",
                    "-v", f"{Path.home()}/.cache/torch:/root/.cache/torch",
                    "-e", "HOST=0.0.0.0",
                    "-e", "PORT=8000",
                    "--name", f"sparkstation-{model_id}",
                    "species-server:latest",
                ]

                # Mount MegaDetector weights if found
                if md_host_path:
                    docker_cmd.insert(-1, "-v")
                    docker_cmd.insert(-1, f"{md_host_path}:/app/models/md_v5a.0.0.pt:ro")
                    docker_cmd.insert(-1, "-e")
                    docker_cmd.insert(-1, "MEGADETECTOR_PATH=/app/models/md_v5a.0.0.pt")

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

                # Check if container is still running (same host)
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
                    backend=Backend.SPECIES,
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
                raise LaunchError("Species Detection subprocess mode not implemented. Please use Docker mode.")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to launch Species Detection: {e.stderr}"
            logger.error(error_msg)
            raise LaunchError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error launching Species Detection: {e}")
            raise LaunchError(str(e))

    async def stop(self, instance: ModelInstance) -> bool:
        """Stop species detection server (on the host it was launched on)."""
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
                    logger.info(f"Stopped species container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
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
            logger.error(f"Error stopping species instance: {e}")
            return False

    async def suspend(self, instance: ModelInstance) -> bool:
        """Suspend species detection (pause Docker container on its host)."""
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
                    logger.info(f"Paused species container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
                    return True
                else:
                    logger.error(f"Failed to pause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error suspending species instance: {e}")
            return False

    async def resume(self, instance: ModelInstance) -> bool:
        """Resume species detection (unpause Docker container on its host)."""
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
                    logger.info(f"Unpaused species container: {instance.container_id[:12]} on host={instance.host or 'primary'}")
                    return True
                else:
                    logger.error(f"Failed to unpause container: {result.stderr}")
                    return False
            else:
                logger.warning(f"No container_id found for instance {instance.id}")
                return False

        except Exception as e:
            logger.error(f"Error resuming species instance: {e}")
            return False

    async def health_check(self, instance: ModelInstance) -> bool:
        """Perform health check for species detection server."""
        try:
            response = await self.client.get(
                f"{instance.base_url}/health",
                timeout=settings.health_check_timeout_seconds,
            )
            if response.status_code == 200:
                data = response.json()
                # All three sub-models should be loaded
                models = data.get("models", {})
                return models.get("megadetector", False)
            return False

        except Exception as e:
            logger.debug(f"Health check failed for {instance.id}: {e}")
            return False
