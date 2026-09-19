"""
Reflex launcher: a Jev-style "System One" decision model in a docker container.

reflex (github.com/kshetrajna12/reflex) takes a block of *state* (text, JSON,
images) and typed *questions* (noul / choice / score) and returns calibrated
probability distributions over the answers you supplied — never free text.
It is NOT an OpenAI chat API: the only inference route is POST /v1/systemone
(TypeSafe Jev wire-compatible). The gateway proxy forwards that path straight
to this container (LiteLLM never sees it) and gateway_sync excludes
model_type "decision" from the LiteLLM model list, like "voice".

Image: docker/reflex (reflex-server:latest — reflex's own uv-locked env +
a thin entrypoint that adds /health). Must exist on the target host.

Config expectations (models.yaml):
  backend: reflex
  model_type: decision
  name: Qwen/Qwen3.5-4B                 # HF id of the base checkpoint
  memory_gb: 12                         # 4B bf16 ≈ 9 GB + state cache
  extra_args:
    runs_dir: ~/src/github.com/reflex/runs   # mounted read-only at /runs
                                              # (absolute per-machine path in
                                              #  .sparkstation.local.yaml)
    adapter: lora-mix                   # LoRA dir under runs_dir (optional)
    calibration: lora-mix/calibration.json   # temperatures under runs_dir (optional)
    max_pack_tokens: 8192               # branch-token budget per forward
    max_image_pixels: 1048576           # image token cost bound (~1k tok/MP)
    dtype: bfloat16
    state_cache_entries: 8
    hf_cache: ~/.cache/huggingface      # host HF hub cache to mount
"""
import asyncio
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

import httpx

from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.launchers.host_memory import check_memory_headroom
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend, ModelType
from supervisor.config import settings
from supervisor.cluster_helpers import merged_env, base_url_for_host

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "reflex-server:latest"
CONTAINER_PORT = 8000
RUNS_MOUNT = "/runs"
HF_CACHE_MOUNT = "/root/.cache/huggingface"


def _expand(path: str) -> str:
    """Expand ~ for the docker host. For host=primary that is this machine;
    remote roles should carry an absolute path in .sparkstation.local.yaml
    (a `~` would be expanded with the SUPERVISOR's home, which may differ)."""
    return os.path.expanduser(str(path))


def build_docker_cmd(config: ModelConfig, model_id: str, port: int) -> list[str]:
    """The `docker run` argv for a reflex container (pure; unit-tested)."""
    xa = config.extra_args or {}
    image = config.docker_image or DEFAULT_IMAGE
    hf_cache = _expand(xa.get("hf_cache") or f"{Path.home()}/.cache/huggingface")

    env = {
        "HOST": "0.0.0.0",
        "PORT": str(CONTAINER_PORT),
        "MODEL_PATH": config.model_name,
        "SERVED_MODEL_NAME": config.model_alias or config.model_name.split("/")[-1],
    }
    knobs = {
        "max_pack_tokens": "REFLEX_MAX_PACK_TOKENS",
        "max_image_pixels": "REFLEX_MAX_IMAGE_PIXELS",
        "dtype": "REFLEX_DTYPE",
        "state_cache_entries": "REFLEX_STATE_CACHE_ENTRIES",
    }
    for key, var in knobs.items():
        if xa.get(key) is not None:
            env[var] = str(xa[key])

    volumes = [f"{hf_cache}:{HF_CACHE_MOUNT}"]
    runs_dir = xa.get("runs_dir")
    adapter, calibration = xa.get("adapter"), xa.get("calibration")
    if adapter or calibration:
        if not runs_dir:
            raise LaunchError("reflex: extra_args.adapter/calibration need extra_args.runs_dir")
        volumes.append(f"{_expand(runs_dir)}:{RUNS_MOUNT}:ro")
        if adapter:
            env["REFLEX_ADAPTER"] = f"{RUNS_MOUNT}/{adapter}"
        if calibration:
            env["REFLEX_CALIBRATION"] = f"{RUNS_MOUNT}/{calibration}"
    elif runs_dir:
        volumes.append(f"{_expand(runs_dir)}:{RUNS_MOUNT}:ro")
    volumes.extend(config.volumes or [])
    env.update(config.env_vars or {})

    cmd = [
        "docker", "run", "-d",
        "--platform", "linux/arm64",
        "--gpus", "all",
        "--shm-size", "8g",
        "--ipc=host",
        "-p", f"{port}:{CONTAINER_PORT}",
        "--name", f"sparkstation-{model_id}",
    ]
    for v in volumes:
        cmd += ["-v", v]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    return cmd


class ReflexLauncher(ModelLauncher):
    """Docker launcher for the reflex decision-model server."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        await self.client.aclose()

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        logger.info(f"Launching reflex decision model: {config.model_name} on host={config.host} port {port}")
        if not settings.use_docker:
            raise LaunchError("reflex subprocess mode not implemented. Please use Docker mode.")

        image = config.docker_image or DEFAULT_IMAGE
        try:
            check_memory_headroom(config.host, memory_gb)
            subprocess_env = merged_env(config.host)

            check_image = subprocess.run(
                ["docker", "images", "-q", image],
                capture_output=True, text=True, env=subprocess_env,
            )
            if not check_image.stdout.strip():
                raise LaunchError(
                    f"reflex Docker image {image!r} not found on host={config.host}. Build it there:\n"
                    "  cd docker/reflex\n"
                    "  docker build --platform linux/arm64 -t reflex-server:latest ."
                )

            docker_cmd = build_docker_cmd(config, model_id, port)
            logger.debug(f"Docker command: {' '.join(docker_cmd)}")
            result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True, env=subprocess_env)
            container_id = result.stdout.strip()
            logger.info(f"Docker container started on host={config.host}: {container_id[:12]}, model_id={model_id}")

            await asyncio.sleep(3)
            check = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True, text=True, env=subprocess_env,
            )
            if check.stdout.strip() != "true":
                logs = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True, env=subprocess_env)
                raise LaunchError(f"Docker container failed to start. Logs:\n{(logs.stdout + logs.stderr)[-1500:]}")

            return ModelInstance(
                id=model_id,
                model_name=config.model_name,
                model_alias=config.model_alias,
                backend=Backend.REFLEX,
                model_type=ModelType.DECISION,
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
        except LaunchError:
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to launch reflex: {e.stderr}")
            raise LaunchError(f"Failed to launch reflex: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error launching reflex: {e}")
            raise LaunchError(str(e))

    def _docker(self, instance: ModelInstance, verb: str, timeout: int) -> bool:
        if not instance.container_id:
            logger.warning(f"No container_id found for instance {instance.id}")
            return False
        host = instance.host or "primary"
        try:
            result = subprocess.run(
                ["docker", verb, instance.container_id],
                capture_output=True, text=True, timeout=timeout, env=merged_env(host),
            )
        except Exception as e:
            logger.error(f"Error running docker {verb} for reflex instance {instance.id}: {e}")
            return False
        if result.returncode != 0:
            logger.error(f"docker {verb} failed for reflex container {instance.container_id[:12]}: {result.stderr}")
            return False
        logger.info(f"docker {verb} reflex container {instance.container_id[:12]} on host={host}")
        return True

    async def stop(self, instance: ModelInstance) -> bool:
        if not self._docker(instance, "stop", 30):
            return False
        self._docker(instance, "rm", 30)
        return True

    async def suspend(self, instance: ModelInstance) -> bool:
        return self._docker(instance, "pause", 10)

    async def resume(self, instance: ModelInstance) -> bool:
        return self._docker(instance, "unpause", 10)

    async def health_check(self, instance: ModelInstance) -> bool:
        try:
            r = await self.client.get(f"{instance.base_url}/health", timeout=settings.health_check_timeout_seconds)
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {instance.id}: {e}")
            return False
