"""
Reflex launcher: a Jev-style "System One" decision model in a docker container.

reflex (github.com/kshetrajna12/reflex) takes a block of *state* (text, JSON,
images) and typed *questions* (noul / choice / score) and returns calibrated
probability distributions over the answers you supplied — never free text.
It is NOT an OpenAI chat API: the only inference route is POST /v1/systemone
(TypeSafe Jev wire-compatible). The gateway proxy forwards that path straight
to this container (LiteLLM never sees it) and gateway_sync excludes
model_type "decision" from the LiteLLM model list, like "voice".

Which reflex to run is reflex's decision, not ours. The reflex repo keeps a
git tag `stable` on the commit it recommends and a `serving/stable.json` at
that commit naming the adapter / calibration / prompt to serve. On EVERY
launch this launcher resolves the tag (`git ls-remote`), and if no image for
that commit exists on the target host it builds one from docker/reflex
(`reflex-server:<sha12>`, also tagged `:latest`) before starting the
container. So a `sparkstation models swap reflex` (or a full restart) always
comes up on the current stable — nobody has to remember to rebuild. If the
remote cannot be reached, the newest local image is used and a warning logged.

Config (models.yaml):
  backend: reflex
  model_type: decision
  name: Qwen/Qwen3.5-4B                 # HF id of the base checkpoint
  memory_gb: 12                         # 4B bf16 ≈ 9 GB + state cache
  docker_image: reflex-server:<tag>     # OPTIONAL pin; set it and no tracking/building happens
  extra_args:
    track: stable                       # git ref to follow (default "stable"; "none" = never build)
    runs_dir: ~/src/github.com/reflex/runs   # mounted read-only at /runs (local adapters)
    adapter: lora-xyz                   # OVERRIDE the manifest's adapter (dir under runs_dir)
    calibration: lora-xyz/calibration.json   # OVERRIDE the manifest's calibration
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
import time
from datetime import datetime
from pathlib import Path

import httpx

from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.launchers.host_memory import check_memory_headroom
from supervisor.models import ModelConfig, ModelInstance, ModelStatus, HealthStatus, Backend, ModelType
from supervisor.config import settings
from supervisor.cluster_helpers import merged_env, base_url_for_host

logger = logging.getLogger(__name__)

IMAGE_REPO = "reflex-server"
DEFAULT_IMAGE = f"{IMAGE_REPO}:latest"
REFLEX_GIT = "https://github.com/kshetrajna12/reflex.git"
DEFAULT_TRACK = "stable"
DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker" / "reflex"
BUILD_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
CONTAINER_PORT = 8000
RUNS_MOUNT = "/runs"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
LS_REMOTE_TIMEOUT_S = 20
BUILD_TIMEOUT_S = 60 * 60
KEEP_OLD_IMAGES = 1   # previous stable kept for a quick rollback; older ones pruned


def _expand(path: str) -> str:
    """Expand ~ for the docker host. For host=primary that is this machine;
    remote roles should carry an absolute path in .sparkstation.local.yaml
    (a `~` would be expanded with the SUPERVISOR's home, which may differ)."""
    return os.path.expanduser(str(path))


# ── which image: resolve the tracked ref, decide whether to build ──────────

def parse_ls_remote(output: str, ref: str) -> str | None:
    """Commit sha for REF from `git ls-remote` output. An annotated tag lists the
    tag object as `refs/tags/x` and the commit it points at as `refs/tags/x^{}`;
    the commit is what REFLEX_REF must be, so the peeled line wins."""
    tag, peeled, branch = None, None, None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, name = parts
        if name == f"refs/tags/{ref}^{{}}":
            peeled = sha
        elif name == f"refs/tags/{ref}":
            tag = sha
        elif name == f"refs/heads/{ref}":
            branch = sha
    return peeled or tag or branch


def resolve_ref(ref: str, repo: str = REFLEX_GIT, env: dict | None = None) -> str | None:
    """The commit sha REF points at on the remote, or None if the remote is
    unreachable (offline, GitHub down) — the caller then falls back."""
    try:
        r = subprocess.run(
            ["git", "ls-remote", "--tags", "--heads", repo, ref, f"{ref}^{{}}"],
            capture_output=True, text=True, timeout=LS_REMOTE_TIMEOUT_S, env=env,
        )
    except Exception as e:  # timeout, git missing
        logger.warning(f"reflex: git ls-remote {repo} {ref} failed: {e}")
        return None
    if r.returncode != 0:
        logger.warning(f"reflex: git ls-remote {repo} {ref} failed: {r.stderr.strip()[:300]}")
        return None
    return parse_ls_remote(r.stdout, ref)


def image_for_commit(sha: str) -> str:
    return f"{IMAGE_REPO}:{sha[:12]}"


def plan_image(docker_image: str | None, track: str, sha: str | None, local_images: list[str]) -> tuple[str, bool, str | None]:
    """Pure decision: (image to run, must build it first, commit sha or None).

    * an explicit `docker_image` in models.yaml is a pin: run it, never build;
    * track "none" / "": same, on `reflex-server:latest`;
    * otherwise the image named after the resolved commit, built if absent;
    * remote unreachable: the newest local image, with no build.
    """
    if docker_image:
        return docker_image, False, None
    if not track or track.lower() == "none":
        return DEFAULT_IMAGE, False, None
    if sha is None:
        if DEFAULT_IMAGE not in local_images:
            raise LaunchError(
                f"reflex: cannot resolve '{track}' on {REFLEX_GIT} and no local {DEFAULT_IMAGE} to fall back on"
            )
        return DEFAULT_IMAGE, False, None
    image = image_for_commit(sha)
    return image, image not in local_images, sha


def _local_images(env: dict) -> list[str]:
    r = subprocess.run(
        ["docker", "images", IMAGE_REPO, "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, env=env,
    )
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def build_image(sha: str, image: str, env: dict) -> None:
    """`docker build` docker/reflex at commit SHA, tagged IMAGE and :latest.
    Blocking (run it in a thread); the full log goes to logs/reflex-build-<sha12>.log."""
    if not (DOCKER_DIR / "Dockerfile").exists():
        raise LaunchError(f"reflex: {DOCKER_DIR}/Dockerfile not found; cannot build {image}")
    BUILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = BUILD_LOG_DIR / f"reflex-build-{sha[:12]}.log"
    cmd = [
        "docker", "build", "--platform", "linux/arm64",
        "--build-arg", f"REFLEX_REF={sha}",
        "-t", image, "-t", DEFAULT_IMAGE, str(DOCKER_DIR),
    ]
    logger.info(f"reflex: building {image} from {REFLEX_GIT}@{sha[:12]} (log: {log_path})")
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=BUILD_TIMEOUT_S, env=env)
        except subprocess.TimeoutExpired:
            raise LaunchError(f"reflex: docker build of {image} exceeded {BUILD_TIMEOUT_S // 60} min (see {log_path})")
    if r.returncode != 0:
        tail = log_path.read_text()[-1500:]
        raise LaunchError(f"reflex: docker build of {image} failed (see {log_path}):\n{tail}")
    logger.info(f"reflex: built {image} in {(time.monotonic() - t0) / 60:.1f} min")


def prune_old_images(keep_image: str, env: dict) -> None:
    """Drop reflex-server:<sha> images older than the KEEP_OLD_IMAGES most recent
    besides KEEP_IMAGE (7 GB each). Images in use by a container fail to remove and
    are simply left alone."""
    r = subprocess.run(
        ["docker", "images", IMAGE_REPO, "--format", "{{.Tag}}\t{{.CreatedAt}}"],
        capture_output=True, text=True, env=env,
    )
    rows = []
    for line in r.stdout.splitlines():
        tag, _, created = line.partition("\t")
        if tag and tag not in ("latest", "<none>") and f"{IMAGE_REPO}:{tag}" != keep_image:
            rows.append((created, tag))
    for _, tag in sorted(rows, reverse=True)[KEEP_OLD_IMAGES:]:
        rm = subprocess.run(["docker", "image", "rm", f"{IMAGE_REPO}:{tag}"], capture_output=True, text=True, env=env)
        logger.info(f"reflex: pruned old image {IMAGE_REPO}:{tag}" if rm.returncode == 0
                    else f"reflex: kept {IMAGE_REPO}:{tag} ({rm.stderr.strip()[:120]})")


# ── the container ──────────────────────────────────────────────────────────

def build_docker_cmd(config: ModelConfig, model_id: str, port: int, image: str | None = None, commit: str | None = None) -> list[str]:
    """The `docker run` argv for a reflex container (pure; unit-tested)."""
    xa = config.extra_args or {}
    image = image or config.docker_image or DEFAULT_IMAGE
    hf_cache = _expand(xa.get("hf_cache") or f"{Path.home()}/.cache/huggingface")

    env = {
        "HOST": "0.0.0.0",
        "PORT": str(CONTAINER_PORT),
        "MODEL_PATH": config.model_name,
        "SERVED_MODEL_NAME": config.model_alias or config.model_name.split("/")[-1],
    }
    if commit:
        env["REFLEX_COMMIT"] = commit
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

    async def _ensure_image(self, config: ModelConfig, env: dict) -> tuple[str, str | None]:
        """Resolve the tracked ref and make sure its image exists on the host.
        Returns (image, commit sha or None). The ls-remote and any build run in a
        thread so the supervisor's event loop keeps serving meanwhile."""
        xa = config.extra_args or {}
        track = str(xa.get("track", DEFAULT_TRACK))
        sha = None
        if not config.docker_image and track and track.lower() != "none":
            sha = await asyncio.to_thread(resolve_ref, track, REFLEX_GIT, env)
            if sha is None:
                logger.warning(f"reflex: '{track}' unresolvable on {REFLEX_GIT}; falling back to the newest local image")
        local = await asyncio.to_thread(_local_images, env)
        image, needs_build, sha = plan_image(config.docker_image, track, sha, local)
        if needs_build:
            await asyncio.to_thread(build_image, sha, image, env)
            await asyncio.to_thread(prune_old_images, image, env)
        elif image not in local:
            raise LaunchError(
                f"reflex Docker image {image!r} not found on host={config.host}. Build it there:\n"
                f"  cd docker/reflex\n  docker build --platform linux/arm64 -t {image} ."
            )
        else:
            logger.info(f"reflex: image {image} already present" + (f" (stable = {sha[:12]})" if sha else ""))
        return image, sha

    async def launch(self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None) -> ModelInstance:
        logger.info(f"Launching reflex decision model: {config.model_name} on host={config.host} port {port}")
        if not settings.use_docker:
            raise LaunchError("reflex subprocess mode not implemented. Please use Docker mode.")

        try:
            check_memory_headroom(config.host, memory_gb)
            subprocess_env = merged_env(config.host)
            image, commit = await self._ensure_image(config, subprocess_env)

            docker_cmd = build_docker_cmd(config, model_id, port, image=image, commit=commit)
            logger.debug(f"Docker command: {' '.join(docker_cmd)}")
            result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True, env=subprocess_env)
            container_id = result.stdout.strip()
            logger.info(f"Docker container started on host={config.host}: {container_id[:12]}, model_id={model_id}, image={image}")

            await asyncio.sleep(3)
            check = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True, text=True, env=subprocess_env,
            )
            if check.stdout.strip() != "true":
                logs = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True, env=subprocess_env)
                raise LaunchError(f"Docker container failed to start. Logs:\n{(logs.stdout + logs.stderr)[-1500:]}")

            extra = dict(config.extra_args or {})
            extra["image"] = image
            if commit:
                extra["reflex_commit"] = commit
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
                extra_args=extra,
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
