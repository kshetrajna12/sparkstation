"""
Voicecascade launcher: the cascade voice stack (Kyutai STT -> routed brain ->
Qwen3-TTS) on a single remote Spark. Replaced the Nemotron VoiceChat backend
2026-08-30 (see voicecascade/DESIGN.md decision record).

The stack is three pieces on the worker, all owned by this launcher:
  1. TTS containers (docker: qwen3-tts-clone/-vd/-cv; pre-created on the
     worker, `docker start`-ed here — specs in voicecascade/DESIGN.md)
  2. gateway tunnel: the gateway binds loopback on primary, so the brain is
     reached via `ssh -N -L <tunnel_port>:127.0.0.1:8000 <primary>` running
     ON the worker; started here if nothing listens on the tunnel port
  3. the Pipecat bot (:7860 runner + :7861 prometheus metrics), launched from
     the worker checkout's venv

Like voicechat before it, every lifecycle command runs over SSH using the
cluster role's ssh_user@ip from .sparkstation.local.yaml, and this is NOT an
OpenAI API — gateway_sync excludes model_type "voice" from LiteLLM. Clients
(browser playground /client/, Reachy/OpenClaw bridge /ws-client) connect
straight to http://<worker_ip>:7860/.

Config expectations (models.yaml):
  backend: voicecascade
  model_type: voice
  host: worker2
  extra_args:
    cascade_dir: /home/kshetrajna/cascade-bot     # bot checkout (module voicecascade)
    port: 7860
    metrics_port: 7861                            # prometheus (http_sd uses this)
    tunnel_port: 18000                            # loopback gateway tunnel on worker
    tts_containers: [qwen3-tts-clone, qwen3-tts-vd, qwen3-tts-cv]
    launch_timeout_seconds: 300                   # STT preload ~40s + TTS warmups
    extra_env: {CASCADE_VOICE: K, ...}            # optional bot knobs
"""
import asyncio
import logging
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import httpx

from supervisor.launchers.base import ModelLauncher, LaunchError
from supervisor.models import (
    Backend,
    HealthStatus,
    ModelConfig,
    ModelInstance,
    ModelStatus,
)
from supervisor.config import settings
from supervisor.cluster_helpers import base_url_for_host
from supervisor.models_config import get_cluster_config

logger = logging.getLogger(__name__)

DEFAULT_CASCADE_DIR = "/home/kshetrajna/cascade-bot"
DEFAULT_PORT = 7860
DEFAULT_METRICS_PORT = 7861
DEFAULT_TUNNEL_PORT = 18000
DEFAULT_TTS_CONTAINERS = ["qwen3-tts-clone", "qwen3-tts-vd", "qwen3-tts-cv"]
DEFAULT_LAUNCH_TIMEOUT_S = 300
STOP_TIMEOUT_S = 120
READY_POLL_INTERVAL_S = 5


class VoicecascadeLauncher(ModelLauncher):
    """Runs the cascade voice stack (TTS containers + tunnel + bot) over SSH."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        await self.client.aclose()

    def _ssh_target(self, host: str) -> str:
        cluster = get_cluster_config()
        entry = cluster.hosts.get(host)
        if entry is None or entry.ip is None or not entry.ssh_user:
            raise LaunchError(
                f"voicecascade backend needs a remote cluster role with ip+ssh_user; "
                f"host {host!r} is not one (check .sparkstation.local.yaml)"
            )
        return f"{entry.ssh_user}@{entry.ip}"

    def _primary_ssh(self) -> str:
        """user@ip of the primary as seen FROM the worker (QSFP)."""
        cluster = get_cluster_config()
        primary = cluster.hosts.get("primary")
        # primary's QSFP IP toward the workers; ssh_user shared cluster-wide
        if primary is None or primary.ip is None or not primary.ssh_user:
            raise LaunchError("cluster config lacks a primary role with ip+ssh_user")
        return f"{primary.ssh_user}@{primary.ip}"

    async def _ssh(
        self, target: str, remote_cmd: str, timeout: int, log_path: Path
    ) -> subprocess.CompletedProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def _run():
            with open(log_path, "a") as log_file:
                log_file.write(
                    f"\n=== {datetime.now().isoformat()} ssh {target} :: {remote_cmd} ===\n"
                )
                log_file.flush()
                return subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", target, remote_cmd],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )

        return await asyncio.to_thread(_run)

    async def launch(
        self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None
    ) -> ModelInstance:
        """Bring up containers + tunnel + bot, then poll :port until ready.

        `port` from the supervisor's 8001+ pool is ignored — the runner
        listens on extra_args.port. Idempotent: a bot already answering is
        adopted (containers/tunnel are still ensured first, which no-ops
        when they're already up)."""
        xa = config.extra_args or {}
        target = self._ssh_target(config.host)
        cascade_dir = xa.get("cascade_dir", DEFAULT_CASCADE_DIR)
        api_port = int(xa.get("port", DEFAULT_PORT))
        tunnel_port = int(xa.get("tunnel_port", DEFAULT_TUNNEL_PORT))
        containers = xa.get("tts_containers") or DEFAULT_TTS_CONTAINERS
        timeout_s = int(xa.get("launch_timeout_seconds", DEFAULT_LAUNCH_TIMEOUT_S))
        log_path = Path("data/model_logs") / f"{model_id}.log"
        base_url = base_url_for_host(config.host, api_port)

        env = ""
        for key, value in (xa.get("extra_env") or {}).items():
            if not str(key).replace("_", "").isalnum():
                raise LaunchError(f"invalid extra_env key: {key!r}")
            env += f"{key}={shlex.quote(str(value))} "

        # 1) TTS containers ('docker start' on a running container is a no-op)
        names = " ".join(shlex.quote(c) for c in containers)
        result = await self._ssh(target, f"docker start {names}", 60, log_path)
        if result.returncode != 0:
            raise LaunchError(
                f"voicecascade: docker start {names} failed (exit {result.returncode}); "
                f"containers must be pre-created on {config.host} — see voicecascade/DESIGN.md"
            )

        # 2) gateway tunnel (worker loopback :tunnel_port -> primary :8000).
        # extra_args.gateway_ssh overrides (QSFP IPs are per-link: worker1
        # and worker2 see different primary addresses).
        primary = xa.get("gateway_ssh") or self._primary_ssh()
        tunnel_cmd = (
            f"nc -z 127.0.0.1 {tunnel_port} 2>/dev/null || "
            f"(setsid nohup ssh -o BatchMode=yes -N "
            f"-L {tunnel_port}:127.0.0.1:8000 {primary} "
            f"> /tmp/cascade-tunnel.log 2>&1 < /dev/null & sleep 1; "
            f"nc -z 127.0.0.1 {tunnel_port})"
        )
        result = await self._ssh(target, tunnel_cmd, 30, log_path)
        if result.returncode != 0:
            raise LaunchError(
                f"voicecascade: gateway tunnel :{tunnel_port} not up on {config.host} "
                f"(see /tmp/cascade-tunnel.log there)"
            )

        # 3) the bot — adopt if it already answers
        already_up = False
        try:
            resp = await self.client.get(f"{base_url}/", timeout=5)
            already_up = resp.status_code < 500
        except Exception:
            pass

        if already_up:
            logger.info(f"voicecascade bot already answering at {base_url}; adopting")
        else:
            up_cmd = (
                f"cd {shlex.quote(cascade_dir)} && "
                f"CASCADE_GATEWAY=http://127.0.0.1:{tunnel_port}/v1 {env}"
                f"setsid nohup {shlex.quote(cascade_dir)}/.venv/bin/python "
                f"-m voicecascade.bot -t webrtc --host 0.0.0.0 --port {api_port} "
                f"> /tmp/cascade-bot.log 2>&1 < /dev/null & echo started"
            )
            # sshd can hold the channel open past the detached dispatch
            # (same behavior the voicechat launcher hit) — a hung dispatch is
            # NOT a failed launch; the readiness poll below is the arbiter.
            try:
                result = await self._ssh(target, up_cmd, 30, log_path)
                if result.returncode != 0:
                    raise LaunchError(
                        f"voicecascade bot dispatch failed (ssh exit {result.returncode}), see {log_path}"
                    )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "voicecascade bot dispatch ssh still open after 30s; "
                    "proceeding to readiness poll"
                )

        deadline = asyncio.get_event_loop().time() + timeout_s
        ready = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await self.client.get(f"{base_url}/", timeout=10)
                if resp.status_code < 500:
                    ready = True
                    break
            except Exception:
                pass
            await asyncio.sleep(READY_POLL_INTERVAL_S)

        if not ready:
            await self._ssh(target, "tail -5 /tmp/cascade-bot.log", 30, log_path)
            raise LaunchError(
                f"voicecascade bot not answering on {base_url} after {timeout_s}s "
                f"(remote log tailed to {log_path})"
            )

        instance = ModelInstance(
            id=model_id,
            model_name=config.model_name,
            model_alias=config.model_alias,
            backend=Backend.VOICECASCADE,
            model_type=config.model_type,
            host=config.host,
            status=ModelStatus.STARTING,
            health_status=HealthStatus.UNKNOWN,
            port=api_port,
            gpu_ids=[0],
            base_url=base_url,
            # First TTS container name; startup reconcile inspects this to
            # detect a dead stack. stop() uses names from extra_args.
            container_id=containers[0],
            started_at=datetime.now(),
            auto_suspend_enabled=False,  # real-time voice stack — never idle-suspend
            idle_timeout_minutes=config.idle_timeout_minutes,
            extra_args=config.extra_args,
        )
        logger.info(f"voicecascade stack up at {base_url} (model_id={model_id})")
        return instance

    async def stop(self, instance: ModelInstance) -> bool:
        """Kill the bot and stop the TTS containers (tunnel is left up —
        it's a passive loopback forward other tools reuse)."""
        try:
            target = self._ssh_target(instance.host)
        except LaunchError as e:
            logger.error(str(e))
            return False
        xa = instance.extra_args or {}
        containers = xa.get("tts_containers") or DEFAULT_TTS_CONTAINERS
        names = " ".join(shlex.quote(c) for c in containers)
        log_path = Path("data/model_logs") / f"{instance.id}.log"
        # [v] bracket so pkill's own remote cmdline never matches itself.
        down_cmd = (
            f"pkill -f '[v]oicecascade.bot' 2>/dev/null; "
            f"docker stop {names} >/dev/null 2>&1; true"
        )
        try:
            result = await self._ssh(target, down_cmd, STOP_TIMEOUT_S, log_path)
        except subprocess.TimeoutExpired:
            logger.error(f"voicecascade stop timed out after {STOP_TIMEOUT_S}s")
            return False
        if result.returncode != 0:
            logger.error(
                f"voicecascade stop failed (exit {result.returncode}), see {log_path}"
            )
            return False
        logger.info(f"voicecascade stack stopped ({instance.model_name})")
        return True

    async def health_check(self, instance: ModelInstance) -> bool:
        """Liveness = the pipecat runner answers HTTP (any status < 500)."""
        try:
            response = await self.client.get(
                f"{instance.base_url}/",
                timeout=settings.health_check_timeout_seconds,
            )
            if response.status_code < 500:
                return True
            logger.warning(
                f"voicecascade health check failed for {instance.id}: {response.status_code}"
            )
            return False
        except Exception as e:
            logger.error(f"voicecascade health check failed for {instance.id}: {e}")
            return False
