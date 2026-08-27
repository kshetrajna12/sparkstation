"""
Voicechat launcher: NVIDIA NemotronLabs VoiceChat 11B (pipecat runtime) on a
single remote Spark.

Wraps the pipecat-ai/nemotron-voicechat-dgx-spark checkout's `./voicechat`
CLI (github.com/pipecat-ai/nemotron-voicechat-dgx-spark) the same way the
dspark backend wraps the Anemll stack scripts — the runtime handles its own
Docker compose (CUDA model server) plus a host-side Pipecat bot from a pinned
uv env, so we do NOT build docker commands here.

Unlike dspark (whose scripts run locally on the head node), the voicechat
stack lives on a WORKER: every lifecycle command is executed over SSH using
the cluster role's ssh_user@ip from .sparkstation.local.yaml.

  launch() -> ssh: nohup ./voicechat up --host 0.0.0.0 --port <port>
              then poll http://<host_ip>:<port>/ until it answers
  stop()   -> ssh: ./voicechat down  (+ pkill the detached `up` supervisor)
  health   -> GET / on the pipecat runner port

This model speaks WebSocket/WebRTC audio, NOT the OpenAI API — gateway_sync
excludes model_type "voice" from LiteLLM. Clients (e.g. the Reachy Mini
bridge) connect straight to http://<worker_ip>:<port>/.

Config expectations (models.yaml):
  backend: voicechat
  model_type: voice
  host: worker2
  extra_args:
    voicechat_dir: /home/kshetrajna/nemotron-voicechat   # runtime checkout
    port: 7860                                           # pipecat runner port
    launch_timeout_seconds: 900                          # ~7 min normal start
"""
import asyncio
import logging
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

DEFAULT_VOICECHAT_DIR = "/home/kshetrajna/nemotron-voicechat"
DEFAULT_PORT = 7860
# README: `./voicechat up` takes ~7 min (compose up + model load + Pipecat
# runner). 15 min leaves room for a cold docker start without masking a hang.
DEFAULT_LAUNCH_TIMEOUT_S = 900
STOP_TIMEOUT_S = 180
READY_POLL_INTERVAL_S = 10


class VoicechatLauncher(ModelLauncher):
    """Wraps the remote `./voicechat up/down` CLI as a supervisor backend."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        await self.client.aclose()

    def _ssh_target(self, host: str) -> str:
        cluster = get_cluster_config()
        entry = cluster.hosts.get(host)
        if entry is None or entry.ip is None or not entry.ssh_user:
            raise LaunchError(
                f"voicechat backend needs a remote cluster role with ip+ssh_user; "
                f"host {host!r} is not one (check .sparkstation.local.yaml)"
            )
        return f"{entry.ssh_user}@{entry.ip}"

    async def _ssh(
        self, target: str, remote_cmd: str, timeout: int, log_path: Path
    ) -> subprocess.CompletedProcess:
        """Run a remote command via ssh, teeing output to the model log."""
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
        """Start the stack detached on the worker, then poll until ready.

        `port` from the supervisor's 8001+ pool is ignored — the pipecat
        runner listens on extra_args.port (7860)."""
        target = self._ssh_target(config.host)
        vc_dir = config.extra_args.get("voicechat_dir", DEFAULT_VOICECHAT_DIR)
        api_port = int(config.extra_args.get("port", DEFAULT_PORT))
        timeout_s = int(
            config.extra_args.get("launch_timeout_seconds", DEFAULT_LAUNCH_TIMEOUT_S)
        )
        log_path = Path("data/model_logs") / f"{model_id}.log"
        base_url = base_url_for_host(config.host, api_port)

        # PATH: uv lives in ~/.local/bin on the worker. --host 0.0.0.0 so the
        # runner is reachable from the cluster/LAN, not just loopback.
        # ALL THREE stdio fds must be detached from the ssh channel (stdin
        # included) or sshd keeps the session open until `up` exits and the
        # dispatch times out — that exact hang failed the first managed launch.
        up_cmd = (
            f"cd {vc_dir} && PATH=$HOME/.local/bin:$PATH "
            f"setsid nohup ./voicechat up --host 0.0.0.0 --port {api_port} "
            f"> /tmp/voicechat-up.log 2>&1 < /dev/null & echo started"
        )
        logger.info(
            f"Launching voicechat stack on {config.host} ({target}), "
            f"port {api_port}, timeout {timeout_s}s"
        )
        # Idempotent launch: a stack already answering on the port (e.g. a
        # prior dispatch whose ssh bookkeeping failed, or a manual `up`) is
        # adopted, not double-started — the runtime can't share the port and
        # a second `up` would just crash out.
        already_up = False
        try:
            resp = await self.client.get(f"{base_url}/", timeout=5)
            already_up = resp.status_code < 500
        except Exception:
            pass

        if already_up:
            logger.info(f"voicechat stack already answering at {base_url}; adopting")
        else:
            # sshd can keep the channel open past the dispatch even with all
            # three stdio fds detached (observed with this runtime's child
            # tree). A hung dispatch is NOT a failed launch — the stack keeps
            # starting on the worker — so fall through to the readiness poll,
            # which is the real arbiter either way.
            try:
                result = await self._ssh(target, up_cmd, 60, log_path)
                if result.returncode != 0:
                    raise LaunchError(
                        f"voicechat up dispatch failed (ssh exit {result.returncode}), see {log_path}"
                    )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "voicechat up dispatch ssh still open after 60s; stack is "
                    "likely starting — proceeding to readiness poll"
                )

        # Poll the pipecat runner until it answers. Any HTTP response counts —
        # we only need the server socket up, not a specific route/status.
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
            # Leave teardown to the caller's stop() path; grab the tail of the
            # remote log for the error message first.
            tail = await self._ssh(
                target, "tail -5 /tmp/voicechat-up.log", 30, log_path
            )
            raise LaunchError(
                f"voicechat runner not answering on {base_url} after {timeout_s}s "
                f"(remote log tailed to {log_path}, ssh exit {tail.returncode})"
            )

        instance = ModelInstance(
            id=model_id,
            model_name=config.model_name,
            model_alias=config.model_alias,
            backend=Backend.VOICECHAT,
            model_type=config.model_type,
            host=config.host,
            status=ModelStatus.STARTING,
            health_status=HealthStatus.UNKNOWN,
            port=api_port,
            gpu_ids=[0],
            base_url=base_url,
            # Compose container name of the CUDA model server; startup
            # reconcile inspects this to detect a dead stack. Our stop()
            # ignores it and uses `./voicechat down`.
            container_id=config.extra_args.get("container_name", "nemotron-voicechat"),
            started_at=datetime.now(),
            auto_suspend_enabled=False,  # real-time voice stack — never idle-suspend
            idle_timeout_minutes=config.idle_timeout_minutes,
            extra_args=config.extra_args,
        )
        logger.info(f"voicechat stack up at {base_url} (model_id={model_id})")
        return instance

    async def stop(self, instance: ModelInstance) -> bool:
        """`./voicechat down` on the worker, then sweep the detached `up`."""
        try:
            target = self._ssh_target(instance.host)
        except LaunchError as e:
            logger.error(str(e))
            return False
        vc_dir = (instance.extra_args or {}).get("voicechat_dir", DEFAULT_VOICECHAT_DIR)
        log_path = Path("data/model_logs") / f"{instance.id}.log"
        down_cmd = (
            f"cd {vc_dir} && PATH=$HOME/.local/bin:$PATH ./voicechat down; "
            # [v] pattern so pkill's own cmdline never matches itself.
            f"pkill -f '[v]oicechat up' 2>/dev/null; true"
        )
        try:
            result = await self._ssh(target, down_cmd, STOP_TIMEOUT_S, log_path)
        except subprocess.TimeoutExpired:
            logger.error(f"voicechat down timed out after {STOP_TIMEOUT_S}s")
            return False
        if result.returncode != 0:
            logger.error(
                f"voicechat down failed (exit {result.returncode}), see {log_path}"
            )
            return False
        logger.info(f"voicechat stack stopped ({instance.model_name})")
        return True

    async def health_check(self, instance: ModelInstance) -> bool:
        """Liveness = the pipecat runner answers HTTP. Cheap by design: an
        audio round-trip probe would hold a conversation slot on a real-time
        half/full-duplex stack."""
        try:
            response = await self.client.get(
                f"{instance.base_url}/",
                timeout=settings.health_check_timeout_seconds,
            )
            if response.status_code < 500:
                return True
            logger.warning(
                f"voicechat health check failed for {instance.id}: {response.status_code}"
            )
            return False
        except Exception as e:
            logger.error(f"voicechat health check failed for {instance.id}: {e}")
            return False
