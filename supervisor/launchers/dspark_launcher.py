"""
DSpark launcher: Anemll DeepSeek-V4-Flash 2-node TP=2 stack on 2x DGX Spark.

This backend does NOT build docker commands itself. The dspark stack
(~/dsv4-2xspark, MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) ships
battle-tested orchestration scripts that handle everything a naive launcher
would get wrong on GB10: RoCEv2 GID auto-resolution per node, worker-side
compose over SSH, hotfix syncing, TCPStore rendezvous ordering (worker rank
up before head), repack-aware health wait loops (~25 min budget), and a
final smoke chat request. We wrap those scripts:

  launch() -> start-deepseek-v4-flash-dspark.sh   (blocks until API healthy)
  stop()   -> stop-deepseek-v4-flash-dspark.sh    (tears down BOTH nodes,
              sweeps zombie VL sidecars and legacy compose projects)

Because stop() takes down the whole 2-node stack and launch() brings it back
in the correct order, RestartManager's generic stop->launch cycle IS the
"restart the pair together" semantics — including recovery from the known
TCPStore deadlock (head dies mid-handshake, compose revives head, stale
worker rank never rejoins). On a failed launch we additionally run stop()
once and retry, which force-removes the stale worker rank.

Config expectations (models.yaml):
  backend: dspark
  host: primary            # head node; worker comes from the stack's .env.dspark
  extra_args:
    dspark_dir: /home/kshetrajna/dsv4-2xspark   # stack checkout (default)
    port: 8888                                  # fixed API port (NOT from the
                                                # supervisor's 8001+ pool)
    launch_timeout_seconds: 2700                # cold load + MoE repack budget
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

logger = logging.getLogger(__name__)

DEFAULT_DSPARK_DIR = str(Path.home() / "dsv4-2xspark")
DEFAULT_API_PORT = 8888
# Cold start worst case: image pull cached, but MoE expert repack + weight
# load ~4 min with OMP_NUM_THREADS=20, plus the script's own 100x15s API
# wait. 45 min leaves room for one slow boot without masking a true hang.
DEFAULT_LAUNCH_TIMEOUT_S = 2700
STOP_TIMEOUT_S = 300


class DsparkLauncher(ModelLauncher):
    """Wraps the dsv4-2xspark start/stop scripts as a supervisor backend."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        await self.client.aclose()

    def _script_dir(self, config: ModelConfig) -> Path:
        return Path(config.extra_args.get("dspark_dir", DEFAULT_DSPARK_DIR)).expanduser()

    def _api_port(self, config: ModelConfig) -> int:
        return int(config.extra_args.get("port", DEFAULT_API_PORT))

    async def _run_script(
        self, script: Path, args: list, timeout: int, log_path: Path
    ) -> subprocess.CompletedProcess:
        """Run a stack script, teeing combined output to a per-model log file."""
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def _run():
            with open(log_path, "a") as log_file:
                log_file.write(
                    f"\n=== {datetime.now().isoformat()} {script.name} {' '.join(args)} ===\n"
                )
                log_file.flush()
                return subprocess.run(
                    [str(script), *args],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=script.parent,
                    timeout=timeout,
                )

        return await asyncio.to_thread(_run)

    async def launch(
        self, config: ModelConfig, model_id: str, port: int, memory_gb: float = None
    ) -> ModelInstance:
        """
        Launch the 2-node DSpark stack. `port` (from the supervisor's pool) is
        ignored — the stack listens on a fixed port (extra_args.port, 8888).
        Blocks until the stack's own health wait + smoke request pass.
        """
        script_dir = self._script_dir(config)
        # Script names are per-stack (default: the DSV4 dspark stack). Other
        # 2-node stacks (e.g. GLM-5.3-Flash DFlash2) reuse this launcher by
        # naming their own scripts in extra_args.
        start_script = script_dir / config.extra_args.get(
            "start_script", "start-deepseek-v4-flash-dspark.sh"
        )
        stop_script = script_dir / config.extra_args.get(
            "stop_script", "stop-deepseek-v4-flash-dspark.sh"
        )
        if not start_script.exists():
            raise LaunchError(f"DSpark start script not found: {start_script}")

        api_port = self._api_port(config)
        timeout_s = int(
            config.extra_args.get("launch_timeout_seconds", DEFAULT_LAUNCH_TIMEOUT_S)
        )
        log_path = Path("data/model_logs") / f"{model_id}.log"
        # Bind to all interfaces so worker-node consumers and the gateway can
        # reach the head API over the cluster network, not just loopback.
        script_args = ["--host", "0.0.0.0", "--port", str(api_port)]

        logger.info(
            f"Launching DSpark stack: {config.model_name} via {start_script} "
            f"(port {api_port}, timeout {timeout_s}s, log {log_path})"
        )

        for attempt in (1, 2):
            try:
                result = await self._run_script(
                    start_script, script_args, timeout_s, log_path
                )
            except subprocess.TimeoutExpired:
                raise LaunchError(
                    f"DSpark start timed out after {timeout_s}s (see {log_path}). "
                    f"Stack may still be loading — inspect before relaunching."
                )

            if result.returncode == 0:
                break

            if attempt == 1:
                # Known failure mode: TCPStore deadlock — head died
                # mid-handshake and the revived head can't rendezvous with the
                # stale worker rank. The stop script force-removes both ranks
                # on both nodes, which is exactly the recovery.
                logger.warning(
                    f"DSpark start failed (exit {result.returncode}); running "
                    f"full 2-node stop to clear stale ranks, then retrying once"
                )
                if stop_script.exists():
                    try:
                        await self._run_script(stop_script, [], STOP_TIMEOUT_S, log_path)
                    except subprocess.TimeoutExpired:
                        logger.error("DSpark stop timed out during launch recovery")
            else:
                raise LaunchError(
                    f"DSpark start failed twice (exit {result.returncode}), "
                    f"see {log_path}"
                )

        # Script exit 0 means: /v1/models answered AND a smoke chat completion
        # succeeded on the head. Register as STARTING; the generic health
        # checker promotes to RUNNING on its next pass.
        instance = ModelInstance(
            id=model_id,
            model_name=config.model_name,
            model_alias=config.model_alias,
            backend=Backend.DSPARK,
            model_type=config.model_type,
            host=config.host,
            status=ModelStatus.STARTING,
            health_status=HealthStatus.UNKNOWN,
            port=api_port,
            gpu_ids=[0],
            base_url=base_url_for_host(config.host, api_port),
            # Head container's compose name. Startup reconcile inspects this to
            # detect a dead stack (it skips instances with no container_id);
            # our own stop() ignores it and uses the stack scripts.
            container_id=config.extra_args.get(
                "head_container", "deepseek-v4-flash-vllm-dspark-1"
            ),
            started_at=datetime.now(),
            auto_suspend_enabled=False,  # never idle-suspend a 2-node stack
            idle_timeout_minutes=config.idle_timeout_minutes,
            extra_args=config.extra_args,
        )
        logger.info(
            f"DSpark stack up: {config.model_name} at {instance.base_url} "
            f"(model_id={model_id})"
        )
        return instance

    async def stop(self, instance: ModelInstance) -> bool:
        """Tear down the whole 2-node stack (both ranks + sidecar sweep)."""
        script_dir = Path(
            (instance.extra_args or {}).get("dspark_dir", DEFAULT_DSPARK_DIR)
        ).expanduser()
        stop_script = script_dir / (instance.extra_args or {}).get(
            "stop_script", "stop-deepseek-v4-flash-dspark.sh"
        )
        if not stop_script.exists():
            logger.error(f"DSpark stop script not found: {stop_script}")
            return False

        log_path = Path("data/model_logs") / f"{instance.id}.log"
        try:
            result = await self._run_script(stop_script, [], STOP_TIMEOUT_S, log_path)
        except subprocess.TimeoutExpired:
            logger.error(f"DSpark stop timed out after {STOP_TIMEOUT_S}s")
            return False
        if result.returncode != 0:
            logger.error(f"DSpark stop failed (exit {result.returncode}), see {log_path}")
            return False
        logger.info(f"DSpark stack stopped ({instance.model_name})")
        return True

    async def health_check(self, instance: ModelInstance) -> bool:
        """
        Cheap liveness probe: /v1/models on the head API. A 1-token chat
        completion (what the vLLM launcher does) is too expensive here — under
        long-context coding load a probe completion can queue for minutes and
        false-fail the check, and a false FAILED triggers a ~10 min 2-node
        restart. Liveness of the head API implies the TP ranks are joined;
        rank death kills the head server process.
        """
        try:
            response = await self.client.get(
                f"{instance.base_url}/v1/models",
                timeout=settings.health_check_timeout_seconds,
            )
            if response.status_code == 200:
                return True
            logger.warning(
                f"DSpark health check failed for {instance.id}: {response.status_code}"
            )
            return False
        except Exception as e:
            logger.error(f"DSpark health check failed for {instance.id}: {e}")
            return False
