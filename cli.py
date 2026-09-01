#!/usr/bin/env python3
"""
Sparkstation CLI - Unified interface for managing Sparkstation and models.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
import httpx


DEFAULT_SUPERVISOR_URL = "http://127.0.0.1:9001"
DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000"


def _find_project_root() -> Path:
    """Find the sparkstation repo root (delegates to models_config.find_sparkstation_root).

    Discovery: walk up from cwd → $SPARKSTATION_HOME → ~/.sparkstation/root
    breadcrumb (dropped below on every successful in-repo run) — so commands
    like `sparkstation init` work from ANY directory, not just inside the repo.

    Why this exists: cli.py used to compute PROJECT_ROOT as Path(__file__).resolve().parent,
    which works in development (cli.py in repo root) but BREAKS for the installed CLI
    (`uv tool install`) — there __file__ resolves to site-packages, and
    site-packages/models.yaml does not exist. Every helper that touched models.yaml or
    gateway/litellm.yaml fell over. The systemd unit got away with it because it sets
    WorkingDirectory to the repo and `cd=PROJECT_ROOT` for spawned subprocesses, but
    that means anything outside that unit was broken. Walking up from cwd is the
    standard "find your project root" pattern (git, npm, etc.).
    """
    from supervisor.models_config import find_sparkstation_root
    root = find_sparkstation_root()
    if (root / "models.yaml").exists():
        # Breadcrumb so future invocations from OTHER directories (e.g.
        # `sparkstation init` inside a client project) still find the repo.
        try:
            crumb = Path.home() / ".sparkstation" / "root"
            crumb.parent.mkdir(parents=True, exist_ok=True)
            if not crumb.exists() or crumb.read_text().strip() != str(root):
                crumb.write_text(str(root) + "\n")
        except OSError:
            pass
    return root


PROJECT_ROOT = _find_project_root()
RUN_DIR = Path.home() / ".sparkstation"
LOG_DIR = RUN_DIR / "logs"
PID_DIR = RUN_DIR / "pids"
# Last profile explicitly started — `start` without --profile resumes it, so
# systemd crash-restarts (ExecStart has no --profile) don't silently fall back
# to default_profile while e.g. `deep` was running.
LAST_PROFILE_FILE = PROJECT_ROOT / "data" / "last_profile"
# `bounce` touches this; `stop` then keeps models + DB and only restarts the
# supervisor/gateway processes. Survives the systemctl-delegation hop (the
# ExecStop invocation consumes it, not the delegating one).
BOUNCE_SENTINEL = PROJECT_ROOT / "data" / ".bounce-keep-models"


# ─── Helpers ────────────────────────────────────────────────────────────────


def _ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)


def _write_pid(name: str, pid: int):
    (PID_DIR / f"{name}.pid").write_text(str(pid))


def _read_pid(name: str) -> Optional[int]:
    pidfile = PID_DIR / f"{name}.pid"
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            # Check if process is alive
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pidfile.unlink(missing_ok=True)
    return None


def _kill_pid(name: str, sig=signal.SIGTERM) -> bool:
    """Kill a process by PID file. Returns True if process was found."""
    pid = _read_pid(name)
    if pid:
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            pass
    (PID_DIR / f"{name}.pid").unlink(missing_ok=True)
    return False


def _kill_and_wait(name: str, timeout: int = 10):
    """Send SIGTERM, wait, then SIGKILL if needed."""
    if not _kill_pid(name, signal.SIGTERM):
        return
    for _ in range(timeout):
        if _read_pid(name) is None:
            return
        time.sleep(1)
    _kill_pid(name, signal.SIGKILL)
    time.sleep(1)
    (PID_DIR / f"{name}.pid").unlink(missing_ok=True)


def _wait_no_process(pattern: str, timeout: int = 15) -> bool:
    """Poll (1s) until no process matches `pattern` (pgrep -f), up to timeout.

    Returns True once clear, False if a match still exists at timeout. Used to
    make teardown VERIFIED rather than timed: a fixed sleep after `pkill -9`
    raced the kernel actually reaping the process and releasing its SQLite DB
    handle, which is what made `restart` flaky (DB-lock race) while a manual
    `stop && start` — two separate processes with more settling time — usually
    got away with it.
    """
    for _ in range(max(1, timeout)):
        # pgrep returns non-zero when nothing matches. pgrep never matches its
        # own pid, and the CLI's argv is `sparkstation …`, not the pattern.
        if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode != 0:
            return True
        time.sleep(1)
    return False


# LiteLLM listens internally on this port; the public :8000 is owned by the
# sparkstation metrics/auto-resume proxy (gateway/proxy.py), which stays up
# across LiteLLM bounces so model swaps no longer drop the public API port.
# 7999 deliberately sits BELOW the model port range (8001-8100) — 8002 was
# bge-m3's container port.
GATEWAY_INTERNAL_PORT = 7999          # blue
GATEWAY_INTERNAL_PORT_GREEN = 7998    # green (blue-green idle slot)
GATEWAY_PORT_POINTER = "gateway/.litellm-port"  # active-port pointer (proxy follows this)
GATEWAY_CLIENTS_FILE = "gateway/clients.yaml"           # per-client policy (gitignored; secrets)
GATEWAY_CLIENTS_EXAMPLE = "gateway/clients.example.yaml"


def _ensure_clients_config() -> None:
    """Seed gateway/clients.yaml from the committed template on first run.

    The real file holds API keys (secrets) so it's gitignored; the proxy
    hot-reloads it. If it's missing we copy the example so the gateway comes up
    with a working (allow-all, unenforced) policy instead of no policy at all.
    """
    dst = PROJECT_ROOT / GATEWAY_CLIENTS_FILE
    src = PROJECT_ROOT / GATEWAY_CLIENTS_EXAMPLE
    if not dst.exists() and src.exists():
        import shutil
        shutil.copyfile(src, dst)


def _start_litellm() -> "subprocess.Popen":
    """Launch LiteLLM under the BLUE-GREEN manager; returns the manager process.

    litellm-bluegreen.sh runs litellm on one of two ports and, on a config
    change, brings the new litellm up on the idle port, health-checks it, then
    atomically flips gateway/.litellm-port — the :8000 proxy follows that pointer
    and swaps its upstream live, so reloads never drop the public API (the old
    per-model restart-in-place dropped it ~2s each, 502-ing clients during a
    profile bring-up). The 'gateway' pidfile holds the MANAGER pid; SIGTERM to it
    stops the active litellm child too.
    """
    gw_log = LOG_DIR / "gateway.log"
    # Rotate instead of truncating: the previous manager's log is the only
    # record of a blue-green desync once it has happened.
    try:
        if gw_log.exists() and gw_log.stat().st_size > 0:
            gw_log.replace(LOG_DIR / "gateway.prev.log")
    except OSError:
        pass
    gw_env = os.environ.copy()
    gw_env.pop("SUPERVISOR_DATABASE_URL", None)
    # Prefer the project venv's python: it has the LOCKED litellm version.
    # The uv tool env (where the installed `sparkstation` entrypoint lives)
    # resolves fresh on every `uv tool install` and has drifted to litellm
    # versions whose proxy is broken (`No module named 'proxy_server'`).
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    gw_env["LITELLM_PYTHON"] = str(venv_python) if venv_python.exists() else sys.executable
    with open(gw_log, "w") as lf:
        proc = subprocess.Popen(
            ["bash", "gateway/litellm-bluegreen.sh",
             str(GATEWAY_INTERNAL_PORT), str(GATEWAY_INTERNAL_PORT_GREEN), GATEWAY_PORT_POINTER],
            stdout=lf, stderr=subprocess.STDOUT,
            env=gw_env, cwd=PROJECT_ROOT,
            start_new_session=True,
        )
    _write_pid("gateway", proc.pid)
    return proc


def _ensure_proxy() -> None:
    """Start the gateway proxy on :8000 if it isn't already serving.

    Idempotent on purpose: _restart_gateway() calls this every model refresh,
    but the proxy is config-free and survives LiteLLM bounces — it only ever
    needs (re)starting after `sparkstation stop` or a crash.
    """
    if _is_port_open("127.0.0.1", 8000):
        return
    log = LOG_DIR / "gateway-proxy.log"
    _ensure_clients_config()
    px_env = os.environ.copy()
    px_env["SPARKSTATION_LITELLM_PORT_FILE"] = GATEWAY_PORT_POINTER  # blue-green pointer to follow
    px_env["SPARKSTATION_CLIENTS_FILE"] = GATEWAY_CLIENTS_FILE       # per-client policy file
    with open(log, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "gateway.proxy:app",
             "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
            stdout=lf, stderr=subprocess.STDOUT,
            env=px_env, cwd=PROJECT_ROOT, start_new_session=True,
        )
    _write_pid("gateway-proxy", proc.pid)
    for _ in range(15):
        if _is_port_open("127.0.0.1", 8000):
            return
        time.sleep(1)
    click.secho("     Warning: gateway proxy did not bind :8000", fg="yellow")


def _is_port_open(host: str, port: int) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def _supervisor_healthy() -> bool:
    try:
        r = httpx.get(f"{DEFAULT_SUPERVISOR_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _gateway_healthy() -> bool:
    try:
        # /health is the unauthenticated liveness path (enforce_auth would 401
        # a keyless /v1/models probe and make a healthy gateway look down)
        r = httpx.get(f"{DEFAULT_GATEWAY_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _docker_containers(only_running: bool = True) -> list[str]:
    flag = "" if only_running else "-a"
    cmd = f"docker ps {flag} --filter name=sparkstation- --format {{{{.Names}}}}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return [n.strip() for n in r.stdout.strip().split("\n") if n.strip()]


def _systemd_unit_exists() -> bool:
    return subprocess.run(["systemctl", "cat", "sparkstation"],
                          capture_output=True).returncode == 0


def _systemd_active() -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", "sparkstation"],
                          capture_output=True).returncode == 0


def _under_systemd() -> bool:
    """True when this process was spawned by systemd (ExecStart/ExecStop)."""
    return bool(os.environ.get("INVOCATION_ID"))


def _systemctl(verb: str) -> bool:
    """Run `sudo -n systemctl <verb> sparkstation`. Non-interactive on purpose:
    needs the NOPASSWD sudoers rule (see infra repo); returns False if absent."""
    return subprocess.run(["sudo", "-n", "systemctl", verb, "sparkstation"],
                          capture_output=True).returncode == 0


def _stop_models_via_api() -> int:
    """Gracefully stop all models through the supervisor before killing it.

    This is the only path that tears models down CORRECTLY across backends
    and hosts: the supervisor's launchers know how (dspark = 2-node script
    teardown, vLLM-on-worker1 = docker over SSH). The local docker sweep in
    stop() only sees sparkstation-* names on the PRIMARY daemon — on its own
    it leaves worker-host containers and the whole DSV4 stack running
    (holding ~94GB that then fails the next start's memory check).
    """
    stopped = 0
    try:
        r = httpx.get(f"{DEFAULT_SUPERVISOR_URL}/models/detailed", timeout=5)
        models = r.json().get("models", [])
    except Exception:
        return 0
    for m in models:
        if m["status"] not in ("running", "starting"):
            continue
        name = m.get("alias") or m["model_name"]
        try:
            # dspark teardown crosses two nodes over SSH — give it time
            httpx.post(f"{DEFAULT_SUPERVISOR_URL}/models/{m['id']}/stop", timeout=180)
            click.echo(f"     stopped {name}")
            stopped += 1
        except Exception as e:
            click.secho(f"     Warning: failed to stop {name}: {e}", fg="yellow")
    return stopped


def _docker_stop_all():
    """Stop and remove all sparkstation containers."""
    # Kill running
    names = _docker_containers(only_running=True)
    if names:
        subprocess.run(["docker", "kill"] + names, capture_output=True)
        time.sleep(2)

    # Remove all (running + stopped)
    names = _docker_containers(only_running=False)
    if names:
        subprocess.run(["docker", "rm", "-f"] + names, capture_output=True)

    return len(names)


def _write_gateway_yaml():
    """Fetch running models from supervisor and write gateway/litellm.yaml."""
    import yaml
    try:
        r = httpx.get(f"{DEFAULT_SUPERVISOR_URL}/models/detailed", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        click.secho(f"     Warning: could not fetch models from supervisor: {e}", fg="yellow")
        return False

    config_path = PROJECT_ROOT / "gateway" / "litellm.yaml"
    if config_path.exists():
        with open(config_path) as f:
            gw = yaml.safe_load(f) or {}
    else:
        gw = {}

    model_list = []
    for m in data.get("models", []):
        if m["status"] != "running":
            continue
        # Voice models speak WebSocket/WebRTC audio, not the OpenAI API — a
        # LiteLLM route would just 404/502 (mirrors gateway_sync exclusion).
        if (m.get("model_type") or "chat") == "voice":
            continue
        alias = m.get("alias") or m["model_name"].split("/")[-1]
        # Use base_url from supervisor rather than hardcoding 127.0.0.1 — for
        # models on remote cluster hosts (e.g. `host: worker1`) the URL is
        # the QSFP-side IP so LiteLLM's proxy reaches the container over the
        # 200 GbE direct link, not the LAN.
        base = m.get("base_url") or f"http://127.0.0.1:{m['port']}"
        lp = {
            "model": f"openai/{m['model_name']}",
            "api_base": f"{base}/v1",
            "api_key": "EMPTY",
        }
        # Chat models: FORWARD reasoning-control params (thinking level, budget)
        # so a client's per-request thinking actually reaches the backend.
        # drop_params:true would silently strip chat_template_kwargs and every
        # request would fall back to the server default (2026-08-19). Keep
        # dropping for embeddings/detection. Mirrors supervisor gateway_sync.
        # drop_params:false routes unknown fields (chat_template_kwargs, etc.)
        # into the downstream extra_body. Do NOT use allowed_openai_params — it
        # makes LiteLLM pass them as OpenAI SDK kwargs, which errors.
        if (m.get("model_type") or "chat") == "chat":
            lp["drop_params"] = False
        else:
            lp["drop_params"] = True
        entry = {"model_name": alias, "litellm_params": lp}
        model_list.append(entry)
        if m.get("is_default"):
            model_list.append({
                "model_name": "default",
                "litellm_params": dict(entry["litellm_params"]),
            })
        # "vision" alias → profile's vision_default model. Mirror is_default so
        # the CLI's gateway-restart path stays consistent with the supervisor's
        # own gateway_sync (which already emits vision); without this, `gateway
        # restart` silently dropped the vision alias (2026-08-18).
        if m.get("is_vision"):
            model_list.append({
                "model_name": "vision",
                "litellm_params": dict(entry["litellm_params"]),
            })

    gw["model_list"] = model_list
    with open(config_path, "w") as f:
        yaml.dump(gw, f, default_flow_style=False)

    names = [e["model_name"] for e in model_list]
    click.echo(f"     Wrote {len(model_list)} models: {names}")
    return True


# ─── CLI ────────────────────────────────────────────────────────────────────


@click.group()
@click.option("--supervisor-url", default=DEFAULT_SUPERVISOR_URL, help="Supervisor API URL")
@click.pass_context
def cli(ctx, supervisor_url):
    """Sparkstation - LLM orchestration CLI."""
    ctx.ensure_object(dict)
    ctx.obj["supervisor_url"] = supervisor_url


@cli.command()
@click.option("--detach", "-d", is_flag=True, help="Run in background")
@click.option("--profile", "-p", help="Load models from named profile")
@click.option("--wait", "wait_models", is_flag=True,
              help="Block until all models finish loading (default: return once "
                   "supervisor + gateway are up; models load in the background)")
@click.pass_context
def start(ctx, detach, profile, wait_models):
    """Start Sparkstation (supervisor + gateway); models load in the background."""
    _ensure_dirs()

    # Sticky profile: remember explicit choices; resume the last one when none
    # is given (systemd ExecStart passes none — without this a crash-restart
    # boots default_profile even if another profile was live).
    profile_was_explicit = bool(profile)
    if profile:
        try:
            LAST_PROFILE_FILE.write_text(profile)
        except OSError:
            pass
    elif LAST_PROFILE_FILE.exists():
        remembered = LAST_PROFILE_FILE.read_text().strip()
        if remembered:
            profile = remembered
            click.echo(f"  (no --profile given — resuming last profile: {profile})")

    # Check if already running
    if _read_pid("supervisor") and _supervisor_healthy():
        click.secho("Sparkstation supervisor is already running.", fg="yellow")
        click.echo("Run 'sparkstation stop' first, or 'sparkstation restart'.")
        return

    # systemd delegation: when the unit exists and no profile override is
    # asked for, start through systemctl so systemd tracks the supervisor it
    # launched (Type=forking + PIDFile) and its crash-restart works on OUR
    # process instead of racing it. With --profile we intentionally start
    # outside systemd (the unit always starts the default profile); the unit
    # is inactive at that point, so nothing races.
    # (sticky-resumed profiles still delegate — ExecStart re-resolves the same
    # sticky file, so systemd manages the identical start.)
    if detach and not _under_systemd() and _systemd_unit_exists() and not profile_was_explicit:
        click.echo("Starting via systemd (unit exists)...")
        if _systemctl("start"):
            for _ in range(120):
                if _supervisor_healthy():
                    break
                time.sleep(1)
            if _supervisor_healthy():
                click.secho("✓ Sparkstation is running (via systemd)!", fg="green")
                click.echo(f"  Gateway:    {DEFAULT_GATEWAY_URL} (OpenAI-compatible API)")
                click.echo(f"  Supervisor: {DEFAULT_SUPERVISOR_URL} (model management)")
                click.echo("  Models load in the background — watch with 'sparkstation status'.")
            else:
                click.secho("systemd start issued but supervisor not healthy yet — check 'systemctl status sparkstation'.", fg="yellow")
            return
        click.secho("  passwordless systemctl unavailable — starting directly (systemd will not track this instance).", fg="yellow")

    # Reap any stray supervisor/gateway processes from a prior session that
    # outlived their pidfile. Without this, a prior start whose pidfile was
    # overwritten still has its supervisor running — and its reconciliation
    # loop will SIGKILL containers launched by the NEW supervisor, treating
    # them as orphans. We've seen containers with exitCode=137 within 60s of
    # a fresh start for exactly this reason. `stop()` already does this same
    # pkill; mirror it here so `start` is idempotent even after bad prior states.
    strays = subprocess.run(
        ["pgrep", "-f", "uvicorn supervisor.main:app"],
        capture_output=True, text=True,
    )
    if strays.stdout.strip():
        click.echo(f"  → Reaping {len(strays.stdout.strip().splitlines())} stray supervisor process(es) before start")
        subprocess.run(["pkill", "-9", "-f", "uvicorn supervisor.main:app"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "gateway/litellm-"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "litellm.proxy.proxy_cli"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "uvicorn gateway.proxy:app"], capture_output=True)
    time.sleep(1)  # let the kernel finish reaping before we spawn the replacement

    if profile:
        click.echo(f"Starting Sparkstation with profile: {profile}")
    else:
        click.echo("Starting Sparkstation...")

    if not detach:
        # Foreground mode — supervisor only
        click.echo("Starting supervisor in foreground (Ctrl+C to stop)...")
        env = os.environ.copy()
        if profile:
            env["STARTUP_PROFILE"] = profile
        subprocess.run(
            ["uv", "run", "uvicorn", "supervisor.main:app",
             "--host", "127.0.0.1", "--port", "9001"],
            env=env, cwd=PROJECT_ROOT,
        )
        return

    # ── Detached mode ──

    # NOTE: previously this deleted data/sparkstation.db unconditionally as
    # part of "clean stale state". With registry.reconcile_state() now using
    # full container IDs (fixed orphan-detection — old short-ID match was
    # broken and wiped every legitimate container on every start), reconcile
    # correctly adopts surviving model_instances rows for containers that are
    # still running. Wiping the DB here means the next supervisor has no
    # record of those containers, treats them as orphans, removes them, and
    # reloads every model from cold. Letting reconcile run on the existing
    # DB makes restarts near-instant when no model config has changed.

    # 1) Start supervisor
    click.echo("  → Starting supervisor...")
    env = os.environ.copy()
    if profile:
        env["STARTUP_PROFILE"] = profile

    sup_log = LOG_DIR / "supervisor.log"
    with open(sup_log, "w") as lf:
        proc = subprocess.Popen(
            ["uv", "run", "uvicorn", "supervisor.main:app",
             "--host", "127.0.0.1", "--port", "9001"],
            stdout=lf, stderr=subprocess.STDOUT,
            env=env, cwd=PROJECT_ROOT,
            start_new_session=True,
        )
    _write_pid("supervisor", proc.pid)
    click.echo(f"     PID {proc.pid} — log: {sup_log}")

    # 2) Wait for supervisor health. Model autoload runs as a background task
    #    in the supervisor now, so the port binds within seconds — a long wait
    #    here means DB reconcile is slow or the process died.
    click.echo("  → Waiting for supervisor...")
    max_wait = 120
    for elapsed in range(max_wait):
        # Check the process is still alive
        if proc.poll() is not None:
            click.secho(f"     Supervisor process exited (code {proc.returncode})", fg="red")
            click.secho(f"     Check log: {sup_log}", fg="red")
            return

        if _supervisor_healthy():
            click.secho(f"     Supervisor ready ({elapsed}s)", fg="green")
            break

        if elapsed > 0 and elapsed % 15 == 0:
            click.echo(f"     still waiting... ({elapsed}s)")
        time.sleep(1)
    else:
        click.secho(f"     Supervisor did not start within {max_wait}s", fg="red")
        click.secho(f"     Check log: {sup_log}", fg="red")
        return

    # 3) Optionally block until models finish loading (--wait). Default is to
    #    proceed: gateway_sync publishes each model to the gateway as it comes
    #    up, and `sparkstation status` shows live progress. No fixed timeout —
    #    progress is judged by the supervisor's own status, and a model stuck
    #    in `starting` is the supervisor's (restart manager's) problem, not ours.
    if wait_models:
        click.echo("  → Waiting for models (--wait)...")
        while True:
            try:
                r = httpx.get(f"{DEFAULT_SUPERVISOR_URL}/models/detailed", timeout=5)
                models = r.json().get("models", [])
                starting = [m for m in models if m["status"] == "starting"]
                if not starting:
                    running = [m for m in models if m["status"] == "running"]
                    failed = [m for m in models if m["status"] == "failed"]
                    click.echo(f"     {len(running)} running, {len(failed)} failed")
                    break
                click.echo(f"     {len(starting)} still starting: "
                           f"{[m['alias'] or m['model_name'] for m in starting]}")
            except Exception:
                pass
            time.sleep(10)

    # 4) Write gateway config from supervisor state
    click.echo("  → Writing gateway config...")
    _write_gateway_yaml()

    # 5) Start gateway (LiteLLM on the internal port + metrics proxy on :8000)
    click.echo("  → Starting gateway...")
    gw_proc = _start_litellm()
    click.echo(f"     LiteLLM PID {gw_proc.pid} (:{GATEWAY_INTERNAL_PORT}) — log: {LOG_DIR / 'gateway.log'}")
    _ensure_proxy()
    click.echo(f"     Proxy PID {_read_pid('gateway-proxy')} (:8000) — log: {LOG_DIR / 'gateway-proxy.log'}")

    # 6) Wait for gateway
    for i in range(30):
        if gw_proc.poll() is not None:
            click.secho(f"     Gateway process exited (code {gw_proc.returncode})", fg="red")
            click.secho(f"     Check log: {gw_log}", fg="red")
            break
        if _gateway_healthy():
            click.secho(f"     Gateway ready ({i}s)", fg="green")
            break
        time.sleep(1)
    else:
        click.secho("     Warning: gateway may not have started", fg="yellow")

    # 7) Final status
    click.echo()
    try:
        r = httpx.get(f"{DEFAULT_SUPERVISOR_URL}/models/detailed", timeout=5)
        for m in r.json().get("models", []):
            icon = {"running": "●", "starting": "◐", "stopped": "○", "failed": "✗"}.get(m["status"], "?")
            color = {"running": "green", "starting": "yellow", "failed": "red"}.get(m["status"], "white")
            click.secho(f"  {icon} {m['alias'] or m['model_name']}: {m['status']}", fg=color)
    except Exception:
        pass

    click.echo()
    click.secho("✓ Sparkstation is running!", fg="green")
    click.echo(f"  Gateway:    {DEFAULT_GATEWAY_URL} (OpenAI-compatible API)")
    click.echo(f"  Supervisor: {DEFAULT_SUPERVISOR_URL} (model management)")
    if not wait_models:
        click.echo("  Models load in the background and appear in the gateway as they")
        click.echo("  come up — watch with 'sparkstation status' (or start with --wait).")


@cli.command()
def stop():
    """Stop Sparkstation (gateway + supervisor + containers)."""
    _ensure_dirs()

    # systemd race guard: if the unit is tracking the supervisor (Type=forking
    # + PIDFile), killing it directly reads as a crash → Restart=on-failure
    # relaunches everything ~30s later, and its stray-reap has killed
    # manually-started supervisors mid-model-launch (2026-08-15). Delegate to
    # systemctl so systemd both performs and records the stop. ExecStop runs
    # this same command WITH INVOCATION_ID set, so the guard doesn't recurse.
    if not _under_systemd() and _systemd_active():
        click.echo("systemd unit is active — stopping via systemctl (avoids the auto-restart race)...")
        if _systemctl("stop"):
            click.secho("✓ Sparkstation stopped (via systemd)", fg="green")
            return
        click.secho("  passwordless systemctl unavailable (sudoers rule missing?) — stopping directly.", fg="yellow")
        click.secho("  WARNING: systemd may auto-restart Sparkstation in ~30s!", fg="yellow")

    click.echo("Stopping Sparkstation...")

    # `bounce` sentinel: keep models + DB; only the supervisor/gateway
    # processes go down. Consumed HERE (the invocation doing real work), not
    # in the systemctl-delegating invocation above, so it survives the hop
    # into ExecStop.
    keep_models = BOUNCE_SENTINEL.exists()
    if keep_models:
        BOUNCE_SENTINEL.unlink(missing_ok=True)
        click.secho("  bounce mode: models and DB stay; restarting processes only", fg="cyan")

    # 0) Gracefully stop models through the supervisor while it's still up —
    #    the only path that correctly tears down remote-host and dspark
    #    (2-node) models. The docker sweep below is local-daemon-only backstop.
    if keep_models:
        click.echo("  → Keeping models running (bounce)")
    else:
        click.echo("  → Stopping models via supervisor...")
        n = _stop_models_via_api()
        click.echo(f"     {n} model(s) stopped gracefully")

    # 1) Gateway (proxy + LiteLLM)
    click.echo("  → Stopping gateway...")
    _kill_and_wait("gateway-proxy", timeout=5)
    subprocess.run(["pkill", "-9", "-f", "uvicorn gateway.proxy:app"], capture_output=True)
    _kill_and_wait("gateway", timeout=5)
    # The manager's SIGTERM stops only the ACTIVE litellm. A blue-green flip in
    # the last 5 min leaves a draining old litellm + its drainer subshell alive;
    # the next start then binds blind and the pointer/process desync (twice on
    # 2026-09-01: proxy 502 "All connection attempts failed" after `bounce`).
    subprocess.run(["pkill", "-f", "gateway/litellm-bluegreen.sh"], capture_output=True)
    subprocess.run(["pkill", "-f", "litellm.proxy.proxy_cli"], capture_output=True)
    # Also kill by pattern in case PID file was lost. Watcher FIRST — killing
    # only the LiteLLM child leaves the watcher alive to restart it.
    subprocess.run(["pkill", "-9", "-f", "gateway/litellm-"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "litellm.proxy.proxy_cli"], capture_output=True)
    click.echo("     done")

    # 2) Supervisor
    click.echo("  → Stopping supervisor...")
    _kill_and_wait("supervisor", timeout=10)
    subprocess.run(["pkill", "-9", "-f", "uvicorn supervisor.main:app"], capture_output=True)
    # Verify the supervisor is actually gone before deleting its SQLite DB below
    # — a fixed sleep raced the SIGKILL reap/WAL release and left `restart`
    # reopening a DB the dying process still held (the flaky-restart bug).
    if not _wait_no_process("uvicorn supervisor.main:app", timeout=15):
        click.secho("     warning: supervisor process still present after 15s", fg="yellow")
    click.echo("     done")

    if keep_models:
        # Bounce: containers keep serving and the DB stays — the next
        # supervisor adopts them via reconcile_state() and autoload skips
        # RUNNING models.
        click.secho("\n✓ Supervisor/gateway stopped (models still serving)", fg="green")
        return

    # 3) Docker containers
    click.echo("  → Stopping containers...")
    n = _docker_stop_all()
    click.echo(f"     removed {n} container(s)")

    # 4) Clean DB
    db_path = PROJECT_ROOT / "data" / "sparkstation.db"
    for p in [db_path, db_path.with_suffix(".db-shm"), db_path.with_suffix(".db-wal")]:
        p.unlink(missing_ok=True)

    click.secho("\n✓ Sparkstation stopped", fg="green")


@cli.command()
@click.option("--profile", "-p", help="Profile for the restarted supervisor (default: last used)")
@click.pass_context
def bounce(ctx, profile):
    """Restart supervisor + gateway WITHOUT touching running models.

    Containers keep serving throughout; the fresh supervisor adopts them
    (reconcile keeps live DB rows, autoload skips RUNNING models) and rewrites
    gateway routes from current models.yaml — so alias changes (default /
    vision) and supervisor code changes apply in ~30s with zero model
    downtime. Container-level config changes (flags, images, memory) still
    need a real restart of the affected model.
    """
    _ensure_dirs()
    BOUNCE_SENTINEL.touch()
    ctx.invoke(stop)
    if not _wait_no_process("uvicorn supervisor.main:app", timeout=15):
        click.secho("  warning: old supervisor still present; proceeding", fg="yellow")
    ctx.invoke(start, detach=True, profile=profile)


@cli.command()
@click.option("--profile", "-p", help="Load models from named profile")
@click.pass_context
def restart(ctx, profile):
    """Restart Sparkstation (stop → start) with a verified-clean teardown."""
    ctx.invoke(stop)
    # `stop` is now authoritative — it blocks until the supervisor is gone
    # before clearing the SQLite DB, and (being synchronous) has already
    # deleted the DB by the time it returns. Re-confirm as a cheap barrier so a
    # stop that warned-but-proceeded, or a systemd-delegated stop, can't race
    # start into a half-torn-down DB. Replaces the old blind sleep(2) that was
    # the flaky-restart DB-lock race.
    db = PROJECT_ROOT / "data" / "sparkstation.db"
    if not _wait_no_process("uvicorn supervisor.main:app", timeout=15) or db.exists():
        click.secho("  warning: previous instance not fully torn down; starting anyway", fg="yellow")
    ctx.invoke(start, detach=True, profile=profile)


@cli.command()
@click.pass_context
def status(ctx):
    """Show Sparkstation status."""
    supervisor_url = ctx.obj["supervisor_url"]
    click.echo("Sparkstation status\n")

    # Supervisor
    sup_pid = _read_pid("supervisor")
    if sup_pid and _supervisor_healthy():
        click.secho(f"  ● Supervisor: running (PID {sup_pid})", fg="green")
    elif sup_pid:
        click.secho(f"  ◐ Supervisor: process alive (PID {sup_pid}) but not healthy", fg="yellow")
    else:
        click.secho("  ○ Supervisor: not running", fg="red")
        return

    # Gateway
    gw_pid = _read_pid("gateway")
    if gw_pid and _gateway_healthy():
        click.secho(f"  ● Gateway:    running (PID {gw_pid})", fg="green")
    elif gw_pid:
        click.secho(f"  ◐ Gateway:    process alive (PID {gw_pid}) but not healthy", fg="yellow")
    else:
        click.secho("  ○ Gateway:    not running", fg="red")

    # Models
    try:
        r = httpx.get(f"{supervisor_url}/models/detailed", timeout=5)
        models = r.json().get("models", [])
        click.echo(f"\n  {len(models)} model(s):\n")
        for m in models:
            icon = {"running": "●", "starting": "◐", "stopped": "○", "failed": "✗"}.get(m["status"], "?")
            color = {"running": "green", "starting": "yellow", "failed": "red"}.get(m["status"], "white")
            mem = f" ({m['memory_gb']}GB)" if m.get("memory_gb") else ""
            click.secho(f"    {icon} {m['alias'] or m['model_name']}: {m['status']}{mem}", fg=color)
    except Exception:
        click.echo("\n  (could not fetch model details)")

    # Containers
    containers = _docker_containers(only_running=True)
    click.echo(f"\n  {len(containers)} Docker container(s)")


# ─── Model subcommands ──────────────────────────────────────────────────────


# ─── Per-model lifecycle helpers (used by stop/start/swap) ──────────────────


def _models_yaml_lookup(alias: str, profile: Optional[str] = None) -> Optional[dict]:
    """Find the resolved model config for an alias.

    Resolution order:
      1. Named profile (CLI --profile flag), if given — resolves through that
         profile's overrides. Returns None if the alias isn't enabled in it.
      2. STARTUP_PROFILE env var, if set (same behavior).
      3. `default_profile` from models.yaml.
      4. Base spec (no profile overrides).

    Delegates the deep-merge of base + profile overrides to
    supervisor.models_config.find_model_by_alias so the CLI and supervisor
    always see the same resolved config for a given alias in a given profile.
    """
    from supervisor.models_config import find_model_by_alias

    profile = profile or os.environ.get("STARTUP_PROFILE")  # None → find_model_by_alias uses default_profile
    resolved = find_model_by_alias(alias, profile_name=profile)
    if resolved is None:
        return None
    return resolved.model_dump()


def _start_model_via_api(model_cfg: dict, supervisor_url: str) -> str:
    """POST /models/start with the given models.yaml dict. Returns the new model_id."""
    body = {
        "model_name": model_cfg["name"],
        "backend": model_cfg["backend"],
        "model_type": model_cfg.get("model_type", "chat"),
        "model_alias": model_cfg.get("alias"),
        # Cluster role from models.yaml (defaults to "primary" if not set,
        # matching pre-cluster single-host behavior).
        "host": model_cfg.get("host", "primary"),
        "quantization": model_cfg.get("quantization") or "none",
        "memory_gb": model_cfg.get("memory_gb"),
        "idle_timeout_minutes": model_cfg.get("idle_timeout_minutes", 30),
        "auto_suspend_enabled": model_cfg.get("auto_suspend_enabled", False),
        "extra_args": model_cfg.get("extra_args", {}),
        "docker_image": model_cfg.get("docker_image"),
        "env_vars": model_cfg.get("env_vars", {}),
        "volumes": model_cfg.get("volumes", []),
    }
    if model_cfg.get("speculative_model"):
        body["speculative_model"] = model_cfg["speculative_model"]
    if model_cfg.get("speculative_method"):
        body["speculative_method"] = model_cfg["speculative_method"]
    # num_speculative_tokens is meaningful whenever speculative-config is
    # enabled (model OR method), so set it whenever it's specified.
    if model_cfg.get("num_speculative_tokens") is not None:
        body["num_speculative_tokens"] = model_cfg["num_speculative_tokens"]
    if model_cfg.get("speculative_extra"):
        body["speculative_extra"] = model_cfg["speculative_extra"]
    r = httpx.post(f"{supervisor_url}/models/start", json=body, timeout=60)
    r.raise_for_status()
    return r.json()["model_id"]


def _wait_for_model_ready(model_id: str, supervisor_url: str, timeout_seconds: int = 600) -> bool:
    """Poll /models/{id}/status until status=running AND health_status=healthy."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            r = httpx.get(f"{supervisor_url}/models/{model_id}/status", timeout=5)
            if r.status_code == 200:
                s = r.json()
                if s.get("status") == "running" and s.get("health_status") == "healthy":
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _restart_gateway() -> bool:
    """Refresh litellm.yaml from supervisor state and bounce ONLY LiteLLM.

    The public :8000 (metrics/auto-resume proxy) stays up throughout — during
    the ~3-5s LiteLLM bounce, in-flight requests to the internal port fail
    over to the proxy's 502/503 handling instead of connection-refused on the
    public port. Until LiteLLM grows a hot-reload path this bounce is the only
    way to pick up model_list changes.
    """
    _ensure_dirs()
    _write_gateway_yaml()
    _kill_and_wait("gateway", timeout=5)
    subprocess.run(["pkill", "-9", "-f", "gateway/litellm-"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "litellm.proxy.proxy_cli"], capture_output=True)
    time.sleep(1)
    _start_litellm()
    _ensure_proxy()
    for _ in range(30):
        time.sleep(1)
        if _gateway_healthy():
            return True
    return False


@cli.group()
def gateway():
    """Manage the gateway (proxy + LiteLLM)."""
    pass


@gateway.command("restart")
def gateway_restart():
    """Refresh gateway config from supervisor state and bounce LiteLLM.

    The public :8000 proxy stays up (started if missing); only the internal
    LiteLLM process is restarted. Models are untouched.
    """
    click.echo("Restarting gateway...")
    if _restart_gateway():
        click.secho("✓ Gateway ready", fg="green")
    else:
        click.secho("Gateway did not become healthy within 30s — check logs", fg="red")
        sys.exit(1)


@cli.group()
def models():
    """Manage models."""
    pass


@models.command("list")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def models_list(ctx, output_json):
    """List all models."""
    supervisor_url = ctx.obj["supervisor_url"]
    try:
        r = httpx.get(f"{supervisor_url}/models/detailed", timeout=10)
        r.raise_for_status()
        if output_json:
            click.echo(r.text)
            return
        for m in r.json().get("models", []):
            click.echo(f"  {m['alias'] or m['model_name']}  status={m['status']}  port={m['port']}")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@models.command("stop")
@click.argument("alias")
@click.option("--no-gateway-refresh", is_flag=True,
              help="Skip restarting the gateway after stop (faster if you're about to start something)")
@click.pass_context
def models_stop(ctx, alias, no_gateway_refresh):
    """Stop a running model by alias. Other models keep running."""
    supervisor_url = ctx.obj["supervisor_url"]
    try:
        r = httpx.get(f"{supervisor_url}/models/detailed", timeout=10)
        r.raise_for_status()
    except Exception as e:
        click.secho(f"Supervisor unreachable: {e}", fg="red")
        sys.exit(1)
    # "failed" is stoppable on purpose: stopping a FAILED instance marks it
    # STOPPED, which cancels RestartManager's pending backoff retries (the
    # post-backoff re-fetch guard skips non-FAILED instances). Without this,
    # a failed model could not be detached from auto-recovery at all.
    matching = [m for m in r.json()["models"] if m.get("alias") == alias and m.get("status") in ("running", "starting", "suspended", "failed")]
    if not matching:
        click.secho(f"No live model with alias '{alias}'", fg="red")
        sys.exit(1)
    model_id = matching[0]["id"]
    click.echo(f"  → Stopping {alias} ({model_id})...")
    sr = httpx.post(f"{supervisor_url}/models/{model_id}/stop", timeout=30)
    if sr.status_code != 200:
        click.secho(f"  Stop failed: HTTP {sr.status_code}: {sr.text}", fg="red")
        sys.exit(1)
    if not no_gateway_refresh:
        click.echo("  → Refreshing gateway routes...")
        _restart_gateway()
    click.secho(f"✓ Stopped {alias}", fg="green")


@models.command("start")
@click.argument("alias")
@click.option("--profile", "-p", default=None,
              help="Profile to read config from (default: STARTUP_PROFILE env or autoload section)")
@click.pass_context
def models_start(ctx, alias, profile):
    """Start a model by alias, reading its config from models.yaml."""
    model_cfg = _models_yaml_lookup(alias, profile=profile)
    if not model_cfg:
        where = f"profile '{profile}'" if profile else "active profile / autoload"
        click.secho(f"No model with alias '{alias}' in models.yaml ({where})", fg="red")
        sys.exit(1)
    supervisor_url = ctx.obj["supervisor_url"]
    click.echo(f"  → Starting {alias} ({model_cfg['name']})...")
    try:
        model_id = _start_model_via_api(model_cfg, supervisor_url)
    except httpx.HTTPStatusError as e:
        click.secho(f"  Start failed: HTTP {e.response.status_code}: {e.response.text}", fg="red")
        sys.exit(1)
    click.echo(f"     id: {model_id}, waiting for ready (up to 10 min)...")
    if not _wait_for_model_ready(model_id, supervisor_url, timeout_seconds=600):
        click.secho("  Did not reach RUNNING+HEALTHY in 10 min — check supervisor log", fg="yellow")
        sys.exit(1)
    click.echo("  → Refreshing gateway routes...")
    _restart_gateway()
    click.secho(f"✓ Started {alias}", fg="green")


@models.command("swap")
@click.argument("alias")
@click.option("--profile", "-p", default=None,
              help="Profile to read new config from (default: STARTUP_PROFILE env or autoload section)")
@click.pass_context
def models_swap(ctx, alias, profile):
    """Stop a model and start it back from current models.yaml. Other models keep running.

    Use this after editing models.yaml's entry for ALIAS — picks up new docker_image,
    extra_args, speculative-config, etc. without touching other models.
    """
    # Skip gateway refresh on the intermediate stop — we'll do it once at the end.
    ctx.invoke(models_stop, alias=alias, no_gateway_refresh=True)
    ctx.invoke(models_start, alias=alias, profile=profile)


@models.command("logs")
@click.argument("model_id")
@click.option("--follow", "-f", is_flag=True)
@click.option("--tail", default=50)
def models_logs(model_id, follow, tail):
    """Show model container logs."""
    r = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=sparkstation-{model_id}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    if not r.stdout.strip():
        click.secho(f"No container found for: {model_id}", fg="red")
        sys.exit(1)
    container = r.stdout.strip().split("\n")[0]
    cmd = ["docker", "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(container)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


# ─── Cluster subcommands (Docker-over-SSH workers) ─────────────────────────


def _cluster_docker_env(host_role: str) -> dict:
    """env dict for a subprocess docker call targeting a cluster role.

    Delegates to supervisor.models_config so the base + gitignored-local file
    merge stays in one place. Returns os.environ unchanged for the local role.
    """
    from supervisor.models_config import get_cluster_config

    env = os.environ.copy()
    try:
        docker_host = get_cluster_config().docker_host_env(host_role)
        if docker_host:
            env["DOCKER_HOST"] = docker_host
    except Exception:
        pass
    return env


@cli.group()
def cluster():
    """Manage cluster hosts (Docker-over-SSH workers).

    A cluster is defined in models.yaml under the top-level `cluster:` block.
    Roles are logical labels (primary/worker1/...), not hostnames — you assign
    your own IPs. Every subcommand here reads that block.
    """
    pass


@cluster.command("status")
@click.pass_context
def cluster_status(ctx):
    """Show each host's reachability, Docker version, and running models."""
    from supervisor.models_config import get_cluster_config
    try:
        cluster_cfg = get_cluster_config()
    except Exception as e:
        click.secho(f"Failed to load cluster config: {e}", fg="red")
        sys.exit(1)
    hosts_cfg = {role: h.model_dump() for role, h in cluster_cfg.hosts.items()}

    # Query supervisor for models (best-effort; may not be running)
    supervisor_url = ctx.obj["supervisor_url"]
    models_by_host = {}
    try:
        r = httpx.get(f"{supervisor_url}/models/detailed", timeout=3)
        for m in r.json().get("models", []):
            models_by_host.setdefault(m.get("host") or "primary", []).append(m)
    except Exception:
        pass

    click.echo("Cluster hosts:")
    for role, host_cfg in hosts_cfg.items():
        ip = host_cfg.get("ip")
        label = host_cfg.get("label") or role
        is_local = ip is None or ip in ("127.0.0.1", "localhost", "::1")

        if is_local:
            reach = click.style("local", fg="cyan")
        else:
            ping = subprocess.run(
                ["ping", "-c", "1", "-W", "1", ip],
                capture_output=True, text=True,
            )
            reach = (click.style(f"{ip} reachable", fg="green")
                     if ping.returncode == 0
                     else click.style(f"{ip} UNREACHABLE", fg="red"))

        # Docker version via merged env (uses DOCKER_HOST=ssh://... for remote)
        try:
            dv = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True, text=True, timeout=8,
                env=_cluster_docker_env(role),
            )
            docker_info = (f"docker {dv.stdout.strip()}"
                           if dv.returncode == 0
                           else click.style(f"docker: {dv.stderr.strip()[:80] or 'unreachable'}", fg="red"))
        except Exception as e:
            docker_info = click.style(f"docker: {e}", fg="red")

        click.echo(f"\n  ● {role} ({label})")
        click.echo(f"       reach:  {reach}")
        click.echo(f"       {docker_info}")

        host_models = models_by_host.get(role, [])
        if host_models:
            for m in host_models:
                status_color = "green" if m["status"] == "running" else "yellow"
                click.echo(f"       • {click.style(m['alias'] or m['model_name'], bold=True):20s} "
                           f"{click.style(m['status'], fg=status_color):10s}  "
                           f"{m['health_status']}")
        else:
            click.echo("       (no models assigned)")


@cluster.command("sync-cache")
@click.argument("host_role")
@click.option("--only", multiple=True,
              help="Sync specific HF cache subdirs (e.g. --only models--nvidia--Qwen3.6-35B-A3B-NVFP4). Repeatable.")
@click.option("--dry-run", is_flag=True, help="Show what rsync would do, don't transfer.")
def cluster_sync_cache(host_role, only, dry_run):
    """Mirror ~/.cache/huggingface (or specific model subdirs) to a worker
    over SSH+rsync. Resumable, delta-only after first sync.

    Runs as the local user, so root-owned files inside the HF cache (there
    are many — vLLM containers write as root) are read via 'other' bits and
    will land as owner-user on the worker. First-run tip: this WILL error on
    root-owned dirs whose mode is 0600. If you see permission errors, chown
    the HF cache to your user on both nodes once (fixes it forever).
    """
    from supervisor.models_config import get_cluster_config
    cluster_cfg = get_cluster_config()
    if host_role not in cluster_cfg.hosts:
        click.secho(f"Unknown host role '{host_role}'. Known: {list(cluster_cfg.hosts.keys())}", fg="red")
        sys.exit(1)
    host_cfg = cluster_cfg.hosts[host_role]
    ip = host_cfg.ip
    ssh_user = host_cfg.ssh_user
    if ip is None or ip in ("127.0.0.1", "localhost", "::1"):
        click.secho(f"Host '{host_role}' is local — nothing to sync.", fg="yellow")
        return
    if not ssh_user:
        click.secho(f"Host '{host_role}' has ip={ip} but no ssh_user (set it in .sparkstation.local.yaml)", fg="red")
        sys.exit(1)

    src_root = Path.home() / ".cache" / "huggingface" / "hub"
    dst_path = host_cfg.hf_cache_path or f"/home/{ssh_user}/.cache/huggingface/hub"

    # Ensure remote dir exists
    subprocess.run(["ssh", f"{ssh_user}@{ip}", f"mkdir -p {dst_path}"], check=False)

    subdirs = list(only) if only else [""]
    for sub in subdirs:
        src = src_root / sub if sub else src_root
        if not src.exists():
            click.secho(f"Source not found, skipping: {src}", fg="yellow")
            continue
        # Trailing slash matters for rsync (copies contents, not the dir itself)
        src_str = str(src) + ("/" if sub else "/")
        dst = f"{ssh_user}@{ip}:{dst_path}/{sub}/" if sub else f"{ssh_user}@{ip}:{dst_path}/"
        cmd = ["rsync", "-avz", "--partial", "--info=progress2"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend([src_str, dst])
        click.echo(click.style(f"→ {' '.join(cmd)}", fg="cyan"))
        # Inherit stdout so rsync's progress line renders live
        subprocess.run(cmd, check=False)


@cluster.command("ncclbench")
@click.option("--script", default=None,
              help="Path to nccl-bench.sh (default: ../homecloud-infra/qsfp/nccl-bench.sh next to sparkstation repo)")
def cluster_ncclbench(script):
    """Run the two-node NCCL all_gather benchmark over the QSFP link.

    Delegates to homecloud-infra/qsfp/nccl-bench.sh which mpirun's the
    NVIDIA NCCL test suite across the two Sparks. Assumes setup-nccl.sh has
    already been run on both nodes (creates ~/nccl and ~/nccl-tests).

    IMPORTANT: There's a live forum report (June 2026) of GB10 QSFP links
    negotiating 200G but delivering ~12 Gbps payload. Run this to confirm
    your link actually delivers before designing anything around 200 Gb/s.
    """
    if script is None:
        # Default: look for the homecloud-infra checkout as a sibling repo
        candidate = PROJECT_ROOT.parent / "homecloud-infra" / "qsfp" / "nccl-bench.sh"
        if candidate.exists():
            script = str(candidate)
        else:
            click.secho(f"nccl-bench.sh not found at expected sibling repo: {candidate}", fg="red")
            click.secho("Pass --script /path/to/nccl-bench.sh explicitly.", fg="yellow")
            sys.exit(1)
    click.echo(click.style(f"Running: {script}", fg="cyan"))
    subprocess.run(["bash", script], check=False)


# ─── Init command (generates CLAUDE.md) ────────────────────────────────────


SPARKSTATION_START_MARKER = "<!-- SPARKSTATION-START -->"
SPARKSTATION_END_MARKER = "<!-- SPARKSTATION-END -->"


@cli.command()
@click.option("--profile", "-p", help="Generate docs for a specific profile")
def init(profile):
    """Add Sparkstation instructions to CLAUDE.md."""
    # Delegate to the existing init logic — it's large but stable,
    # so we import it from the original module to avoid duplication.
    from cli_init import run_init
    run_init(profile)


@cli.command()
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def cleanup(ctx, force):
    """Full cleanup: stop everything, remove containers, reset database."""
    if not force:
        click.confirm("Stop all services, remove containers, and reset DB?", abort=True)
    ctx.invoke(stop)
    click.secho("✓ Cleanup complete", fg="green")


# ─── voice: Voice Studio over the supervisor's /voice/* API ──────────────────
# Same endpoints the Console uses (supervisor/voice.py). The registry lives on
# the voice role (worker2 ~/cascade-tts/config), never in git.


def _voice_api(ctx, method: str, path: str, timeout: float = 30.0, **kwargs):
    supervisor_url = ctx.obj["supervisor_url"]
    try:
        r = httpx.request(method, f"{supervisor_url}{path}", timeout=timeout, **kwargs)
    except httpx.HTTPError as e:
        click.secho(f"Error: cannot reach supervisor ({e})", fg="red")
        sys.exit(1)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        click.secho(f"Error {r.status_code}: {detail}", fg="red")
        sys.exit(1)
    return r


@cli.group()
def voice():
    """Voice Studio: Sparky's voice registry (clone / design / stock)."""
    pass


@voice.command("status")
@click.pass_context
def voice_status(ctx):
    """Bot + TTS engine health and the default voice."""
    st = _voice_api(ctx, "GET", "/voice/status").json()
    bot = "up" if st["bot"]["ok"] else "DOWN"
    click.echo(f"voice stack on {st['role']}: bot {bot} (:{st['bot']['port']})")
    for name, e in st["engines"].items():
        state = "ready" if e["ok"] else "down"
        if e.get("apply") and e["apply"]["state"] == "restarting":
            state = "restarting"
        elif e.get("apply") and e["apply"]["state"] == "error":
            state = f"restart failed: {e['apply']['error']}"
        click.echo(f"  {name:<7} :{e['port']}  {state}")
    d = st.get("default")
    click.echo(f"default: {d['voice']} ({d['engine']})" if d else "default: (none set — bot uses CASCADE_VOICE)")


@voice.command("list")
@click.option("--json", "output_json", is_flag=True, help="Raw JSON")
@click.pass_context
def voice_list(ctx, output_json):
    """List every registered voice across the three engines."""
    r = _voice_api(ctx, "GET", "/voice/voices")
    if output_json:
        click.echo(r.text)
        return
    for v in r.json():
        star = "★" if v["is_default"] else " "
        detail = v.get("instruct") or (v.get("ref_text") or "")[:60]
        click.echo(f" {star} {v['id']:<24} {v['engine']:<7} {v['language']:<9} {detail}")


@voice.command("default")
@click.argument("voice_id")
@click.option("--engine", type=click.Choice(["clone", "stock", "design"]), help="Needed if the id exists in more than one engine")
@click.pass_context
def voice_default(ctx, voice_id, engine):
    """Make VOICE_ID the voice Sparky speaks in (new sessions)."""
    if engine is None:
        matches = [v for v in _voice_api(ctx, "GET", "/voice/voices").json() if v["id"] == voice_id]
        if not matches:
            click.secho(f"Error: no voice {voice_id!r}", fg="red"); sys.exit(1)
        if len(matches) > 1:
            click.secho(f"Error: {voice_id!r} exists in {[m['engine'] for m in matches]} — pass --engine", fg="red"); sys.exit(1)
        engine = matches[0]["engine"]
    d = _voice_api(ctx, "POST", f"/voice/voices/{engine}/{voice_id}/default").json()["default"]
    click.secho(f"✓ default voice: {d['voice']} ({d['engine']})", fg="green")


@voice.command("design")
@click.argument("voice_id")
@click.argument("description")
@click.option("--language", default="English", show_default=True)
@click.pass_context
def voice_design(ctx, voice_id, description, language):
    """Create a designed voice from a text DESCRIPTION (live immediately)."""
    _voice_api(ctx, "POST", "/voice/voices/design", json={"id": voice_id, "instruct": description, "language": language})
    click.secho(f"✓ designed voice {voice_id} saved", fg="green")


@voice.command("clone")
@click.argument("voice_id")
@click.argument("clip", type=click.Path(exists=True, dir_okay=False))
@click.option("--transcript", required=True, help="Exact transcript of the clip (required by the cloner)")
@click.option("--language", default="English", show_default=True)
@click.option("--instruct", default="", help="Optional style direction applied to every line")
@click.pass_context
def voice_clone(ctx, voice_id, clip, transcript, language, instruct):
    """Register a cloned voice from an 8-12 s reference CLIP (restarts the clone engine, ~45 s)."""
    with open(clip, "rb") as f:
        r = _voice_api(ctx, "POST", "/voice/voices/clone", timeout=180,
                       data={"id": voice_id, "ref_text": transcript, "language": language, "instruct": instruct},
                       files={"file": (os.path.basename(clip), f)})
    body = r.json()
    click.secho(f"✓ clone voice {voice_id} registered ({body['duration_seconds']}s clip); clone engine restarting", fg="green")
    if body.get("warning"):
        click.secho(f"  warning: {body['warning']}", fg="yellow")


@voice.command("instruct")
@click.argument("engine", type=click.Choice(["clone", "stock", "design"]))
@click.argument("voice_id")
@click.argument("text")
@click.pass_context
def voice_instruct(ctx, engine, voice_id, text):
    """Set a voice's style instruct (design: its identity). Empty TEXT clears it."""
    body = _voice_api(ctx, "PATCH", f"/voice/voices/{engine}/{voice_id}", json={"instruct": text}).json()
    click.secho("✓ saved" + (" — engine restarting (~45 s)" if body.get("applying") else ""), fg="green")


@voice.command("delete")
@click.argument("engine", type=click.Choice(["clone", "design"]))
@click.argument("voice_id")
@click.confirmation_option(prompt="Delete this voice (and its reference clip, for clones)?")
@click.pass_context
def voice_delete(ctx, engine, voice_id):
    """Remove a cloned or designed voice."""
    body = _voice_api(ctx, "DELETE", f"/voice/voices/{engine}/{voice_id}").json()
    click.secho("✓ deleted" + (" — engine restarting (~45 s)" if body.get("applying") else ""), fg="green")


@voice.command("sample")
@click.argument("voice_id")
@click.option("--text", default=None, help="What to say (default: a short Sparky line)")
@click.option("--instruct", default=None, help="Per-line direction (stock/design)")
@click.option("-o", "--output", default=None, help="WAV path (default: ./<voice_id>.wav)")
@click.pass_context
def voice_sample(ctx, voice_id, text, instruct, output):
    """Synthesize a WAV sample with a registered voice."""
    body = {"voice": voice_id}
    if text:
        body["text"] = text
    if instruct:
        body["instruct"] = instruct
    r = _voice_api(ctx, "POST", "/voice/speak", timeout=120, json=body)
    out = output or f"{voice_id}.wav"
    with open(out, "wb") as f:
        f.write(r.content)
    click.secho(f"✓ wrote {out} ({len(r.content) // 1024} KB)", fg="green")


if __name__ == "__main__":
    cli(obj={})
