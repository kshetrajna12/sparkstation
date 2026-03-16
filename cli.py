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
PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = Path.home() / ".sparkstation"
LOG_DIR = RUN_DIR / "logs"
PID_DIR = RUN_DIR / "pids"


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
        r = httpx.get(f"{DEFAULT_GATEWAY_URL}/v1/models",
                       headers={"Authorization": "Bearer dummy-key"}, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _docker_containers(only_running: bool = True) -> list[str]:
    flag = "" if only_running else "-a"
    cmd = f"docker ps {flag} --filter name=sparkstation- --format {{{{.Names}}}}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return [n.strip() for n in r.stdout.strip().split("\n") if n.strip()]


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
        alias = m.get("alias") or m["model_name"].split("/")[-1]
        entry = {
            "model_name": alias,
            "litellm_params": {
                "model": f"openai/{m['model_name']}",
                "api_base": f"http://127.0.0.1:{m['port']}/v1",
                "api_key": "EMPTY",
                "drop_params": True,
            },
        }
        model_list.append(entry)
        if m.get("is_default"):
            model_list.append({
                "model_name": "default",
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
@click.pass_context
def start(ctx, detach, profile):
    """Start Sparkstation (supervisor + gateway) and wait for models to be ready."""
    _ensure_dirs()

    # Check if already running
    if _read_pid("supervisor") and _supervisor_healthy():
        click.secho("Sparkstation supervisor is already running.", fg="yellow")
        click.echo("Run 'sparkstation stop' first, or 'sparkstation restart'.")
        return

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

    # Clean stale state
    db_path = PROJECT_ROOT / "data" / "sparkstation.db"
    if db_path.exists():
        db_path.unlink()

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

    # 2) Wait for supervisor health (port bind + models loaded).
    #    The supervisor loads ALL models in its lifespan before binding the
    #    HTTP port, so "connection refused" is expected for several minutes.
    click.echo("  → Waiting for supervisor (models loading)...")
    max_wait = 900  # 15 minutes — large models can take a while
    for elapsed in range(max_wait):
        # Check the process is still alive
        if proc.poll() is not None:
            click.secho(f"     Supervisor process exited (code {proc.returncode})", fg="red")
            click.secho(f"     Check log: {sup_log}", fg="red")
            return

        if _supervisor_healthy():
            click.secho(f"     Supervisor ready ({elapsed}s)", fg="green")
            break

        if elapsed > 0 and elapsed % 30 == 0:
            click.echo(f"     still loading... ({elapsed}s)")
        time.sleep(1)
    else:
        click.secho(f"     Supervisor did not start within {max_wait}s", fg="red")
        click.secho(f"     Check log: {sup_log}", fg="red")
        return

    # 3) Wait for all models to finish starting
    click.echo("  → Waiting for models...")
    for _ in range(300):
        try:
            r = httpx.get(f"{DEFAULT_SUPERVISOR_URL}/models/detailed", timeout=5)
            models = r.json().get("models", [])
            starting = [m for m in models if m["status"] == "starting"]
            if not starting:
                running = [m for m in models if m["status"] == "running"]
                failed = [m for m in models if m["status"] == "failed"]
                click.echo(f"     {len(running)} running, {len(failed)} failed")
                break
        except Exception:
            pass
        time.sleep(2)

    # 4) Write gateway config from supervisor state
    click.echo("  → Writing gateway config...")
    _write_gateway_yaml()

    # 5) Start gateway
    click.echo("  → Starting gateway...")
    gw_log = LOG_DIR / "gateway.log"
    gw_env = os.environ.copy()
    gw_env.pop("SUPERVISOR_DATABASE_URL", None)

    with open(gw_log, "w") as lf:
        gw_proc = subprocess.Popen(
            [sys.executable, "-m", "litellm.proxy.proxy_cli",
             "--config", "gateway/litellm.yaml",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=lf, stderr=subprocess.STDOUT,
            env=gw_env, cwd=PROJECT_ROOT,
            start_new_session=True,
        )
    _write_pid("gateway", gw_proc.pid)
    click.echo(f"     PID {gw_proc.pid} — log: {gw_log}")

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


@cli.command()
def stop():
    """Stop Sparkstation (gateway + supervisor + containers)."""
    _ensure_dirs()
    click.echo("Stopping Sparkstation...")

    # 1) Gateway
    click.echo("  → Stopping gateway...")
    _kill_and_wait("gateway", timeout=5)
    # Also kill by pattern in case PID file was lost
    subprocess.run(["pkill", "-9", "-f", "litellm.proxy.proxy_cli"], capture_output=True)
    click.echo("     done")

    # 2) Supervisor
    click.echo("  → Stopping supervisor...")
    _kill_and_wait("supervisor", timeout=10)
    subprocess.run(["pkill", "-9", "-f", "uvicorn supervisor.main:app"], capture_output=True)
    time.sleep(2)
    click.echo("     done")

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
@click.option("--profile", "-p", help="Load models from named profile")
@click.pass_context
def restart(ctx, profile):
    """Restart Sparkstation (stop → start)."""
    ctx.invoke(stop)
    time.sleep(2)
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


if __name__ == "__main__":
    cli(obj={})
