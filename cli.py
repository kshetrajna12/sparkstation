#!/usr/bin/env python3
"""
Sparkstation CLI - Unified interface for managing Sparkstation and models.
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
import httpx


DEFAULT_SUPERVISOR_URL = "http://localhost:9001"
DEFAULT_GATEWAY_URL = "http://localhost:8000"


@click.group()
@click.option("--supervisor-url", default=DEFAULT_SUPERVISOR_URL, help="Supervisor API URL")
@click.pass_context
def cli(ctx, supervisor_url):
    """Sparkstation - LLM orchestration CLI."""
    ctx.ensure_object(dict)
    ctx.obj["supervisor_url"] = supervisor_url


@cli.command()
@click.option("--detach", "-d", is_flag=True, help="Run in background")
@click.pass_context
def start(ctx, detach):
    """Start Sparkstation (supervisor + gateway) and wait for models to be ready."""
    click.echo("Starting Sparkstation...")

    if detach:
        # Start supervisor in background
        click.echo("  → Starting supervisor...")
        subprocess.Popen(
            ["uv", "run", "uvicorn", "supervisor.main:app", "--host", "127.0.0.1", "--port", "9001"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for supervisor to be ready
        supervisor_url = ctx.obj['supervisor_url']
        for i in range(30):  # Try for 30 seconds
            try:
                response = httpx.get(f"{supervisor_url}/health", timeout=2)
                if response.status_code == 200:
                    click.echo("     Supervisor is running")
                    break
            except:
                pass
            time.sleep(1)
        else:
            click.secho("✗ Supervisor failed to start", fg="red")
            return

        # Wait for all models to be ready (RUNNING or FAILED)
        click.echo("  → Waiting for models to load...")
        max_wait = 180  # 3 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = httpx.get(f"{supervisor_url}/models/detailed", timeout=5)
                if response.status_code == 200:
                    models = response.json()["models"]

                    if not models:
                        # No models to wait for
                        break

                    # Check if all models are done loading (RUNNING or FAILED)
                    starting_models = [m for m in models if m["status"] == "starting"]

                    if not starting_models:
                        # All models are done
                        running = [m for m in models if m["status"] == "running"]
                        failed = [m for m in models if m["status"] == "failed"]

                        click.echo(f"     {len(running)} model(s) running, {len(failed)} failed")
                        break
                    else:
                        # Still loading
                        click.echo(f"     {len(starting_models)} model(s) still loading...", nl=False)
                        click.echo("\r", nl=False)

            except Exception as e:
                pass

            time.sleep(5)

        # Start gateway in background
        click.echo("  → Starting gateway...")
        # Remove SUPERVISOR_DATABASE_URL from environment - gateway runs as simple router without DB features
        # TODO: Add gateway database support for usage tracking, API key management, rate limiting
        gateway_env = os.environ.copy()
        gateway_env.pop("SUPERVISOR_DATABASE_URL", None)
        subprocess.Popen(
            ["uv", "run", "litellm", "--config", "gateway/litellm.yaml",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=gateway_env,
        )

        # Wait for gateway to be ready
        gateway_url = "http://localhost:8000"
        for i in range(15):  # Try for 15 seconds
            try:
                response = httpx.get(f"{gateway_url}/health", timeout=2)
                if response.status_code == 200:
                    click.echo("     Gateway is running")
                    break
            except:
                pass
            time.sleep(1)
        else:
            click.secho("     Warning: Gateway may not have started", fg="yellow")

        # Final status
        click.echo("\n")
        try:
            response = httpx.get(f"{supervisor_url}/models/detailed", timeout=5)
            if response.status_code == 200:
                models = response.json()["models"]
                for model in models:
                    status_icon = {"running": "●", "starting": "◐", "stopped": "○", "failed": "✗"}.get(model["status"], "?")
                    status_color = {"running": "green", "starting": "yellow", "stopped": "white", "failed": "red"}.get(model["status"], "white")

                    click.echo(f"  {status_icon} ", nl=False)
                    click.secho(f"{model['alias'] or model['model_name']}: {model['status']}", fg=status_color)

                click.echo("")
                click.secho("✓ Sparkstation is running!", fg="green")
                click.echo(f"Gateway: http://localhost:8000 (OpenAI-compatible API)")
                click.echo(f"Supervisor: {supervisor_url} (Model management)")
        except:
            click.secho("✓ Supervisor started (could not verify models)", fg="yellow")
    else:
        # Start in foreground (supervisor only)
        click.echo("Starting supervisor in foreground...")
        click.echo("Note: Run with -d to also start gateway")
        subprocess.run(
            ["uv", "run", "uvicorn", "supervisor.main:app", "--host", "127.0.0.1", "--port", "9001"]
        )


@cli.command()
def stop():
    """Stop Sparkstation (gateway + supervisor) and all model containers."""
    click.echo("Stopping Sparkstation...")

    # Stop gateway
    click.echo("  → Stopping gateway...")
    result = subprocess.run(
        ["pkill", "-f", "litellm.*gateway/litellm.yaml"],
        capture_output=True,
    )

    if result.returncode == 0:
        click.echo("     Gateway stopped")
    else:
        click.echo("     No gateway process found")

    # Stop supervisor
    click.echo("  → Stopping supervisor...")
    result = subprocess.run(
        ["pkill", "-f", "uvicorn supervisor.main:app"],
        capture_output=True,
    )

    if result.returncode == 0:
        click.echo("     Supervisor stopped")
    else:
        click.echo("     No supervisor process found")

    # Stop all model containers
    click.echo("  → Stopping model containers...")
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=sparkstation-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        container_names = result.stdout.strip().split("\n")
        for container_name in container_names:
            if container_name:
                subprocess.run(["docker", "stop", container_name], capture_output=True)
                click.echo(f"     Stopping: {container_name}")

        # Wait for containers to actually stop
        click.echo("  → Waiting for containers to stop...")
        for i in range(30):  # Wait up to 30 seconds
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=sparkstation-", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                click.echo("     All containers stopped")
                break
            time.sleep(1)
        else:
            click.secho("     Warning: Some containers may still be stopping", fg="yellow")

        click.secho("\n✓ Sparkstation stopped", fg="green")
    else:
        click.echo("     No model containers running")
        click.secho("\n✓ Sparkstation stopped", fg="green")


@cli.command()
@click.pass_context
def restart(ctx, detach=True):
    """Restart Sparkstation supervisor."""
    ctx.invoke(stop)
    time.sleep(2)
    ctx.invoke(start, detach=detach)


@cli.command()
@click.pass_context
def status(ctx):
    """Show Sparkstation status."""
    supervisor_url = ctx.obj["supervisor_url"]

    # Check supervisor
    click.echo("Checking Sparkstation status...\n")

    try:
        response = httpx.get(f"{supervisor_url}/health", timeout=5)
        if response.status_code == 200:
            click.secho("✓ Supervisor: RUNNING", fg="green")
        else:
            click.secho("✗ Supervisor: UNHEALTHY", fg="red")
            return
    except Exception as e:
        click.secho(f"✗ Supervisor: NOT RUNNING ({e})", fg="red")
        click.echo("\nRun 'sparkstation start' to start the supervisor")
        return

    # Get models
    try:
        response = httpx.get(f"{supervisor_url}/models/detailed", timeout=10)
        if response.status_code == 200:
            models = response.json()["models"]

            if not models:
                click.echo("\nNo models loaded")
                return

            click.echo(f"\n{len(models)} model(s) loaded:\n")

            for model in models:
                status_icon = {
                    "running": "●",
                    "starting": "◐",
                    "stopped": "○",
                    "failed": "✗",
                }.get(model["status"], "?")

                status_color = {
                    "running": "green",
                    "starting": "yellow",
                    "stopped": "white",
                    "failed": "red",
                }.get(model["status"], "white")

                click.echo(f"  {status_icon} ", nl=False)
                click.secho(f"{model['alias'] or model['model_name']}", fg=status_color, bold=True)
                click.echo(f"     Model: {model['model_name']}")
                click.echo(f"     Status: {model['status']} | Port: {model['port']} | Backend: {model['backend']}")

                if model.get("memory_gb"):
                    click.echo(f"     Memory: {model['memory_gb']}GB")

                click.echo()

    except Exception as e:
        click.secho(f"Failed to get model status: {e}", fg="red")


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
        response = httpx.get(f"{supervisor_url}/models/detailed", timeout=10)
        response.raise_for_status()

        if output_json:
            click.echo(response.text)
            return

        models = response.json()["models"]

        if not models:
            click.echo("No models loaded")
            return

        click.echo(f"Found {len(models)} model(s):\n")

        for model in models:
            click.echo(f"ID: {model['id']}")
            click.echo(f"  Name: {model['model_name']}")
            click.echo(f"  Alias: {model['alias']}")
            click.echo(f"  Status: {model['status']}")
            click.echo(f"  Backend: {model['backend']}")
            click.echo(f"  Port: {model['port']}")
            click.echo()

    except Exception as e:
        click.secho(f"Failed to list models: {e}", fg="red")
        sys.exit(1)


@models.command("start")
@click.argument("model_name", required=False)
@click.option("--profile", help="Load models from named profile")
@click.option("--alias", help="Model alias")
@click.option("--backend", default="vllm", help="Backend to use (default: vllm)")
@click.option("--quantization", default="fp8", help="Quantization type (default: fp8)")
@click.pass_context
def models_start(ctx, model_name, profile, alias, backend, quantization):
    """Start a model (from config, profile, or by name)."""
    supervisor_url = ctx.obj["supervisor_url"]

    if profile:
        # Load models from profile
        import yaml
        try:
            with open("models.yaml") as f:
                config = yaml.safe_load(f)
                profile_models = config.get("profiles", {}).get(profile, [])

            if not profile_models:
                click.secho(f"✗ Profile not found: {profile}", fg="red")
                sys.exit(1)

            click.echo(f"Loading {len(profile_models)} model(s) from profile '{profile}'...")
            for model_config in profile_models:
                _start_model(supervisor_url, model_config)

        except Exception as e:
            click.secho(f"✗ Failed to load profile: {e}", fg="red")
            sys.exit(1)

    elif model_name:
        # Start specific model
        model_config = {
            "model_name": model_name,
            "backend": backend,
            "model_alias": alias,
            "quantization": quantization,
        }
        _start_model(supervisor_url, model_config)

    else:
        click.secho("✗ Specify either a model name or --profile", fg="red")
        sys.exit(1)


def _start_model(supervisor_url: str, model_config: dict):
    """Helper to start a single model."""
    try:
        response = httpx.post(
            f"{supervisor_url}/models/start",
            json=model_config,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        model_name = model_config.get("model_alias") or model_config["model_name"]
        click.secho(f"✓ Started model: {model_name}", fg="green")

    except httpx.HTTPStatusError as e:
        model_name = model_config.get("model_alias") or model_config["model_name"]
        click.secho(f"✗ Failed to start {model_name}: {e.response.text}", fg="red")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red")


@models.command("stop")
@click.argument("model_id")
@click.pass_context
def models_stop(ctx, model_id):
    """Stop a running model."""
    supervisor_url = ctx.obj["supervisor_url"]

    try:
        response = httpx.post(f"{supervisor_url}/models/{model_id}/stop", timeout=30)
        response.raise_for_status()

        result = response.json()
        click.secho(f"✓ Stopped model: {result['model_id']}", fg="green")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            click.secho(f"✗ Model not found: {model_id}", fg="red")
        else:
            click.secho(f"✗ Failed to stop model: {e.response.text}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red")
        sys.exit(1)


@models.command("logs")
@click.argument("model_id")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--tail", default=50, help="Number of lines to show")
def models_logs(model_id, follow, tail):
    """Show model container logs."""
    # Find container name from model_id
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=sparkstation-{model_id}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        click.secho(f"✗ No container found for model: {model_id}", fg="red")
        sys.exit(1)

    container_name = result.stdout.strip().split("\n")[0]

    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.extend(["--tail", str(tail), container_name])

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


@cli.command()
@click.option("--force", is_flag=True, help="Force cleanup without confirmation")
@click.pass_context
def cleanup(ctx, force):
    """Clean up database and orphaned containers."""
    if not force:
        click.confirm("This will stop all models, clean up containers, and reset the database. Continue?", abort=True)

    click.echo("Cleaning up...")

    # Stop gateway
    click.echo("  → Stopping gateway...")
    subprocess.run(["pkill", "-9", "-f", "litellm.*gateway/litellm.yaml"], capture_output=True)

    # Stop supervisor
    click.echo("  → Stopping supervisor...")
    subprocess.run(["pkill", "-9", "-f", "uvicorn supervisor.main:app"], capture_output=True)

    # Stop and remove ALL Sparkstation containers (running and stopped)
    click.echo("  → Stopping and removing Sparkstation containers...")
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=sparkstation-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        container_names = result.stdout.strip().split("\n")
        for container_name in container_names:
            if container_name:  # Skip empty lines
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                click.echo(f"     Removed: {container_name}")
    else:
        click.echo("     No Sparkstation containers found")

    # Clean database
    click.echo("  → Cleaning database...")
    db_path = Path("data/sparkstation.db")
    if db_path.exists():
        db_path.unlink()
        click.echo("     Database removed")

    for ext in ["-shm", "-wal"]:
        db_file = Path(f"data/sparkstation.db{ext}")
        if db_file.exists():
            db_file.unlink()

    click.secho("\n✓ Cleanup complete!", fg="green")
    click.echo("\nRun 'sparkstation start' to restart with a clean state")


if __name__ == "__main__":
    cli(obj={})
