#!/usr/bin/env python3
"""
Sparkstation Deploy

Safe deployment operations: profile switching, restart, rebuild, verify.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SUPERVISOR_URL = os.environ.get("SPARKSTATION_SUPERVISOR_URL", "http://127.0.0.1:9001")
GATEWAY_URL = os.environ.get("SPARKSTATION_GATEWAY_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("SPARKSTATION_API_KEY", "dummy-key")

# ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DRY_RUN = False


def log(msg, level="info"):
    icons = {"info": "→", "ok": "✅", "warn": "⚠️ ", "error": "❌", "step": "•"}
    colors = {"info": "", "ok": GREEN, "warn": YELLOW, "error": RED, "step": DIM}
    icon = icons.get(level, "→")
    color = colors.get(level, "")
    prefix = f"{YELLOW}[DRY RUN]{RESET} " if DRY_RUN else ""
    print(f"  {prefix}{color}{icon} {msg}{RESET}")


def http_get(url, headers=None, timeout=5):
    try:
        req = urllib.request.Request(url, headers=headers or {})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {"error": str(e)}
        return body, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def is_supervisor_running():
    _, status = http_get(f"{SUPERVISOR_URL}/health")
    return status in (200, 503)


def is_gateway_running():
    # Use /v1/models which is lighter than /health (which does full backend health checks)
    _, status = http_get(f"{GATEWAY_URL}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    return status == 200


def get_running_models():
    data, status = http_get(f"{SUPERVISOR_URL}/models/detailed")
    if status == 200:
        return data.get("models", [])
    return []


def load_profiles():
    """Load available profiles from models.yaml."""
    try:
        import yaml
        with open(PROJECT_ROOT / "models.yaml") as f:
            config = yaml.safe_load(f)
        return config.get("profiles", {})
    except Exception as e:
        log(f"Cannot load models.yaml: {e}", "error")
        return {}


def validate_profile(profile_name):
    """Pre-flight: check profile exists and memory fits."""
    profiles = load_profiles()
    if profile_name not in profiles:
        log(f"Profile '{profile_name}' not found. Available: {list(profiles.keys())}", "error")
        return False

    models = profiles[profile_name]
    total_mem = sum(m.get("memory_gb", 0) or 0 for m in models)

    # Load limit from .env
    hard_limit = 113  # default
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("MEMORY_HARD_LIMIT_GB"):
                    try:
                        hard_limit = float(line.split("=")[1].split("#")[0].strip())
                    except Exception:
                        pass

    if total_mem > hard_limit:
        log(f"Profile '{profile_name}' needs {total_mem:.1f} GB but limit is {hard_limit:.1f} GB", "error")
        return False

    log(f"Profile '{profile_name}': {len(models)} models, {total_mem:.1f}/{hard_limit:.1f} GB", "info")
    return True


def run_cmd(cmd, check=True, timeout=None, env=None):
    """Run a shell command."""
    if DRY_RUN:
        log(f"Would run: {' '.join(cmd) if isinstance(cmd, list) else cmd}", "step")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=PROJECT_ROOT,
        )
        if check and result.returncode != 0:
            log(f"Command failed: {result.stderr[:200]}", "error")
            return False
        return True
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s", "error")
        return False
    except Exception as e:
        log(f"Command error: {e}", "error")
        return False


def stop_all():
    """Stop everything: models, supervisor, gateway.

    Strategy: Stop processes first (graceful SIGTERM, then SIGKILL), then
    stop Docker containers in parallel for speed.
    """
    log("Stopping Sparkstation...", "info")

    if DRY_RUN:
        log("Would stop supervisor, gateway, and all containers", "step")
        return True

    # 1) Gracefully stop supervisor and gateway processes (SIGTERM first)
    for pattern in ["supervisor.main:app", "litellm"]:
        try:
            pids = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True
            )
            if pids.stdout.strip():
                for pid in pids.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid:
                        subprocess.run(["kill", pid], capture_output=True)
                        log(f"Sent SIGTERM to {pattern} (pid {pid})", "step")
        except Exception:
            pass

    # Give processes 3s to shut down gracefully
    time.sleep(3)

    # 2) Force-kill any remaining supervisor/gateway processes
    for pattern in ["supervisor.main:app", "litellm"]:
        try:
            pids = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True
            )
            if pids.stdout.strip():
                for pid in pids.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                        log(f"Force-killed {pattern} (pid {pid})", "step")
        except Exception:
            pass

    # 3) Stop all sparkstation Docker containers in PARALLEL
    containers = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=sparkstation-"],
        capture_output=True, text=True, timeout=10,
    )
    if containers.stdout.strip():
        cids = [c.strip() for c in containers.stdout.strip().split("\n") if c.strip()]
        log(f"Stopping {len(cids)} Docker containers in parallel...", "info")

        # docker stop accepts multiple container IDs — stops them in parallel
        subprocess.run(
            ["docker", "stop", "-t", "5"] + cids,  # 5s grace period (default is 10)
            capture_output=True, timeout=30,
        )

        # Force-remove any that didn't stop
        remaining = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=sparkstation-"],
            capture_output=True, text=True, timeout=5,
        )
        if remaining.stdout.strip():
            leftover = [c.strip() for c in remaining.stdout.strip().split("\n") if c.strip()]
            log(f"Force-killing {len(leftover)} remaining containers", "warn")
            subprocess.run(["docker", "kill"] + leftover, capture_output=True, timeout=10)

    # 4) Remove stopped sparkstation containers to avoid name conflicts on restart
    subprocess.run(
        ["docker", "container", "prune", "-f", "--filter", "label=sparkstation"],
        capture_output=True, timeout=10,
    )
    # Also remove by name pattern (containers may not have the label)
    stopped = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=sparkstation-"],
        capture_output=True, text=True, timeout=5,
    )
    if stopped.stdout.strip():
        rm_ids = [c.strip() for c in stopped.stdout.strip().split("\n") if c.strip()]
        subprocess.run(["docker", "rm", "-f"] + rm_ids, capture_output=True, timeout=10)

    # 5) Clean stale DB to prevent ghost model IDs on next startup
    db_path = PROJECT_ROOT / "data" / "sparkstation.db"
    if db_path.exists():
        db_path.unlink()
        log("Cleaned stale database", "step")

    log("All services stopped", "ok")
    return True


def start_with_profile(profile=None):
    """Start supervisor + gateway with optional profile.

    Starts both as background processes (like the CLI does) and returns
    immediately. Use wait_for_healthy() after to confirm models are ready.
    """
    if DRY_RUN:
        profile_str = f" --profile {profile}" if profile else ""
        log(f"Would run: sparkstation start -d{profile_str}", "step")
        return True

    log(f"Starting Sparkstation{' with profile ' + profile if profile else ''}...", "info")

    import time as _time

    # Create log directory
    log_dir = Path.home() / ".sparkstation" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Start supervisor ---
    log("Starting supervisor...", "step")
    supervisor_env = os.environ.copy()
    if profile:
        supervisor_env["STARTUP_PROFILE"] = profile

    supervisor_log = log_dir / "supervisor.log"
    with open(supervisor_log, "w") as lf:
        subprocess.Popen(
            ["uv", "run", "uvicorn", "supervisor.main:app", "--host", "127.0.0.1", "--port", "9001"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=supervisor_env,
            cwd=PROJECT_ROOT,
            start_new_session=True,  # Detach from parent so it survives script exit
        )
    log(f"Supervisor log: {supervisor_log}", "step")

    # Wait for supervisor to accept connections.
    # NOTE: Supervisor loads ALL models in its lifespan before binding the port.
    # On DGX Spark with large models (30B+), this can take 5-10 minutes.
    wait_secs = 600
    log(f"Waiting for supervisor (up to {wait_secs}s — model loading happens before port binds)...", "step")
    for i in range(wait_secs):
        if is_supervisor_running():
            log(f"Supervisor is up (took {i}s)", "ok")
            break
        if i > 0 and i % 30 == 0:
            log(f"Still waiting... ({i}s)", "step")
        _time.sleep(1)
    else:
        log(f"Supervisor did not start within {wait_secs}s", "error")
        return False

    # --- Start gateway ---
    log("Starting gateway...", "step")
    gateway_env = os.environ.copy()
    gateway_env.pop("SUPERVISOR_DATABASE_URL", None)

    gateway_log = log_dir / "gateway.log"
    with open(gateway_log, "w") as lf:
        subprocess.Popen(
            ["uv", "run", "litellm", "--config", "gateway/litellm.yaml",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=gateway_env,
            cwd=PROJECT_ROOT,
            start_new_session=True,  # Detach from parent
        )
    log(f"Gateway log: {gateway_log}", "step")

    # Wait for gateway (up to 60s — litellm startup can be slow)
    for i in range(60):
        if is_gateway_running():
            log(f"Gateway is up (took {i}s)", "ok")
            break
        if i > 0 and i % 15 == 0:
            log(f"Still waiting for gateway... ({i}s)", "step")
        _time.sleep(1)
    else:
        log("Gateway did not start within 60s", "error")
        return False

    log("Services started", "ok")
    return True


def wait_for_healthy(timeout_sec=600):
    """Wait for supervisor to be healthy and all models RUNNING."""
    if DRY_RUN:
        log("Would wait for all models to be RUNNING", "step")
        return True

    log("Waiting for models to be ready...", "info")
    start = time.time()

    while time.time() - start < timeout_sec:
        # Check supervisor health
        data, status = http_get(f"{SUPERVISOR_URL}/health")
        if status == 503:
            elapsed = int(time.time() - start)
            log(f"Supervisor starting... ({elapsed}s)", "step")
            time.sleep(10)
            continue
        elif status != 200:
            time.sleep(5)
            continue

        # Check models
        models = get_running_models()
        if not models:
            time.sleep(5)
            continue

        running = sum(1 for m in models if m["status"] == "running")
        starting = sum(1 for m in models if m["status"] == "starting")
        failed = sum(1 for m in models if m["status"] == "failed")

        if starting == 0:
            elapsed = int(time.time() - start)
            if failed > 0:
                log(f"{running} running, {failed} FAILED after {elapsed}s", "warn")
                for m in models:
                    if m["status"] == "failed":
                        log(f"  Failed: {m.get('alias', m.get('model_name', '?'))}", "error")
                return False
            else:
                log(f"All {running} models RUNNING in {elapsed}s", "ok")
                return True

        elapsed = int(time.time() - start)
        starting_names = [m.get("alias", "?") for m in models if m["status"] == "starting"]
        log(f"{running} running, {starting} starting ({elapsed}s): {starting_names}", "step")
        time.sleep(10)

    log(f"Timed out after {timeout_sec}s", "error")
    return False


def restart_gateway():
    """Restart the gateway to pick up updated litellm.yaml.

    After all models are healthy, we write the final litellm.yaml ourselves
    from the supervisor's model list, then restart the gateway. This avoids
    depending on the async gateway sync (which runs every 60s).
    """
    log("Restarting gateway to pick up final model list...", "info")

    # Write litellm.yaml from the supervisor's /models/detailed endpoint.
    # This gives us the full model names, aliases, and ports needed for the gateway.
    try:
        detailed, status = http_get(f"{SUPERVISOR_URL}/models/detailed")
        if status == 200 and detailed and "models" in detailed:
            import yaml as _yaml
            config_path = PROJECT_ROOT / "gateway" / "litellm.yaml"

            # Read existing config to preserve general/router settings
            if config_path.exists():
                with open(config_path, "r") as f:
                    gw_config = _yaml.safe_load(f) or {}
            else:
                gw_config = {}

            # Build model_list from running models
            model_list = []
            default_alias = None
            for m in detailed["models"]:
                if m["status"] != "running":
                    continue
                alias = m.get("alias") or m["model_name"].split("/")[-1]
                if m.get("is_default"):
                    default_alias = alias
                model_list.append({
                    "model_name": alias,
                    "litellm_params": {
                        "model": f"openai/{m['model_name']}",
                        "api_base": f"http://127.0.0.1:{m['port']}/v1",
                        "api_key": "EMPTY",
                        "drop_params": True,
                    },
                })

            # Add "default" alias for the default model
            if default_alias:
                for entry in model_list:
                    if entry["model_name"] == default_alias:
                        model_list.append({
                            "model_name": "default",
                            "litellm_params": dict(entry["litellm_params"]),
                        })
                        break

            gw_config["model_list"] = model_list
            with open(config_path, "w") as f:
                _yaml.dump(gw_config, f, default_flow_style=False)

            names = [m["model_name"] for m in model_list]
            log(f"Wrote litellm.yaml with {len(model_list)} models: {names}", "step")
    except Exception as e:
        log(f"Failed to write litellm.yaml: {e}", "warn")

    # Kill existing gateway
    for pattern in ["litellm"]:
        try:
            pids = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True
            )
            if pids.stdout.strip():
                for pid in pids.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid:
                        subprocess.run(["kill", pid], capture_output=True)
        except Exception:
            pass

    time.sleep(2)

    # Force-kill any remaining
    try:
        subprocess.run(["pkill", "-9", "-f", "litellm"], capture_output=True)
    except Exception:
        pass
    time.sleep(1)

    # Start fresh gateway
    log_dir = Path.home() / ".sparkstation" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gateway_env = os.environ.copy()
    gateway_env.pop("SUPERVISOR_DATABASE_URL", None)

    gateway_log = log_dir / "gateway.log"
    with open(gateway_log, "w") as lf:
        subprocess.Popen(
            ["uv", "run", "litellm", "--config", "gateway/litellm.yaml",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=gateway_env,
            cwd=PROJECT_ROOT,
            start_new_session=True,
        )

    import time as _time
    for i in range(30):
        if is_gateway_running():
            log(f"Gateway restarted (took {i}s)", "ok")
            return True
        _time.sleep(1)

    log("Gateway restart may have failed", "warn")
    return False


def verify_deployment():
    """Quick integration check after deploy."""
    log("Verifying deployment...", "info")

    if DRY_RUN:
        log("Would run integration checks", "step")
        return True

    checks_passed = 0
    checks_total = 0

    # Check supervisor
    checks_total += 1
    _, status = http_get(f"{SUPERVISOR_URL}/health")
    if status == 200:
        checks_passed += 1
        log("Supervisor healthy", "ok")
    else:
        log("Supervisor not healthy", "error")

    # Check gateway
    checks_total += 1
    _, status = http_get(f"{GATEWAY_URL}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    if status == 200:
        checks_passed += 1
        log("Gateway responding", "ok")
    else:
        log(f"Gateway error (HTTP {status})", "error")

    # Quick chat test
    models_data, _ = http_get(f"{GATEWAY_URL}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    chat_model = None
    if models_data and "data" in models_data:
        for m in models_data["data"]:
            if m["id"] != "default" and "bge" not in m["id"] and "clip" not in m["id"]:
                chat_model = m["id"]
                break

    if chat_model:
        checks_total += 1
        try:
            body = json.dumps({
                "model": chat_model,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 8,
            }).encode()
            req = urllib.request.Request(
                f"{GATEWAY_URL}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            )
            resp = urllib.request.urlopen(req, timeout=60)
            if resp.status == 200:
                checks_passed += 1
                log(f"Chat model '{chat_model}' responding", "ok")
            else:
                log(f"Chat model '{chat_model}' returned {resp.status}", "error")
        except Exception as e:
            log(f"Chat test failed: {e}", "error")

    if checks_passed == checks_total:
        log(f"All {checks_total} checks passed", "ok")
        return True
    else:
        log(f"{checks_passed}/{checks_total} checks passed", "warn")
        return checks_passed > 0


def rebuild_backend(backend):
    """Rebuild Docker image for a backend."""
    valid = {"clip", "flux", "species"}
    if backend not in valid:
        log(f"Invalid backend '{backend}'. Valid: {valid}", "error")
        return False

    dockerfile = PROJECT_ROOT / "docker" / backend / "Dockerfile"
    if not dockerfile.exists():
        log(f"Dockerfile not found: {dockerfile}", "error")
        return False

    tag = f"sparkstation-{backend}:latest"
    log(f"Rebuilding Docker image: {tag}", "info")

    if DRY_RUN:
        log(f"Would build: docker build -t {tag} docker/{backend}/", "step")
        return True

    result = subprocess.run(
        ["docker", "build", "-t", tag, f"docker/{backend}/"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        log(f"Build failed: {result.stderr[:300]}", "error")
        return False

    log(f"Image rebuilt: {tag}", "ok")
    return True


# ─── Commands ───

def cmd_stop(args):
    print(f"\n{BOLD}═══ SPARKSTATION STOP ═══{RESET}\n")
    ok = stop_all()
    print()
    return ok


def cmd_start(args):
    print(f"\n{BOLD}═══ SPARKSTATION START ═══{RESET}\n")

    if args.profile:
        if not validate_profile(args.profile):
            return False

    ok = start_with_profile(args.profile)
    if ok and not DRY_RUN:
        ok = wait_for_healthy()
    print()
    return ok


def cmd_restart(args):
    print(f"\n{BOLD}═══ SPARKSTATION RESTART ═══{RESET}\n")

    if args.profile:
        if not validate_profile(args.profile):
            return False

    ok = stop_all()
    if not ok and not DRY_RUN:
        log("Stop had issues, continuing with start...", "warn")

    time.sleep(3) if not DRY_RUN else None

    ok = start_with_profile(args.profile)
    if ok and not DRY_RUN:
        ok = wait_for_healthy()
        if ok:
            # Restart gateway to pick up final model list (litellm.yaml rewritten by sync)
            restart_gateway()
            verify_deployment()
    print()
    return ok


def cmd_switch_profile(args):
    profile = args.profile_name
    print(f"\n{BOLD}═══ SWITCH PROFILE → {profile.upper()} ═══{RESET}\n")

    # Pre-flight
    if not validate_profile(profile):
        return False

    # Show what's currently running
    if is_supervisor_running():
        models = get_running_models()
        if models:
            current_names = [m.get("alias", m.get("model_name", "?")) for m in models]
            log(f"Currently running: {current_names}", "info")

    # Stop
    ok = stop_all()
    if not ok and not DRY_RUN:
        log("Stop had issues, continuing...", "warn")

    if not DRY_RUN:
        time.sleep(3)

    # Start with new profile
    ok = start_with_profile(profile)
    if ok and not DRY_RUN:
        ok = wait_for_healthy()
        if ok:
            restart_gateway()
            verify_deployment()

    if ok:
        log(f"Profile '{profile}' deployed successfully!", "ok")
    else:
        log(f"Profile switch had issues", "error")

    print()
    return ok


def cmd_rebuild(args):
    print(f"\n{BOLD}═══ REBUILD {args.backend.upper()} ═══{RESET}\n")
    ok = rebuild_backend(args.backend)
    print()
    return ok


def cmd_verify(args):
    print(f"\n{BOLD}═══ VERIFY DEPLOYMENT ═══{RESET}\n")

    if not is_supervisor_running():
        log("Supervisor not running", "error")
        return False

    models = get_running_models()
    running = sum(1 for m in models if m["status"] == "running")
    failed = sum(1 for m in models if m["status"] == "failed")
    starting = sum(1 for m in models if m["status"] == "starting")

    log(f"Models: {running} running, {starting} starting, {failed} failed", "info")

    if starting > 0:
        log("Models still starting, waiting...", "info")
        wait_for_healthy()

    ok = verify_deployment()
    print()
    return ok


def cmd_full(args):
    print(f"\n{BOLD}═══ FULL DEPLOYMENT → {args.profile.upper() if args.profile else 'DEFAULT'} ═══{RESET}\n")

    if args.profile:
        if not validate_profile(args.profile):
            return False

    # Stop
    stop_all()
    if not DRY_RUN:
        time.sleep(3)

    # Rebuild if specified
    if args.rebuild:
        for backend in args.rebuild.split(","):
            rebuild_backend(backend.strip())

    # Start
    ok = start_with_profile(args.profile)
    if ok and not DRY_RUN:
        ok = wait_for_healthy()
        if ok:
            restart_gateway()
            ok = verify_deployment()

    if ok:
        log("Full deployment complete!", "ok")
    print()
    return ok


def main():
    global DRY_RUN

    parser = argparse.ArgumentParser(description="Sparkstation Deploy")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    subparsers = parser.add_subparsers(dest="command", help="Deploy command")

    # stop
    subparsers.add_parser("stop", help="Stop all services")

    # start
    p_start = subparsers.add_parser("start", help="Start services")
    p_start.add_argument("--profile", help="Profile name")

    # restart
    p_restart = subparsers.add_parser("restart", help="Full restart")
    p_restart.add_argument("--profile", help="Profile name")

    # switch-profile
    p_switch = subparsers.add_parser("switch-profile", help="Switch to a different profile")
    p_switch.add_argument("profile_name", help="Target profile")

    # rebuild
    p_rebuild = subparsers.add_parser("rebuild", help="Rebuild Docker image")
    p_rebuild.add_argument("backend", choices=["clip", "flux", "species"], help="Backend to rebuild")

    # verify
    subparsers.add_parser("verify", help="Verify deployment health")

    # full
    p_full = subparsers.add_parser("full", help="Full deploy: stop → rebuild → start → verify")
    p_full.add_argument("--profile", help="Profile name")
    p_full.add_argument("--rebuild", help="Backends to rebuild (comma-separated)")

    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "stop": cmd_stop,
        "start": cmd_start,
        "restart": cmd_restart,
        "switch-profile": cmd_switch_profile,
        "rebuild": cmd_rebuild,
        "verify": cmd_verify,
        "full": cmd_full,
    }

    ok = commands[args.command](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
