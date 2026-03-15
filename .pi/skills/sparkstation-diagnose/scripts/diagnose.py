#!/usr/bin/env python3
"""
Sparkstation Diagnostics

One-command health check: GPU, models, containers, gateway, logs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

# Config
SUPERVISOR_URL = os.environ.get("SPARKSTATION_SUPERVISOR_URL", "http://127.0.0.1:9001")
GATEWAY_URL = os.environ.get("SPARKSTATION_GATEWAY_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("SPARKSTATION_API_KEY", "dummy-key")
LOG_DIR = Path(os.environ.get("SPARKSTATION_LOG_DIR", "data"))

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[96m"


def ok(msg):
    return f"{GREEN}✅ {msg}{RESET}"

def warn(msg):
    return f"{YELLOW}⚠️  {msg}{RESET}"

def fail(msg):
    return f"{RED}❌ {msg}{RESET}"

def header(title, status=None):
    status_str = ""
    if status == "ok":
        status_str = f"{GREEN}✅ OK{RESET}"
    elif status == "warn":
        status_str = f"{YELLOW}⚠️  WARNING{RESET}"
    elif status == "fail":
        status_str = f"{RED}❌ FAILED{RESET}"
    line = f"\n{BOLD}{title}{RESET}"
    if status_str:
        padding = max(1, 50 - len(title))
        line += " " * padding + status_str
    print(line)


def http_get(url, headers=None, timeout=5):
    """Simple HTTP GET."""
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


def check_gpu():
    """Check GPU status via nvidia-smi."""
    result = {"status": "unknown"}

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if out.returncode != 0:
            header("GPU STATUS", "fail")
            print(f"  nvidia-smi failed: {out.stderr.strip()}")
            result["status"] = "error"
            return result

        def _safe_float(val, default=0.0):
            val = val.strip()
            if val in ("[N/A]", "N/A", ""):
                return default
            try:
                return float(val)
            except ValueError:
                return default

        lines = out.stdout.strip().split("\n")
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpu = {
                    "name": parts[0],
                    "memory_used_mib": _safe_float(parts[1]),
                    "memory_total_mib": _safe_float(parts[2]),
                    "temperature_c": _safe_float(parts[3]),
                    "power_draw_w": _safe_float(parts[4]),
                    "power_limit_w": _safe_float(parts[5]),
                    "utilization_pct": _safe_float(parts[6]),
                }
                gpus.append(gpu)

        result["gpus"] = gpus

        # Determine overall status
        max_temp = max(g["temperature_c"] for g in gpus) if gpus else 0
        if max_temp >= 80:
            status = "fail"
        elif max_temp >= 70:
            status = "warn"
        else:
            status = "ok"

        result["status"] = status
        header("GPU STATUS", status)

        for i, gpu in enumerate(gpus):
            temp_color = GREEN if gpu["temperature_c"] < 70 else (YELLOW if gpu["temperature_c"] < 80 else RED)

            print(f"  {gpu['name']}")

            # DGX Spark uses unified memory — nvidia-smi may report 0/N/A
            if gpu["memory_total_mib"] > 0:
                mem_pct = gpu["memory_used_mib"] / gpu["memory_total_mib"] * 100
                mem_used_gib = gpu["memory_used_mib"] / 1024
                mem_total_gib = gpu["memory_total_mib"] / 1024
                print(f"    Memory:      {mem_used_gib:.1f} / {mem_total_gib:.1f} GiB ({mem_pct:.1f}%)")
            else:
                print(f"    Memory:      Unified memory (see RESOURCES section)")

            print(f"    Temperature: {temp_color}{gpu['temperature_c']:.0f}°C{RESET} (threshold: 80°C)")
            if gpu["power_limit_w"] > 0:
                print(f"    Power:       {gpu['power_draw_w']:.0f}W / {gpu['power_limit_w']:.0f}W")
            else:
                print(f"    Power:       {gpu['power_draw_w']:.0f}W")
            print(f"    Utilization: {gpu['utilization_pct']:.0f}%")

    except FileNotFoundError:
        header("GPU STATUS", "fail")
        print("  nvidia-smi not found")
        result["status"] = "error"
    except Exception as e:
        header("GPU STATUS", "fail")
        print(f"  Error: {e}")
        result["status"] = "error"

    return result


def check_supervisor():
    """Check supervisor health."""
    result = {"status": "unknown"}

    data, status = http_get(f"{SUPERVISOR_URL}/health")

    if status == 200:
        result["status"] = "ok"
        result["data"] = data
        header("SUPERVISOR", "ok")
        print(f"  Status: {data.get('status', 'unknown')}")
    elif status == 503:
        result["status"] = "warn"
        result["data"] = data
        header("SUPERVISOR", "warn")
        print(f"  Status: {data.get('status', 'starting')} — {data.get('message', '')}")
    else:
        result["status"] = "fail"
        header("SUPERVISOR", "fail")
        if status == 0:
            print(f"  Connection refused — is supervisor running?")
            print(f"  Start with: sparkstation start -d")
        else:
            print(f"  HTTP {status}: {data}")

    return result


def check_models():
    """Check model statuses."""
    result = {"status": "unknown", "models": []}

    data, status = http_get(f"{SUPERVISOR_URL}/models/detailed")

    if status != 200:
        header("MODELS", "fail")
        print(f"  Cannot reach supervisor ({status})")
        result["status"] = "fail"
        return result

    models = data.get("models", [])
    result["models"] = models

    running = sum(1 for m in models if m["status"] == "running")
    failed = sum(1 for m in models if m["status"] == "failed")
    suspended = sum(1 for m in models if m["status"] == "suspended")
    starting = sum(1 for m in models if m["status"] == "starting")

    if failed > 0:
        overall = "warn"
    elif running == 0 and len(models) > 0:
        overall = "fail"
    else:
        overall = "ok"

    result["status"] = overall
    header("MODELS", overall)
    print(f"  {running} running, {suspended} suspended, {failed} failed, {starting} starting")
    print()

    for m in models:
        alias = m.get("alias") or m.get("model_name", "unknown")
        status_str = m.get("status", "unknown")
        port = m.get("port", "?")
        mem = m.get("memory_gb", 0) or 0
        idle = m.get("idle_seconds")
        health = m.get("health_status", "unknown")

        if status_str == "running":
            icon = f"{GREEN}✅{RESET}"
        elif status_str == "suspended":
            icon = f"{YELLOW}💤{RESET}"
        elif status_str == "starting":
            icon = f"{CYAN}🔄{RESET}"
        else:
            icon = f"{RED}❌{RESET}"

        idle_str = ""
        if idle is not None:
            if idle < 60:
                idle_str = f"idle:{idle:.0f}s"
            elif idle < 3600:
                idle_str = f"idle:{idle/60:.0f}m"
            else:
                idle_str = f"idle:{idle/3600:.1f}h"

        health_str = ""
        if health == "unhealthy":
            health_str = f" {RED}[unhealthy]{RESET}"
        elif health == "unknown":
            health_str = f" {DIM}[health:?]{RESET}"

        print(f"  {icon} {alias:<20s} {status_str:<10s} port:{port}  mem:{mem:.1f}GB  {idle_str}{health_str}")

    return result


def check_containers():
    """Check Docker container status."""
    result = {"status": "unknown", "containers": []}

    try:
        out = subprocess.run(
            [
                "docker", "ps", "-a",
                "--filter", "name=sparkstation-",
                "--format", "{{.Names}}\t{{.Status}}\t{{.State}}\t{{.Ports}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if out.returncode != 0:
            header("CONTAINERS", "fail")
            print(f"  docker ps failed: {out.stderr.strip()}")
            result["status"] = "error"
            return result

        lines = out.stdout.strip().split("\n")
        containers = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                c = {
                    "name": parts[0],
                    "status": parts[1],
                    "state": parts[2],
                    "ports": parts[3] if len(parts) > 3 else "",
                }
                containers.append(c)

        result["containers"] = containers

        running = sum(1 for c in containers if c["state"] == "running")
        exited = sum(1 for c in containers if c["state"] == "exited")

        if exited > 0:
            overall = "warn"
        elif running == 0 and containers:
            overall = "fail"
        else:
            overall = "ok"

        result["status"] = overall
        header("CONTAINERS", overall)

        if not containers:
            print("  No sparkstation containers found")
        else:
            for c in containers:
                if c["state"] == "running":
                    icon = f"{GREEN}✅{RESET}"
                elif c["state"] == "exited":
                    icon = f"{RED}❌{RESET}"
                else:
                    icon = f"{YELLOW}⚠️{RESET} "

                # Extract restart count from status
                restart_match = re.search(r"Restarting \((\d+)\)", c["status"])
                restart_str = f"  restarts:{restart_match.group(1)}" if restart_match else ""

                print(f"  {icon} {c['name']:<40s} {c['status']}{restart_str}")

    except FileNotFoundError:
        header("CONTAINERS", "warn")
        print("  Docker not found — using subprocess mode?")
        result["status"] = "warn"
    except Exception as e:
        header("CONTAINERS", "fail")
        print(f"  Error: {e}")
        result["status"] = "error"

    return result


def check_gateway():
    """Check LiteLLM gateway."""
    result = {"status": "unknown"}

    # Check health
    data, status = http_get(
        f"{GATEWAY_URL}/health",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )

    if status == 200:
        result["health"] = data
    elif status == 0:
        header("GATEWAY", "fail")
        print(f"  Connection refused — is gateway running?")
        print(f"  Start with: sparkstation start -d")
        result["status"] = "fail"
        return result

    # Check models
    models_data, models_status = http_get(
        f"{GATEWAY_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )

    if models_status == 200:
        models = models_data.get("data", [])
        result["models"] = [m["id"] for m in models]
        model_count = len([m for m in models if m["id"] != "default"])
        result["status"] = "ok"
        header("GATEWAY", "ok")
        print(f"  Models registered: {model_count}")
        for m in models:
            if m["id"] != "default":
                print(f"    • {m['id']}")
    else:
        result["status"] = "warn"
        header("GATEWAY", "warn")
        print(f"  Health OK but models endpoint returned {models_status}")

    return result


def check_resources():
    """Check resource allocation."""
    result = {"status": "unknown"}

    data, status = http_get(f"{SUPERVISOR_URL}/resources")

    if status != 200:
        header("RESOURCES", "fail")
        print(f"  Cannot reach supervisor ({status})")
        result["status"] = "fail"
        return result

    result["data"] = data
    mem_used = data.get("unified_memory_used_gb", 0)
    mem_limit = data.get("unified_memory_limit_gb", 0)
    mem_pct = (mem_used / mem_limit * 100) if mem_limit > 0 else 0
    temp = data.get("gpu_temperature_c", 0)
    models_count = data.get("resident_models_count", 0)

    if mem_pct > 90:
        overall = "warn"
    else:
        overall = "ok"

    result["status"] = overall
    header("RESOURCES", overall)
    print(f"  Allocated memory: {mem_used:.1f} / {mem_limit:.1f} GiB ({mem_pct:.1f}%)")
    print(f"  GPU temperature:  {temp:.0f}°C")
    print(f"  Resident models:  {models_count}")
    print(f"  Power draw:       {data.get('gpu_power_draw_w', 0):.0f}W")

    return result


def check_logs(minutes=15):
    """Check recent log entries for errors."""
    result = {"status": "ok", "errors": [], "warnings": []}

    log_file = LOG_DIR / "sparkstation.log"
    if not log_file.exists():
        header("RECENT LOGS", "ok")
        print(f"  No log file found at {log_file}")
        return result

    try:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        errors = []
        warnings = []

        with open(log_file, "r") as f:
            # Read last 500 lines
            lines = f.readlines()[-500:]

        for line in lines:
            # Parse timestamp
            ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if ts_match:
                try:
                    ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if ts < cutoff:
                        continue
                except ValueError:
                    continue

            if " ERROR " in line or " CRITICAL " in line:
                errors.append(line.strip())
            elif " WARNING " in line:
                warnings.append(line.strip())

        result["errors"] = errors[-10:]  # Last 10
        result["warnings"] = warnings[-5:]  # Last 5

        if errors:
            result["status"] = "warn"
            header(f"RECENT LOGS (last {minutes}min)", "warn")
        else:
            header(f"RECENT LOGS (last {minutes}min)", "ok")

        if errors:
            print(f"  {RED}{len(errors)} error(s):{RESET}")
            for e in errors[-10:]:
                # Truncate long lines
                if len(e) > 120:
                    e = e[:117] + "..."
                print(f"    {RED}{e}{RESET}")

        if warnings:
            print(f"  {YELLOW}{len(warnings)} warning(s):{RESET}")
            for w in warnings[-5:]:
                if len(w) > 120:
                    w = w[:117] + "..."
                print(f"    {DIM}{w}{RESET}")

        if not errors and not warnings:
            print(f"  No errors or warnings in the last {minutes} minutes")

    except Exception as e:
        header("RECENT LOGS", "fail")
        print(f"  Error reading logs: {e}")
        result["status"] = "error"

    return result


def main():
    parser = argparse.ArgumentParser(description="Sparkstation Diagnostics")
    parser.add_argument("--quick", action="store_true", help="Quick check (skip logs)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--component",
        choices=["gpu", "models", "containers", "gateway", "logs", "resources"],
        help="Check specific component only",
    )

    args = parser.parse_args()

    # Suppress printed output in JSON mode by redirecting stdout
    import io as _io
    _real_stdout = sys.stdout
    if args.json:
        sys.stdout = _io.StringIO()

    if not args.json:
        print(f"\n{BOLD}{'═' * 50}{RESET}")
        print(f"{BOLD}  SPARKSTATION DIAGNOSTICS{RESET}")
        print(f"{BOLD}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        print(f"{BOLD}{'═' * 50}{RESET}")

    all_results = {}

    checks = {
        "gpu": check_gpu,
        "supervisor": check_supervisor,
        "models": check_models,
        "containers": check_containers,
        "gateway": check_gateway,
        "resources": check_resources,
        "logs": lambda: check_logs(15),
    }

    if args.component:
        if args.component == "logs":
            all_results["logs"] = check_logs(30)
        elif args.component in checks:
            all_results[args.component] = checks[args.component]()
    else:
        for name, check_fn in checks.items():
            if args.quick and name == "logs":
                continue
            all_results[name] = check_fn()

    if args.json:
        sys.stdout = _real_stdout
        print(json.dumps(all_results, indent=2, default=str))
    else:
        # Summary
        statuses = [r.get("status", "unknown") for r in all_results.values()]
        fails = statuses.count("fail") + statuses.count("error")
        warns = statuses.count("warn")

        print(f"\n{BOLD}{'─' * 50}{RESET}")
        if fails > 0:
            print(f"{RED}{BOLD}  OVERALL: {fails} issue(s) need attention{RESET}")
        elif warns > 0:
            print(f"{YELLOW}{BOLD}  OVERALL: {warns} warning(s){RESET}")
        else:
            print(f"{GREEN}{BOLD}  OVERALL: All systems healthy ✅{RESET}")
        print()


if __name__ == "__main__":
    main()
