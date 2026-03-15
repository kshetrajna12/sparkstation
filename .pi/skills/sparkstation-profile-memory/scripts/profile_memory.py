#!/usr/bin/env python3
"""
Sparkstation Memory Profiler

Measures actual GPU/process memory per model container and compares
to declared memory_gb in models.yaml.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SUPERVISOR_URL = os.environ.get("SPARKSTATION_SUPERVISOR_URL", "http://127.0.0.1:9001")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def http_get(url, timeout=5):
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), resp.status
    except Exception as e:
        return {"error": str(e)}, 0


def get_container_memory():
    """Get memory usage from docker stats for sparkstation containers."""
    try:
        out = subprocess.run(
            [
                "docker", "stats", "--no-stream",
                "--format", "{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return {}

        containers = {}
        for line in out.stdout.strip().split("\n"):
            if not line.strip() or "sparkstation-" not in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                name = parts[0].strip()
                mem_str = parts[1].strip()

                # Parse memory like "48.2GiB / 124.6GiB" or "1.8GiB / 124.6GiB"
                match = re.match(r"([\d.]+)\s*(GiB|MiB|KiB|GB|MB|KB)", mem_str)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    if unit in ("MiB", "MB"):
                        value /= 1024
                    elif unit in ("KiB", "KB"):
                        value /= (1024 * 1024)
                    containers[name] = value

        return containers
    except Exception as e:
        print(f"Error getting docker stats: {e}", file=sys.stderr)
        return {}


def get_model_details():
    """Get model details from supervisor."""
    data, status = http_get(f"{SUPERVISOR_URL}/models/detailed")
    if status != 200:
        return []
    return data.get("models", [])


def load_models_yaml():
    """Load models.yaml to get declared memory_gb."""
    yaml_path = PROJECT_ROOT / "models.yaml"
    if not yaml_path.exists():
        return {}

    try:
        import yaml
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        # Build lookup by alias
        declared = {}
        
        # Check autoload models
        autoload = config.get("autoload", {}).get("models", [])
        for m in autoload:
            alias = m.get("alias", m.get("name", ""))
            if m.get("memory_gb"):
                declared[alias] = m["memory_gb"]
        
        # Check all profiles
        for profile_name, profile_models in config.get("profiles", {}).items():
            for m in profile_models:
                alias = m.get("alias", m.get("name", ""))
                if m.get("memory_gb"):
                    declared[alias] = m["memory_gb"]
        
        return declared
    except ImportError:
        # Fallback: parse yaml manually for memory_gb
        declared = {}
        current_alias = None
        with open(yaml_path) as f:
            for line in f:
                alias_match = re.search(r'alias:\s*"?([^"\s]+)"?', line)
                if alias_match:
                    current_alias = alias_match.group(1)
                mem_match = re.search(r'memory_gb:\s*([\d.]+)', line)
                if mem_match and current_alias:
                    declared[current_alias] = float(mem_match.group(1))
        return declared


def match_container_to_model(container_name, models):
    """Match a Docker container name to a model entry."""
    # Container names are like: sparkstation-qwen3-vl-30b-a3b-instruct-fp8-a6e79ca0
    container_lower = container_name.lower()
    for m in models:
        model_name = (m.get("model_name") or "").lower().replace("/", "-").replace(".", "-")
        alias = (m.get("alias") or "").lower()
        # Check if model name parts appear in container name
        if alias and alias.replace("-", "") in container_lower.replace("-", ""):
            return m
        # Check by model name fragments
        name_parts = model_name.split("-")
        if len(name_parts) >= 2:
            key_parts = [p for p in name_parts if len(p) > 2]
            if key_parts and all(p in container_lower for p in key_parts[:3]):
                return m
    return None


def main():
    parser = argparse.ArgumentParser(description="Sparkstation Memory Profiler")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--compare", action="store_true", help="Compare with models.yaml")
    args = parser.parse_args()

    models = get_model_details()
    container_mem = get_container_memory()
    declared_mem = load_models_yaml()

    if not models:
        print("ERROR: No models found. Is Sparkstation running?", file=sys.stderr)
        sys.exit(1)

    # Build profile data
    profiles = []
    for m in models:
        alias = m.get("alias") or m.get("model_name", "unknown")
        supervisor_mem = m.get("memory_gb", 0) or 0

        # Find matching container
        actual_mem = 0
        matched_container = None
        for cname, cmem in container_mem.items():
            matched = match_container_to_model(cname, [m])
            if matched:
                actual_mem = cmem
                matched_container = cname
                break

        # Get declared memory from models.yaml
        yaml_mem = declared_mem.get(alias, supervisor_mem)

        delta = actual_mem - yaml_mem if actual_mem > 0 else 0
        pct_used = (actual_mem / yaml_mem * 100) if yaml_mem > 0 and actual_mem > 0 else 0

        profiles.append({
            "alias": alias,
            "status": m.get("status", "unknown"),
            "declared_gb": yaml_mem,
            "supervisor_gb": supervisor_mem,
            "actual_gb": round(actual_mem, 2),
            "delta_gb": round(delta, 2),
            "utilization_pct": round(pct_used, 1),
            "container": matched_container,
        })

    if args.json:
        print(json.dumps(profiles, indent=2))
        return

    # Display
    print(f"\n{BOLD}═══ SPARKSTATION MEMORY PROFILE ═══{RESET}\n")
    print(f"{BOLD}MODEL MEMORY USAGE{RESET}\n")
    print(f"  {'Model':<20s} {'Declared':>10s} {'Actual':>10s} {'Delta':>10s} {'Used%':>8s} Status")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*12}")

    total_declared = 0
    total_actual = 0

    for p in profiles:
        alias = p["alias"]
        declared = p["declared_gb"]
        actual = p["actual_gb"]
        delta = p["delta_gb"]
        pct = p["utilization_pct"]
        status = p["status"]

        total_declared += declared
        total_actual += actual

        # Status indicator
        if actual == 0:
            indicator = f"{DIM}[no data]{RESET}"
        elif pct > 95:
            indicator = f"{RED}⚠️  tight{RESET}"
        elif pct < 50:
            indicator = f"{YELLOW}⚠️  over-allocated{RESET}"
        elif pct < 70:
            indicator = f"{YELLOW}📊 room to tighten{RESET}"
        else:
            indicator = f"{GREEN}✅ OK{RESET}"

        actual_str = f"{actual:.1f} GB" if actual > 0 else "N/A"
        delta_str = f"{delta:+.1f} GB" if actual > 0 else "N/A"
        pct_str = f"{pct:.0f}%" if actual > 0 else "N/A"

        print(f"  {alias:<20s} {declared:>8.1f} GB {actual_str:>10s} {delta_str:>10s} {pct_str:>8s} {indicator}")

    # Summary
    potential_savings = total_declared - total_actual if total_actual > 0 else 0
    savings_pct = (potential_savings / total_declared * 100) if total_declared > 0 else 0

    print(f"\n{BOLD}SUMMARY{RESET}")
    print(f"  Total declared:   {total_declared:.1f} GB")
    if total_actual > 0:
        print(f"  Total actual:     {total_actual:.1f} GB")
        if potential_savings > 0:
            print(f"  Potential savings: {YELLOW}{potential_savings:.1f} GB ({savings_pct:.1f}%){RESET}")
        else:
            print(f"  {RED}Memory pressure: over-using by {abs(potential_savings):.1f} GB{RESET}")
    else:
        print(f"  {DIM}Actual memory not available (docker stats may need a moment){RESET}")

    # DGX Spark note
    print(f"\n  {DIM}NOTE: On DGX Spark, docker stats shows RSS (host memory) only.")
    print(f"  GPU/CUDA allocations (KV cache, model weights) use unified memory")
    print(f"  and may not appear in docker stats. The 'declared' value accounts")
    print(f"  for total GPU+host memory needed by the model.{RESET}")

    # Recommendations
    recs = []
    for p in profiles:
        if p["actual_gb"] > 0:
            if p["utilization_pct"] < 50:
                rec_gb = max(p["actual_gb"] * 1.2, p["actual_gb"] + 1)  # 20% headroom min
                recs.append(f"  • {p['alias']}: Reduce memory_gb from {p['declared_gb']:.1f} to ~{rec_gb:.1f} GB")
            elif p["utilization_pct"] > 95:
                rec_gb = p["actual_gb"] * 1.3  # 30% headroom
                recs.append(f"  • {p['alias']}: Increase memory_gb from {p['declared_gb']:.1f} to ~{rec_gb:.1f} GB")

    if recs:
        print(f"\n{BOLD}RECOMMENDATIONS{RESET}")
        for r in recs:
            print(r)

    print()


if __name__ == "__main__":
    main()
