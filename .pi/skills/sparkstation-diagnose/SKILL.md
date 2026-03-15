---
name: sparkstation-diagnose
description: One-command health check for Sparkstation. Shows GPU status, model states, container health, recent log errors, memory usage, and thermal status. Use when asked to "diagnose", "check health", "what's wrong", "system status", or "why is X slow/broken".
---

# Sparkstation Diagnose

Comprehensive system health check in a single command.

## Usage

```bash
# Full diagnostic report
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py

# Quick status only (no log analysis)
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py --quick

# JSON output for programmatic use
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py --json

# Check specific component
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py --component gpu
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py --component models
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py --component containers
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py --component logs
python3 .pi/skills/sparkstation-diagnose/scripts/diagnose.py --component gateway
```

## What It Checks

1. **GPU**: Memory usage, temperature, power draw, utilization, thermal warnings
2. **Supervisor**: Health endpoint, running/failed/suspended model counts
3. **Models**: Per-model status, health, idle time, memory allocation vs actual
4. **Containers**: Docker container status, restarts, resource usage
5. **Gateway**: LiteLLM health, registered models, routing status
6. **Logs**: Recent errors/warnings from sparkstation.log (last 100 lines)
7. **Resources**: Memory allocation vs limits, port usage, disk space for models

## Output

Color-coded terminal output with ✅/⚠️/❌ indicators:

```
═══ SPARKSTATION DIAGNOSTICS ═══

GPU STATUS                                    ✅ OK
  Memory:     45.2 / 128.0 GiB (35.3%)
  Temperature: 52°C (threshold: 80°C)
  Power:      180W / 400W

SUPERVISOR                                    ✅ OK
  Status: healthy
  Models: 4 running, 0 suspended, 0 failed

MODELS
  ✅ qwen3-vl-30b    RUNNING  port:8001  mem:55.0GB  idle:42s
  ✅ bge-m3           RUNNING  port:8002  mem:2.5GB   idle:15s
  ✅ clip-vit         RUNNING  port:8003  mem:5.0GB   idle:30s
  ❌ flux-dev         FAILED   port:8004  mem:35.0GB

CONTAINERS
  ✅ sparkstation-qwen3-vl-30b   Up 2 hours   restarts:0
  ✅ sparkstation-bge-m3          Up 2 hours   restarts:0
  ⚠️ sparkstation-flux-dev       Exited (1)   restarts:3

RECENT ERRORS (last 15min)
  [14:32:01] ERROR: Failed to health-check flux-dev: Connection refused
  [14:31:45] ERROR: Container sparkstation-flux-dev exited unexpectedly
```
