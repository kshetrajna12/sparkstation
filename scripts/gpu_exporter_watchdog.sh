#!/usr/bin/env bash
# Watchdog for nvidia-gpu-exporter: the NVIDIA container toolkit can silently
# revoke a running container's GPU device access after host cgroup events
# (driver update, systemctl daemon-reload, ...). The container stays "up" and
# serves /metrics, but every nvidia-smi call inside fails with
# "Failed to initialize NVML" — so the Prometheus target looks healthy while
# all nvidia_smi_* series vanish. A container restart fixes it.
#
# Deployed on every Spark via user crontab (*/2 min). No sudo needed: the
# user is in the docker group.
set -u

METRICS_URL="http://localhost:9835/metrics"
CONTAINER="nvidia-gpu-exporter"

m=$(curl -sm 5 "$METRICS_URL" || true)

healthy=true
if [ -z "$m" ]; then
  healthy=false
  reason="exporter unresponsive"
elif echo "$m" | grep -Eq '^nvidia_smi_command_exit_code [1-9]'; then
  healthy=false
  reason="nvidia-smi failing inside container (NVML lost)"
elif ! echo "$m" | grep -q '^nvidia_smi_utilization_gpu_ratio'; then
  healthy=false
  reason="no GPU series in scrape"
fi

if ! $healthy; then
  logger -t gpu-exporter-watchdog "$reason — restarting $CONTAINER"
  docker restart "$CONTAINER" >/dev/null 2>&1
fi
