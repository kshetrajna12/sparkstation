"""
Prometheus metrics for Sparkstation Supervisor.

Every model-scoped metric carries a `host` label so multi-host cluster
deployments can be broken down per Spark in Grafana. For single-host
deployments this is invisible (label is always "primary"). System-level
metrics (unified memory, GPU temp/power) carry a `host` label too — but
today the supervisor only scrapes its own local values, so those come
out as "primary". Adding a worker-side scraper is a future step; the
label is there so the dashboards don't need to change again when it lands.
"""
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


# Memory metrics (per-host — today only primary is populated). Adding a
# per-worker sample requires a scraper that exports nvidia-smi + /proc/meminfo
# on the worker; see docs/cluster.md (TODO) for the pattern.
unified_memory_used_bytes = Gauge(
    "unified_memory_used_bytes", "Total unified memory usage in bytes", ["host"]
)

unified_memory_limit_bytes = Gauge(
    "unified_memory_limit_bytes", "Unified memory hard limit in bytes", ["host"]
)

# Per-model memory. host label makes "which Spark is this container on?" queryable.
model_memory_used_bytes = Gauge(
    "model_memory_used_bytes", "Estimated memory usage per model in bytes",
    ["host", "model_name"],
)

# GPU metrics (per-host — same story as unified memory).
gpu_temperature_celsius = Gauge(
    "gpu_temperature_celsius", "GPU temperature in Celsius", ["host"]
)

gpu_power_draw_watts = Gauge(
    "gpu_power_draw_watts", "GPU power draw in Watts", ["host"]
)

# Model status: 0=stopped, 1=starting, 2=running, 3=suspended, 4=failed.
model_status = Gauge(
    "model_status",
    "Model status (0=stopped, 1=starting, 2=running, 3=suspended, 4=failed)",
    ["host", "model_name", "model_id"],
)

model_last_request_timestamp = Gauge(
    "model_last_request_timestamp",
    "Unix timestamp of last request served by model",
    ["host", "model_name"],
)

# Request metrics — declared for future wiring; not populated today.
# host label preserved for consistency so downstream queries always
# match the same label set.
model_requests_total = Counter(
    "model_requests_total", "Total requests served by model",
    ["host", "model_name"],
)

model_requests_failed = Counter(
    "model_requests_failed", "Failed requests by model",
    ["host", "model_name"],
)

model_request_latency_seconds = Histogram(
    "model_request_latency_seconds", "Request latency in seconds",
    ["host", "model_name"],
)

# System counters — cluster-wide, no host label needed.
resident_models_count = Gauge("resident_models_count", "Number of currently resident (running) models")

suspended_models_count = Gauge("suspended_models_count", "Number of suspended models")


def metrics_response() -> Response:
    """Generate Prometheus metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
