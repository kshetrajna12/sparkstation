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


# Supervisor's allocation-estimate view of memory. Distinct name from
# `unified_memory_used_bytes` (which the node_exporter recording rule now
# owns — that's actual /proc/meminfo pressure). Two different numbers with
# distinct meanings: the supervisor sums each model's declared memory_gb
# budget, node_exporter reports what the kernel is actually holding.
sparkstation_allocated_memory_bytes = Gauge(
    "sparkstation_allocated_memory_bytes",
    "Sum of running-model memory budgets (allocation estimate, not actual pressure)",
    ["host"],
)

sparkstation_memory_budget_bytes = Gauge(
    "sparkstation_memory_budget_bytes",
    "Configured hard memory budget the supervisor enforces on the primary Spark",
    ["host"],
)

# Per-model memory. host label makes "which Spark is this container on?" queryable.
model_memory_used_bytes = Gauge(
    "model_memory_used_bytes", "Estimated memory usage per model in bytes",
    ["host", "model_name"],
)

# GPU temp/power gauges were removed (2026-07-02): nvidia_gpu_exporter runs on
# every host and the Prometheus recording rules own those metric names now.
# The supervisor emitting host="primary" copies double-counted that host.

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
