"""
Prometheus metrics for Sparkstation Supervisor (DGX Spark optimized).
"""
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


# Memory metrics (DGX Spark unified memory)
unified_memory_used_bytes = Gauge(
    "unified_memory_used_bytes", "Total unified memory usage in bytes (DGX Spark)"
)

unified_memory_limit_bytes = Gauge(
    "unified_memory_limit_bytes", "Unified memory hard limit in bytes (110 GB)"
)

model_memory_used_bytes = Gauge(
    "model_memory_used_bytes", "Estimated memory usage per model in bytes", ["model_name"]
)

# GPU metrics
gpu_temperature_celsius = Gauge("gpu_temperature_celsius", "GPU temperature in Celsius")

gpu_power_draw_watts = Gauge("gpu_power_draw_watts", "GPU power draw in Watts")

# Model status
model_status = Gauge(
    "model_status",
    "Model status (0=stopped, 1=starting, 2=running, 3=suspended, 4=failed)",
    ["model_name", "model_id"],
)

model_last_request_timestamp = Gauge(
    "model_last_request_timestamp", "Unix timestamp of last request served by model", ["model_name"]
)

# Request metrics
model_requests_total = Counter("model_requests_total", "Total requests served by model", ["model_name"])

model_requests_failed = Counter("model_requests_failed", "Failed requests by model", ["model_name"])

model_request_latency_seconds = Histogram(
    "model_request_latency_seconds", "Request latency in seconds", ["model_name"]
)

# System metrics
resident_models_count = Gauge("resident_models_count", "Number of currently resident (running) models")

suspended_models_count = Gauge("suspended_models_count", "Number of suspended models")


def metrics_response() -> Response:
    """Generate Prometheus metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
