"""
Sparkstation gateway proxy — engine-agnostic request metrics + model lifecycle
smoothing, in front of LiteLLM.

Why this exists:
- Request rate / latency / TTFT are measured HERE, at the OpenAI-API boundary,
  so the Grafana request panels work identically whether a model is served by
  vLLM, SGLang, TensorRT-LLM, or anything else behind LiteLLM.
- Auto-resume: requests to a SUSPENDED model trigger a supervisor resume and
  wait, instead of failing (the old auto_resume_middleware was never wired —
  this is its production home).
- Swap smoothing: requests to a model that is STARTING (mid-swap/restart) get
  a clean 503 + Retry-After instead of a connection error, and the public
  port :8000 stays up across LiteLLM bounces because the proxy owns it.

Topology:  client -> proxy :8000 -> LiteLLM :7999 -> engine backends

Run:  uvicorn gateway.proxy:app --host 127.0.0.1 --port 8000
(started/stopped by `sparkstation start/stop`; LiteLLM moves to :7999)
"""
import asyncio
import json
import logging
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

logger = logging.getLogger("gateway.proxy")

# 7999: below the model port range (8001-8100) — see cli.GATEWAY_INTERNAL_PORT.
UPSTREAM_URL = os.environ.get("SPARKSTATION_LITELLM_URL", "http://127.0.0.1:7999")
SUPERVISOR_URL = os.environ.get("SPARKSTATION_SUPERVISOR_URL", "http://127.0.0.1:9001")
try:
    # Read the supervisor's settings (.env-aware) so auto-resume calls carry
    # the API key when one is configured. Falls back to plain env.
    from supervisor.config import settings as _supervisor_settings
    SUPERVISOR_API_KEY = _supervisor_settings.api_key
except Exception:
    SUPERVISOR_API_KEY = os.environ.get("API_KEY")

# ─── Metrics (engine-agnostic: measured at the OpenAI API boundary) ─────────

REQUESTS_TOTAL = Counter(
    "sparkstation_gateway_requests_total",
    "Requests through the gateway",
    ["alias", "method", "code"],
)

REQUEST_DURATION = Histogram(
    "sparkstation_gateway_request_duration_seconds",
    "End-to-end request duration (incl. streaming) per model alias",
    ["alias"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300],
)

TTFT = Histogram(
    "sparkstation_gateway_ttft_seconds",
    "Time to first response byte per model alias",
    ["alias"],
    buckets=[0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

RESUMES_TOTAL = Counter(
    "sparkstation_gateway_autoresume_total",
    "Auto-resume attempts triggered by incoming requests",
    ["alias", "outcome"],
)

# Hop-by-hop headers must not be forwarded (RFC 7230 §6.1).
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

app = FastAPI(title="Sparkstation Gateway Proxy", docs_url=None, redoc_url=None)

client: Optional[httpx.AsyncClient] = None
supervisor_client: Optional[httpx.AsyncClient] = None

# Tiny TTL cache of supervisor model state so per-request checks don't hammer it.
_models_cache: dict = {"ts": 0.0, "models": []}
_MODELS_CACHE_TTL = 2.0


@app.on_event("startup")
async def _startup():
    global client, supervisor_client
    # No read timeout: long generations stream for minutes.
    client = httpx.AsyncClient(
        base_url=UPSTREAM_URL,
        timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )
    supervisor_client = httpx.AsyncClient(base_url=SUPERVISOR_URL, timeout=10)
    logger.info(f"Gateway proxy up: upstream={UPSTREAM_URL} supervisor={SUPERVISOR_URL}")


@app.on_event("shutdown")
async def _shutdown():
    if client:
        await client.aclose()
    if supervisor_client:
        await supervisor_client.aclose()


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/proxy/health")
async def proxy_health():
    """Liveness of the proxy itself (upstream health is /health via passthrough)."""
    return {"status": "ok"}


async def _get_models() -> list:
    now = time.monotonic()
    if now - _models_cache["ts"] > _MODELS_CACHE_TTL:
        try:
            r = await supervisor_client.get("/models/detailed")
            _models_cache["models"] = r.json().get("models", [])
            _models_cache["ts"] = now
        except Exception as e:
            logger.warning(f"Supervisor model lookup failed: {e}")
    return _models_cache["models"]


def _find_model(models: list, alias: str) -> Optional[dict]:
    for m in models:
        if m.get("alias") == alias or m.get("model_name") == alias:
            return m
        if alias == "default" and m.get("is_default"):
            return m
    return None


async def _ensure_available(alias: str) -> Optional[Response]:
    """Auto-resume suspended models; 503+Retry-After for models mid-start.

    Returns a Response to short-circuit with, or None to proceed upstream.
    """
    models = await _get_models()
    model = _find_model(models, alias)
    if model is None:
        return None  # unknown alias — let LiteLLM produce its own 4xx

    status = model.get("status")
    if status == "running":
        return None

    if status == "starting":
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "15"},
            content={
                "error": {
                    "message": f"Model '{alias}' is starting (swap/restart in progress). Retry shortly.",
                    "type": "model_starting",
                }
            },
        )

    if status == "suspended":
        logger.info(f"Auto-resuming suspended model '{alias}'")
        headers = {"X-API-Key": SUPERVISOR_API_KEY} if SUPERVISOR_API_KEY else {}
        try:
            r = await supervisor_client.post(f"/models/{model['id']}/resume", headers=headers)
            if r.status_code != 200:
                RESUMES_TOTAL.labels(alias=alias, outcome="error").inc()
                return JSONResponse(
                    status_code=503,
                    headers={"Retry-After": "30"},
                    content={"error": {"message": f"Resume of '{alias}' failed: {r.text[:200]}", "type": "model_suspended"}},
                )
            # Wait for the model to come back (resume returns while STARTING;
            # the supervisor's startup monitor promotes it once /health passes).
            for _ in range(60):
                await asyncio.sleep(2)
                _models_cache["ts"] = 0  # force refresh
                m = _find_model(await _get_models(), alias)
                if m and m.get("status") == "running":
                    RESUMES_TOTAL.labels(alias=alias, outcome="ok").inc()
                    return None
            RESUMES_TOTAL.labels(alias=alias, outcome="timeout").inc()
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "30"},
                content={"error": {"message": f"Model '{alias}' did not resume within 120s", "type": "model_suspended"}},
            )
        except Exception as e:
            RESUMES_TOTAL.labels(alias=alias, outcome="error").inc()
            logger.error(f"Auto-resume of '{alias}' failed: {e}")
            return None  # fall through; upstream will error visibly

    # stopped / failed — surface a clear error instead of an upstream 404
    return JSONResponse(
        status_code=503,
        content={"error": {"message": f"Model '{alias}' is {status}", "type": f"model_{status}"}},
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def forward(request: Request, path: str):
    body = await request.body()

    # Extract the model alias from JSON bodies on /v1/* (chat, embeddings, ...)
    alias = "none"
    if request.method == "POST" and path.startswith("v1/") and body:
        try:
            alias = json.loads(body).get("model") or "none"
        except (json.JSONDecodeError, AttributeError):
            pass

    if alias != "none":
        short_circuit = await _ensure_available(alias)
        if short_circuit is not None:
            REQUESTS_TOTAL.labels(alias=alias, method=request.method, code=str(short_circuit.status_code)).inc()
            return short_circuit

    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    upstream_request = client.build_request(
        request.method,
        f"/{path}",
        params=request.url.query,
        headers=headers,
        content=body,
    )

    start = time.perf_counter()
    try:
        upstream = await client.send(upstream_request, stream=True)
    except Exception as e:
        REQUESTS_TOTAL.labels(alias=alias, method=request.method, code="502").inc()
        return JSONResponse(status_code=502, content={"error": {"message": f"Upstream gateway error: {e}", "type": "bad_gateway"}})

    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
    measured = alias != "none"

    async def stream():
        first = True
        try:
            async for chunk in upstream.aiter_raw():
                if first:
                    first = False
                    if measured:
                        TTFT.labels(alias=alias).observe(time.perf_counter() - start)
                yield chunk
        finally:
            await upstream.aclose()
            if measured:
                REQUEST_DURATION.labels(alias=alias).observe(time.perf_counter() - start)
            REQUESTS_TOTAL.labels(alias=alias, method=request.method, code=str(upstream.status_code)).inc()

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )
