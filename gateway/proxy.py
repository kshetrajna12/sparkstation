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
    Gauge,
    Histogram,
    generate_latest,
)

from gateway.clients import (
    DENY_UNKNOWN_KEY,
    Registry,
    extract_key,
)
from gateway.reasoning import DialectResolver
from gateway.reasoning import apply_client_default as apply_client_reasoning
from gateway.reasoning import normalize as normalize_reasoning

logger = logging.getLogger("gateway.proxy")

# 7999: below the model port range (8001-8100) — see cli.GATEWAY_INTERNAL_PORT.
UPSTREAM_URL = os.environ.get("SPARKSTATION_LITELLM_URL", "http://127.0.0.1:7999")
SUPERVISOR_URL = os.environ.get("SPARKSTATION_SUPERVISOR_URL", "http://127.0.0.1:9001")

# Blue-green: the litellm-bluegreen.sh manager runs litellm on one of two ports
# and writes the ACTIVE port into this pointer file, flipping it atomically once
# a freshly-reloaded litellm is healthy. The proxy watches the pointer and swaps
# its upstream client live — so config reloads never drop the public :8000 port.
LITELLM_PORT_FILE = os.environ.get("SPARKSTATION_LITELLM_PORT_FILE", "gateway/.litellm-port")
# In-flight streams hold a reference to the old client; keep it alive this long
# after a flip so long generations finish before the old connections close.
_UPSTREAM_DRAIN_GRACE = float(os.environ.get("SPARKSTATION_UPSTREAM_DRAIN_GRACE", "300"))

# Per-client access control (keys + model allow-lists + rate/concurrency limits
# + attribution). Hot-reloaded from this YAML; see gateway/clients.py.
CLIENTS_FILE = os.environ.get("SPARKSTATION_CLIENTS_FILE", "gateway/clients.yaml")
clients = Registry(CLIENTS_FILE)

# Reasoning-control normalization: translate whatever thinking knobs a client
# sends into the dialect the currently-loaded backend actually honors, so client
# configs survive model swaps (see gateway/reasoning.py + reasoning.yaml).
reasoning = DialectResolver()


def _default_port() -> int:
    try:
        return int(UPSTREAM_URL.rsplit(":", 1)[1])
    except Exception:
        return 7999


def _read_active_port() -> int:
    try:
        return int(open(LITELLM_PORT_FILE).read().strip())
    except Exception:
        return _default_port()


def _new_upstream_client(port: int) -> "httpx.AsyncClient":
    return httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )
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

# Per-client attribution + policy enforcement (kept separate from the alias-only
# metrics above so existing dashboards/recording rules are untouched).
CLIENT_REQUESTS = Counter(
    "sparkstation_gateway_client_requests_total",
    "Requests through the gateway, attributed to the resolved client",
    ["client", "alias", "code"],
)

CLIENT_DENIED = Counter(
    "sparkstation_gateway_client_denied_total",
    "Requests rejected by per-client policy (auth / allow-list / limits)",
    ["client", "alias", "reason"],
)

CLIENT_INFLIGHT = Gauge(
    "sparkstation_gateway_client_inflight",
    "In-flight requests currently attributed to each client",
    ["client"],
)

# Hop-by-hop headers must not be forwarded (RFC 7230 §6.1).
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

app = FastAPI(title="Sparkstation Gateway Proxy", docs_url=None, redoc_url=None)

client: Optional[httpx.AsyncClient] = None
supervisor_client: Optional[httpx.AsyncClient] = None
_current_port: Optional[int] = None
_upstream_task: Optional["asyncio.Task"] = None

# Tiny TTL cache of supervisor model state so per-request checks don't hammer it.
_models_cache: dict = {"ts": 0.0, "models": []}
_MODELS_CACHE_TTL = 2.0


async def _close_later(c: "httpx.AsyncClient", delay: float):
    """Close a retired upstream client after a grace so in-flight streams drain."""
    try:
        await asyncio.sleep(delay)
        await c.aclose()
    except Exception:
        pass


async def _upstream_watcher():
    """Follow the blue-green pointer file; swap the upstream client on a flip.

    New requests use the freshly-assigned global `client`; requests already
    streaming hold their own connection on the old client, which is closed only
    after _UPSTREAM_DRAIN_GRACE. Result: config reloads never break the API.
    """
    global client, _current_port
    while True:
        try:
            port = _read_active_port()
            if port != _current_port:
                new = _new_upstream_client(port)
                old = client
                client = new
                _current_port = port
                logger.info(f"Gateway proxy upstream flipped -> 127.0.0.1:{port}")
                if old is not None:
                    asyncio.create_task(_close_later(old, _UPSTREAM_DRAIN_GRACE))
        except Exception as e:
            logger.warning(f"upstream watcher error: {e}")
        clients.maybe_reload()    # pick up clients.yaml edits without a restart
        reasoning.maybe_reload()  # pick up reasoning.yaml / litellm.yaml changes
        await asyncio.sleep(1)


@app.on_event("startup")
async def _startup():
    global client, supervisor_client, _current_port, _upstream_task
    _current_port = _read_active_port()
    # No read timeout: long generations stream for minutes.
    client = _new_upstream_client(_current_port)
    supervisor_client = httpx.AsyncClient(base_url=SUPERVISOR_URL, timeout=10)
    _upstream_task = asyncio.create_task(_upstream_watcher())
    logger.info(f"Gateway proxy up: upstream=127.0.0.1:{_current_port} (blue-green) supervisor={SUPERVISOR_URL}")


@app.on_event("shutdown")
async def _shutdown():
    if _upstream_task:
        _upstream_task.cancel()
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

    # Resolve the client up front — its reasoning default is applied before
    # dialect normalization below.
    policy = clients.resolve(extract_key(request.headers))

    # Extract the model alias from JSON bodies on /v1/* (chat, embeddings, ...)
    # and, for chat, apply the client's reasoning default then normalize the
    # thinking controls to the backend's dialect.
    alias = "none"
    if request.method == "POST" and path.startswith("v1/") and body:
        try:
            parsed = json.loads(body)
            alias = parsed.get("model") or "none"
            if alias != "none" and path.endswith("chat/completions") and isinstance(parsed, dict):
                dialect = reasoning.dialect_for(alias)
                if dialect != "passthrough":
                    if policy is not None:
                        apply_client_reasoning(parsed, policy.reasoning)
                    normalize_reasoning(parsed, dialect)
                    body = json.dumps(parsed).encode()
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Per-client access control (auth + allow-list + limits + attribution) ──
    if policy is None:  # enforce_auth on + unknown key
        CLIENT_DENIED.labels(client="unknown", alias=alias, reason=DENY_UNKNOWN_KEY).inc()
        REQUESTS_TOTAL.labels(alias=alias, method=request.method, code="401").inc()
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid or missing API key", "type": "invalid_api_key"}},
        )
    if alias != "none" and not policy.allows_model(alias):
        CLIENT_DENIED.labels(client=policy.name, alias=alias, reason="model_not_allowed").inc()
        REQUESTS_TOTAL.labels(alias=alias, method=request.method, code="403").inc()
        return JSONResponse(
            status_code=403,
            content={"error": {"message": f"Client '{policy.name}' is not permitted to use model '{alias}'", "type": "model_not_allowed"}},
        )

    if alias != "none":
        short_circuit = await _ensure_available(alias)
        if short_circuit is not None:
            CLIENT_REQUESTS.labels(client=policy.name, alias=alias, code=str(short_circuit.status_code)).inc()
            REQUESTS_TOTAL.labels(alias=alias, method=request.method, code=str(short_circuit.status_code)).inc()
            return short_circuit

    # Rate / concurrency limits (counters committed here; released in stream()).
    admitted, reason, retry_after = policy.admit(alias, time.monotonic())
    if not admitted:
        CLIENT_DENIED.labels(client=policy.name, alias=alias, reason=reason).inc()
        REQUESTS_TOTAL.labels(alias=alias, method=request.method, code="429").inc()
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after or 1)},
            content={"error": {"message": f"Client '{policy.name}' {reason} (retry after {retry_after or 1}s)", "type": reason}},
        )
    CLIENT_INFLIGHT.labels(client=policy.name).set(policy.inflight)

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
        policy.release()
        CLIENT_INFLIGHT.labels(client=policy.name).set(policy.inflight)
        CLIENT_REQUESTS.labels(client=policy.name, alias=alias, code="502").inc()
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
            policy.release()
            CLIENT_INFLIGHT.labels(client=policy.name).set(policy.inflight)
            if measured:
                REQUEST_DURATION.labels(alias=alias).observe(time.perf_counter() - start)
            CLIENT_REQUESTS.labels(client=policy.name, alias=alias, code=str(upstream.status_code)).inc()
            REQUESTS_TOTAL.labels(alias=alias, method=request.method, code=str(upstream.status_code)).inc()

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )
