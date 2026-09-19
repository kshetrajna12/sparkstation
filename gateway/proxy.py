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
from gateway.sse_normalize import normalize_sse

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
# Backend-direct client for routes LiteLLM does not proxy (/v1/systemone).
direct_client: Optional[httpx.AsyncClient] = None
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
    global client, supervisor_client, direct_client, _current_port, _upstream_task
    _current_port = _read_active_port()
    # No read timeout: long generations stream for minutes.
    client = _new_upstream_client(_current_port)
    supervisor_client = httpx.AsyncClient(base_url=SUPERVISOR_URL, timeout=10)
    direct_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
    )
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
    if direct_client:
        await direct_client.aclose()


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


# ─── Decision models (reflex): POST /v1/systemone bypasses LiteLLM ──────────
#
# A "decision" model (models.yaml model_type: decision, backend: reflex) is a
# Jev-style System One server: typed questions in, calibrated probabilities
# out, over its own POST /v1/systemone route. LiteLLM has no notion of that
# endpoint, so gateway_sync keeps decision models OUT of litellm.yaml and the
# proxy forwards the path straight to the model's container instead. Every
# other gateway concern (client keys, allow-lists, rate/concurrency limits,
# auto-resume, per-alias metrics) applies exactly as for chat.
SYSTEMONE_PATH = "v1/systemone"
DECISION_MODEL_TYPE = "decision"
# reflex's schema default / the name TypeSafe's SDK sends — treated as "the
# loaded decision model", so Jev client code works unchanged with model unset.
_DECISION_WILDCARDS = {"reflex-latest", "decision", "systemone", ""}
_DECISION_STATUS_ORDER = {"running": 0, "starting": 1, "suspended": 2}


def _resolve_decision_target(parsed: object, models: list) -> Optional[tuple[str, str]]:
    """Pick the decision model a /v1/systemone request goes to.

    `parsed` is the JSON body (or None). A body `model` naming a specific
    decision alias/model_name selects it; a missing or wildcard name picks the
    loaded decision model, preferring running > starting > suspended so
    _ensure_available can then resume / 503 it. Returns (alias, base_url) or
    None when nothing matches.
    """
    wanted = parsed.get("model") if isinstance(parsed, dict) else None
    candidates = [m for m in models if m.get("model_type") == DECISION_MODEL_TYPE]
    if isinstance(wanted, str) and wanted not in _DECISION_WILDCARDS:
        m = _find_model(candidates, wanted)
    else:
        m = min(candidates, key=lambda x: _DECISION_STATUS_ORDER.get(x.get("status"), 9), default=None)
    if m is None or not m.get("base_url"):
        return None
    return (m.get("alias") or m.get("model_name"), m["base_url"])


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

    # GET /health is unauthenticated liveness: monitors and the CLI must be
    # able to see "gateway up" without holding a client key (enforce_auth
    # would otherwise 401 the probe and every health check would lie).
    if request.method == "GET" and path == "health":
        try:
            # litellm's /health itself demands auth; /health/liveliness is its
            # unauthenticated liveness route and answers 200 when it's up.
            upstream = await client.get("/health/liveliness")
            return Response(content=upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type"))
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {"message": f"Upstream gateway error: {e}", "type": "bad_gateway"}})

    # Resolve the client up front — its reasoning default is applied before
    # dialect normalization below.
    policy = clients.resolve(extract_key(request.headers))

    # Extract the model alias from JSON bodies on /v1/* (chat, embeddings, ...)
    # and, for chat, apply the client's reasoning default then normalize the
    # thinking controls to the backend's dialect.
    alias = "none"
    parsed = None
    if request.method == "POST" and path.startswith("v1/") and body:
        try:
            parsed = json.loads(body)
            alias = parsed.get("model") or "none"
            if alias != "none" and path.endswith("chat/completions") and isinstance(parsed, dict):
                dialect, efforts, effort_aliases = reasoning.policy_for(alias)
                if dialect != "passthrough":
                    if policy is not None:
                        apply_client_reasoning(parsed, policy.reasoning)
                    normalize_reasoning(parsed, dialect, efforts, effort_aliases)
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

    # /v1/systemone → the loaded decision model, directly (after auth so an
    # unknown key cannot probe which models exist; before the allow-list so
    # the policy is checked against the RESOLVED alias).
    direct_url: Optional[str] = None
    if request.method == "POST" and path == SYSTEMONE_PATH:
        target = _resolve_decision_target(parsed, await _get_models())
        if target is None:
            CLIENT_DENIED.labels(client=policy.name, alias=alias, reason="no_decision_model").inc()
            REQUESTS_TOTAL.labels(alias=alias, method=request.method, code="404").inc()
            return JSONResponse(
                status_code=404,
                content={"error": {
                    "message": "No decision model is loaded (models.yaml model_type: decision, e.g. the "
                               "`reflex` alias) or the requested one is unknown — /v1/systemone has nowhere to go",
                    "type": "model_not_found",
                }},
            )
        alias, direct_url = target[0], f"{target[1]}/{SYSTEMONE_PATH}"
        if isinstance(parsed, dict) and parsed.get("model") != alias:
            parsed["model"] = alias  # the backend echoes its own served name anyway
            body = json.dumps(parsed).encode()

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
    if direct_url:
        # Backend-direct: the gateway key stays at the gateway.
        headers.pop("authorization", None)
        headers.pop("x-api-key", None)
        send_client = direct_client
        upstream_request = send_client.build_request(
            request.method, direct_url, params=request.url.query, headers=headers, content=body,
        )
    else:
        send_client = client
        upstream_request = send_client.build_request(
            request.method,
            f"/{path}",
            params=request.url.query,
            headers=headers,
            content=body,
        )

    start = time.perf_counter()
    try:
        upstream = await send_client.send(upstream_request, stream=True)
    except Exception as e:
        policy.release()
        CLIENT_INFLIGHT.labels(client=policy.name).set(policy.inflight)
        CLIENT_REQUESTS.labels(client=policy.name, alias=alias, code="502").inc()
        REQUESTS_TOTAL.labels(alias=alias, method=request.method, code="502").inc()
        return JSONResponse(status_code=502, content={"error": {"message": f"Upstream gateway error: {e}", "type": "bad_gateway"}})

    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
    measured = alias != "none"

    # Chat SSE streams get their deltas normalized (a delta carrying both
    # reasoning and content is split in two — see gateway/sse_normalize.py);
    # everything else is a byte-for-byte passthrough.
    is_chat_sse = path.endswith("chat/completions") and upstream.headers.get(
        "content-type", ""
    ).lower().startswith("text/event-stream")
    upstream_iter = normalize_sse(upstream.aiter_raw()) if is_chat_sse else upstream.aiter_raw()

    async def stream():
        first = True
        try:
            async for chunk in upstream_iter:
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
