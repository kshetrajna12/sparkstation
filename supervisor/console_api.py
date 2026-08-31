"""
Console support endpoints: profiles, start-by-alias, and log tailing.

These exist for the Console's Cluster & models and Logs sections (console/),
but are plain supervisor API like everything else — the CLI could use them
too. Model lifecycle stays in main.py; this router only adds the pieces the
SPA can't do with the existing endpoints:

- GET  /profiles                      models.yaml profiles + the active one
- POST /models/{alias}/start-by-alias resolve alias through a profile and
                                      start it (the CLI's `models start` flow,
                                      server-side, so the SPA never needs the
                                      full ModelStartRequest config)
- GET  /logs                          available log sources
- GET  /logs/{source}?lines=N         tail one source (supervisor file,
                                      model log file, or `docker logs`)

Log sources are matched against the registry / the model_logs directory
listing — ids are never interpolated into shell strings (they contain spaces
and parens) and never treated as paths beyond a basename equality check.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time

import httpx
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from supervisor.auth import require_api_key
from supervisor.cluster_helpers import merged_env
from supervisor.config import settings
from supervisor.models_config import find_model_by_alias, load_models_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["console"])

MODEL_LOGS_DIR = Path("data/model_logs")
MAX_TAIL_LINES = 5000


def _registry(request: Request):
    from supervisor import main
    if main.registry is None:
        raise HTTPException(503, "Registry not initialized")
    return main.registry


# ── profiles ─────────────────────────────────────────────────────────────────

@router.get("/profiles")
async def list_profiles_endpoint():
    """Profiles from models.yaml, which aliases each enables, and the active one.

    `active` is the profile this supervisor was started with (STARTUP_PROFILE),
    falling back to models.yaml default_profile — the same resolution the CLI
    and autoload use.
    """
    cfg = load_models_config()
    active = settings.startup_profile or os.environ.get("STARTUP_PROFILE") or cfg.default_profile
    profiles = {name: sorted(overrides.keys()) for name, overrides in cfg.profiles.items()}
    return {
        "active": active,
        "default": cfg.default_profile,
        "profiles": profiles,
        "all_aliases": sorted(cfg.models.keys()),
        # base-spec placement facts so pickers can say where a model would run
        "aliases": {a: {"host": m.host, "memory_gb": m.memory_gb} for a, m in cfg.models.items()},
    }


# ── start by alias ───────────────────────────────────────────────────────────

@router.post("/models/{alias}/start-by-alias", dependencies=[Depends(require_api_key)])
async def start_by_alias(alias: str, profile: Optional[str] = Query(None)):
    """Start a model by its models.yaml alias, resolved through a profile.

    Mirrors the CLI's `sparkstation models start <alias> -p <profile>`:
    find_model_by_alias() deep-merges base spec + profile overrides, then the
    result feeds the same start path as POST /models/start. Returns that
    endpoint's response (the model comes up in the background; poll
    /models/detailed)."""
    from supervisor.main import start_model
    from supervisor.models import Backend, ModelStartRequest, ModelType

    resolved = find_model_by_alias(alias, profile_name=profile or None)
    if resolved is None:
        raise HTTPException(404, f"no model alias {alias!r}" + (f" in profile {profile!r}" if profile else ""))
    cfg = resolved.model_dump()
    req = ModelStartRequest(
        model_name=cfg["name"],
        backend=Backend(cfg["backend"]),
        model_type=ModelType(cfg.get("model_type", "chat")),
        model_alias=cfg.get("alias") or alias,
        host=cfg.get("host", "primary"),
        quantization=cfg.get("quantization") or "none",
        memory_gb=cfg.get("memory_gb"),
        idle_timeout_minutes=cfg.get("idle_timeout_minutes", 30),
        auto_suspend_enabled=cfg.get("auto_suspend_enabled", False),
        speculative_model=cfg.get("speculative_model"),
        speculative_method=cfg.get("speculative_method"),
        num_speculative_tokens=cfg.get("num_speculative_tokens", 5),
        speculative_extra=cfg.get("speculative_extra") or {},
        extra_args=cfg.get("extra_args") or {},
        docker_image=cfg.get("docker_image"),
        env_vars=cfg.get("env_vars") or {},
        volumes=cfg.get("volumes") or [],
    )
    return await start_model(req)


# ── per-host resources ───────────────────────────────────────────────────────
# /resources only reads the local machine (nvidia-smi + /proc/meminfo on the
# supervisor host). The console's Cluster section wants the workers too, so
# this probes every cluster role over the same SSH the launchers use. Probes
# run in parallel and the result is cached briefly — the SPA polls every 10 s.

_HOST_PROBE_CMD = (
    "cat /proc/meminfo; "
    "nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null"
)
_hosts_cache: dict = {"at": 0.0, "data": None}
_HOSTS_CACHE_S = 8.0


def _parse_host_probe(text: str) -> dict:
    mem = {}
    gpu_temp = gpu_power = None
    for line in text.splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            key = line.split(":")[0]
            try:
                mem[key] = int(line.split()[1]) / (1024.0 ** 2)  # kB → GB
            except (IndexError, ValueError):
                pass
        elif "," in line and "kB" not in line:
            parts = [p.strip() for p in line.split(",")]
            try:
                gpu_temp, gpu_power = float(parts[0]), float(parts[1])
            except (IndexError, ValueError):
                pass
    total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
    return {
        "ok": total is not None,
        "mem_total_gb": total,
        "mem_available_gb": avail,
        "mem_used_gb": (total - avail) if total is not None and avail is not None else None,
        "gpu_temp_c": gpu_temp,
        "gpu_power_w": gpu_power,
    }


def _probe_host(role: str) -> dict:
    from supervisor.models_config import get_cluster_config
    cluster = get_cluster_config()
    entry = cluster.hosts.get(role)
    local = entry is None or entry.ip is None or entry.ip in ("127.0.0.1", "localhost", "::1")
    if not local and not entry.ssh_user:
        return {"ok": False, "error": "no ssh_user configured"}
    argv = ["bash", "-c", _HOST_PROBE_CMD] if local else         ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", f"{entry.ssh_user}@{entry.ip}", _HOST_PROBE_CMD]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=12)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": str(e)[:120]}
    if r.returncode != 0 and "MemTotal" not in r.stdout:
        return {"ok": False, "error": r.stderr.strip()[:120] or f"exit {r.returncode}"}
    out = _parse_host_probe(r.stdout)
    if entry is not None and entry.label:
        out["label"] = entry.label
    return out


@router.get("/resources/hosts")
async def resources_per_host():
    """Memory + GPU snapshot for every cluster role (briefly cached)."""
    now = time.monotonic()
    if _hosts_cache["data"] is not None and now - _hosts_cache["at"] < _HOSTS_CACHE_S:
        return _hosts_cache["data"]
    from supervisor.models_config import get_cluster_config
    roles = list(get_cluster_config().hosts.keys())
    results = await asyncio.gather(*(asyncio.to_thread(_probe_host, r) for r in roles))
    data = {"hosts": dict(zip(roles, results))}
    _hosts_cache.update(at=now, data=data)
    return data


# ── profile switching ────────────────────────────────────────────────────────
# The CLI's full switch is `sparkstation stop && start -d --profile X` (kills
# the supervisor too). This is the lighter, supervisor-resident version: stop
# the live models the target profile doesn't include, start the ones it lacks,
# then repoint everything that carries "the active profile" at runtime —
# settings.startup_profile, main.default_model_alias / vision_model_alias,
# gateway_sync's copies — and the CLI's sticky file (data/last_profile), so a
# later plain `sparkstation start` resumes the same profile. One switch at a
# time; progress is polled via GET /profiles/switch-status.

LAST_PROFILE_FILE = Path("data/last_profile")
LIVE_STATUSES = ("running", "starting", "suspended")

_switch_state: dict = {"state": "idle"}
_switch_task: Optional[asyncio.Task] = None


async def _profile_plan(profile: str) -> dict:
    from supervisor import main
    from supervisor.models_config import get_profile_models

    target = {m.alias: m for m in get_profile_models(profile)}
    live = {}
    for m in await main.registry.list_all():
        if str(m.status) in LIVE_STATUSES:
            live[m.model_alias or m.model_name] = m
    return {
        "profile": profile,
        "stop": sorted(a for a in live if a not in target),
        "start": sorted(a for a in target if a not in live),
        "keep": sorted(a for a in live if a in target),
    }


@router.get("/profiles/{profile}/plan")
async def profile_switch_plan(profile: str, request: Request):
    _registry(request)
    cfg = load_models_config()
    if profile not in cfg.profiles:
        raise HTTPException(404, f"no profile {profile!r}")
    return await _profile_plan(profile)


async def _run_switch(profile: str) -> None:
    from supervisor import main
    from supervisor.models_config import get_default_model_alias, get_vision_model_alias

    steps = []
    _switch_state.update(state="switching", profile=profile, steps=steps, error=None,
                         started_at=time.strftime("%H:%M:%S"))

    async def step(action, alias, coro):
        entry = {"action": action, "alias": alias, "status": "running"}
        steps.append(entry)
        try:
            await coro
            entry["status"] = "done"
        except Exception as e:
            detail = getattr(e, "detail", None) or str(e)
            entry["status"] = "error"
            entry["error"] = str(detail)[:200]
            logger.error(f"profile switch {profile}: {action} {alias} failed: {detail}")
            return False
        return True

    try:
        plan = await _profile_plan(profile)
        ok = True
        # stops first: free the memory the incoming models need
        for alias in plan["stop"]:
            m = next((x for x in await main.registry.list_all()
                      if (x.model_alias or x.model_name) == alias and str(x.status) in LIVE_STATUSES), None)
            if m is not None:
                ok = await step("stop", alias, main.stop_model(m.id)) and ok
        for alias in plan["start"]:
            ok = await step("start", alias, start_by_alias(alias, profile=profile)) and ok

        if not ok:
            _switch_state.update(state="error", error="one or more steps failed (see steps)")
            return

        # repoint runtime + sticky profile state
        from supervisor.config import settings as live_settings
        live_settings.startup_profile = profile
        main.default_model_alias = get_default_model_alias(profile)
        main.vision_model_alias = get_vision_model_alias(profile)
        if main.gateway_sync is not None:
            main.gateway_sync.default_model_alias = main.default_model_alias
            main.gateway_sync.vision_model_alias = main.vision_model_alias
            try:
                await main.gateway_sync.sync_models()  # don't wait for the 60 s pass
            except Exception as e:
                logger.warning(f"post-switch gateway sync failed (periodic sync will retry): {e}")
        try:
            LAST_PROFILE_FILE.write_text(profile)
        except OSError as e:
            logger.warning(f"could not persist sticky profile: {e}")
        _switch_state.update(state="done")
        logger.info(f"profile switch complete: {profile} "
                    f"(default={main.default_model_alias}, vision={main.vision_model_alias})")
    except Exception as e:
        logger.exception(f"profile switch to {profile} crashed")
        _switch_state.update(state="error", error=str(e)[:300])


@router.post("/profiles/{profile}/activate", dependencies=[Depends(require_api_key)])
async def activate_profile(profile: str, request: Request):
    """Switch the running system to `profile` (background; poll /profiles/switch-status).

    Stops models outside the profile, starts the missing ones (through the
    normal start path, so per-host memory gates still apply), updates the
    default/vision aliases + gateway, and persists the sticky profile."""
    global _switch_task
    _registry(request)
    cfg = load_models_config()
    if profile not in cfg.profiles:
        raise HTTPException(404, f"no profile {profile!r}")
    if _switch_task is not None and not _switch_task.done():
        raise HTTPException(409, f"a profile switch is already running ({_switch_state.get('profile')})")
    plan = await _profile_plan(profile)
    _switch_task = asyncio.create_task(_run_switch(profile))
    return {"ok": True, "profile": profile, "plan": plan, "status_url": "/profiles/switch-status"}


@router.get("/profiles/switch-status")
async def switch_status():
    return _switch_state


# ── playground (gateway proxy) ───────────────────────────────────────────────
# The gateway binds loopback on primary, so a browser on console.<domain>
# can't reach it directly; these two endpoints proxy it through the
# supervisor. Chat is a byte-for-byte passthrough (SSE streaming included) —
# the SPA speaks plain OpenAI chat-completions shapes.

_gw = httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=600.0))


def _gateway_base() -> str:
    return settings.litellm_admin_url.rstrip("/")


@router.get("/playground/models")
async def playground_models():
    """Chat-capable model ids the gateway currently serves."""
    try:
        r = await _gw.get(f"{_gateway_base()}/v1/models", headers={"Authorization": "Bearer console"})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"gateway unreachable: {e}")
    ids = [m.get("id") for m in r.json().get("data", [])]
    # embeddings/detection aliases aren't chattable; the registry knows types
    from supervisor import main
    non_chat = set()
    if main.registry is not None:
        for m in await main.registry.list_all():
            if str(getattr(m, "model_type", "chat")) != "chat":
                non_chat.add(m.model_alias or m.model_name)
    chat_ids = [i for i in ids if i and i not in non_chat]
    # stable, default-first ordering
    chat_ids.sort(key=lambda i: (i not in ("default", "vision"), i))
    return {"models": chat_ids}


@router.post("/playground/chat")
async def playground_chat(request: Request):
    """Proxy a chat-completions request to the gateway, streaming the reply."""
    body = await request.body()
    if len(body) > 30 * 1024 * 1024:
        raise HTTPException(413, "request too large")
    upstream = _gw.build_request(
        "POST", f"{_gateway_base()}/v1/chat/completions",
        content=body,
        headers={"Authorization": "Bearer console", "Content-Type": "application/json"},
    )
    try:
        resp = await _gw.send(upstream, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"gateway unreachable: {e}")

    async def body_iter():
        try:
            async for chunk in resp.aiter_bytes(8192):
                yield chunk
        finally:
            await resp.aclose()

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        body_iter(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
        headers={"Cache-Control": "no-store"},
    )


# ── logs ─────────────────────────────────────────────────────────────────────

@router.get("/logs")
async def list_log_sources(request: Request):
    """Log sources the console can tail.

    - `supervisor`: the rotating supervisor log file
    - one per registry model: its model_logs file (launcher output) when one
      exists, and `container` when the model has a container to `docker logs`.
    """
    registry = _registry(request)
    files = {p.name for p in MODEL_LOGS_DIR.glob("*.log")} if MODEL_LOGS_DIR.is_dir() else set()
    sources = [{"id": "supervisor", "label": "supervisor", "kinds": ["file"]}]
    for m in await registry.list_all():
        kinds = []
        if f"{m.id}.log" in files:
            kinds.append("file")
        if m.container_id:
            kinds.append("container")
        if kinds:
            # ModelInstance uses use_enum_values, so status is already a str
            sources.append({"id": m.id, "label": m.model_alias or m.model_name,
                            "status": str(m.status), "host": m.host, "kinds": kinds})
    return {"sources": sources}


def _tail_file(path: Path, lines: int) -> str:
    """Last `lines` lines of a file without reading the whole thing."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, max(lines * 250, 16384))
            f.seek(size - chunk)
            data = f.read()
    except OSError as e:
        raise HTTPException(502, f"cannot read log: {e}")
    text = data.decode("utf-8", errors="replace")
    tail = text.splitlines()[-lines:]
    if chunk < size and tail:
        tail = tail[1:]  # first line is probably cut mid-way
    return "\n".join(tail) + "\n"


@router.get("/logs/{source_id}", response_class=PlainTextResponse)
async def tail_log(
    request: Request,
    source_id: str,
    lines: int = Query(200, ge=10, le=MAX_TAIL_LINES),
    kind: str = Query("auto", pattern="^(auto|file|container)$"),
):
    """Tail one log source as plain text."""
    if source_id == "supervisor":
        return _tail_file(Path(settings.log_file_path), lines)

    registry = _registry(request)
    model = await registry.get(source_id)
    if model is None:
        raise HTTPException(404, f"unknown log source {source_id!r}")

    log_file = MODEL_LOGS_DIR / f"{model.id}.log"
    if kind == "file" or (kind == "auto" and log_file.is_file() and not model.container_id):
        if not log_file.is_file():
            raise HTTPException(404, "this model has no log file")
        return _tail_file(log_file, lines)

    if not model.container_id:
        if log_file.is_file():
            return _tail_file(log_file, lines)
        raise HTTPException(404, "this model has neither a container nor a log file")

    # `docker logs` on the model's host; args passed as a list (ids/names are
    # not shell-interpolated), env carries DOCKER_HOST for remote roles.
    def _run():
        return subprocess.run(
            ["docker", "logs", "--tail", str(lines), model.container_id],
            capture_output=True, timeout=20, env=merged_env(model.host),
        )

    try:
        r = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"docker logs timed out on {model.host}")
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace")[:300]
        # container gone but a launcher log file exists → fall back
        if log_file.is_file():
            return _tail_file(log_file, lines)
        raise HTTPException(502, f"docker logs failed: {err}")
    # container runtimes interleave stdout/stderr; show both, stdout first
    out = r.stdout.decode(errors="replace")
    err = r.stderr.decode(errors="replace")
    return out + (("\n--- stderr ---\n" + err) if err.strip() else "")
