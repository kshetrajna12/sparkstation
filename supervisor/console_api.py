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
