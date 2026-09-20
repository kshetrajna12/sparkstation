#!/usr/bin/env python3
"""Sparkstation entrypoint for the reflex decision-model server.

Thin wrapper over `reflex.server.create_app` (which owns POST /v1/systemone):
  * configuration comes from env vars — the supervisor's reflex launcher
    passes them with `docker run -e ...`;
  * adds the `/health` liveness route the supervisor's health checker probes
    (reflex itself only serves /healthz) and an OpenAI-shaped `/v1/models`;
  * warms the engine with one tiny request BEFORE binding the port, so the
    ~20 s of Triton kernel compilation lands in the STARTING phase instead of
    on the first client call.

Env:
  MODEL_PATH                 HF id / path (default Qwen/Qwen3.5-4B)
  SERVED_MODEL_NAME          name echoed in responses (the sparkstation alias)
  REFLEX_ADAPTER             LoRA adapter dir from reflex-calibrate (optional; overrides the manifest)
  REFLEX_CALIBRATION         calibration.json with per-primitive temperatures (optional; overrides the manifest)
  REFLEX_PROMPT_TEXTS        prompt.json wording override (optional; overrides the manifest)
  REFLEX_STABLE_MANIFEST     reflex's serving/stable.json (default: the checkout baked into the image);
                             adapter / calibration / prompt defaults come from here
  REFLEX_COMMIT              reflex commit the image was built from (set by the launcher; /health echoes it)
  REFLEX_DTYPE               bfloat16 | float16 | float32
  REFLEX_MAX_PACK_TOKENS     branch-token budget per forward (default 8192)
  REFLEX_MAX_IMAGE_PIXELS    image token cost bound (default 1 MP ≈ 1k tokens)
  REFLEX_STATE_CACHE_ENTRIES LRU size of cached state prefixes (default 8)
  REFLEX_WARMUP              "0" to skip the warm-up request
"""
import logging
import os
import time

import torch
import uvicorn
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from reflex.engine import Engine
from reflex.schema import SystemOneRequest
from reflex.server import create_app

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("reflex.sparkstation")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3.5-4B")
SERVED_NAME = os.getenv("SERVED_MODEL_NAME", "reflex")
ADAPTER = os.getenv("REFLEX_ADAPTER") or None
CALIBRATION = os.getenv("REFLEX_CALIBRATION") or None
PROMPT_TEXTS = os.getenv("REFLEX_PROMPT_TEXTS") or None
STABLE_MANIFEST = os.getenv("REFLEX_STABLE_MANIFEST", "/app/reflex/serving/stable.json")
COMMIT = os.getenv("REFLEX_COMMIT") or (
    open("/app/REFLEX_COMMIT").read().strip() if os.path.exists("/app/REFLEX_COMMIT") else None
)
DTYPE = os.getenv("REFLEX_DTYPE", "bfloat16")
MAX_PACK_TOKENS = int(os.getenv("REFLEX_MAX_PACK_TOKENS", "8192"))
MAX_IMAGE_PIXELS = int(os.getenv("REFLEX_MAX_IMAGE_PIXELS", str(1024 * 1024)))
STATE_CACHE_ENTRIES = int(os.getenv("REFLEX_STATE_CACHE_ENTRIES", "8"))
WARMUP = os.getenv("REFLEX_WARMUP", "1") != "0"

WARMUP_REQUEST = SystemOneRequest(
    state={"ticket": "The export button crashes in Safari but works in Chrome."},
    questions={
        "browser_specific": {"type": "noul", "instructions": "Is the bug browser-specific?"},
        "area": {
            "type": "choice",
            "instructions": "Which area is affected?",
            "criteria": {"frontend": "UI, browser", "backend": "API, database", "other": None},
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this?",
            "criteria": ["cosmetic", "degraded with a workaround", "blocking"],
        },
    },
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available (run with --gpus all)")
    # What to serve is reflex's call: serving/stable.json at the baked commit names
    # the adapter / calibration / prompt it recommends; env vars override per knob.
    manifest = {}
    try:
        from reflex.serving import engine_kwargs, load_stable

        manifest = load_stable(STABLE_MANIFEST)
        kw = engine_kwargs(manifest, adapter_path=ADAPTER, calibration_path=CALIBRATION, prompt_texts=PROMPT_TEXTS)
    except ImportError:  # a reflex checkout from before serving/stable.json existed
        kw = {"adapter_path": ADAPTER, "calibration_path": CALIBRATION}
    kw["model_id"] = MODEL_PATH
    for label, path in (("adapter", kw.get("adapter_path")), ("calibration", kw.get("calibration_path"))):
        # absolute paths must exist inside the container; anything else is a hub id
        if path and path.startswith("/") and not os.path.exists(path):
            raise FileNotFoundError(f"{label}={path} does not exist inside the container")

    t0 = time.perf_counter()
    engine = Engine.load(
        dtype=getattr(torch, DTYPE),
        max_pack_tokens=MAX_PACK_TOKENS,
        max_image_pixels=MAX_IMAGE_PIXELS,
        state_cache_entries=STATE_CACHE_ENTRIES,
        **kw,
    )
    engine.model_name = SERVED_NAME  # responses report the sparkstation alias
    log.info(
        "loaded %s@%s in %.1fs (adapter=%s calibration=%s prompt_texts=%s strategy=%s multimodal=%s)",
        MODEL_PATH, (COMMIT or "?")[:12], time.perf_counter() - t0, kw.get("adapter_path"),
        engine.cal.temperature, kw.get("prompt_texts"), engine.strategy, engine.multimodal,
    )

    if WARMUP:
        t0 = time.perf_counter()
        engine.answer(WARMUP_REQUEST)
        warm = time.perf_counter()
        engine.answer(WARMUP_REQUEST)
        log.info(
            "warm-up: cold %.0f ms, state-cached %.0f ms",
            (warm - t0) * 1000, (time.perf_counter() - warm) * 1000,
        )

    app = create_app(engine)
    started = int(time.time())
    # newer reflex serves /health itself; ours (below) carries the deployment
    # facts the supervisor and operators want, so it must be the one that answers
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/health"]

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": MODEL_PATH,
            "served_name": SERVED_NAME,
            "reflex_commit": COMMIT,
            "stable": {k: manifest.get(k) for k in ("released", "adapter", "calibration", "prompt_texts")} if manifest else None,
            "adapter": kw.get("adapter_path"),
            "calibration": engine.cal.temperature,
            "strategy": engine.strategy,
            "multimodal": engine.multimodal,
            "device": str(engine.device),
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": SERVED_NAME, "object": "model", "created": started, "owned_by": "reflex"}],
        }

    @app.exception_handler(HTTPException)
    async def _openai_shaped_errors(_: Request, exc: HTTPException):
        # Same envelope the gateway uses, so clients see one error shape.
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc.detail), "type": "systemone_error"}},
        )

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
