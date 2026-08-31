"""
/voice/* — Voice Studio API for the Sparkstation Console.

The cascade voice stack (backend `voicecascade`) runs on ONE cluster role
(worker2 today) and consists of the Pipecat bot (:7860, `/ws-client`) plus
three Qwen3-TTS servers, each with its own voice registry file in the same
config directory (bind-mounted into every TTS container as /config):

  engine   server                    registry file              hot-reload
  clone    VoiceClone   (:8023)      voices.json                 no  (restart ~40 s)
  stock    CustomVoice  (:8024)      customvoice_voices.json     no  (restart ~35 s)
  design   VoiceDesign  (:8025)      voicedesign_voices.json     yes

This module presents them as ONE registry (`GET /voice/voices`) and hides the
transport: registry files + clone reference clips are read/written over SSH
to the role's host (cluster.hosts.<role> from .sparkstation.local.yaml — no
hostname lives in this file); TTS requests are proxied over HTTP; the Talk tab
gets a WebSocket relay to the bot's `/ws-client` (the WebRTC playground can't
traverse the zero-trust tunnel, a plain WS can).

Console-owned state lives in ONE extra file in the same directory,
`console.json` — currently just the default voice, which the bot reads at
session start (voicecascade/bot.py, CASCADE_VOICE_CONFIG_DIR). The registry
directory holds biometric data (K's reference clips) and is deliberately NOT
in any git repo; back it up out-of-band.

Every path is derived from the voicecascade model spec in models.yaml
(extra_args.tts_config_dir / tts_engines) and validated voice ids
(`^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$`), so the API can't be steered outside the
registry directory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from supervisor.auth import require_api_key
from supervisor.config import settings
from supervisor.models_config import get_cluster_config, load_models_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")

# Engine table. `container` names match the launcher's tts_containers; the
# registry file names are what each server script reads (worker's
# ~/cascade-tts/config/run_*server.py).
ENGINES: Dict[str, Dict[str, Any]] = {
    "clone": {
        "port": 8023,
        "container": "qwen3-tts-clone",
        "registry": "voices.json",
        "hot_reload": False,
        "label": "Cloned (VoiceClone)",
    },
    "stock": {
        "port": 8024,
        "container": "qwen3-tts-cv",
        "registry": "customvoice_voices.json",
        "hot_reload": False,
        "label": "Stock (CustomVoice)",
    },
    "design": {
        "port": 8025,
        "container": "qwen3-tts-vd",
        "registry": "voicedesign_voices.json",
        "hot_reload": True,
        "label": "Designed (VoiceDesign)",
    },
}
CONSOLE_FILE = "console.json"
SPEAKERS_DIR = "speakers"
DEFAULT_CONFIG_DIR = "~/cascade-tts/config"
DEFAULT_CONFIG_MOUNT = "/config"  # where the config dir appears INSIDE the TTS containers
DEFAULT_BOT_PORT = 7860
SAMPLE_TEXT = (
    "Hi, I'm Sparky. This is a short sample so you can hear how this voice sounds."
)
# Clone reference length guard: ICL cloning re-processes the whole reference
# on every request; ~12 s runs 1.45x realtime, 30 s dropped to 1.05x (choppy).
CLONE_REF_MAX_S = 20.0
CLONE_REF_WARN_S = 14.0
SSH_TIMEOUT_S = 30


# ── configuration ────────────────────────────────────────────────────────────

class VoiceStackConfig(BaseModel):
    role: str
    ip: Optional[str]            # None = local machine
    ssh_user: Optional[str]
    config_dir: str              # on the role's host (may start with ~)
    config_mount: str            # same dir as seen inside the TTS containers
    bot_port: int
    engines: Dict[str, Dict[str, Any]]

    @property
    def is_local(self) -> bool:
        return self.ip is None or self.ip in ("127.0.0.1", "localhost", "::1")

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.ip}"

    @property
    def http_host(self) -> str:
        return "127.0.0.1" if self.is_local else self.ip

    def engine_url(self, engine: str) -> str:
        return f"http://{self.http_host}:{self.engines[engine]['port']}"

    @property
    def bot_ws_url(self) -> str:
        return f"ws://{self.http_host}:{self.bot_port}/ws-client"


def load_stack_config() -> VoiceStackConfig:
    """Locate the voicecascade spec in models.yaml and the role's SSH identity."""
    cfg = load_models_config()
    spec = None
    for defn in cfg.models.values():
        if defn.backend == "voicecascade":
            spec = defn
            break
    if spec is None:
        raise HTTPException(503, "no voicecascade model spec in models.yaml — Voice Studio is unavailable")
    xa = spec.extra_args or {}
    engines: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in ENGINES.items()}
    for name, override in (xa.get("tts_engines") or {}).items():
        if name in engines and isinstance(override, dict):
            engines[name].update(override)
    cluster = get_cluster_config()
    host = cluster.hosts.get(spec.host)
    ip = host.ip if host else None
    ssh_user = host.ssh_user if host else None
    if ip is not None and not cluster.is_local(spec.host) and not ssh_user:
        raise HTTPException(503, f"cluster role {spec.host!r} has no ssh_user (check .sparkstation.local.yaml)")
    return VoiceStackConfig(
        role=spec.host,
        ip=None if cluster.is_local(spec.host) else ip,
        ssh_user=ssh_user,
        config_dir=str(xa.get("tts_config_dir") or DEFAULT_CONFIG_DIR),
        config_mount=str(xa.get("tts_config_mount") or DEFAULT_CONFIG_MOUNT),
        bot_port=int(xa.get("port") or DEFAULT_BOT_PORT),
        engines=engines,
    )


# ── remote file / command access ─────────────────────────────────────────────

class VoiceHost:
    """Shell access to the voice role's config directory (SSH or local).

    Paths handed to these methods are always built by `registry_path()` /
    `speaker_path()` from validated ids, never from raw request input.
    """

    def __init__(self, cfg: VoiceStackConfig):
        self.cfg = cfg

    async def run(self, cmd: str, stdin: Optional[bytes] = None, timeout: int = SSH_TIMEOUT_S) -> subprocess.CompletedProcess:
        if self.cfg.is_local:
            argv = ["bash", "-c", cmd]
        else:
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.cfg.ssh_target, cmd]

        def _run():
            return subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)

        try:
            return await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"voice host command timed out: {cmd[:80]}")

    def path(self, *parts: str) -> str:
        return "/".join([self.cfg.config_dir.rstrip("/"), *parts])

    def q(self, *parts: str) -> str:
        """Shell-quoted path; a leading ~ must stay unquoted to expand."""
        p = self.path(*parts)
        if p.startswith("~/"):
            return "~/" + shlex.quote(p[2:])
        return shlex.quote(p)

    async def read_json(self, name: str) -> Dict[str, Any]:
        r = await self.run(f"cat {self.q(name)} 2>/dev/null || echo '{{}}'")
        if r.returncode != 0:
            raise HTTPException(502, f"cannot read {name} on {self.cfg.role}: {r.stderr.decode(errors='replace')[:200]}")
        try:
            data = json.loads(r.stdout.decode() or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(502, f"{name} on {self.cfg.role} is not valid JSON: {e}")
        return data if isinstance(data, dict) else {}

    async def write_json(self, name: str, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode() + b"\n"
        await self.put_file(name, payload)

    async def put_file(self, name: str, payload: bytes) -> None:
        # atomic: write to a temp name in the same dir, then mv over the target
        target, tmp = self.q(name), self.q(name + ".tmp")
        r = await self.run(f"mkdir -p $(dirname {target}) && cat > {tmp} && mv -f {tmp} {target}", stdin=payload, timeout=120)
        if r.returncode != 0:
            raise HTTPException(502, f"cannot write {name} on {self.cfg.role}: {r.stderr.decode(errors='replace')[:200]}")

    async def remove_file(self, name: str) -> None:
        await self.run(f"rm -f {self.q(name)}")

    async def list_speakers(self) -> List[str]:
        r = await self.run(f"ls -1 {self.q(SPEAKERS_DIR)} 2>/dev/null || true")
        return [ln.strip() for ln in r.stdout.decode(errors="replace").splitlines() if ln.strip()]

    async def restart_container(self, name: str) -> None:
        r = await self.run(f"docker restart {shlex.quote(name)}", timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"docker restart {name}: {r.stderr.decode(errors='replace')[:200]}")


# ── engine restart tracking (clone/stock servers read their registry at boot) ─

_apply_state: Dict[str, Dict[str, Any]] = {}   # engine -> {state, since, error}
_apply_tasks: Dict[str, asyncio.Task] = {}
_http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0))


async def _engine_health(cfg: VoiceStackConfig, engine: str) -> Dict[str, Any]:
    url = cfg.engine_url(engine)
    try:
        r = await _http.get(f"{url}/health", timeout=3.0)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return {"ok": r.status_code == 200 and body.get("model_loaded", True), "port": cfg.engines[engine]["port"]}
    except Exception:
        return {"ok": False, "port": cfg.engines[engine]["port"]}


async def _apply_engine(cfg: VoiceStackConfig, engine: str) -> None:
    """Restart an engine's container and wait for /health (background task)."""
    host = VoiceHost(cfg)
    _apply_state[engine] = {"state": "restarting", "since": time.time(), "error": None}
    try:
        await host.restart_container(cfg.engines[engine]["container"])
        deadline = time.time() + 240
        while time.time() < deadline:
            await asyncio.sleep(3)
            if (await _engine_health(cfg, engine))["ok"]:
                _apply_state[engine] = {"state": "ready", "since": time.time(), "error": None}
                logger.info(f"voice engine {engine} restarted and healthy")
                return
        _apply_state[engine] = {"state": "error", "since": time.time(), "error": "timed out waiting for /health"}
    except Exception as e:
        logger.error(f"voice engine {engine} restart failed: {e}")
        _apply_state[engine] = {"state": "error", "since": time.time(), "error": str(e)}


def schedule_apply(cfg: VoiceStackConfig, engine: str) -> bool:
    """Restart the engine if its server can't hot-reload. Returns True if scheduled."""
    if cfg.engines[engine].get("hot_reload"):
        return False
    task = _apply_tasks.get(engine)
    if task and not task.done():
        return True  # already restarting; the new registry is on disk and will be picked up
    _apply_tasks[engine] = asyncio.create_task(_apply_engine(cfg, engine))
    return True


# ── registry model ───────────────────────────────────────────────────────────

class Voice(BaseModel):
    id: str
    engine: str
    language: str = "English"
    instruct: str = ""                   # design: identity; clone/stock: style direction
    ref_text: Optional[str] = None       # clone only (transcript of the reference clip)
    ref_audio: Optional[str] = None      # clone only (container path)
    speaker: Optional[str] = None        # stock only (bundled speaker name)
    chunk_size: Optional[int] = None
    is_default: bool = False


def _voice_from_entry(vid: str, engine: str, entry: Dict[str, Any], default: Dict[str, Any]) -> Voice:
    return Voice(
        id=vid,
        engine=engine,
        language=str(entry.get("language") or "English"),
        instruct=str(entry.get("instruct") or ""),
        ref_text=entry.get("ref_text"),
        ref_audio=entry.get("ref_audio"),
        speaker=entry.get("speaker"),
        chunk_size=entry.get("chunk_size"),
        is_default=(default.get("voice") == vid and default.get("engine") == engine),
    )


async def _read_all(host: VoiceHost) -> Dict[str, Any]:
    cfg = host.cfg
    names = [cfg.engines[e]["registry"] for e in cfg.engines] + [CONSOLE_FILE]
    results = await asyncio.gather(*(host.read_json(n) for n in names))
    registries = dict(zip(cfg.engines.keys(), results[:-1]))
    console = results[-1]
    return {"registries": registries, "console": console}


def _default_of(console: Dict[str, Any]) -> Dict[str, Any]:
    d = console.get("default") or {}
    return d if isinstance(d, dict) else {}


def _merge_voices(registries: Dict[str, Dict[str, Any]], console: Dict[str, Any]) -> List[Voice]:
    default = _default_of(console)
    out: List[Voice] = []
    for engine, reg in registries.items():
        for vid, entry in reg.items():
            if isinstance(entry, dict):
                out.append(_voice_from_entry(vid, engine, entry, default))
    return out


def _find(voices: List[Voice], vid: str, engine: Optional[str] = None) -> Optional[Voice]:
    for v in voices:
        if v.id == vid and (engine is None or v.engine == engine):
            return v
    return None


def _check_id(vid: str) -> str:
    if not VOICE_ID_RE.match(vid or ""):
        raise HTTPException(400, "voice id must be 1-40 chars of letters, digits, '_' or '-'")
    return vid


def _check_engine(engine: str) -> str:
    if engine not in ENGINES:
        raise HTTPException(400, f"unknown engine {engine!r}; one of {sorted(ENGINES)}")
    return engine


# ── read endpoints ───────────────────────────────────────────────────────────

@router.get("/status")
async def voice_status():
    """Stack + engine health and the console-owned default voice."""
    cfg = load_stack_config()
    host = VoiceHost(cfg)
    console, healths = await asyncio.gather(
        host.read_json(CONSOLE_FILE),
        asyncio.gather(*(_engine_health(cfg, e) for e in cfg.engines)),
    )
    bot_ok = False
    try:
        r = await _http.get(f"http://{cfg.http_host}:{cfg.bot_port}/", timeout=3.0, follow_redirects=False)
        bot_ok = r.status_code < 500
    except Exception:
        pass
    engines = {}
    for name, h in zip(cfg.engines, healths):
        engines[name] = {**h, "label": cfg.engines[name]["label"], "hot_reload": cfg.engines[name]["hot_reload"],
                         "apply": _apply_state.get(name)}
    return {
        "role": cfg.role,
        "bot": {"ok": bot_ok, "port": cfg.bot_port},
        "engines": engines,
        "default": _default_of(console) or None,
        "clone_ref_max_seconds": CLONE_REF_MAX_S,
        "clone_ref_recommended_seconds": [8, 12],
    }


@router.get("/voices", response_model=List[Voice])
async def list_voices():
    host = VoiceHost(load_stack_config())
    data = await _read_all(host)
    voices = _merge_voices(data["registries"], data["console"])
    voices.sort(key=lambda v: (not v.is_default, v.engine, v.id.lower()))
    return voices


# ── speech (proxy to the right engine) ───────────────────────────────────────

class SpeakRequest(BaseModel):
    text: str = Field(SAMPLE_TEXT, max_length=2000)
    voice: Optional[str] = None            # registered voice id (any engine)
    engine: Optional[str] = None           # required when previewing an unregistered design
    instruct: Optional[str] = None         # per-request direction (design: appended to identity)
    language: Optional[str] = None
    format: str = "wav"                    # wav | pcm


@router.post("/speak")
async def speak(req: SpeakRequest):
    """Synthesize `text` and stream audio back (24 kHz mono PCM16 in a WAV wrapper).

    Two modes: a registered voice (looked up across engines), or
    engine="design" + instruct for a preview of a voice that isn't saved yet.
    """
    cfg = load_stack_config()
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if req.format not in ("wav", "pcm"):
        raise HTTPException(400, "format must be wav or pcm")

    engine = req.engine
    voice_id = req.voice
    if voice_id:
        _check_id(voice_id)
        data = await _read_all(VoiceHost(cfg))
        v = _find(_merge_voices(data["registries"], data["console"]), voice_id, engine)
        if v is None:
            raise HTTPException(404, f"voice {voice_id!r} is not registered")
        engine = v.engine
    else:
        if engine != "design" or not (req.instruct or "").strip():
            raise HTTPException(400, "give a registered `voice`, or engine='design' with an `instruct` to preview")
        voice_id = "_console_preview"  # unregistered → the VoiceDesign server uses the request instruct
    _check_engine(engine)

    body: Dict[str, Any] = {"model": "tts-1", "input": text, "voice": voice_id, "response_format": req.format}
    if req.instruct and engine != "clone":  # the clone server has no per-request instruct
        body["instruct"] = req.instruct.strip()
    if req.language:
        body["language"] = req.language

    url = f"{cfg.engine_url(engine)}/v1/audio/speech"
    try:
        upstream = _http.build_request("POST", url, json=body)
        resp = await _http.send(upstream, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"{engine} engine unreachable: {e}")
    if resp.status_code != 200:
        detail = (await resp.aread()).decode(errors="replace")[:400]
        await resp.aclose()
        raise HTTPException(resp.status_code if resp.status_code < 500 else 502, f"{engine} engine: {detail}")

    async def body_iter():
        try:
            async for chunk in resp.aiter_bytes(8192):
                yield chunk
        finally:
            await resp.aclose()

    media = "audio/wav" if req.format == "wav" else "audio/pcm"
    return StreamingResponse(body_iter(), media_type=media, headers={"Cache-Control": "no-store"})


# ── registry mutations ───────────────────────────────────────────────────────

class DesignRequest(BaseModel):
    id: str
    instruct: str = Field(..., min_length=3, max_length=1000, description="voice description = identity")
    language: str = "English"


@router.post("/voices/design", dependencies=[Depends(require_api_key)], status_code=201)
async def create_design_voice(req: DesignRequest):
    cfg = load_stack_config()
    host = VoiceHost(cfg)
    vid = _check_id(req.id)
    reg = await host.read_json(cfg.engines["design"]["registry"])
    if vid in reg:
        raise HTTPException(409, f"design voice {vid!r} already exists (PATCH it instead)")
    reg[vid] = {"instruct": req.instruct.strip(), "language": req.language}
    await host.write_json(cfg.engines["design"]["registry"], reg)
    return {"ok": True, "voice": vid, "engine": "design", "applying": schedule_apply(cfg, "design")}


def _probe_wav(path: Path) -> float:
    """Duration in seconds of a WAV (wave module; the file is our own ffmpeg output)."""
    import wave
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


def _convert_to_wav(src: Path, dst: Path) -> None:
    """Normalize any browser upload (webm/opus, m4a, mp3, wav...) to 24 kHz mono PCM16."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if src.suffix.lower() == ".wav":
            shutil.copyfile(src, dst)
            return
        raise HTTPException(501, "ffmpeg is not installed on the supervisor host; upload a WAV file instead")
    r = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(dst)],
        capture_output=True, timeout=120,
    )
    if r.returncode != 0:
        raise HTTPException(400, f"could not decode the audio file: {r.stderr.decode(errors='replace')[:200]}")


@router.post("/voices/clone", dependencies=[Depends(require_api_key)], status_code=201)
async def create_clone_voice(
    id: str = Form(...),
    ref_text: str = Form(..., description="exact transcript of the reference clip"),
    language: str = Form("English"),
    instruct: str = Form(""),
    chunk_size: int = Form(16),
    file: UploadFile = File(...),
):
    """Register a cloned voice from a reference clip + its transcript.

    The clip is normalized to 24 kHz mono WAV on the supervisor host, stored
    as speakers/<id>.wav in the registry dir, and the VoiceClone server is
    restarted (it reads voices.json only at boot). Keep clips ~8-12 s.
    """
    cfg = load_stack_config()
    host = VoiceHost(cfg)
    vid = _check_id(id)
    ref_text = ref_text.strip()
    if len(ref_text) < 5:
        raise HTTPException(400, "ref_text must be the transcript of the clip (required by the cloner)")
    if not (4 <= chunk_size <= 64):
        raise HTTPException(400, "chunk_size must be 4-64")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(413, "reference clip larger than 50 MB")

    suffix = Path(file.filename or "clip").suffix.lower() or ".bin"
    with tempfile.TemporaryDirectory(prefix="sparkvoice-") as td:
        src = Path(td) / f"in{suffix}"
        dst = Path(td) / "ref.wav"
        src.write_bytes(raw)
        await asyncio.to_thread(_convert_to_wav, src, dst)
        duration = await asyncio.to_thread(_probe_wav, dst)
        if duration > CLONE_REF_MAX_S:
            raise HTTPException(
                400,
                f"reference clip is {duration:.1f}s; keep it under {CLONE_REF_MAX_S:.0f}s "
                f"(8-12 s is ideal — long references make cloned speech choppy)",
            )
        wav_bytes = dst.read_bytes()

    reg = await host.read_json(cfg.engines["clone"]["registry"])
    if vid in reg:
        raise HTTPException(409, f"clone voice {vid!r} already exists (delete it first to re-record)")
    await host.put_file(f"{SPEAKERS_DIR}/{vid}.wav", wav_bytes)
    entry: Dict[str, Any] = {
        "ref_audio": f"{cfg.config_mount.rstrip('/')}/{SPEAKERS_DIR}/{vid}.wav",
        "ref_text": ref_text,
        "language": language,
        "chunk_size": chunk_size,
    }
    if instruct.strip():
        entry["instruct"] = instruct.strip()
    reg[vid] = entry
    await host.write_json(cfg.engines["clone"]["registry"], reg)
    warning = None
    if duration > CLONE_REF_WARN_S:
        warning = f"clip is {duration:.1f}s — cloned speech may be choppy; 8-12 s clips stream faster"
    return {"ok": True, "voice": vid, "engine": "clone", "duration_seconds": round(duration, 2),
            "applying": schedule_apply(cfg, "clone"), "warning": warning}


class VoicePatch(BaseModel):
    instruct: Optional[str] = Field(None, max_length=1000)
    language: Optional[str] = None
    ref_text: Optional[str] = Field(None, max_length=4000)
    chunk_size: Optional[int] = Field(None, ge=4, le=64)


@router.patch("/voices/{engine}/{voice_id}", dependencies=[Depends(require_api_key)])
async def patch_voice(engine: str, voice_id: str, patch: VoicePatch):
    """Per-voice style/identity edits. Design edits are live; clone/stock edits restart their server."""
    cfg = load_stack_config()
    host = VoiceHost(cfg)
    _check_engine(engine)
    vid = _check_id(voice_id)
    reg_name = cfg.engines[engine]["registry"]
    reg = await host.read_json(reg_name)
    if vid not in reg:
        raise HTTPException(404, f"{engine} voice {vid!r} not found")
    entry = dict(reg[vid])
    if patch.instruct is not None:
        if engine == "design" and not patch.instruct.strip():
            raise HTTPException(400, "a designed voice's instruct is its identity and cannot be empty")
        if patch.instruct.strip():
            entry["instruct"] = patch.instruct.strip()
        else:
            entry.pop("instruct", None)
    if patch.language:
        entry["language"] = patch.language
    if patch.ref_text is not None:
        if engine != "clone":
            raise HTTPException(400, "ref_text only applies to clone voices")
        entry["ref_text"] = patch.ref_text.strip()
    if patch.chunk_size is not None:
        if engine != "clone":
            raise HTTPException(400, "chunk_size only applies to clone voices")
        entry["chunk_size"] = patch.chunk_size
    reg[vid] = entry
    await host.write_json(reg_name, reg)
    return {"ok": True, "voice": vid, "engine": engine, "applying": schedule_apply(cfg, engine)}


@router.delete("/voices/{engine}/{voice_id}", dependencies=[Depends(require_api_key)])
async def delete_voice(engine: str, voice_id: str):
    cfg = load_stack_config()
    host = VoiceHost(cfg)
    _check_engine(engine)
    vid = _check_id(voice_id)
    data = await _read_all(host)
    default = _default_of(data["console"])
    if default.get("voice") == vid and default.get("engine") == engine:
        raise HTTPException(409, "this is the default voice — pick another default first")
    reg = data["registries"][engine]
    if vid not in reg:
        raise HTTPException(404, f"{engine} voice {vid!r} not found")
    if engine == "stock":
        raise HTTPException(400, "stock speakers are bundled with the model; clear the instruct instead")
    if engine == "clone" and len(reg) == 1:
        raise HTTPException(409, "the VoiceClone server needs at least one registered voice to boot")
    entry = reg.pop(vid)
    await host.write_json(cfg.engines[engine]["registry"], reg)
    if engine == "clone":
        ref = str(entry.get("ref_audio") or "")
        # only remove clips that live in OUR speakers dir (never a hand-placed file elsewhere)
        prefix = f"{cfg.config_mount.rstrip('/')}/{SPEAKERS_DIR}/"
        if ref.startswith(prefix) and VOICE_ID_RE.match(Path(ref).stem or "") and "/" not in ref[len(prefix):]:
            await host.remove_file(f"{SPEAKERS_DIR}/{Path(ref).name}")
    return {"ok": True, "voice": vid, "engine": engine, "applying": schedule_apply(cfg, engine)}


@router.post("/voices/{engine}/{voice_id}/default", dependencies=[Depends(require_api_key)])
async def set_default_voice(engine: str, voice_id: str):
    """Make this the voice Sparky speaks in (new sessions; the bot reads console.json per session)."""
    cfg = load_stack_config()
    host = VoiceHost(cfg)
    _check_engine(engine)
    vid = _check_id(voice_id)
    data = await _read_all(host)
    if vid not in data["registries"][engine]:
        raise HTTPException(404, f"{engine} voice {vid!r} not found")
    console = data["console"]
    console["default"] = {"voice": vid, "engine": engine, "set_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    await host.write_json(CONSOLE_FILE, console)
    return {"ok": True, "default": console["default"]}


@router.post("/engines/{engine}/apply", dependencies=[Depends(require_api_key)])
async def apply_engine(engine: str):
    """Restart an engine so it re-reads its registry (automatic after edits; manual escape hatch)."""
    cfg = load_stack_config()
    _check_engine(engine)
    task = _apply_tasks.get(engine)
    if task and not task.done():
        return {"ok": True, "engine": engine, "applying": True}
    _apply_tasks[engine] = asyncio.create_task(_apply_engine(cfg, engine))
    return {"ok": True, "engine": engine, "applying": True}


# ── Talk: WebSocket relay to the bot's /ws-client ────────────────────────────

@router.websocket("/talk")
async def talk_relay(ws: WebSocket):
    """Byte-for-byte relay browser <-> bot `/ws-client` (Pipecat protobuf frames).

    The browser can't reach the voice role directly from outside the LAN, and
    the zero-trust tunnel won't carry WebRTC media — a plain WS relay is the
    one transport that works everywhere. Single session: the bot closes extra
    connections with 4429 and that close code is forwarded.
    """
    import websockets

    try:
        cfg = load_stack_config()
    except HTTPException as e:
        await ws.close(code=1011, reason=str(e.detail)[:120])
        return
    await ws.accept()
    try:
        upstream = await websockets.connect(cfg.bot_ws_url, max_size=None, open_timeout=10)
    except Exception as e:
        logger.warning(f"talk relay: cannot reach bot at {cfg.bot_ws_url}: {e}")
        await ws.close(code=1011, reason="voice bot unreachable — is voicecascade running?")
        return

    async def browser_to_bot():
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                await upstream.send(msg["bytes"])
            elif msg.get("text") is not None:
                await upstream.send(msg["text"])

    async def bot_to_browser():
        async for data in upstream:
            if isinstance(data, (bytes, bytearray)):
                await ws.send_bytes(bytes(data))
            else:
                await ws.send_text(data)

    t1 = asyncio.create_task(browser_to_bot())
    t2 = asyncio.create_task(bot_to_browser())
    close_code, close_reason = 1000, ""
    try:
        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if exc is not None and isinstance(exc, websockets.exceptions.ConnectionClosed):
                close_code = exc.rcvd.code if exc.rcvd else 1011
                close_reason = (exc.rcvd.reason if exc.rcvd else "")[:120]
    finally:
        try:
            await upstream.close()
        except Exception:
            pass
        try:
            await ws.close(code=close_code, reason=close_reason)
        except Exception:
            pass
