"""
Tests for the Voice Studio API (supervisor/voice.py).

The remote side (SSH to the voice role, HTTP to the TTS servers) is replaced
by an in-memory fake, so these cover registry merging, id/path validation,
the mutation rules (default guard, engine restart scheduling, clone clip
handling) and the request/response shapes the console relies on.
"""
import io
import json
import struct
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from supervisor import voice as voice_mod
from supervisor.voice import (
    CONSOLE_FILE,
    ENGINES,
    VOICE_ID_RE,
    VoiceStackConfig,
    _merge_voices,
    router,
)


class FakeHost:
    """In-memory stand-in for VoiceHost: a dict of filename -> bytes."""

    files: dict = {}
    restarted: list = []

    def __init__(self, cfg):
        self.cfg = cfg

    async def read_json(self, name):
        raw = FakeHost.files.get(name)
        return json.loads(raw) if raw else {}

    async def write_json(self, name, data):
        FakeHost.files[name] = json.dumps(data).encode()

    async def put_file(self, name, payload):
        FakeHost.files[name] = payload

    async def remove_file(self, name):
        FakeHost.files.pop(name, None)

    async def list_speakers(self):
        return [n.split("/", 1)[1] for n in FakeHost.files if n.startswith("speakers/")]

    async def restart_container(self, name):
        FakeHost.restarted.append(name)


def _cfg():
    return VoiceStackConfig(
        role="worker2", ip="10.0.0.2", ssh_user="u", config_dir="~/cascade-tts/config",
        config_mount="/config", bot_port=7860, engines={k: dict(v) for k, v in ENGINES.items()},
    )


@pytest.fixture
def client(monkeypatch):
    FakeHost.files = {
        "voices.json": json.dumps({
            "K": {"ref_audio": "/config/speakers/K_ref12.wav", "ref_text": "hello there", "language": "English", "chunk_size": 16},
            "K_full": {"ref_audio": "/config/speakers/K_ref.wav", "ref_text": "longer", "language": "English"},
        }).encode(),
        "customvoice_voices.json": json.dumps({"Ryan": {"speaker": "Ryan", "language": "English", "instruct": ""}}).encode(),
        "voicedesign_voices.json": json.dumps({"vd_british_male": {"instruct": "A British male", "language": "English"}}).encode(),
        CONSOLE_FILE: json.dumps({"default": {"voice": "K", "engine": "clone"}}).encode(),
        "speakers/K_ref12.wav": b"RIFF",
    }
    FakeHost.restarted = []
    monkeypatch.setattr(voice_mod, "VoiceHost", FakeHost)
    monkeypatch.setattr(voice_mod, "load_stack_config", _cfg)
    # don't spawn real restart tasks — record the intent instead
    scheduled = []
    monkeypatch.setattr(voice_mod, "schedule_apply", lambda cfg, engine: scheduled.append(engine) or not cfg.engines[engine]["hot_reload"])
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    c.scheduled = scheduled
    return c


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_voice_id_regex():
    assert VOICE_ID_RE.match("K")
    assert VOICE_ID_RE.match("sparky_warm-male2")
    assert not VOICE_ID_RE.match("../etc")
    assert not VOICE_ID_RE.match("a b")
    assert not VOICE_ID_RE.match("")
    assert not VOICE_ID_RE.match("-lead")
    assert not VOICE_ID_RE.match("x" * 41)


def test_merge_voices_tags_engine_and_default():
    voices = _merge_voices(
        {"clone": {"K": {"ref_text": "t"}}, "stock": {"Ryan": {"speaker": "Ryan"}}, "design": {}},
        {"default": {"voice": "K", "engine": "clone"}},
    )
    by_id = {v.id: v for v in voices}
    assert by_id["K"].engine == "clone" and by_id["K"].is_default
    assert by_id["Ryan"].engine == "stock" and not by_id["Ryan"].is_default
    assert by_id["Ryan"].speaker == "Ryan"


def test_merge_ignores_non_dict_entries():
    voices = _merge_voices({"clone": {"bad": "nope", "ok": {}}, "stock": {}, "design": {}}, {})
    assert [v.id for v in voices] == ["ok"]


def test_stack_config_urls():
    cfg = _cfg()
    assert cfg.engine_url("clone") == "http://10.0.0.2:8023"
    assert cfg.bot_ws_url == "ws://10.0.0.2:7860/ws-client"
    assert cfg.ssh_target == "u@10.0.0.2"
    local = VoiceStackConfig(role="primary", ip=None, ssh_user=None, config_dir="/x", config_mount="/config",
                             bot_port=7860, engines=cfg.engines)
    assert local.is_local and local.http_host == "127.0.0.1"


def test_voicehost_quotes_paths():
    host = voice_mod.VoiceHost(_cfg())
    # ~ stays unquoted so the remote shell expands it; safe names need no quotes
    assert host.q("voices.json") == "~/cascade-tts/config/voices.json"
    assert host.q("speakers", "K ref.wav") == "~/'cascade-tts/config/speakers/K ref.wav'"
    abs_host = voice_mod.VoiceHost(VoiceStackConfig(role="w", ip="1.2.3.4", ssh_user="u", config_dir="/srv/tts cfg",
                                                    config_mount="/config", bot_port=1, engines=ENGINES))
    assert abs_host.q("a b.json") == "'/srv/tts cfg/a b.json'"


# ── listing ──────────────────────────────────────────────────────────────────

def test_list_voices_merged_and_default_first(client):
    r = client.get("/voice/voices")
    assert r.status_code == 200
    voices = r.json()
    assert voices[0]["id"] == "K" and voices[0]["is_default"] is True
    engines = {v["engine"] for v in voices}
    assert engines == {"clone", "stock", "design"}
    k = next(v for v in voices if v["id"] == "K")
    assert k["ref_text"] == "hello there" and k["chunk_size"] == 16


# ── design ───────────────────────────────────────────────────────────────────

def test_design_create_conflict_and_delete(client):
    r = client.post("/voice/voices/design", json={"id": "sea_captain", "instruct": "A gravelly old sea captain"})
    assert r.status_code == 201
    assert r.json()["applying"] is False  # VoiceDesign hot-reloads
    reg = json.loads(FakeHost.files["voicedesign_voices.json"])
    assert reg["sea_captain"] == {"instruct": "A gravelly old sea captain", "language": "English"}

    assert client.post("/voice/voices/design", json={"id": "sea_captain", "instruct": "again"}).status_code == 409
    assert client.post("/voice/voices/design", json={"id": "../x", "instruct": "bad id"}).status_code == 400

    r = client.delete("/voice/voices/design/sea_captain")
    assert r.status_code == 200
    assert "sea_captain" not in json.loads(FakeHost.files["voicedesign_voices.json"])


def test_design_patch_cannot_blank_identity(client):
    r = client.patch("/voice/voices/design/vd_british_male", json={"instruct": "  "})
    assert r.status_code == 400
    r = client.patch("/voice/voices/design/vd_british_male", json={"instruct": "A British male, warmer"})
    assert r.status_code == 200
    assert json.loads(FakeHost.files["voicedesign_voices.json"])["vd_british_male"]["instruct"] == "A British male, warmer"


# ── clone ────────────────────────────────────────────────────────────────────

def _wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return buf.getvalue()


def test_clone_create_stores_clip_and_restarts_engine(client, monkeypatch):
    # bypass ffmpeg: copy the wav as-is
    monkeypatch.setattr(voice_mod, "_convert_to_wav", lambda src, dst: dst.write_bytes(src.read_bytes()))
    r = client.post(
        "/voice/voices/clone",
        data={"id": "K_studio", "ref_text": "this is the transcript", "language": "English", "instruct": "calm"},
        files={"file": ("clip.wav", _wav_bytes(10.0), "audio/wav")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["applying"] is True and body["warning"] is None
    assert 9.9 < body["duration_seconds"] < 10.1
    assert client.scheduled == ["clone"]
    reg = json.loads(FakeHost.files["voices.json"])
    assert reg["K_studio"]["ref_audio"] == "/config/speakers/K_studio.wav"
    assert reg["K_studio"]["ref_text"] == "this is the transcript"
    assert reg["K_studio"]["instruct"] == "calm"
    assert "speakers/K_studio.wav" in FakeHost.files


def test_clone_rejects_long_clip_and_missing_transcript(client, monkeypatch):
    monkeypatch.setattr(voice_mod, "_convert_to_wav", lambda src, dst: dst.write_bytes(src.read_bytes()))
    r = client.post("/voice/voices/clone", data={"id": "too_long", "ref_text": "long transcript here"},
                    files={"file": ("clip.wav", _wav_bytes(25.0), "audio/wav")})
    assert r.status_code == 400 and "under 20s" in r.json()["detail"]
    assert "too_long" not in json.loads(FakeHost.files["voices.json"])

    r = client.post("/voice/voices/clone", data={"id": "no_text", "ref_text": "hi"},
                    files={"file": ("clip.wav", _wav_bytes(5.0), "audio/wav")})
    assert r.status_code == 400


def test_clone_warns_on_longish_clip(client, monkeypatch):
    monkeypatch.setattr(voice_mod, "_convert_to_wav", lambda src, dst: dst.write_bytes(src.read_bytes()))
    r = client.post("/voice/voices/clone", data={"id": "longish", "ref_text": "a fine transcript"},
                    files={"file": ("clip.wav", _wav_bytes(16.0), "audio/wav")})
    assert r.status_code == 201 and "choppy" in r.json()["warning"]


def test_delete_clone_removes_only_our_clip_and_guards(client):
    # default voice can't be deleted
    assert client.delete("/voice/voices/clone/K").status_code == 409
    # non-default clone: registry entry + clip under speakers/ removed, engine restart scheduled
    FakeHost.files["speakers/K_ref.wav"] = b"RIFF"
    r = client.delete("/voice/voices/clone/K_full")
    assert r.status_code == 200 and r.json()["applying"] is True
    assert "K_full" not in json.loads(FakeHost.files["voices.json"])
    assert "speakers/K_ref.wav" not in FakeHost.files
    # last remaining clone voice can't be removed (server needs one to boot)
    client.post("/voice/voices/design/vd_british_male/default")
    assert client.delete("/voice/voices/clone/K").status_code == 409


def test_stock_voices_cannot_be_deleted_but_take_instruct(client):
    assert client.delete("/voice/voices/stock/Ryan").status_code == 400
    r = client.patch("/voice/voices/stock/Ryan", json={"instruct": "calm and slow"})
    assert r.status_code == 200 and r.json()["applying"] is True  # CustomVoice reads registry at boot
    assert json.loads(FakeHost.files["customvoice_voices.json"])["Ryan"]["instruct"] == "calm and slow"


# ── default ──────────────────────────────────────────────────────────────────

def test_set_default_writes_console_file(client):
    r = client.post("/voice/voices/stock/Ryan/default")
    assert r.status_code == 200
    console = json.loads(FakeHost.files[CONSOLE_FILE])
    assert console["default"]["voice"] == "Ryan" and console["default"]["engine"] == "stock"
    voices = client.get("/voice/voices").json()
    assert [v["id"] for v in voices if v["is_default"]] == ["Ryan"]
    assert client.post("/voice/voices/clone/missing/default").status_code == 404
    assert client.post("/voice/voices/nope/Ryan/default").status_code == 400


# ── speak validation (no upstream) ───────────────────────────────────────────

def test_speak_validation(client):
    assert client.post("/voice/speak", json={"text": "   "}).status_code == 400
    assert client.post("/voice/speak", json={"voice": "ghost"}).status_code == 404
    assert client.post("/voice/speak", json={"voice": "../x"}).status_code == 400
    # preview needs engine=design + instruct
    assert client.post("/voice/speak", json={"text": "hi"}).status_code == 400
    assert client.post("/voice/speak", json={"text": "hi", "engine": "stock"}).status_code == 400
    assert client.post("/voice/speak", json={"voice": "K", "format": "ogg"}).status_code == 400
