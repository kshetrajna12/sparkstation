"""
Tests for the console support endpoints (supervisor/console_api.py):
profiles listing, start-by-alias resolution, and log tailing (path safety,
file tails, docker fallback behavior).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from supervisor import console_api
from supervisor.console_api import _tail_file, router


class FakeRegistry:
    def __init__(self, models):
        self.models = models

    async def list_all(self):
        return self.models

    async def get(self, model_id):
        return next((m for m in self.models if m.id == model_id), None)


def _model(id="m-1", alias="qwen", container=None, host="primary", status="running"):
    return SimpleNamespace(
        id=id, model_alias=alias, model_name="org/model", host=host,
        container_id=container, status=status,  # plain str, like the real registry (use_enum_values)
    )


@pytest.fixture
def client(monkeypatch, tmp_path):
    import supervisor.main as main_mod

    logs_dir = tmp_path / "model_logs"
    logs_dir.mkdir()
    monkeypatch.setattr(console_api, "MODEL_LOGS_DIR", logs_dir)

    sup_log = tmp_path / "sparkstation.log"
    sup_log.write_text("".join(f"line {i}\n" for i in range(500)))
    monkeypatch.setattr(console_api.settings, "log_file_path", str(sup_log))

    registry = FakeRegistry([_model()])
    monkeypatch.setattr(main_mod, "registry", registry)

    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    c.logs_dir = logs_dir
    c.registry = registry
    return c


# ── tail helper ──────────────────────────────────────────────────────────────

def test_tail_file_returns_last_lines(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("".join(f"row {i}\n" for i in range(1000)))
    out = _tail_file(f, 10).splitlines()
    assert out[-1] == "row 999" and len(out) == 10


def test_tail_file_small_file(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("only\n")
    assert _tail_file(f, 100) == "only\n"


# ── profiles ─────────────────────────────────────────────────────────────────

def test_profiles_lists_yaml_profiles(client):
    d = client.get("/profiles").json()
    assert d["default"] in d["profiles"]
    assert "voice" in d["profiles"] and "voicecascade" in d["profiles"]["voice"]
    assert "qwen-flash-next" in d["all_aliases"]
    assert d["aliases"]["voicecascade"]["host"] == "worker2"
    assert d["aliases"]["qwen-flash-next"]["memory_gb"]
    assert d["active"]  # env/default resolution never leaves it empty


# ── start-by-alias ───────────────────────────────────────────────────────────

def test_start_by_alias_unknown(client):
    assert client.post("/models/nope-model/start-by-alias").status_code == 404
    # NB an alias outside the profile falls back to its base spec (same as the
    # CLI's `models start <alias> -p <profile>`), so that is not a 404.


def test_start_by_alias_resolves_and_forwards(client, monkeypatch):
    import supervisor.main as main_mod

    captured = {}

    async def fake_start(req):
        captured["req"] = req
        return {"model_id": "x", "status": "starting"}

    monkeypatch.setattr(main_mod, "start_model", fake_start)
    r = client.post("/models/gemma4-2b/start-by-alias?profile=voice")
    assert r.status_code == 200, r.text
    req = captured["req"]
    assert req.model_alias == "gemma4-2b"
    assert req.backend.value in ("vllm", "sglang")  # from models.yaml, not hardcoded


# ── log sources ──────────────────────────────────────────────────────────────

def test_log_sources_lists_supervisor_and_models(client):
    (client.logs_dir / "m-1.log").write_text("hello\n")
    d = client.get("/logs").json()
    ids = {s["id"] for s in d["sources"]}
    assert "supervisor" in ids and "m-1" in ids
    m = next(s for s in d["sources"] if s["id"] == "m-1")
    assert m["kinds"] == ["file"] and m["label"] == "qwen"


def test_log_sources_skips_models_without_logs(client):
    d = client.get("/logs").json()
    assert [s["id"] for s in d["sources"]] == ["supervisor"]


def test_tail_supervisor_and_model_file(client):
    r = client.get("/logs/supervisor?lines=10")
    assert r.status_code == 200
    assert r.text.splitlines()[-1] == "line 499"

    (client.logs_dir / "m-1.log").write_text("model says hi\n")
    r = client.get("/logs/m-1?lines=50")
    assert r.status_code == 200 and "model says hi" in r.text


def test_tail_unknown_source_and_bounds(client):
    assert client.get("/logs/ghost").status_code == 404
    assert client.get("/logs/supervisor?lines=999999").status_code == 422
    assert client.get("/logs/supervisor?lines=1").status_code == 422


def test_tail_container_falls_back_to_file_when_docker_fails(client, monkeypatch):
    client.registry.models[0] = _model(container="deadbeef")
    (client.logs_dir / "m-1.log").write_text("launcher output\n")

    def fake_run(argv, **kw):
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"no such container")

    monkeypatch.setattr(console_api.subprocess, "run", fake_run)
    r = client.get("/logs/m-1")
    assert r.status_code == 200 and "launcher output" in r.text


# ── per-host resources ───────────────────────────────────────────────────────

def test_parse_host_probe():
    from supervisor.console_api import _parse_host_probe
    out = _parse_host_probe("MemTotal:       125000000 kB\nMemAvailable:    80000000 kB\n55, 12.34\n")
    assert out["ok"] and 118 < out["mem_total_gb"] < 120
    assert out["gpu_temp_c"] == 55 and out["gpu_power_w"] == 12.34
    assert abs(out["mem_used_gb"] - (out["mem_total_gb"] - out["mem_available_gb"])) < 1e-9
    # no nvidia-smi line → memory still parses
    out = _parse_host_probe("MemTotal: 1000000 kB\nMemAvailable: 400000 kB\n")
    assert out["ok"] and out["gpu_temp_c"] is None


def test_resources_hosts_lists_cluster_roles(client, monkeypatch):
    monkeypatch.setattr(console_api, "_probe_host", lambda role: {"ok": True, "mem_total_gb": 1.0,
        "mem_available_gb": 0.5, "mem_used_gb": 0.5, "gpu_temp_c": None, "gpu_power_w": None})
    console_api._hosts_cache.update(at=0.0, data=None)
    d = client.get("/resources/hosts").json()
    assert set(d["hosts"]) >= {"primary", "worker1", "worker2"}
    console_api._hosts_cache.update(at=0.0, data=None)
