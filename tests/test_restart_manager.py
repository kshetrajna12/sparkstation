"""
Auto-restart recovery (2026-10-06 gemma4-2b incident).

1. A failed model still holds its own resident slot from the original launch.
   With the primary at max_resident_models, the capacity check counted that
   slot and the model blocked its own restart forever.
2. Restart's stop() docker-rm's the dead container, deleting the only record
   of why it died. Logs must be saved before that.
"""
import os
import subprocess
from datetime import datetime
from types import SimpleNamespace

import pytest

from supervisor import restart_manager as rm_mod
from supervisor.models import (
    Backend,
    ModelConfig,
    ModelInstance,
    ModelStatus,
    ModelType,
    build_saved_config,
)
from supervisor.resources import ResourceManager


def _failed_gemma(container_id="c4795f04c88cb54d") -> ModelInstance:
    config = ModelConfig(
        model_name="google/gemma-4-E2B-it-qat-w4a16-ct",
        backend=Backend.VLLM,
        model_type=ModelType.CHAT,
        model_alias="gemma4-2b",
        host="primary",
    )
    return ModelInstance(
        id="gemma-4-e2b-it-qat-w4a16-ct-c8f311c8",
        model_name=config.model_name,
        model_alias="gemma4-2b",
        backend=Backend.VLLM,
        status=ModelStatus.FAILED,
        port=8004,
        gpu_ids=[0],
        base_url="http://127.0.0.1:8004",
        container_id=container_id,
        started_at=datetime.now(),
        memory_gb=16.0,
        saved_config=build_saved_config(config, gpu_ids=[0], port=8004, memory_gb=16.0),
    )


class _Registry:
    def __init__(self):
        self.updates = []

    async def update(self, model):
        self.updates.append(model.status)


class _Launcher:
    def __init__(self):
        self.stopped = 0
        self.launched = 0

    async def stop(self, instance):
        self.stopped += 1
        return True

    async def launch(self, config, model_id, port, memory_gb=None):
        self.launched += 1
        return SimpleNamespace(pid=None, container_id="new-container", started_at=datetime.now())


def _manager(monkeypatch, resident_others: int, mem_available_gb: float = 94.0):
    resources = ResourceManager()
    monkeypatch.setattr(resources, "get_mem_available_gb", lambda: mem_available_gb)
    model = _failed_gemma()
    # The failed model's own stale slot + the other live primary models.
    resources.model_memory_usage[model.id] = 16.0
    for i in range(resident_others):
        resources.model_memory_usage[f"other-{i}"] = 1.0
    launcher = _Launcher()
    manager = rm_mod.RestartManager(
        registry=_Registry(),
        launcher_factory=SimpleNamespace(get_launcher=lambda backend: launcher),
        resource_manager=resources,
    )
    # No docker in unit tests.
    monkeypatch.setattr(manager, "_preserve_crash_logs", lambda model: None)
    return manager, resources, launcher, model


async def test_failed_model_at_slot_cap_does_not_block_its_own_restart(monkeypatch):
    # 4 live models + the failed model's stale slot == max_resident_models (5).
    manager, resources, launcher, model = _manager(monkeypatch, resident_others=4)
    assert len(resources.model_memory_usage) == resources.max_resident_models == 5

    await manager._restart_model(model)

    assert launcher.launched == 1
    assert model.status == ModelStatus.STARTING
    assert resources.model_memory_usage[model.id] == 16.0
    assert len(resources.model_memory_usage) == 5


async def test_restart_still_refused_when_other_models_fill_every_slot(monkeypatch):
    # 5 OTHER live models: genuinely full, the restart must wait.
    manager, resources, launcher, model = _manager(monkeypatch, resident_others=5)

    await manager._restart_model(model)

    assert launcher.launched == 0
    assert model.id not in resources.model_memory_usage


def _fake_docker(calls, inspect_rc=0):
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "inspect":
            return subprocess.CompletedProcess(cmd, inspect_rc, "status=exited exit=255 oom_killed=false", "")
        return subprocess.CompletedProcess(cmd, 0, "", "EngineDeadError: boom\n")
    return run


def _bare_manager():
    return rm_mod.RestartManager(registry=None, launcher_factory=None, resource_manager=None)


def test_preserve_crash_logs_writes_state_and_stderr(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rm_mod.subprocess, "run", _fake_docker(calls))
    monkeypatch.setattr(rm_mod.settings, "crash_log_dir", str(tmp_path))

    path = _bare_manager()._preserve_crash_logs(_failed_gemma())

    assert path is not None and path.parent == tmp_path
    assert path.name.startswith("gemma4-2b-c4795f04c88c-")
    text = path.read_text()
    assert "exit=255" in text
    assert "EngineDeadError: boom" in text  # vLLM writes to stderr
    assert [c[1] for c in calls] == ["inspect", "logs"]


def test_preserve_crash_logs_skips_missing_container(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rm_mod.subprocess, "run", _fake_docker(calls, inspect_rc=1))
    monkeypatch.setattr(rm_mod.settings, "crash_log_dir", str(tmp_path))

    assert _bare_manager()._preserve_crash_logs(_failed_gemma()) is None
    assert list(tmp_path.iterdir()) == []
    assert _bare_manager()._preserve_crash_logs(_failed_gemma(container_id=None)) is None


def test_preserve_crash_logs_prunes_to_keep_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(rm_mod.subprocess, "run", _fake_docker([]))
    monkeypatch.setattr(rm_mod.settings, "crash_log_dir", str(tmp_path))
    monkeypatch.setattr(rm_mod.settings, "crash_log_keep", 2)
    for i in range(3):
        old = tmp_path / f"old-{i}.log"
        old.write_text("x")
        os.utime(old, (i, i))

    path = _bare_manager()._preserve_crash_logs(_failed_gemma())

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["old-2.log", path.name])


def test_preserve_crash_logs_never_raises(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("docker", 20)
    monkeypatch.setattr(rm_mod.subprocess, "run", boom)
    monkeypatch.setattr(rm_mod.settings, "crash_log_dir", str(tmp_path))

    assert _bare_manager()._preserve_crash_logs(_failed_gemma()) is None
