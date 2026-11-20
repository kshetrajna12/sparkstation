import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from supervisor import main, models_config
from supervisor.models import ModelStatus


@pytest.mark.asyncio
async def test_primary_aux_loads_during_worker_stack_boot_and_shutdown_cancels_task(monkeypatch):
    selected = {"glm5.3-flash", "gemma4-2b", "clip-vit"}
    models = [model for model in models_config.get_profile_models("deep") if model.alias in selected]
    assert {model.alias for model in models} == selected

    worker_started = asyncio.Event()
    worker_cancelled = asyncio.Event()
    aux_started = asyncio.Event()
    instances = {}

    async def launch(config, model_id, port, memory_gb):
        if config.model_alias == "glm5.3-flash":
            worker_started.set()
            try:
                await asyncio.Future()
            finally:
                worker_cancelled.set()
        if config.model_alias == "clip-vit":
            assert "gemma4-2b" in instances
            aux_started.set()
        return SimpleNamespace(id=config.model_alias, status=ModelStatus.RUNNING, gpu_ids=[0])

    async def register(instance):
        instances[instance.id] = instance

    fake_registry = SimpleNamespace(
        initialize=AsyncMock(),
        reconcile_state=AsyncMock(return_value=dict(fixed_stopped=0, fixed_running=0, removed_orphaned=0)),
        list_all=AsyncMock(side_effect=lambda: list(instances.values())),
        get_by_alias=AsyncMock(side_effect=instances.get),
        generate_id=lambda name: name,
        create=register,
        delete=AsyncMock(),
    )
    launcher = SimpleNamespace(launch=launch)
    manager = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(main, "ModelRegistry", lambda: fake_registry)
    monkeypatch.setattr(main, "ResourceManager", lambda: SimpleNamespace(allocate_model=lambda *a, **kw: 8001))
    monkeypatch.setattr(main, "LauncherFactory", lambda: SimpleNamespace(get_launcher=lambda backend: launcher))
    monkeypatch.setattr(main, "build_saved_config", lambda *a: {})
    for manager_name in ("AutoSuspendManager", "GatewaySync", "HealthCheckManager", "RestartManager"):
        monkeypatch.setattr(main, manager_name, lambda *a, **kw: manager)
    for name in ("registry", "resource_manager", "launcher_factory", "auto_suspend_manager", "gateway_sync",
                 "health_check_manager", "restart_manager", "default_model_alias", "vision_model_alias",
                 "vision_capable_aliases", "startup_complete"):
        monkeypatch.setattr(main, name, getattr(main, name))
    monkeypatch.setattr(main.settings, "startup_profile", "deep")
    monkeypatch.setattr(models_config, "get_profile_models", lambda profile: models)

    async with main.lifespan(FastAPI()):
        await asyncio.wait_for(worker_started.wait(), timeout=2)
        await asyncio.wait_for(aux_started.wait(), timeout=2)
        assert not worker_cancelled.is_set()
    assert worker_cancelled.is_set()
