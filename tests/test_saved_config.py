"""
Relaunch fidelity: saved_config written at launch must reconstruct the exact
original ModelConfig. Auto-restart and resume both depend on this round trip —
a partial reconstruction historically relaunched worker-pinned models on the
primary host with the default Docker image and no memory budget.
"""
from supervisor.models import Backend, ModelConfig, ModelType, build_saved_config


def _full_config() -> ModelConfig:
    return ModelConfig(
        model_name="nvidia/Qwen3.6-35B-A3B-NVFP4",
        backend=Backend.VLLM,
        model_type=ModelType.CHAT,
        model_alias="qwen3.8-27b",
        host="worker1",
        num_gpus=1,
        quantization="modelopt",
        idle_timeout_minutes=60,
        auto_suspend_enabled=False,
        speculative_method="mtp",
        num_speculative_tokens=3,
        speculative_extra={"moe_backend": "triton"},
        extra_args={"max_model_len": 32768, "kv_cache_dtype": "fp8"},
        docker_image="vllm/vllm-openai:nightly",
        env_vars={"VLLM_USE_V1": "1"},
        volumes=["./patches:/patches"],
    )


def test_saved_config_round_trip_preserves_all_launch_fields():
    config = _full_config()
    saved = build_saved_config(config, gpu_ids=[0], port=8001, memory_gb=100.0)

    rebuilt = ModelConfig.from_saved_config(saved)

    assert rebuilt.model_name == config.model_name
    assert rebuilt.backend == config.backend
    assert rebuilt.model_type == config.model_type
    assert rebuilt.model_alias == config.model_alias
    assert rebuilt.host == "worker1"
    assert rebuilt.quantization == "modelopt"
    assert rebuilt.idle_timeout_minutes == 60
    assert rebuilt.auto_suspend_enabled is False
    assert rebuilt.speculative_method == "mtp"
    assert rebuilt.num_speculative_tokens == 3
    assert rebuilt.speculative_extra == {"moe_backend": "triton"}
    assert rebuilt.extra_args == config.extra_args
    assert rebuilt.docker_image == "vllm/vllm-openai:nightly"
    assert rebuilt.env_vars == {"VLLM_USE_V1": "1"}
    assert rebuilt.volumes == ["./patches:/patches"]


def test_saved_config_records_memory_budget():
    saved = build_saved_config(_full_config(), gpu_ids=[0], port=8001, memory_gb=100.0)
    assert saved["memory_gb"] == 100.0
    assert saved["port"] == 8001


def test_from_saved_config_tolerates_legacy_minimal_snapshot():
    # Rows written by old code lack model_type/host/docker_image/etc.
    legacy = {
        "model_name": "BAAI/bge-m3",
        "backend": "vllm",
        "model_alias": "bge-m3",
        "gpu_ids": [0],
        "port": 8002,
        "quantization": None,
        "auto_suspend_enabled": False,
        "idle_timeout_minutes": 30,
        "extra_args": {},
    }
    rebuilt = ModelConfig.from_saved_config(legacy)
    assert rebuilt.host == "primary"
    assert rebuilt.model_type == ModelType.CHAT
    assert rebuilt.docker_image is None
    assert rebuilt.env_vars == {}
