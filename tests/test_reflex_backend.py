"""Unit tests for the reflex decision-model backend (no GPU, no network).

Covers the three seams that wire reflex into sparkstation:
  * the launcher's `docker run` argv (env/volume mapping from models.yaml),
  * the gateway proxy's /v1/systemone target resolution,
  * gateway_sync / CLI keeping model_type "decision" out of LiteLLM.
"""
import pytest

from gateway.proxy import _resolve_decision_target
from supervisor.launchers.base import LaunchError
from supervisor.launchers.reflex_launcher import DEFAULT_IMAGE, build_docker_cmd
from supervisor.models import Backend, ModelConfig, ModelType

pytestmark = pytest.mark.unit


def _cfg(**extra_args):
    return ModelConfig(
        model_name="Qwen/Qwen3.5-4B",
        backend=Backend.REFLEX,
        model_type=ModelType.DECISION,
        model_alias="reflex",
        host="primary",
        quantization="none",
        extra_args=extra_args,
    )


def _env(cmd):
    return dict(a.split("=", 1) for f, a in zip(cmd, cmd[1:]) if f == "-e")


def _volumes(cmd):
    return [a for f, a in zip(cmd, cmd[1:]) if f == "-v"]


class TestDockerCmd:
    def test_minimal_config(self):
        cmd = build_docker_cmd(_cfg(), "qwen3.5-4b-abc12345", 8007)
        assert cmd[:3] == ["docker", "run", "-d"]
        assert cmd[-1] == DEFAULT_IMAGE
        assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "8007:8000"
        assert cmd[cmd.index("--name") + 1] == "sparkstation-qwen3.5-4b-abc12345"
        env = _env(cmd)
        assert env["MODEL_PATH"] == "Qwen/Qwen3.5-4B"
        assert env["SERVED_MODEL_NAME"] == "reflex"
        assert "REFLEX_ADAPTER" not in env and "REFLEX_CALIBRATION" not in env
        # the HF hub cache is always mounted so the checkpoint is not re-downloaded
        assert any(v.endswith(":/root/.cache/huggingface") for v in _volumes(cmd))

    def test_adapter_and_calibration_live_under_runs_mount(self):
        cmd = build_docker_cmd(
            _cfg(runs_dir="/srv/reflex/runs", adapter="lora-mix", calibration="lora-mix/calibration.json",
                 max_pack_tokens=4096, max_image_pixels=512 * 512),
            "id", 8010,
        )
        env = _env(cmd)
        assert env["REFLEX_ADAPTER"] == "/runs/lora-mix"
        assert env["REFLEX_CALIBRATION"] == "/runs/lora-mix/calibration.json"
        assert env["REFLEX_MAX_PACK_TOKENS"] == "4096"
        assert env["REFLEX_MAX_IMAGE_PIXELS"] == str(512 * 512)
        assert "/srv/reflex/runs:/runs:ro" in _volumes(cmd)

    def test_adapter_without_runs_dir_is_a_config_error(self):
        with pytest.raises(LaunchError):
            build_docker_cmd(_cfg(adapter="lora-mix"), "id", 8010)

    def test_tilde_runs_dir_is_expanded_for_the_docker_host(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        cmd = build_docker_cmd(_cfg(runs_dir="~/reflex/runs", calibration="c.json"), "id", 8010)
        assert "/home/tester/reflex/runs:/runs:ro" in _volumes(cmd)

    def test_docker_image_override_and_env_passthrough(self):
        cfg = _cfg()
        cfg.docker_image = "reflex-server:next"
        cfg.env_vars = {"REFLEX_WARMUP": "0"}
        cmd = build_docker_cmd(cfg, "id", 8010)
        assert cmd[-1] == "reflex-server:next"
        assert _env(cmd)["REFLEX_WARMUP"] == "0"


MODELS = [
    {"alias": "qwen-flash-next", "model_name": "qwen-flash-next", "model_type": "chat",
     "status": "running", "base_url": "http://w1:8888", "is_default": True},
    {"alias": "reflex", "model_name": "Qwen/Qwen3.5-4B", "model_type": "decision",
     "status": "running", "base_url": "http://127.0.0.1:8007"},
]


class TestSystemoneResolution:
    def test_model_unset_picks_the_loaded_decision_model(self):
        assert _resolve_decision_target({"state": "x", "questions": {}}, MODELS) == ("reflex", "http://127.0.0.1:8007")

    def test_jev_sdk_default_name_is_a_wildcard(self):
        assert _resolve_decision_target({"model": "reflex-latest"}, MODELS)[0] == "reflex"

    def test_explicit_alias_or_model_name(self):
        assert _resolve_decision_target({"model": "reflex"}, MODELS)[0] == "reflex"
        assert _resolve_decision_target({"model": "Qwen/Qwen3.5-4B"}, MODELS)[0] == "reflex"

    def test_chat_model_is_never_a_systemone_target(self):
        assert _resolve_decision_target({"model": "qwen-flash-next"}, MODELS) is None
        assert _resolve_decision_target({"model": "default"}, MODELS) is None

    def test_no_decision_model_loaded(self):
        assert _resolve_decision_target({}, [MODELS[0]]) is None
        assert _resolve_decision_target(None, []) is None

    def test_prefers_running_over_starting_over_suspended(self):
        stale = dict(MODELS[1], alias="reflex-old", status="suspended", base_url="http://127.0.0.1:8001")
        booting = dict(MODELS[1], alias="reflex-new", status="starting", base_url="http://127.0.0.1:8002")
        assert _resolve_decision_target({}, [stale, booting])[0] == "reflex-new"
        assert _resolve_decision_target({}, [stale, booting, MODELS[1]])[0] == "reflex"
        # a suspended one is still returned when it is all there is, so the
        # proxy's auto-resume path gets a chance
        assert _resolve_decision_target({}, [stale])[0] == "reflex-old"


class TestLitellmExclusion:
    async def test_gateway_sync_skips_decision_models(self, monkeypatch, tmp_path):
        from supervisor.gateway_sync import GatewaySync
        from supervisor.models import HealthStatus, ModelInstance, ModelStatus
        from datetime import datetime

        def inst(alias, mt, port):
            return ModelInstance(
                id=alias, model_name=alias, model_alias=alias, backend=Backend.VLLM, model_type=mt,
                status=ModelStatus.RUNNING, health_status=HealthStatus.HEALTHY, port=port, gpu_ids=[0],
                base_url=f"http://127.0.0.1:{port}", started_at=datetime.now(),
            )

        class Reg:
            async def list_running(self):
                return [inst("gemma4-2b", ModelType.CHAT, 8002), inst("reflex", ModelType.DECISION, 8007)]

        captured = {}

        async def fake_write(self, model_list):
            captured["list"] = model_list

        monkeypatch.setattr(GatewaySync, "_fallback_yaml_reload", fake_write)
        sync = GatewaySync(Reg(), default_model_alias="gemma4-2b")
        await sync.sync_models()
        names = [m["model_name"] for m in captured["list"]]
        assert "reflex" not in names
        assert names == ["gemma4-2b", "default"]
