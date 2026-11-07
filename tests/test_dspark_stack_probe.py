"""stack_probe: a 2-node stack's health_script decides health past the head's /health."""
import asyncio
from datetime import datetime

from supervisor.launchers.dspark_launcher import stack_probe
from supervisor.models import Backend, ModelInstance, ModelStatus


def _instance(tmp_path, script_body=None):
    extra = {"dspark_dir": str(tmp_path)}
    if script_body is not None:
        s = tmp_path / "health.sh"
        s.write_text("#!/usr/bin/env bash\n" + script_body)
        s.chmod(0o755)
        extra["health_script"] = "health.sh"
    return ModelInstance(id="m", model_name="m", backend=Backend.DSPARK, model_type="chat",
                         status=ModelStatus.RUNNING, port=8009, gpu_ids=[0],
                         base_url="http://x:8009", started_at=datetime.now(), extra_args=extra)


def test_no_script_is_healthy(tmp_path):
    assert asyncio.run(stack_probe(_instance(tmp_path))) == (True, "")


def test_script_ok(tmp_path):
    assert asyncio.run(stack_probe(_instance(tmp_path, "echo ok\n"))) == (True, "ok")


def test_script_failure_reports_last_line(tmp_path):
    ok, reason = asyncio.run(stack_probe(_instance(tmp_path, "echo checking\necho rank 1 down\nexit 1\n")))
    assert not ok and reason == "rank 1 down"


def test_missing_script_is_unhealthy(tmp_path):
    inst = _instance(tmp_path)
    inst.extra_args["health_script"] = "nope.sh"
    ok, reason = asyncio.run(stack_probe(inst))
    assert not ok and "nope.sh" in reason
