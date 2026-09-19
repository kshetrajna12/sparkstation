"""Live /v1/systemone smoke test through the gateway (auto-skipped unless the
gateway AND a decision model (reflex) are up).

Asserts the contract a client can rely on: the proxy forwards the Jev-shaped
request to the loaded decision model, the answer carries every question with
a proper distribution, and a second call over the same state hits the server's
state cache (cheaper, identical numbers).
"""
import json
import os
import urllib.error
import urllib.request

import pytest

GATEWAY = "http://127.0.0.1:8000"
_key = os.environ.get("SPARK_KEY") or next(
    (l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("GATEWAY_LOCAL_KEY=")), "missing-key"
) if os.path.exists(".env") else "missing-key"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {_key}"}

REQUEST = {
    "state": {"ticket": "My payouts have failed three times this week and nobody has replied to my emails."},
    "questions": {
        "queue": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"payments": "payouts, refunds, invoices", "account": "login, 2FA", "other": None}},
        "escalate": {"type": "noul", "instructions": "Should this be escalated to a manager?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["can wait a week", "handle today", "blocked right now"]},
    },
}


def _post(body):
    req = urllib.request.Request(f"{GATEWAY}/v1/systemone", data=json.dumps(body).encode(), headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except Exception:
        return None, None


def _decision_model_up() -> bool:
    try:
        with urllib.request.urlopen(f"{GATEWAY}/health", timeout=3) as r:
            if r.status != 200:
                return False
        with urllib.request.urlopen("http://127.0.0.1:9001/models/detailed", timeout=3) as r:
            models = json.loads(r.read()).get("models", [])
        return any(m.get("model_type") == "decision" and m.get("status") == "running" for m in models)
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _decision_model_up(), reason="gateway or decision model (reflex) not running"),
]


def test_systemone_answers_every_question_with_distributions():
    status, out = _post(REQUEST)
    assert status == 200, out
    answers = out["answers"]
    assert set(answers) == set(REQUEST["questions"])
    q = answers["queue"]
    assert q["type"] == "choice" and q["choice"] in REQUEST["questions"]["queue"]["criteria"]
    assert abs(sum(q["probabilities"].values()) - 1.0) < 1e-3
    assert 0.0 <= answers["escalate"]["noul"] <= 1.0
    assert 0.0 <= answers["urgency"]["score"] <= 2.0
    # a payouts complaint routes to payments with real confidence
    assert q["choice"] == "payments" and q["probabilities"]["payments"] > 0.5
    assert out["usage"]["output_tokens"] == 0  # nothing is generated


def test_second_call_hits_state_cache_and_is_deterministic():
    _, first = _post(REQUEST)
    _, second = _post(REQUEST)
    assert second["usage"]["state_cache_hit"] is True
    assert first["answers"] == second["answers"]


def test_model_unset_and_reflex_latest_both_resolve():
    body = dict(REQUEST, model="reflex-latest")
    status, out = _post(body)
    assert status == 200, out
    assert out["model"]  # the served alias is echoed


def test_bad_question_is_a_422_not_a_502():
    body = dict(REQUEST, questions={"q": {"type": "choice", "instructions": "?", "criteria": {"only-one": None}}})
    status, out = _post(body)
    assert status == 422, out
