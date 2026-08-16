"""Live gateway smoke tests (auto-skipped when the gateway isn't up).

Regression for the 2026-08-15 desire-foundry incident: a reasoning model
(DSV4-Flash) with a modest max_tokens spent its ENTIRE budget on reasoning
and returned empty content with finish_reason=length. Root cause was the
DSpark stack's server-side default reasoning effort being set to `max`
(.env.dspark DEFAULT_THINKING) — requests that send no chat_template_kwargs
inherited unbounded-feeling reasoning (~12K tokens on complex structured
prompts). Fix: DEFAULT_THINKING=low; clients wanting deep reasoning pass
their own kwargs per request (pi does).

These tests assert the CONTRACT that fix restores: a structured-output
request with a 4K budget and no thinking kwargs must yield non-empty,
valid-JSON content from the default chat model.
"""
import json
import urllib.error
import urllib.request

import pytest

GATEWAY = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer dummy-key"}

# Deliberately reasoning-heavy prompt — with a `max`-effort default this
# reliably consumed >4096 tokens of pure reasoning.
PROMPT = (
    "Analyze the following business scenario thoroughly, considering at "
    "least 8 distinct factors, second-order effects, and counterarguments "
    "before concluding: A regional grocery chain with 40 stores is deciding "
    "whether to build its own delivery fleet or partner with a gig-economy "
    "platform. Margins are 2.1%, median basket $62, urban/suburban split 30/70."
)
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "analysis",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "factors": {"type": "array", "items": {"type": "string"}},
                "score": {"type": "number"},
            },
            "required": ["summary", "factors", "score"],
        },
    },
}


def _gateway_up() -> bool:
    try:
        req = urllib.request.Request(f"{GATEWAY}/v1/models", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _chat(payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{GATEWAY}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=HEADERS,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


live = pytest.mark.skipif(not _gateway_up(), reason="gateway not reachable on :8000")


@live
def test_structured_output_modest_budget_returns_content():
    """A 4K-budget structured request with NO thinking kwargs must produce
    non-empty valid-JSON content — the default reasoning effort has to leave
    room for the answer. Fails with empty content + finish_reason=length when
    the server default is max effort."""
    # 8192, not 4096: low-effort reasoning length is highly variable
    # (observed 1.9K-11K chars on identical prompts), so 4096 flakes at the
    # boundary. 8192 still discriminates the regression cleanly — under the
    # old `max` default this request consumed 12,043 tokens before content.
    body = _chat({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 8192,
        "response_format": SCHEMA,
    })
    choice = body["choices"][0]
    content = choice["message"].get("content") or ""
    assert choice["finish_reason"] == "stop", (
        f"finish_reason={choice['finish_reason']} — budget exhausted before "
        f"content (server default reasoning effort too high?)"
    )
    assert content, "empty content: reasoning consumed the whole budget"
    parsed = json.loads(content)
    assert set(parsed) >= {"summary", "factors", "score"}


@live
def test_explicit_thinking_kwargs_still_override_default():
    """Per-request chat_template_kwargs must keep overriding the server
    default (pi depends on this for xhigh reasoning)."""
    body = _chat({
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 64,
        "thinking_token_budget": 1,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"},
    })
    msg = body["choices"][0]["message"]
    assert (msg.get("content") or "").strip(), "explicit kwargs path broken"
