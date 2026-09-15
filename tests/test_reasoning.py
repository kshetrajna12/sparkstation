"""Reasoning-control normalization: clients express intent, the gateway
translates it into the loaded template's dialect AND vocabulary.

The vocabulary half matters because Qwen3.8-Flash-Next's chat template does not
ignore an unsupported effort - it raises:
    'Unexpected reasoning effort high. Supported types are xhigh, medium, low.'
so an un-mapped "high" from a client is a 400, not a degraded answer.
"""
import pytest

from gateway.reasoning import DialectResolver, normalize

pytestmark = pytest.mark.unit

FLASH_NEXT = dict(efforts=["low", "medium", "xhigh"],
                  effort_aliases={"high": "xhigh", "max": "xhigh", "minimal": "low"})


def _norm(body, **kw):
    return normalize(body, "qwen_enable_thinking", kw.get("efforts"), kw.get("effort_aliases"))


class TestKeyTranslation:
    def test_thinking_false_becomes_enable_thinking(self):
        """`thinking` is not a variable in this template; `enable_thinking` is.

        Sending `thinking: false` verbatim leaves enable_thinking undefined, and
        the template treats undefined as ON - the opposite of the client's ask.
        """
        body = {"chat_template_kwargs": {"thinking": False}}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_top_level_reasoning_bool_translated(self):
        body = {"reasoning": False}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        assert "reasoning" not in body

    def test_effort_rides_inside_chat_template_kwargs(self):
        """Top-level reasoning_effort is consumed by LiteLLM and never forwarded."""
        body = {"reasoning_effort": "low"}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"]["reasoning_effort"] == "low"
        assert "reasoning_effort" not in body

    def test_effort_implies_thinking_on(self):
        body = {"reasoning_effort": "medium"}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"]["enable_thinking"] is True


class TestEffortVocabulary:
    def test_high_maps_to_xhigh_instead_of_raising(self):
        body = {"reasoning_effort": "high"}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"]["reasoning_effort"] == "xhigh"

    def test_unknown_effort_degrades_not_raises(self):
        body = {"reasoning_effort": "turbo"}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"]["reasoning_effort"] == "medium"

    def test_supported_value_passes_through(self):
        body = {"reasoning_effort": "xhigh"}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"]["reasoning_effort"] == "xhigh"

    def test_no_vocabulary_declared_leaves_value_alone(self):
        """Dialects without a recorded vocabulary keep the old behaviour."""
        body = {"reasoning_effort": "high"}
        _norm(body)
        assert body["chat_template_kwargs"]["reasoning_effort"] == "high"

    def test_effort_off_word_disables_thinking(self):
        body = {"reasoning_effort": "none"}
        _norm(body, **FLASH_NEXT)
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        assert "reasoning_effort" not in body["chat_template_kwargs"]


class TestNoSignal:
    def test_request_without_any_signal_untouched(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        _norm(body, **FLASH_NEXT)
        assert body == {"messages": [{"role": "user", "content": "hi"}]}


class TestResolver:
    def test_flash_next_rule_matches_both_served_names(self, tmp_path):
        cfg = tmp_path / "reasoning.yaml"
        cfg.write_text(
            "dialects:\n"
            '  - match: "flash-next"\n'
            "    dialect: qwen_enable_thinking\n"
            "    efforts: [low, medium, xhigh]\n"
            "    effort_aliases: {high: xhigh}\n"
            'default: passthrough\n')
        lite = tmp_path / "litellm.yaml"
        lite.write_text(
            "model_list:\n"
            "  - model_name: default\n"
            "    litellm_params: {model: openai/qwen3.8-flash-next}\n"
            "  - model_name: rollback\n"
            "    litellm_params: {model: openai/qwen-flash-next}\n"
            "  - model_name: other\n"
            "    litellm_params: {model: openai/glm-5.3-flash}\n")
        r = DialectResolver(str(cfg), str(lite))
        for alias in ("default", "rollback"):
            dialect, efforts, aliases = r.policy_for(alias)
            assert dialect == "qwen_enable_thinking"
            assert efforts == ["low", "medium", "xhigh"]
            assert aliases["high"] == "xhigh"
        assert r.policy_for("other")[0] == "passthrough"
