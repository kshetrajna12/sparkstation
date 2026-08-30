"""Per-turn brain routing for the cascade: gemma for the quick lane, qwen for
the think lane — decided from the user's utterance at ~zero cost (rules).

The gateway serves both aliases; routing is just choosing `model` per request.
Escalation philosophy (K): the slow brain STREAMS too — routing replaces the
speaker, it never makes a fast model wait on a slow one's full answer.
"""
from __future__ import annotations

import re

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

# Utterances that deserve the big brain: analytical/expository asks, multi-step
# requests, or anything the user explicitly wants thought through.
_THINK = re.compile(
    r"\b(why|how|explain|compare|plan|analy[sz]e|think|debug|design|write|code|"
    r"summari[sz]e|difference|history|prove|calculate|convert)\b", re.I)


class RouterLLMService(OpenAILLMService):
    def __init__(self, *, fast_model: str, think_model: str, think_word_threshold: int = 22, **kwargs):
        super().__init__(model=fast_model, **kwargs)
        self._fast = fast_model
        self._think = think_model
        self._threshold = think_word_threshold

    def _last_user_text(self, context: LLMContext) -> str:
        for m in reversed(context.get_messages()):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "user":
                c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                return str(c or "")
        return ""

    def _pick(self, text: str) -> str:
        if len(text.split()) > self._threshold or _THINK.search(text):
            return self._think
        return self._fast

    async def _process_context(self, context: LLMContext):
        text = self._last_user_text(context)
        model = self._pick(text)
        if model != self._settings.model:
            logger.info("Router: {} -> {} ({!r})", self._settings.model, model, text[:60])
            self._settings.model = model  # read per-request by get_chat_completions
        await super()._process_context(context)
