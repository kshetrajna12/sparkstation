"""Per-turn brain routing for the cascade: gemma for the quick lane, qwen for
the think lane — decided from the user's utterance at ~zero cost (rules).

The gateway serves both aliases; routing is just choosing `model` per request.
Escalation philosophy (K): the slow brain STREAMS too — routing replaces the
speaker, it never makes a fast model wait on a slow one's full answer.
"""
from __future__ import annotations

import re

from loguru import logger

from . import metrics
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

# Utterances that deserve the big brain: analytical/expository asks, multi-step
# requests, or anything the user explicitly wants thought through.
# NB: no bare "how"/"write" — casual voice questions ("how many eggs...")
# must stay in the fast lane; latency is the product.
_THINK = re.compile(
    r"\b(why|explain|compare|plan|analy[sz]e|think|debug|design|code|"
    r"summari[sz]e|difference|prove|calculate|step by step|in detail)\b", re.I)


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

    async def get_chat_completions(self, context: LLMContext, *args, **kwargs):
        """TOOLTRACE: on turns with tools registered, log every streamed chunk
        at the parse boundary (content / tool_call deltas / finish_reason) so a
        dropped call names its layer: model-as-text vs parser vs forwarder."""
        stream = await super().get_chat_completions(context, *args, **kwargs)
        tools = context.tools
        n_tools = len(getattr(tools, "standard_tools", []) or []) if tools else 0
        if not n_tools:
            return stream
        logger.info("TOOLTRACE request: model={} tools={}", self._settings.model, n_tools)

        async def traced():
            i = 0
            try:
                async for chunk in stream:
                    i += 1
                    ch = chunk.choices[0] if chunk.choices else None
                    d = ch.delta if ch else None
                    if d is not None and d.tool_calls:
                        for tc in d.tool_calls:
                            fn = tc.function
                            logger.info("TOOLTRACE #{} tool_call idx={} id={} name={!r} args={!r}",
                                        i, tc.index, tc.id, fn.name if fn else None,
                                        (fn.arguments if fn else None))
                    elif d is not None and d.content:
                        logger.info("TOOLTRACE #{} content={!r}", i, d.content[:120])
                    elif d is not None and getattr(d, "reasoning_content", None):
                        logger.info("TOOLTRACE #{} reasoning={!r}", i, str(d.reasoning_content)[:80])
                    if ch is not None and ch.finish_reason:
                        logger.info("TOOLTRACE #{} finish_reason={}", i, ch.finish_reason)
                    yield chunk
            finally:
                logger.info("TOOLTRACE stream end after {} chunks", i)

        class _Wrapped:
            """Quacks like the openai AsyncStream pipecat expects (aiter + close)."""
            def __init__(self, gen, inner):
                self._gen, self._inner = gen, inner
            def __aiter__(self):
                return self._gen.__aiter__()
            async def close(self):
                await self._gen.aclose()
                close = getattr(self._inner, "close", None)
                if close:
                    await close()

        return _Wrapped(traced(), stream)

    async def _process_context(self, context: LLMContext):
        text = self._last_user_text(context)
        model = self._pick(text)
        if model != self._settings.model:
            logger.info("Router: {} -> {} ({!r})", self._settings.model, model, text[:60])
            self._settings.model = model  # read per-request by get_chat_completions
        # Voice is latency-critical: reasoning burns seconds before the first
        # audible word, so suppress it even on the think lane (the win there
        # is the bigger model, not chain-of-thought).
        self._settings.extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if model == self._think else {}
        metrics.turn_routed("think" if model == self._think else "fast")
        await super()._process_context(context)
