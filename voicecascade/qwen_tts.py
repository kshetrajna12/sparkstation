"""Streaming TTS for the cascade: Faster-Qwen3-TTS (CustomVoice server).

Subclass of pipecat's OpenAITTSService that (a) skips the OpenAI voice-name
whitelist (our voices are Ryan/Aiden/... or cloned ones), and (b) speaks to
the local server's /v1/audio/speech with response_format=pcm (24 kHz mono
PCM16, streamed while generating — measured 0.46-0.56 s to first chunk).

Also holds FirstClauseAggregator: the text aggregator that decides WHEN a
piece of the LLM stream is handed to the TTS server.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

from loguru import logger
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

from .echo_guard import RecentBotText
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Clause boundaries we are willing to speak on: the comma family plus sentence
# punctuation (the parent only ever cuts on the latter, and only after NLTK).
CLAUSE_BOUNDARY = frozenset(",;:—–.!?…")


class FirstClauseAggregator(SimpleTextAggregator):
    """Speak the FIRST clause of a response instead of waiting for its first
    full sentence.

    pipecat's SimpleTextAggregator only releases text to the TTS at a sentence
    boundary (confirmed by NLTK, after a non-whitespace lookahead char), so the
    first audible word waits on the whole opening sentence — ~0.3 s of dead air
    at the front of every turn, where it is most audible. Here the first
    aggregation of each response is cut early, at the first clause boundary
    after MIN_WORDS words (or hard-cut at MAX_WORDS); everything after it falls
    back to the parent's sentence behaviour, which chunks better for prosody.

    Env: CASCADE_TTS_FIRST_MIN_WORDS (4), CASCADE_TTS_FIRST_MAX_WORDS (9),
    CASCADE_TTS_FIRST_CLAUSE=0 to disable (exactly the parent's behaviour).
    """

    def __init__(self, *, min_words: int | None = None, max_words: int | None = None,
                 enabled: bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self._min_words = min_words if min_words is not None else _env_int("CASCADE_TTS_FIRST_MIN_WORDS", 4)
        self._max_words = max_words if max_words is not None else _env_int("CASCADE_TTS_FIRST_MAX_WORDS", 9)
        self._enabled = enabled if enabled is not None else os.environ.get("CASCADE_TTS_FIRST_CLAUSE", "1") != "0"
        self._first_pending = True

    def _early_cut(self) -> str | None:
        """Prefix of self._text to speak now, or None. Consumes the prefix."""
        text = self._text
        n_words = len(text.split())
        last = text[-1]
        if last in CLAUSE_BOUNDARY and n_words >= self._min_words:
            # "...it is 3." is a decimal or an abbreviation as often as a clause.
            if not (last == "." and len(text) > 1 and text[-2].isdigit()):
                self._text = ""
                return text
        if n_words >= self._max_words:
            i = len(text) - 1
            while i >= 0 and not text[i].isspace():
                i -= 1
            if i > 0:
                self._text = text[i:]
                return text[:i]
        return None

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Same char-by-char loop as the parent, with the first-clause rule
        layered on top while the first aggregation is still pending."""
        if not self._enabled or self._aggregation_type == AggregationType.TOKEN:
            async for aggregation in super().aggregate(text):
                yield aggregation
            return

        for char in text:
            self._text += char

            if self._first_pending:
                early = self._early_cut()
                if early:
                    self._first_pending = False
                    self._needs_lookahead = False
                    yield Aggregation(text=early.strip(" "), type=AggregationType.SENTENCE)
                    continue

            result = await self._check_sentence_with_lookahead(char)
            if result:
                self._first_pending = False
                yield result

    async def flush(self) -> Aggregation | None:
        """End of an LLM response: the next one starts a new first clause."""
        result = await super().flush()
        self._first_pending = True
        return result

    async def handle_interruption(self):
        await super().handle_interruption()
        self._first_pending = True

    async def reset(self):
        await super().reset()
        self._first_pending = True


class QwenTTSService(OpenAITTSService):
    async def run_tts(self, text: str, context_id: str):
        # Echo guard learns the bot's words HERE, at synthesis start: this is
        # >=0.6 s ahead of any acoustic echo of them (TTS first audio 0.35 s +
        # speaker->mic path + Kyutai's 0.5 s text delay). TTSTextFrame is
        # emitted too late for the first fragment.
        RecentBotText.record(text)
        voice = self._settings.voice or "Ryan"
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)
            async with self._client.audio.speech.with_streaming_response.create(
                input=text,
                model=self._settings.model or "tts-1",
                voice=voice,  # no whitelist — server-defined voices
                response_format="pcm",
            ) as r:
                if r.status_code != 200:
                    yield ErrorFrame(error=f"QwenTTS status {r.status_code}: {await r.text()}")
                    return
                first = True
                async for chunk in r.iter_bytes(8192):
                    if not chunk:
                        continue
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    yield TTSAudioRawFrame(
                        audio=chunk, sample_rate=self.sample_rate, num_channels=1,
                        context_id=context_id,
                    )
        except Exception as exc:
            logger.exception("QwenTTS failed: {}", exc)
            yield ErrorFrame(error=f"QwenTTS: {exc}")
        finally:
            yield TTSStoppedFrame(context_id=context_id)
