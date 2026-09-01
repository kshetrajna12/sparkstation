"""Echo suppression for clients without acoustic echo cancellation.

The problem (seen in production logs): the bot's own speech leaves the
client's speaker, comes back in through its microphone, and the pipeline
treats it as the user — the turn-start strategy interrupts the bot mid
sentence, then the STT transcribes the echo ("It has a", "population") and
the brain dutifully answers its own words.

The fix here is text-level, not acoustic: everything the TTS says is recorded
(see `RecentBotText`), and any transcript that is mostly made of words the bot
just said is dropped before it reaches the user aggregator. Bot text is fed in
by `CascadeMetricsTap` in bot.py, which sits between TTS and transport.output()
and therefore sees every `TTSTextFrame` plus the transport's
`BotStarted/StoppedSpeakingFrame`s.

Two thresholds, because the risk is asymmetric:
  * while the bot is speaking (or within ECHO_TAIL_S of stopping) even a
    single matching word is echo — the user has had no chance to say it;
  * once the bot is quiet, only a long, near-verbatim repeat counts, since a
    user may legitimately echo the bot ("forty thousand? really?").

Env knobs:
  CASCADE_ECHO_GUARD   (default "1")    "0" disables the guard entirely
  CASCADE_ECHO_WINDOW_S (default "12")  how long bot text stays comparable
  CASCADE_ECHO_TAIL_S  (default "1.5")  speaker/mic tail after the bot stops
"""
from __future__ import annotations

import os
import re
import time

from loguru import logger
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


ECHO_WINDOW_S = _env_float("CASCADE_ECHO_WINDOW_S", 12.0)
ECHO_TAIL_S = _env_float("CASCADE_ECHO_TAIL_S", 1.5)
ECHO_RATIO_SPEAKING = _env_float("CASCADE_ECHO_RATIO_SPEAKING", 0.5)  # bot speaking / just stopped: 1 word is enough (echo transcripts are noisy)
ECHO_RATIO_QUIET = 0.85      # bot quiet: near-verbatim and >= 3 words only
ECHO_MIN_WORDS_QUIET = 3

_WORD_RE = re.compile(r"[^\w']+", re.UNICODE)


def normalize(text: str) -> list[str]:
    """lowercase, drop possessives/apostrophes, split on non-word chars."""
    t = (text or "").lower().replace("\u2019", "'")
    t = re.sub(r"'s\b", "", t).replace("'", "")
    return [w for w in _WORD_RE.split(t) if w]


class _RecentBotText:
    """Rolling record of what the bot has said, and whether it is speaking."""

    def __init__(self):
        self._entries: list[tuple[float, list[str]]] = []
        self._speaking = False
        self._stopped_at = 0.0
        self.last_ratio = 0.0
        self.last_fresh = False

    # -- recording side (fed by CascadeMetricsTap) ---------------------------
    def record(self, text: str) -> None:
        words = normalize(text)
        if not words:
            return
        self._entries.append((time.monotonic(), words))
        self._prune()

    def speaking(self, flag: bool) -> None:
        if not flag and self._speaking:
            self._stopped_at = time.monotonic()
        self._speaking = bool(flag)

    def reset(self) -> None:
        self._entries.clear()
        self._speaking = False
        self._stopped_at = 0.0

    # -- query side ----------------------------------------------------------
    @property
    def bot_speaking(self) -> bool:
        return self._speaking

    def _prune(self) -> None:
        cutoff = time.monotonic() - ECHO_WINDOW_S
        self._entries = [e for e in self._entries if e[0] >= cutoff]

    def _bag(self) -> dict[str, int]:
        """Multiset of bot words still inside the window (oldest first)."""
        self._prune()
        bag: dict[str, int] = {}
        for _, words in self._entries:
            for w in words:
                bag[w] = bag.get(w, 0) + 1
        return bag

    def is_echo(self, text: str) -> bool:
        words = normalize(text)
        if not words:
            return False
        bag = self._bag()
        if not bag:
            return False
        # bag-of-words match fraction, honouring multiplicity
        # Echo transcripts are noisy ("earth's tilted" -> "earth still"): a
        # word also matches a bot word sharing a >=4-char prefix.
        left = dict(bag)
        hits = 0
        for w in words:
            key = w if left.get(w, 0) else next(
                (b for b, n in left.items()
                 if n and len(w) >= 4 and len(b) >= 4 and b[:4] == w[:4]), None)
            if key is not None:
                left[key] -= 1
                hits += 1
        ratio = hits / len(words)
        self.last_ratio = ratio

        now = time.monotonic()
        fresh = self._speaking or (now - self._stopped_at) < ECHO_TAIL_S
        self.last_fresh = fresh
        if fresh:
            return ratio >= ECHO_RATIO_SPEAKING
        return len(words) >= ECHO_MIN_WORDS_QUIET and ratio >= ECHO_RATIO_QUIET


RecentBotText = _RecentBotText()


def _enabled() -> bool:
    return (os.environ.get("CASCADE_ECHO_GUARD", "1") or "1").strip() not in ("0", "false", "no")


class EchoGuard(FrameProcessor):
    """Drops transcripts that are the bot hearing itself.

    Sits between the STT service and the user aggregator, so a suppressed
    transcript reaches neither the turn-start strategy (no interruption) nor
    the LLM context (no self-answering). Every other frame passes through.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._enabled = _enabled()
        if not self._enabled:
            logger.info("EchoGuard disabled (CASCADE_ECHO_GUARD=0)")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self._enabled and isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            kind = "interim" if isinstance(frame, InterimTranscriptionFrame) else "final"
            if RecentBotText.is_echo(frame.text):
                logger.info("EchoGuard: suppressed {} {!r} (bot_speaking={}, ratio={:.2f})",
                            kind, frame.text, RecentBotText.bot_speaking, RecentBotText.last_ratio)
                _count_suppressed()
                return
            if RecentBotText.last_fresh:
                logger.info("EchoGuard: pass {} {!r} (bot_speaking={}, ratio={:.2f})",
                            kind, frame.text, RecentBotText.bot_speaking, RecentBotText.last_ratio)
        await self.push_frame(frame, direction)


def _count_suppressed() -> None:
    try:
        from . import metrics
        hook = getattr(metrics, "echo_suppressed", None)
    except ImportError:
        hook = None
    if hook:
        hook()
