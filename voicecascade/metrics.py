"""Prometheus metrics for the cascade voice bot.

Import-safe without prometheus_client (all hooks become no-ops) so the bot
never fails to start over a missing metrics dep. Scraped via the supervisor's
http_sd (/prometheus/targets lists <host>:<metrics_port> for the voicecascade
backend), so targets follow placement — nothing here names a host.

Metric names are cascade_* (the voicechat_* family died with the legacy
runtime; dashboards query these).
"""
from __future__ import annotations

import os
import time

from loguru import logger

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _ENABLED = True
except ImportError:  # metrics are optional by design
    _ENABLED = False

if _ENABLED:
    SESSIONS_ACTIVE = Gauge("cascade_sessions_active", "Live voice sessions (0/1: single-session bot)")
    SESSIONS_TOTAL = Counter("cascade_sessions_total", "Voice sessions accepted since start")
    SESSIONS_REJECTED = Counter("cascade_sessions_rejected_total", "Connections rejected busy (4429)")
    TURNS = Counter("cascade_turns_total", "Completed user turns by brain lane", ["lane"])
    TOOL_CALLS = Counter("cascade_tool_calls_total", "Brain tool calls forwarded to the client", ["function"])
    LLM_TTFB = Histogram(
        "cascade_llm_ttfb_seconds", "Brain time-to-first-token", ["lane"],
        buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8))
    TTS_TTFB = Histogram(
        "cascade_tts_ttfb_seconds", "TTS time-to-first-audio per sentence",
        buckets=(0.25, 0.5, 0.75, 1, 1.5, 2, 4))
    VOICE_TO_VOICE = Histogram(
        "cascade_voice_to_voice_seconds", "User speech end -> first bot audio",
        buckets=(1, 1.5, 2, 2.5, 3, 4, 6, 10))
    STT_FINALS = Counter("cascade_stt_finals_total", "Finalized user transcripts")


def start_metrics_server() -> None:
    port = int(os.environ.get("CASCADE_METRICS_PORT", "7861"))
    if not _ENABLED:
        logger.warning("prometheus_client not installed; cascade metrics disabled")
        return
    start_http_server(port)
    logger.info("cascade metrics on :{}", port)


def session_started(rejected: bool = False) -> None:
    if not _ENABLED:
        return
    if rejected:
        SESSIONS_REJECTED.inc()
    else:
        SESSIONS_TOTAL.inc()
        SESSIONS_ACTIVE.inc()


def session_ended() -> None:
    if _ENABLED:
        SESSIONS_ACTIVE.dec()


_last_lane = "fast"


def turn_routed(lane: str) -> None:
    global _last_lane
    _last_lane = lane
    if _ENABLED:
        TURNS.labels(lane=lane).inc()


def tool_call(function_name: str) -> None:
    if _ENABLED:
        TOOL_CALLS.labels(function=function_name).inc()


class MetricsTap:
    """Frame tap placed between TTS and transport.output(): observes pipecat
    MetricsFrames (TTS TTFB) and computes voice-to-voice latency from
    UserStoppedSpeakingFrame -> first TTSAudioRawFrame. Not a FrameProcessor
    subclass to keep pipeline wiring explicit — see bot.py CascadeMetricsTap.
    """

    def __init__(self):
        self._speech_end: float | None = None

    def on_frame(self, frame) -> None:
        if not _ENABLED:
            return
        from pipecat.frames.frames import (
            MetricsFrame,
            TranscriptionFrame,
            TTSAudioRawFrame,
            UserStoppedSpeakingFrame,
        )
        from pipecat.metrics.metrics import TTFBMetricsData

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._speech_end = time.monotonic()
        elif isinstance(frame, TranscriptionFrame):
            STT_FINALS.inc()
        elif isinstance(frame, TTSAudioRawFrame):
            if self._speech_end is not None:
                VOICE_TO_VOICE.observe(time.monotonic() - self._speech_end)
                self._speech_end = None
        elif isinstance(frame, MetricsFrame):
            for d in frame.data or []:
                if isinstance(d, TTFBMetricsData) and "QwenTTSService" in (d.processor or ""):
                    TTS_TTFB.observe(d.value)
                elif isinstance(d, TTFBMetricsData) and "RouterLLMService" in (d.processor or ""):
                    # single-session bot: the last routed lane is this turn's
                    LLM_TTFB.labels(lane=_last_lane).observe(d.value)
