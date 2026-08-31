"""Cascade voice bot: streaming STT -> routed brain -> streaming TTS.

Same Pipecat runner and /ws-client protobuf contract as the VoiceChat bot it
is built to replace: browser playground over SmallWebRTC at /client/, plain
WebSocket for LAN bridges at /ws-client, continuous paced audio in, RTVI
messages for control. Speech starts on the brain's FIRST sentence — the TTS
speaks while the LLM is still streaming (no full-answer wait, by design).

Run: .venv/bin/python -m voicecascade.bot -t webrtc --host 0.0.0.0 --port 7860

Session config (RTVI client-message, BEFORE client-ready, all optional):
  {"t": "configure", "d": {"system_instruction": "...",   # OpenClaw-owned
                            "voice": "Ryan",              # TTS voice slot
                            "engine": "clone"|"stock"|"design",  # which TTS server
                            "brain": "auto"|"gemma4-2b"|"default"}}
Ack: server-message {"type": "configured", "data": {...}}.

Default voice: the Console writes {"default": {"voice", "engine"}} to
<CASCADE_VOICE_CONFIG_DIR>/console.json (the TTS registry dir); it is read at
every session start, falling back to CASCADE_VOICE / the clone engine. Engine
URLs: CASCADE_TTS_CLONE / CASCADE_TTS_STOCK / CASCADE_TTS_DESIGN.
"""
from __future__ import annotations

import os
import time

import warnings

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InputAudioRawFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.runner.types import (
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from . import metrics
from .kyutai_stt import KyutaiSTTService
from .qwen_tts import QwenTTSService
from .router_llm import RouterLLMService

GATEWAY = os.environ.get("CASCADE_GATEWAY", "http://192.168.101.10:8000/v1")
TTS_URL = os.environ.get("CASCADE_TTS", "http://127.0.0.1:8023/v1")  # VoiceClone server
# The three Qwen3-TTS servers side by side (see voice.py in the supervisor for
# the engine table). A voice id is only meaningful together with its engine.
TTS_ENGINES = {
    "clone": os.environ.get("CASCADE_TTS_CLONE", TTS_URL),
    "stock": os.environ.get("CASCADE_TTS_STOCK", "http://127.0.0.1:8024/v1"),
    "design": os.environ.get("CASCADE_TTS_DESIGN", "http://127.0.0.1:8025/v1"),
}
VOICE_CONFIG_DIR = os.path.expanduser(os.environ.get("CASCADE_VOICE_CONFIG_DIR", "~/cascade-tts/config"))


def default_voice() -> tuple[str, str]:
    """(voice, engine) for a new session: console.json (written by the
    Sparkstation Console's Voice Studio) else CASCADE_VOICE on the clone engine."""
    fallback = (os.environ.get("CASCADE_VOICE", "K"), "clone")
    try:
        import json
        with open(os.path.join(VOICE_CONFIG_DIR, "console.json")) as f:
            d = (json.load(f) or {}).get("default") or {}
        voice, engine = d.get("voice"), d.get("engine", "clone")
        if isinstance(voice, str) and voice.strip() and engine in TTS_ENGINES:
            return voice.strip(), engine
    except (OSError, ValueError):
        pass
    return fallback
FAST_BRAIN = os.environ.get("CASCADE_FAST_BRAIN", "gemma4-2b")
THINK_BRAIN = os.environ.get("CASCADE_THINK_BRAIN", "default")

DEV_SYSTEM_INSTRUCTION = (
    "You are a friendly home robot voice assistant (dev voice, not Sparky). "
    "Answer in one or two short spoken-style sentences. Accuracy over "
    "confidence: if you are not sure, say so instead of guessing."
)

WS_CLOSE_BUSY = 4429
_active = 0


def _env_float(name, default):
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class IngressPacer(FrameProcessor):
    """Hold WS client audio to real time (same rationale as the VoiceChat bot:
    overspeed clients build unbounded backlog). Drops silent frames when >0.5s
    ahead."""

    MAX_LEAD_S = 0.5
    SILENCE_PEAK = 300

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._t0 = None
        self._audio_s = 0.0
        self._warned = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and frame.audio:
            import array
            now = time.monotonic()
            rate = frame.sample_rate or 16_000
            dur = len(frame.audio) / (rate * 2 * (frame.num_channels or 1))
            if self._t0 is None:
                self._t0 = now
            self._audio_s += dur
            if self._audio_s - (now - self._t0) > self.MAX_LEAD_S:
                peak = max(abs(x) for x in array.array("h", frame.audio)) if len(frame.audio) % 2 == 0 else 0
                if peak < self.SILENCE_PEAK:
                    self._audio_s -= dur
                    if not self._warned:
                        logger.warning("IngressPacer: dropping silent frames (client ahead of real time)")
                        self._warned = True
                    return
        await self.push_frame(frame, direction)


class CascadeMetricsTap(FrameProcessor):
    """Pass-through that feeds every frame to the prometheus tap."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tap = metrics.MetricsTap()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self._tap.on_frame(frame)
        await self.push_frame(frame, direction)


def create_transport(runner_args):
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        return SmallWebRTCTransport(
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True, audio_out_enabled=True,
                audio_in_sample_rate=16_000, audio_out_sample_rate=24_000,
                audio_in_channels=1, audio_out_channels=1,
            ),
        )
    if isinstance(runner_args, WebSocketRunnerArguments):
        return FastAPIWebsocketTransport(
            websocket=runner_args.websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True, audio_out_enabled=True,
                audio_in_sample_rate=16_000, audio_out_sample_rate=24_000,
                audio_in_channels=1, audio_out_channels=1,
                add_wav_header=False,
                serializer=ProtobufFrameSerializer(),
            ),
        )
    raise TypeError("cascade bot supports SmallWebRTC and plain WebSocket transports")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = KyutaiSTTService(sample_rate=16_000)
    llm = RouterLLMService(
        fast_model=FAST_BRAIN, think_model=THINK_BRAIN,
        base_url=GATEWAY, api_key="dummy-key",
    )
    voice, engine = default_voice()  # K's pick 2026-08-30: cloned voice "K"
    tts = QwenTTSService(
        base_url=TTS_ENGINES[engine], api_key="dummy-key",
        voice=voice, model="tts-1", sample_rate=24_000,
    )
    context = LLMContext(
        messages=[{"role": "system", "content": DEV_SYSTEM_INSTRUCTION}]
    )
    user_agg = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=_env_float("CASCADE_VAD_CONFIDENCE", 0.7),
                min_volume=_env_float("CASCADE_VAD_MIN_VOLUME", 0.6),
                start_secs=_env_float("CASCADE_VAD_START_SECS", 0.2),
                stop_secs=_env_float("CASCADE_VAD_STOP_SECS", 0.2),
            )),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(cpu_count=1),
                    wait_for_transcript=True,
                )],
            ),
            user_turn_stop_timeout=_env_float("CASCADE_TURN_STOP_TIMEOUT", 4.0),
        ),
    )
    assistant_agg = LLMAssistantAggregator(context, _paired_user_aggregator=user_agg)

    pacer = [IngressPacer()] if isinstance(transport, FastAPIWebsocketTransport) else []
    pipeline = Pipeline([
        transport.input(),
        *pacer,
        stt,
        user_agg,
        llm,
        tts,
        CascadeMetricsTap(),
        transport.output(),
        assistant_agg,
    ])

    rtvi = RTVIProcessor()

    worker = PipelineWorker(
        pipeline,
        enable_rtvi=True,
        params=PipelineParams(
            audio_in_sample_rate=16_000, audio_out_sample_rate=24_000,
            enable_metrics=True, enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        rtvi_processor=rtvi,
    )

    async def forward_tool_call(params: FunctionCallParams):
        """Catch-all: every brain tool call is forwarded to the client as an
        RTVI llm-function-call message (contract v1.1 shapes). Deferred: the
        client's llm-function-call-result becomes a FunctionCallResultFrame
        via the RTVI processor, which completes the call and re-runs the LLM."""
        logger.info("Tool call -> client: {}({}) id={}",
                    params.function_name, params.arguments, params.tool_call_id)
        metrics.tool_call(params.function_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            await rtvi.handle_function_call(params)

    llm.register_function(None, forward_tool_call)

    @rtvi.event_handler("on_client_message")
    async def on_client_message(processor, message):
        if message.type == "register-tools":
            d = message.data or {}
            fns = []
            try:
                for t in d.get("tools") or []:
                    f = t.get("function", t)
                    p = f.get("parameters") or {}
                    fns.append(FunctionSchema(
                        name=f["name"], description=f.get("description", ""),
                        properties=p.get("properties", {}), required=p.get("required", []),
                    ))
            except (KeyError, TypeError, AttributeError) as e:
                await processor.send_server_message(
                    {"type": "tools-error", "data": {"error": f"bad tool spec: {e}"}})
                return
            # total replacement, same as contract v1.1
            context.set_tools(ToolsSchema(standard_tools=fns))
            si = d.get("system_instruction")
            if isinstance(si, str) and si.strip():
                context.set_messages([{"role": "system", "content": si.strip()}])
            logger.info("Tools registered: {} ({})", len(fns), [f.name for f in fns])
            await processor.send_server_message(
                {"type": "tools-registered",
                 "data": {"count": len(fns),
                          "system_instruction": "custom" if si else "default"}})
            return
        if message.type != "configure":
            return
        d = message.data or {}
        applied = {}
        si = d.get("system_instruction")
        if isinstance(si, str) and si.strip():
            context.set_messages([{"role": "system", "content": si.strip()}])
            applied["system_instruction"] = "custom"
        v = d.get("voice")
        if isinstance(v, str) and v.strip():
            tts._settings.voice = v.strip()
            applied["voice"] = v.strip()
        eng = d.get("engine")
        if isinstance(eng, str) and eng in TTS_ENGINES:
            # swap the TTS server for this session (openai client base_url is settable)
            tts._client.base_url = TTS_ENGINES[eng]
            applied["engine"] = eng
        b = d.get("brain")
        if isinstance(b, str) and b.strip() and b != "auto":
            llm._fast = llm._think = b.strip()
            applied["brain"] = b.strip()
        logger.info("Session configured: {}", applied)
        await processor.send_server_message({"type": "configured", "data": applied})

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_t, _c):
        logger.info("cascade client connected (brains: {}/{}, tts voice {} @ {})",
                    FAST_BRAIN, THINK_BRAIN, tts._settings.voice, tts._client.base_url)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_t, _c):
        logger.info("cascade client disconnected")
        await worker.cancel()

    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint,
        handle_sigterm=runner_args.handle_sigterm,
    )
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    global _active
    if isinstance(runner_args, WebSocketRunnerArguments) and _active > 0:
        logger.warning("Rejecting WS connection: session active (4429)")
        metrics.session_started(rejected=True)
        await runner_args.websocket.close(code=WS_CLOSE_BUSY, reason="session-busy; retry")
        return
    transport = create_transport(runner_args)
    _active += 1
    metrics.session_started()
    try:
        await run_bot(transport, runner_args)
    finally:
        _active -= 1
        metrics.session_ended()


if __name__ == "__main__":
    from pipecat.runner.run import main

    # Load the STT engine before accepting sessions: first-session utterances
    # must not race a 19 s model load.
    from .kyutai_stt import preload
    preload()
    metrics.start_metrics_server()
    main()
