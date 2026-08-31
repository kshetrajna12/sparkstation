"""Streaming STT for the cascade voice pipeline: Kyutai stt-1b-en_fr via moshi.

Continuous (non-segmented) Pipecat STT service. Every InputAudioRawFrame's
PCM is resampled 16k->24k, buffered into 80 ms mimi frames, and stepped
through the model on a dedicated worker thread (GPU work must not block the
event loop). Decoded word pieces stream out as InterimTranscriptionFrames;
the aggregator's VAD/turn strategies decide turn boundaries, and on
UserStoppedSpeaking we flush the model's ~0.5 s text delay and emit the
final TranscriptionFrame for the turn.
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import time

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

FLUSH_AFTER_TURN_S = 0.9   # model text delay (~0.5s) + margin
TEXT_GAP_FINAL_S = float(os.environ.get("CASCADE_STT_FINAL_GAP", "0.7"))     # no new pieces for this long (with text pending) => final

# Process-global engine: the model loads ONCE per bot process (~19 s) and is
# reused across sessions (sessions are serial — single-session bot). Without
# this, the first utterance of every session raced the per-session load and
# transcribed nothing.
_ENGINE_LOCK = threading.Lock()
_ENGINE_IN: "queue.Queue|None" = None      # (kind, payload): ("audio", (bytes, sink)), ("reset", sink)
_ENGINE_STARTED = threading.Event()


def preload(device: str = "cuda", hf_repo: str = "kyutai/stt-1b-en_fr"):
    """Start the single global engine thread (loads model, owns ALL GPU calls).

    Every mimi/LM call for the life of the process happens on this one thread:
    load, boot self-test, and every session's stepping. This removes any
    thread-affinity questions (streaming state, cuda graphs, streams) that a
    per-session worker thread raised — sessions only exchange queues with it.
    """
    global _ENGINE_IN
    with _ENGINE_LOCK:
        if _ENGINE_IN is not None:
            return
        _ENGINE_IN = queue.Queue()
        threading.Thread(target=_engine_main, args=(device, hf_repo), daemon=True).start()
    _ENGINE_STARTED.wait(timeout=300)


def _engine_main(device: str, hf_repo: str):
    import julius
    import numpy as np
    import torch
    from moshi.models import loaders, LMGen

    logger.info("KyutaiSTT engine thread: loading {} ...", hf_repo)
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
    mimi = info.get_mimi(device=device)
    tokenizer = info.get_text_tokenizer()
    lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    lm_gen = LMGen(lm, temp=0, temp_text=0.0)
    CTX16, BLK16, CTX24, BLK24 = 160, 1280, 240, 1920
    assert BLK24 == mimi.frame_size

    def _reset():
        for m in (mimi, lm_gen):
            if hasattr(m, "reset_streaming"):
                m.reset_streaming()

    with mimi.streaming(1), lm_gen.streaming(1):
        # boot self-test on THIS thread — the same thread sessions will use
        try:
            import os as _os
            import wave
            if _os.path.exists("/tmp/eggs16k.wav"):
                w = wave.open("/tmp/eggs16k.wav")
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
                pcm = np.concatenate([np.zeros(16000, dtype=np.float32), pcm, np.zeros(48000, dtype=np.float32)])
                buf = torch.cat([torch.zeros(CTX16), torch.from_numpy(pcm)]); out = []
                while buf.shape[-1] >= CTX16 + BLK16 + CTX16:
                    win = buf[: CTX16 + BLK16 + CTX16]; buf = buf[BLK16:]
                    chunk = julius.resample_frac(win, 16000, 24000)[CTX24: CTX24 + BLK24]
                    t = lm_gen.step(mimi.encode(chunk.to(device)[None, None]))
                    if t is not None and t[0, 0, 0].item() not in (0, 3):
                        out.append(tokenizer.id_to_piece(t[0, 0, 0].item()).replace("▁", " "))
                logger.info("KyutaiSTT engine self-test: {!r}", "".join(out).strip())
                _reset()
        except Exception:
            logger.exception("engine self-test failed")
        logger.info("KyutaiSTT ready (engine thread)")
        _ENGINE_STARTED.set()

        buf16 = torch.zeros(CTX16)
        sink = None
        nstat = {"n": 0, "pad": 0, "tok": 0, "cmax": 0.0}
        while True:
            kind, payload = _ENGINE_IN.get()
            if kind == "reset":
                _reset()
                # Kyutai's stream needs ~1s of lead-in before it can decode
                # speech; prime with 2s of silence so an utterance that
                # arrives immediately after connect still transcribes.
                buf16 = torch.zeros(CTX16 + 32000)
                sink = payload
                continue
            pcm, s_sink = payload
            if s_sink is not sink:
                continue  # stale session audio
            audio = torch.from_numpy(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
            buf16 = torch.cat([buf16, audio])
            while buf16.shape[-1] >= CTX16 + BLK16 + CTX16:
                win = buf16[: CTX16 + BLK16 + CTX16]; buf16 = buf16[BLK16:]
                chunk = julius.resample_frac(win, 16000, 24000)[CTX24: CTX24 + BLK24]
                t = lm_gen.step(mimi.encode(chunk.to(device)[None, None]))
                nstat["n"] += 1; nstat["cmax"] = max(nstat["cmax"], float(chunk.abs().max()))
                if t is None:
                    continue
                v = t[0, 0, 0].item()
                if v in (0, 3):
                    nstat["pad"] += 1
                else:
                    nstat["tok"] += 1
                    piece = tokenizer.id_to_piece(v).replace("▁", " ")
                    if sink is not None:
                        sink(("text", piece))
                if nstat["n"] % 100 == 0:
                    logger.debug("KyutaiSTT engine steps {}: pad={} tok={} max24k={:.3f}",
                                 nstat["n"], nstat["pad"], nstat["tok"], nstat["cmax"])
                    nstat.update(pad=0, tok=0, cmax=0.0)


class KyutaiSTTService(STTService):
    def __init__(self, *, hf_repo: str = "kyutai/stt-1b-en_fr", device: str = "cuda", **kwargs):
        super().__init__(audio_passthrough=True, **kwargs)
        self._hf_repo = hf_repo
        self._device = device
        self._out_q: asyncio.Queue = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn_text: list[str] = []
        self._last_piece_at = 0.0
        self._flush_task: asyncio.Task | None = None
        # one stable callable per service: the engine compares sink identity
        # to drop stale-session audio, and bound methods are re-created on
        # every attribute access (`self._sink is self._sink` -> False!)
        self._sink_fn = self._sink

    def _sink(self, item):
        if self._loop:
            self._loop.call_soon_threadsafe(self._out_q.put_nowait, item)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._loop = asyncio.get_running_loop()
        preload(self._device, self._hf_repo)
        _ENGINE_IN.put(("reset", self._sink_fn))  # claim the engine for this session
        self._emit_task = self.create_task(self._emit_loop())

    async def stop(self, frame):
        if getattr(self, "_emit_task", None):
            await self.cancel_task(self._emit_task)
        await super().stop(frame)

    async def _emit_loop(self):
        while True:
            try:
                kind, payload = await asyncio.wait_for(self._out_q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                if self._turn_text and time.monotonic() - self._last_piece_at > TEXT_GAP_FINAL_S:
                    await self._emit_final()
                continue
            if kind == "text":
                if not self._turn_text:
                    logger.debug("KyutaiSTT first piece: {!r}", payload)
                self._turn_text.append(payload)
                self._last_piece_at = time.monotonic()
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text="".join(self._turn_text).strip(),
                        user_id="user", timestamp=time_now_iso8601(),
                    )
                )

    async def _emit_final(self):
        text = "".join(self._turn_text).strip()
        self._turn_text = []
        if text:
            logger.info("KyutaiSTT final: {!r}", text)
            await self.push_frame(
                TranscriptionFrame(text=text, user_id="user", timestamp=time_now_iso8601())
            )

    async def run_stt(self, audio: bytes):
        if self.sample_rate != 16000:
            await self.push_error(error_msg=f"KyutaiSTT expects 16k input, got {self.sample_rate}", fatal=True)
        _ENGINE_IN.put(("audio", (audio, self._sink_fn)))
        yield None
