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

FLUSH_AFTER_TURN_S = 0.9  # model text delay (~0.5s) + margin


class KyutaiSTTService(STTService):
    def __init__(self, *, hf_repo: str = "kyutai/stt-1b-en_fr", device: str = "cuda", **kwargs):
        super().__init__(audio_passthrough=True, **kwargs)
        self._hf_repo = hf_repo
        self._device = device
        self._in_q: queue.Queue = queue.Queue()
        self._out_q: asyncio.Queue = asyncio.Queue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._turn_text: list[str] = []
        self._flush_task: asyncio.Task | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._thread:
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        # consume decoded pieces
        self.create_task(self._emit_loop())

    def _worker(self):
        """Owns the model. Feeds 80ms mimi frames, posts decoded text pieces."""
        try:
            import julius
            import numpy as np
            import torch
            from moshi.models import loaders, LMGen

            info = loaders.CheckpointInfo.from_hf_repo(self._hf_repo)
            mimi = info.get_mimi(device=self._device)
            tokenizer = info.get_text_tokenizer()
            lm = info.get_moshi(device=self._device, dtype=torch.bfloat16)
            lm_gen = LMGen(lm, temp=0, temp_text=0.0)
            frame_size = mimi.frame_size  # 1920 @ 24k = 80ms
            logger.info("KyutaiSTT loaded ({}, frame {} @ {}Hz)", self._hf_repo, frame_size, mimi.sample_rate)
            self._post(("ready", None))

            buf = torch.zeros(0)
            with mimi.streaming(1), lm_gen.streaming(1):
                while self._running:
                    try:
                        item = self._in_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    pcm16k, in_rate = item
                    audio = torch.from_numpy(
                        np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    if in_rate != mimi.sample_rate:
                        audio = julius.resample_frac(audio, in_rate, mimi.sample_rate)
                    buf = torch.cat([buf, audio])
                    while buf.shape[-1] >= frame_size:
                        chunk, buf = buf[:frame_size], buf[frame_size:]
                        codes = mimi.encode(chunk.to(self._device)[None, None])
                        text_tokens = lm_gen.step(codes)
                        if text_tokens is None:
                            continue
                        tok = text_tokens[0, 0, 0].item()
                        if tok not in (0, 3):  # EPAD/PAD
                            piece = tokenizer.id_to_piece(tok).replace("▁", " ")
                            self._post(("text", piece))
        except Exception as exc:  # surface loudly; the service is useless without the model
            logger.exception("KyutaiSTT worker died: {}", exc)
            self._post(("error", str(exc)))

    def _post(self, item):
        if self._loop:
            self._loop.call_soon_threadsafe(self._out_q.put_nowait, item)

    async def _emit_loop(self):
        while True:
            kind, payload = await self._out_q.get()
            if kind == "text":
                self._turn_text.append(payload)
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text="".join(self._turn_text).strip(),
                        user_id="user", timestamp=time_now_iso8601(),
                    )
                )
            elif kind == "error":
                await self.push_error(error_msg=f"KyutaiSTT: {payload}", fatal=True)

    async def run_stt(self, audio: bytes):
        # Called once per InputAudioRawFrame (continuous mode): enqueue for the
        # worker; results surface via _emit_loop.
        self._in_q.put((audio, self.sample_rate))
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            # let the model's text delay drain, then finalize the turn
            if self._flush_task:
                self._flush_task.cancel()
            self._flush_task = self.create_task(self._finalize_turn())

    async def _finalize_turn(self):
        await asyncio.sleep(FLUSH_AFTER_TURN_S)
        text = "".join(self._turn_text).strip()
        self._turn_text = []
        if text:
            logger.info("KyutaiSTT final: {!r}", text)
            await self.push_frame(
                TranscriptionFrame(text=text, user_id="user", timestamp=time_now_iso8601())
            )
