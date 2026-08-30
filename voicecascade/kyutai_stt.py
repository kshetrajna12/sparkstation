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

FLUSH_AFTER_TURN_S = 0.9   # model text delay (~0.5s) + margin
TEXT_GAP_FINAL_S = 1.1     # no new pieces for this long (with text pending) => final

# Process-global engine: the model loads ONCE per bot process (~19 s) and is
# reused across sessions (sessions are serial — single-session bot). Without
# this, the first utterance of every session raced the per-session load and
# transcribed nothing.
_ENGINE = {}
_ENGINE_LOCK = threading.Lock()


def preload(device: str = "cuda", hf_repo: str = "kyutai/stt-1b-en_fr"):
    """Load mimi + LM once per process. Safe to call repeatedly."""
    with _ENGINE_LOCK:
        if _ENGINE.get("repo") == hf_repo:
            return _ENGINE
        import contextlib

        import torch
        from moshi.models import loaders, LMGen

        logger.info("KyutaiSTT preloading {} ...", hf_repo)
        info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
        mimi = info.get_mimi(device=device)
        tokenizer = info.get_text_tokenizer()
        lm = info.get_moshi(device=device, dtype=torch.bfloat16)
        lm_gen = LMGen(lm, temp=0, temp_text=0.0)
        # Streaming contexts stay open for the life of the process (entering
        # twice raises "already streaming"); sessions reset state instead.
        es = contextlib.ExitStack()
        es.enter_context(mimi.streaming(1))
        es.enter_context(lm_gen.streaming(1))
        _ENGINE.update(repo=hf_repo, mimi=mimi, tokenizer=tokenizer,
                       lm_gen=lm_gen, _es=es)
        logger.info("KyutaiSTT ready (frame {} @ {} Hz)", mimi.frame_size, mimi.sample_rate)
        return _ENGINE


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
        self._last_piece_at = 0.0
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
        self._emit_task = self.create_task(self._emit_loop())

    def _worker(self):
        """Feeds 80ms mimi frames through the shared engine, posts text pieces."""
        try:
            import numpy as np
            import torch

            eng = preload(self._device, self._hf_repo)
            mimi, tokenizer, lm_gen = eng["mimi"], eng["tokenizer"], eng["lm_gen"]
            frame_size = mimi.frame_size
            # fresh recurrent state for this session
            for m in (mimi, lm_gen):
                if hasattr(m, "reset_streaming"):
                    m.reset_streaming()
            self._post(("ready", None))

            import julius

            # Streaming 16k->24k resample with overlap context: naive per-chunk
            # resampling destroys the filter state at every boundary and the
            # model decodes nothing (first-light bug). We resample 80 ms blocks
            # (1280 @ 16k -> 1920 @ 24k) inside a window padded with 10 ms of
            # real context on each side, keeping only the center.
            CTX16, BLK16 = 160, 1280
            CTX24, BLK24 = 240, 1920
            assert BLK24 == frame_size, "mimi frame size changed?"
            buf16 = torch.zeros(CTX16)  # left context primed with silence

            while self._running:
                try:
                    item = self._in_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                pcm, in_rate = item
                if in_rate != 16000:
                    self._post(("error", f"expected 16000 Hz input, got {in_rate}"))
                    return
                audio = torch.from_numpy(
                    np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                )
                buf16 = torch.cat([buf16, audio])
                while buf16.shape[-1] >= CTX16 + BLK16 + CTX16:
                    window = buf16[: CTX16 + BLK16 + CTX16]
                    buf16 = buf16[BLK16:]  # advance one block; contexts overlap
                    out24 = julius.resample_frac(window, 16000, 24000)
                    chunk = out24[CTX24: CTX24 + BLK24]
                    if True:
                        codes = mimi.encode(chunk.to(self._device)[None, None])
                        text_tokens = lm_gen.step(codes)
                        if text_tokens is None:
                            continue
                        self._steps = getattr(self, "_steps", 0) + 1
                        if self._steps % 50 == 1:
                            logger.debug("KyutaiSTT worker step #{}", self._steps)
                        tok = text_tokens[0, 0, 0].item()
                        if tok not in (0, 3):  # EPAD/PAD
                            piece = tokenizer.id_to_piece(tok).replace("▁", " ")
                            self._post(("text", piece))
        except Exception as exc:
            logger.exception("KyutaiSTT worker died: {}", exc)
            self._post(("error", str(exc)))

    def _post(self, item):
        if self._loop:
            self._loop.call_soon_threadsafe(self._out_q.put_nowait, item)

    async def _emit_loop(self):
        while True:
            try:
                kind, payload = await asyncio.wait_for(self._out_q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                # text-gap finalization: the model has been quiet for a while
                # after producing words -> the utterance is complete
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
            elif kind == "error":
                await self.push_error(error_msg=f"KyutaiSTT: {payload}", fatal=True)

    async def _emit_final(self):
        text = "".join(self._turn_text).strip()
        self._turn_text = []
        if text:
            logger.info("KyutaiSTT final: {!r}", text)
            await self.push_frame(
                TranscriptionFrame(text=text, user_id="user", timestamp=time_now_iso8601())
            )

    async def run_stt(self, audio: bytes):
        # Called once per InputAudioRawFrame (continuous mode): enqueue for the
        # worker; results surface via _emit_loop.
        self._in_q.put((audio, self.sample_rate))
        self._rx = getattr(self, "_rx", 0) + 1
        if self._rx % 100 == 1:
            logger.debug("KyutaiSTT rx frame #{} ({} bytes @ {})", self._rx, len(audio), self.sample_rate)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            # let the model's text delay drain, then finalize the turn
            if self._flush_task:
                self._flush_task.cancel()
            self._flush_task = self.create_task(self._finalize_turn())

    async def stop(self, frame):
        self._running = False
        if getattr(self, "_emit_task", None):
            await self.cancel_task(self._emit_task)
        await super().stop(frame)

    async def _finalize_turn(self):
        # belt-and-braces: if a stop frame ever does reach us, flush then too
        await asyncio.sleep(FLUSH_AFTER_TURN_S)
        await self._emit_final()
