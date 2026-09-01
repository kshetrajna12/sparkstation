"""Streaming STT for the cascade voice pipeline: Kyutai stt-1b-en_fr via moshi.

Continuous (non-segmented) Pipecat STT service. Every InputAudioRawFrame's
PCM is resampled 16k->24k, buffered into 80 ms mimi frames, and stepped
through the model on a dedicated worker thread (GPU work must not block the
event loop). Decoded word pieces stream out as InterimTranscriptionFrames;
the aggregator's VAD/turn strategies decide turn boundaries, and on
UserStoppedSpeaking we flush the model's ~0.5 s text delay and emit the
final TranscriptionFrame for the turn.

Endpointing (semantic VAD + flush trick)
----------------------------------------
PRIMARY trigger: the model's semantic-VAD "pause prediction" heads
(`lm_gen.step_with_extra_heads` -> `(text_tokens, extra_heads)`), softmax over
6 classes with class 0 = "speaker finished", for pauses of 0.5/1.0/2.0/3.0 s
(head 0..3; lower index = more aggressive). We watch head VAD_HEAD (default
2, as Kyutai's Unmute does) and endpoint on `p(class 0) > VAD_THRESH` (0.6)
sustained over VAD_CONSEC (default 2) consecutive 80 ms steps — a single step
over threshold used to fire mid-phrase, splitting "Hey, / Sparky," into two
turns. The counter resets on any step below threshold, on every emitted text
token, and after a flush.
Measured on a real utterance: ~1.0 during pre-speech silence — which is why
the `spoke` gate is essential, not optional — 0.0 while speaking, a brief
mid-sentence wobble to ~0.55, then 0.98 within ~0.2 s of speech end.

The heads are absent from the published PyTorch `kyutai/stt-1b-en_fr`
weights, but the candle sibling repo (CASCADE_STT_VAD_REPO, default
"kyutai/stt-1b-en_fr-candle") holds the same base tensors byte-for-byte plus
`extra_heads.0-3.weight` [6,2048]; we point the checkpoint's `moshi_weights`
at that file and pass `extra_heads_num_heads=4, extra_heads_dim=6`. On any
download/load failure we log and fall back to the plain weights; the startup
line "KyutaiSTT semantic VAD heads: N" says which happened.

SECONDARY trigger (safety net, and the only one when N == 0): AUDIO ENERGY.
The 16 kHz block feeding each 80 ms step is reduced to its peak amplitude,
and a run of consecutive blocks below SILENCE_PEAK (default 0.015 ~ -36 dBFS)
lasting SILENCE_MS (default 400 ms = 5 steps) means the speaker stopped. The
silence run resets to zero on every emitted text token, so a mid-word gap can
never trigger it; the 400 ms default keeps it from pre-empting the semantic
head, which normally fires ~200 ms after speech end.

Either trigger runs the same flush + final path.

Because the model emits text with a fixed ~0.5 s delay, detecting the pause
is not enough: the tail of the utterance is still in flight. So on detection
we run the "flush trick" — push 7 frames (ceil(0.5 s / 80 ms) + 1) of
synthetic silence through the model back-to-back, unpaced, which drains the
delayed text in ~0.25 s of wall clock instead of 0.5 s. Any tokens that come
out are forwarded normally, then a ("final", None) marker tells the emit loop
to push a finalized TranscriptionFrame immediately. (Tokens for the last
spoken word arrive ~6 steps after their audio; that is exactly what the flush
recovers.)

Streaming state is never reset between turns (a reset costs a 1-2 s lead-in);
the flush merely advances the stream with silence and real audio continues
right after.

Env knobs:
  CASCADE_STT_SILENCE_PEAK  (default "0.015") peak amplitude below = silence
  CASCADE_STT_SILENCE_MS    (default "400")   silence needed to endpoint
  CASCADE_STT_VAD_REPO      (default "kyutai/stt-1b-en_fr-candle") head weights
  CASCADE_STT_VAD_HEAD      (default "2")     which pause head to watch
  CASCADE_STT_VAD_THRESH    (default "0.6")   p(finished) threshold
  CASCADE_STT_VAD_CONSEC    (default "2")     steps over threshold to endpoint
  CASCADE_STT_FINAL_GAP     (default "1.0")   text-gap fallback, safety net
"""
from __future__ import annotations

import asyncio
import math
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
TEXT_GAP_FINAL_S = float(os.environ.get("CASCADE_STT_FINAL_GAP", "1.0"))     # safety net: no new pieces for this long (with text pending) => final

# Endpointing knobs (see module docstring).
SILENCE_PEAK = float(os.environ.get("CASCADE_STT_SILENCE_PEAK", "0.015"))   # 16k block peak below this = silent
SILENCE_MS = float(os.environ.get("CASCADE_STT_SILENCE_MS", "400"))         # silence run needed to endpoint (fallback path)
SILENCE_STEPS = max(1, math.ceil(SILENCE_MS / 80.0))                        # 80 ms per step
# Semantic-VAD heads: weights come from the candle sibling repo ("" disables).
VAD_REPO = os.environ.get("CASCADE_STT_VAD_REPO", "kyutai/stt-1b-en_fr-candle")
VAD_HEAD = int(os.environ.get("CASCADE_STT_VAD_HEAD", "2"))
VAD_THRESH = float(os.environ.get("CASCADE_STT_VAD_THRESH", "0.6"))
VAD_CONSEC = max(1, int(os.environ.get("CASCADE_STT_VAD_CONSEC", "2")))     # consecutive steps over VAD_THRESH
VAD_WARMUP_STEPS = 12      # ignore endpointing right after a stream reset
FLUSH_FRAMES = 7           # ceil(0.5 s text delay / 80 ms) + 1

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
    # The published PyTorch weights carry no semantic-VAD heads, but the
    # candle sibling checkpoint is the same base tensors byte-for-byte PLUS
    # extra_heads.0-3.weight [6,2048] — so load the LM from that file and tell
    # the config it has 4 heads of dim 6. Everything else (tokenizer, mimi,
    # config) still comes from hf_repo. CASCADE_STT_VAD_REPO="" opts out.
    lm = None
    if VAD_REPO:
        try:
            from huggingface_hub import snapshot_download
            cand = os.path.join(snapshot_download(VAD_REPO), "model.safetensors")
            info.moshi_weights = cand
            lm = info.get_moshi(device=device, dtype=torch.bfloat16,
                                lm_kwargs_overrides={"extra_heads_num_heads": 4,
                                                     "extra_heads_dim": 6})
            logger.info("KyutaiSTT loaded VAD-head weights from {}", VAD_REPO)
        except Exception:
            logger.exception("semantic-VAD weights unavailable; falling back to plain STT weights")
            lm = None
            info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
    if lm is None:
        lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    lm_gen = LMGen(lm, temp=0, temp_text=0.0)
    try:
        _n_heads = len(getattr(lm, "extra_heads", None) or [])
    except TypeError:
        _n_heads = 0
    has_heads = _n_heads > 0
    logger.info("KyutaiSTT semantic VAD heads: {}", _n_heads)
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

        # 80 ms of digital silence at 24 kHz, reused for every flush step.
        silence24 = torch.zeros(BLK24, device=device)

        def _step(chunk24):
            """One model step. Returns (token_id_or_None, p_finished_or_None)."""
            res = lm_gen.step_with_extra_heads(mimi.encode(chunk24[None, None]))
            if res is None:
                return None, None
            t, extra_heads = res
            pr = None
            try:
                pr = extra_heads[VAD_HEAD][0, 0, 0].item()
            except Exception:  # head layout unexpected — degrade to gap-based final
                pr = None
            if t is None:
                return None, pr
            return t[0, 0, 0].item(), pr

        buf16 = torch.zeros(CTX16)
        sink = None
        steps_since_reset = 0
        spoke = False        # a real (non-pad) token since the last finalize
        flushed = False      # a flush already fired for this utterance
        silent_steps = 0     # consecutive quiet 80 ms blocks since the last token
        pr_hits = 0          # consecutive steps with pr > VAD_THRESH (hysteresis)
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
                steps_since_reset = 0
                spoke = False
                flushed = False
                silent_steps = 0
                pr_hits = 0
                continue
            pcm, s_sink = payload
            if s_sink is not sink:
                continue  # stale session audio
            audio = torch.from_numpy(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
            buf16 = torch.cat([buf16, audio])
            while buf16.shape[-1] >= CTX16 + BLK16 + CTX16:
                win = buf16[: CTX16 + BLK16 + CTX16]; buf16 = buf16[BLK16:]
                # the 16k block this step actually covers (centre of the window)
                peak = float(win[CTX16: CTX16 + BLK16].abs().max())
                chunk = julius.resample_frac(win, 16000, 24000)[CTX24: CTX24 + BLK24]
                v, pr = _step(chunk.to(device))
                steps_since_reset += 1
                silent_steps = silent_steps + 1 if peak < SILENCE_PEAK else 0
                nstat["n"] += 1; nstat["cmax"] = max(nstat["cmax"], float(chunk.abs().max()))
                if v is not None:
                    if v in (0, 3):
                        nstat["pad"] += 1
                    else:
                        nstat["tok"] += 1
                        piece = tokenizer.id_to_piece(v).replace("▁", " ")
                        # new speech after a flush = a new utterance
                        spoke = True
                        flushed = False
                        # count silence only from AFTER the last text token,
                        # so a mid-word gap cannot endpoint the turn
                        silent_steps = 0
                        pr_hits = 0
                        if sink is not None:
                            sink(("text", piece))
                if pr is not None:
                    nstat["prmax"] = max(nstat.get("prmax", 0.0), pr)
                    if spoke and not flushed:
                        nstat["prmax_spoke"] = max(nstat.get("prmax_spoke", 0.0), pr)
                nstat["silmax"] = max(nstat.get("silmax", 0), silent_steps)
                nstat["peakmax"] = max(nstat.get("peakmax", 0.0), peak)
                nstat["peakmin"] = min(nstat.get("peakmin", 1e9), peak)
                if nstat["n"] % 100 == 0:
                    logger.debug("KyutaiSTT engine steps {}: pad={} tok={} max24k={:.3f} prmax={:.2f} "
                                 "prmax_spoke={:.2f} silmax={} peak={:.4f}..{:.4f}",
                                 nstat["n"], nstat["pad"], nstat["tok"], nstat["cmax"],
                                 nstat.get("prmax", 0.0), nstat.get("prmax_spoke", 0.0),
                                 nstat.get("silmax", 0), nstat.get("peakmin", 0.0), nstat.get("peakmax", 0.0))
                    nstat.update(pad=0, tok=0, cmax=0.0, prmax=0.0, prmax_spoke=0.0,
                                 silmax=0, peakmax=0.0, peakmin=1e9)

                # --- endpoint (semantic head primary, energy silence fallback) + flush trick ---
                # hysteresis: VAD_CONSEC steps in a row over the threshold. One
                # step is not enough — a brief mid-phrase pause spikes the head
                # and used to split a single utterance into two turns.
                if has_heads and pr is not None:
                    pr_hits = pr_hits + 1 if pr > VAD_THRESH else 0
                by_head = has_heads and pr is not None and pr_hits >= VAD_CONSEC
                by_silence = silent_steps >= SILENCE_STEPS
                if (spoke and not flushed and steps_since_reset >= VAD_WARMUP_STEPS
                        and (by_head or by_silence)):
                    if by_head:
                        logger.info("KyutaiSTT endpoint: pr={:.2f} x{} after {} steps, flushed {} frames",
                                    pr, pr_hits, steps_since_reset, FLUSH_FRAMES)
                    else:
                        logger.info("KyutaiSTT endpoint: silence {} ms after {} steps, flushed {} frames",
                                    int(silent_steps * 80), steps_since_reset, FLUSH_FRAMES)
                    for _ in range(FLUSH_FRAMES):
                        fv, fpr = _step(silence24)
                        steps_since_reset += 1
                        nstat["n"] += 1
                        if fv is not None and fv not in (0, 3):
                            nstat["tok"] += 1
                            piece = tokenizer.id_to_piece(fv).replace("▁", " ")
                            if sink is not None:
                                sink(("text", piece))
                        elif fv is not None:
                            nstat["pad"] += 1
                        if fpr is not None:
                            logger.debug("KyutaiSTT flush step: pr={:.2f}", fpr)
                    if sink is not None:
                        sink(("final", None))
                    flushed = True
                    spoke = False
                    silent_steps = 0
                    pr_hits = 0


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
            if kind == "final":
                # engine-side semantic VAD says the turn ended (post-flush)
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
                TranscriptionFrame(text=text, user_id="user", timestamp=time_now_iso8601(),
                                   finalized=True)
            )

    async def run_stt(self, audio: bytes):
        if self.sample_rate != 16000:
            await self.push_error(error_msg=f"KyutaiSTT expects 16k input, got {self.sample_rate}", fatal=True)
        _ENGINE_IN.put(("audio", (audio, self._sink_fn)))
        yield None
