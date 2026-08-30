"""Streaming TTS for the cascade: Faster-Qwen3-TTS (CustomVoice server).

Subclass of pipecat's OpenAITTSService that (a) skips the OpenAI voice-name
whitelist (our voices are Ryan/Aiden/... or cloned ones), and (b) speaks to
the local server's /v1/audio/speech with response_format=pcm (24 kHz mono
PCM16, streamed while generating — measured 0.46-0.56 s to first chunk).
"""
from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.openai.tts import OpenAITTSService


class QwenTTSService(OpenAITTSService):
    async def run_tts(self, text: str, context_id: str):
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
