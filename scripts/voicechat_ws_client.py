#!/usr/bin/env python3
"""Smoke-test client for the Nemotron VoiceChat plain-WebSocket transport.

Connects to ws://HOST:7860/ws-client, performs the RTVI client-ready
handshake, streams mic-format audio (16 kHz mono PCM16 — here a quiet tone),
and prints every frame the bot sends (RTVI messages, transcripts, bot audio).

Wire format: Pipecat protobuf frames (pipecat/frames/frames.proto,
ProtobufFrameSerializer). The tiny codec below covers the full wire schema —
useful as reference for a bridge in any language. Requires only `websockets`.

Usage: python ws_smoke_client.py [host] [seconds]
"""
import asyncio
import json
import os
import math
import struct
import sys
import time

import websockets

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.101.12"
RUN_SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
URL = f"ws://{HOST}:7860/ws-client"

IN_RATE = 16_000   # mic path: client -> bot
# bot audio arrives at 24_000 Hz mono PCM16


# ── minimal protobuf wire codec for pipecat/frames/frames.proto ──────────────
def _varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _len_field(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def encode_audio_frame(pcm: bytes, sample_rate: int = IN_RATE, channels: int = 1) -> bytes:
    """Frame{ audio: AudioRawFrame{ audio=3, sample_rate=4, num_channels=5 } }"""
    inner = _len_field(3, pcm) + _tag(4, 0) + _varint(sample_rate) + _tag(5, 0) + _varint(channels)
    return _len_field(2, inner)  # Frame.audio = field 2


def encode_message_frame(data: dict) -> bytes:
    """Frame{ message: MessageFrame{ data=1 } } — the RTVI JSON channel."""
    inner = _len_field(1, json.dumps(data).encode())
    return _len_field(4, inner)  # Frame.message = field 4


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = n = 0
    while True:
        b = buf[i]; i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


def _parse_fields(buf: bytes) -> dict:
    """Parse one protobuf message into {field_number: value or bytes}."""
    i, out = 0, {}
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            out[field], i = _read_varint(buf, i)
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            out[field] = buf[i:i + ln]; i += ln
        else:
            raise ValueError(f"unexpected wire type {wire}")
    return out


FRAME_KINDS = {1: "text", 2: "audio", 3: "transcription", 4: "message", 5: "interruption"}


def decode_frame(buf: bytes):
    top = _parse_fields(buf)
    (field, payload), = top.items()
    kind = FRAME_KINDS.get(field, f"unknown-{field}")
    if kind == "audio":
        f = _parse_fields(payload)
        return kind, {"bytes": len(f.get(3, b"")), "rate": f.get(4), "ch": f.get(5)}
    if kind == "message":
        return kind, json.loads(_parse_fields(payload).get(1, b"{}"))
    if kind in ("text", "transcription"):
        f = _parse_fields(payload)
        return kind, {k: v.decode("utf-8", "replace") for k, v in f.items() if isinstance(v, bytes)}
    return kind, {}


# ── smoke test ───────────────────────────────────────────────────────────────
async def main():
    print(f"connecting {URL}")
    got: dict[str, int] = {}
    audio_in_bytes = 0
    wav_done_at = None
    first_audio_at = None
    bot_ready = asyncio.Event()
    async with websockets.connect(URL, max_size=None) as ws:
        # RTVI handshake: bot answers with bot-ready once the model session is live
        # per-session tool registration (contract v1.1): send BEFORE
        # client-ready — WS ordering then guarantees the registered set is in
        # place before the session's first configuration render.
        if len(sys.argv) > 4:
            specs = json.load(open(sys.argv[4]))
            d = {"tools": specs}
            if os.environ.get("SYS_PROMPT"):
                d["system_instruction"] = os.environ["SYS_PROMPT"]
            await ws.send(encode_message_frame(
                {"id": "smoke-2", "label": "rtvi-ai", "type": "client-message",
                 "data": {"t": "register-tools", "d": d}}))
            print(f"-> register-tools sent ({len(specs)} tools, sys_prompt={'custom' if 'system_instruction' in d else 'default'})")
        await ws.send(encode_message_frame(
            {"id": "smoke-1", "label": "rtvi-ai", "type": "client-ready",
             "data": {"version": "1.0.0"}}))

        async def sender():
            # stream continuous mic audio in 20 ms chunks: a wav file if given
            # (16 kHz mono PCM16), then endless silence — like an open mic.
            chunk = int(IN_RATE * 0.02)
            wav_pcm = b""
            if len(sys.argv) > 3:
                import wave
                w = wave.open(sys.argv[3])
                assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (IN_RATE, 1, 2), "need 16kHz mono PCM16 wav"
                wav_pcm = w.readframes(w.getnframes())
            pos, t = 0, 0
            nonlocal wav_done_at
            while True:
                # hold the utterance until the model session is live —
                # audio sent before bot-ready lands in the warm-up pre-roll
                if pos < len(wav_pcm) and bot_ready.is_set():
                    pcm = wav_pcm[pos:pos + chunk * 2].ljust(chunk * 2, b"\x00")
                    pos += chunk * 2
                    if pos >= len(wav_pcm):
                        wav_done_at = time.monotonic()
                else:
                    pcm = b"".join(
                        struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * (t + i) / IN_RATE)))
                        for i in range(chunk)) if False else b"\x00" * (chunk * 2)
                t += chunk
                # MUTE_AFTER_SPEECH=1: emulate a half-duplex bridge that stops
                # sending mic audio once its utterance is done
                if os.environ.get("MUTE_AFTER_SPEECH") == "1" and pos >= len(wav_pcm) and wav_pcm:
                    await asyncio.sleep(0.02)
                    continue
                await ws.send(encode_audio_frame(pcm))
                await asyncio.sleep(0.02)

        send_task = asyncio.create_task(sender())
        deadline = time.monotonic() + RUN_SECS
        try:
            while time.monotonic() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
                if isinstance(raw, str):
                    print("TEXT-frame(?):", raw[:120]); continue
                kind, info = decode_frame(raw)
                if kind == "audio":
                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                    audio_in_bytes += info["bytes"]
                    got["audio"] = got.get("audio", 0) + 1
                else:
                    got[kind] = got.get(kind, 0) + 1
                    label = info.get("type") if kind == "message" else info
                    if kind == "message" and info.get("type") == "bot-ready":
                        bot_ready.set()
                    if kind == "message" and info.get("type") in ("tools-registered", "tools-error", "server-message"):
                        print(f"<- {info.get('type')}: {info.get('data')}")
                    if kind == "message" and info.get("type") == "llm-function-call":
                        d = info.get("data", {})
                        print(f"   TOOL CALL: {d.get('function_name')}({d.get('args')}) id={d.get('tool_call_id')}")
                        await ws.send(encode_message_frame({
                            "id": "smoke-fc-1", "label": "rtvi-ai",
                            "type": "llm-function-call-result",
                            "data": {"function_name": d.get("function_name"),
                                     "tool_call_id": d.get("tool_call_id"),
                                     "arguments": d.get("args") or {},
                                     "result": {"battery_percent": 87, "state": "discharging"}
                                               if d.get("function_name") == "get_battery_level"
                                               else {"answer": "The agent says: all systems nominal and the weather is sunny."}}}))
                        print("   -> sent llm-function-call-result")
                    if kind == "message" and info.get("type") == "metrics":
                        continue
                    if kind == "message" and info.get("type") in ("bot-transcription", "bot-tts-text", "bot-llm-text"):
                        txt = (info.get("data") or {}).get("text", "")
                        if info.get("type") == "bot-transcription" and txt:
                            print(f"<- BOT SAID: {txt}")
                        continue
                    print(f"<- {kind}: {label}")
        except asyncio.TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed as e:
            print(f"!! server closed the socket ({e.code}) — likely server_busy: one session at a time")
        finally:
            send_task.cancel()

    if wav_done_at and first_audio_at:
        print(f"RESPONSE LATENCY (end of speech -> first bot audio): {first_audio_at - wav_done_at:.2f}s")
    secs = audio_in_bytes / (24_000 * 2)
    print(f"\nsummary: {got} | bot audio: {audio_in_bytes} bytes (~{secs:.1f}s @24kHz)")
    ok = got.get("message", 0) > 0
    print("SMOKE", "PASS — RTVI channel live" + (", bot audio flowing" if audio_in_bytes else ", no bot audio (expected for a tone — VAD wants speech)") if ok else "FAIL")


asyncio.run(main())
