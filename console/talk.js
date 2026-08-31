/*
 * Talk transport for the Voice Studio: browser mic -> /voice/talk (supervisor
 * relay) -> cascade bot /ws-client, and bot audio back.
 *
 * Wire format is Pipecat's ProtobufFrameSerializer (pipecat/frames/frames.proto):
 *   Frame { text=1 | audio=2 | transcription=3 | message=4 | interruption=5 }
 *   AudioRawFrame { audio=3 (bytes, PCM16), sample_rate=4, num_channels=5 }
 *   MessageFrame  { data=1 (JSON bytes) }   <- RTVI control channel
 * The tiny codec below is a port of the dependency-free Python smoke client
 * that validated the bot. Audio: 16 kHz mono PCM16 in, 24 kHz mono PCM16 out,
 * client streams continuously (silence too) — the bot does VAD/turn-taking.
 */
(function () {
  "use strict";

  const IN_RATE = 16000;
  const OUT_RATE = 24000;
  const CHUNK_MS = 20;

  // ── protobuf helpers ──────────────────────────────────────────────────────
  function varint(n) {
    const out = [];
    while (true) {
      const b = n & 0x7f;
      n = Math.floor(n / 128);
      if (n) out.push(b | 0x80); else { out.push(b); return out; }
    }
  }
  function tag(field, wire) { return varint((field << 3) | wire); }
  function lenField(field, payload) {
    const head = tag(field, 2).concat(varint(payload.length));
    const out = new Uint8Array(head.length + payload.length);
    out.set(head, 0); out.set(payload, head.length);
    return out;
  }
  function concat(parts) {
    const n = parts.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(n); let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }
  function encodeAudioFrame(pcmBytes, rate, channels) {
    const inner = concat([
      lenField(3, pcmBytes),
      Uint8Array.from(tag(4, 0).concat(varint(rate))),
      Uint8Array.from(tag(5, 0).concat(varint(channels))),
    ]);
    return lenField(2, inner);
  }
  function encodeMessageFrame(obj) {
    const json = new TextEncoder().encode(JSON.stringify(obj));
    return lenField(4, lenField(1, json));
  }
  function readVarint(buf, i) {
    let shift = 0, n = 0;
    while (true) {
      const b = buf[i++];
      n += (b & 0x7f) * Math.pow(2, shift);
      if (!(b & 0x80)) return [n, i];
      shift += 7;
    }
  }
  function parseFields(buf) {
    let i = 0; const out = {};
    while (i < buf.length) {
      let key; [key, i] = readVarint(buf, i);
      const field = Math.floor(key / 8), wire = key & 7;
      if (wire === 0) { let v; [v, i] = readVarint(buf, i); out[field] = v; }
      else if (wire === 2) { let ln; [ln, i] = readVarint(buf, i); out[field] = buf.subarray(i, i + ln); i += ln; }
      else if (wire === 1) { i += 8; }
      else if (wire === 5) { i += 4; }
      else throw new Error("unexpected wire type " + wire);
    }
    return out;
  }
  const KINDS = { 1: "text", 2: "audio", 3: "transcription", 4: "message", 5: "interruption" };
  const td = new TextDecoder();
  function decodeFrame(buf) {
    const top = parseFields(buf);
    const field = Number(Object.keys(top)[0]);
    const kind = KINDS[field] || ("unknown-" + field);
    const payload = top[field];
    if (kind === "audio") {
      const f = parseFields(payload);
      return { kind, pcm: f[3] || new Uint8Array(0), rate: f[4] || OUT_RATE, channels: f[5] || 1 };
    }
    if (kind === "message") {
      const f = parseFields(payload);
      try { return { kind, data: JSON.parse(td.decode(f[1] || new Uint8Array(0))) }; }
      catch (e) { return { kind, data: null }; }
    }
    if (kind === "text" || kind === "transcription") {
      const f = parseFields(payload);
      return { kind, text: f[3] ? td.decode(f[3]) : "" };
    }
    return { kind };
  }

  // ── mic capture worklet (inline; posts Int16 PCM chunks at 16 kHz) ───────
  const WORKLET_SRC = `
    class MicCapture extends AudioWorkletProcessor {
      constructor() { super(); this.buf = []; this.n = 0; this.frame = ${IN_RATE * CHUNK_MS / 1000}; }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (!ch) return true;
        for (let i = 0; i < ch.length; i++) this.buf.push(ch[i]);
        while (this.buf.length >= this.frame) {
          const slice = this.buf.splice(0, this.frame);
          const pcm = new Int16Array(this.frame);
          let peak = 0;
          for (let i = 0; i < this.frame; i++) {
            const s = Math.max(-1, Math.min(1, slice[i]));
            pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            const a = Math.abs(s); if (a > peak) peak = a;
          }
          this.port.postMessage({ pcm: pcm.buffer, peak }, [pcm.buffer]);
        }
        return true;
      }
    }
    registerProcessor("mic-capture", MicCapture);
  `;

  // ── playback: schedule 24 kHz PCM16 chunks back-to-back ──────────────────
  class Player {
    constructor() { this.ctx = null; this.nextAt = 0; this.sources = new Set(); }
    async start() {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUT_RATE });
      await this.ctx.resume();
      this.nextAt = 0;
    }
    push(pcmBytes, rate) {
      if (!this.ctx || pcmBytes.length < 2) return;
      const n = pcmBytes.length >> 1;
      const i16 = new Int16Array(pcmBytes.buffer, pcmBytes.byteOffset, n);
      const buf = this.ctx.createBuffer(1, n, rate || OUT_RATE);
      const f32 = buf.getChannelData(0);
      for (let i = 0; i < n; i++) f32[i] = i16[i] / 0x8000;
      const src = this.ctx.createBufferSource();
      src.buffer = buf; src.connect(this.ctx.destination);
      const now = this.ctx.currentTime;
      if (this.nextAt < now + 0.02) this.nextAt = now + 0.05; // small jitter buffer
      src.start(this.nextAt);
      this.nextAt += buf.duration;
      this.sources.add(src);
      src.onended = () => this.sources.delete(src);
    }
    flush() { // interruption: drop everything queued
      for (const s of this.sources) { try { s.stop(); } catch (e) {} }
      this.sources.clear();
      this.nextAt = 0;
    }
    async stop() { this.flush(); if (this.ctx) { await this.ctx.close(); this.ctx = null; } }
  }

  // ── session ───────────────────────────────────────────────────────────────
  class TalkSession {
    /**
     * opts: { url, voice, engine, brain, systemInstruction,
     *         onState(str), onMessage(kind, data), onLevel(peak), onStats(obj) }
     */
    constructor(opts) {
      this.o = opts;
      this.ws = null; this.mic = null; this.micCtx = null; this.node = null; this.stream = null;
      this.player = new Player();
      this.stats = { sentFrames: 0, botAudioBytes: 0, ttfbMs: null, turns: 0 };
      this.closed = false;
    }

    async start() {
      const st = (s) => this.o.onState && this.o.onState(s);
      st("requesting microphone…");
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      this.micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: IN_RATE });
      if (this.micCtx.sampleRate !== IN_RATE) {
        throw new Error("browser refused a " + IN_RATE + " Hz capture context (got " + this.micCtx.sampleRate + ")");
      }
      const blob = new Blob([WORKLET_SRC], { type: "application/javascript" });
      const url = URL.createObjectURL(blob);
      await this.micCtx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      await this.player.start();

      st("connecting…");
      await new Promise((resolve, reject) => {
        const ws = new WebSocket(this.o.url);
        ws.binaryType = "arraybuffer";
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("WebSocket connection failed"));
        this.ws = ws;
      });
      this.ws.onmessage = (ev) => this._onFrame(ev.data);
      this.ws.onclose = (ev) => {
        const why = ev.code === 4429 ? "bot busy — another session is active (4429)"
          : ev.reason ? `closed (${ev.code}): ${ev.reason}` : `closed (${ev.code})`;
        this._teardown(why);
      };

      // RTVI: configure BEFORE client-ready (voice/engine/brain/persona for this session)
      const cfg = {};
      if (this.o.voice) cfg.voice = this.o.voice;
      if (this.o.engine) cfg.engine = this.o.engine;
      if (this.o.brain && this.o.brain !== "auto") cfg.brain = this.o.brain;
      if (this.o.systemInstruction) cfg.system_instruction = this.o.systemInstruction;
      this._sendMessage({ id: "console-cfg", label: "rtvi-ai", type: "client-message", data: { t: "configure", d: cfg } });
      this._sendMessage({ id: "console-ready", label: "rtvi-ai", type: "client-ready", data: { version: "1.0.0" } });

      // mic -> worklet -> ws (continuous, including silence: the bot needs a live stream)
      this.mic = this.micCtx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.micCtx, "mic-capture");
      this.node.port.onmessage = (ev) => {
        const pcm = new Uint8Array(ev.data.pcm);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(encodeAudioFrame(pcm, IN_RATE, 1));
          this.stats.sentFrames++;
        }
        if (this.o.onLevel) this.o.onLevel(ev.data.peak);
      };
      this.mic.connect(this.node);
      // no connect to destination: we don't want to hear ourselves
      st("waiting for bot-ready…");
    }

    _sendMessage(obj) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(encodeMessageFrame(obj));
    }

    _onFrame(data) {
      let f;
      try { f = decodeFrame(new Uint8Array(data)); } catch (e) { return; }
      if (f.kind === "audio") {
        this.stats.botAudioBytes += f.pcm.length;
        this.player.push(f.pcm, f.rate);
        return;
      }
      if (f.kind === "interruption") { this.player.flush(); }
      if (f.kind === "message" && f.data) {
        const t = f.data.type;
        if (t === "bot-ready") this.o.onState && this.o.onState("live — speak naturally");
        if (t === "user-started-speaking") { this.player.flush(); this.stats.turns++; }
        if (t === "bot-started-speaking") this.player.nextAt = 0;
        if (t === "metrics" && f.data.data && Array.isArray(f.data.data.ttfb)) {
          for (const m of f.data.data.ttfb) {
            if (typeof m.value === "number" && /tts/i.test(m.processor || "")) this.stats.ttfbMs = Math.round(m.value * 1000);
          }
        }
        if (t === "llm-function-call") {
          // The console registers no tools; answer so the brain isn't left waiting.
          const d = f.data.data || {};
          this._sendMessage({ id: "console-fc", label: "rtvi-ai", type: "llm-function-call-result",
            data: { function_name: d.function_name, tool_call_id: d.tool_call_id, arguments: d.args || {},
                    result: { error: "no tools are available in the console Talk tab" } } });
        }
        if (this.o.onStats) this.o.onStats(this.stats);
      }
      if (this.o.onMessage) this.o.onMessage(f.kind, f.kind === "message" ? f.data : f);
    }

    _teardown(reason) {
      if (this.closed) return;
      this.closed = true;
      try { this.node && this.node.disconnect(); } catch (e) {}
      try { this.mic && this.mic.disconnect(); } catch (e) {}
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      if (this.micCtx) this.micCtx.close().catch(() => {});
      this.player.stop().catch(() => {});
      if (this.o.onState) this.o.onState(reason || "stopped");
      if (this.o.onClosed) this.o.onClosed();
    }

    stop() {
      if (this.ws && this.ws.readyState <= WebSocket.OPEN) { try { this.ws.close(1000, "user stopped"); } catch (e) {} }
      this._teardown("stopped");
    }
  }

  window.SparkTalk = { TalkSession, encodeAudioFrame, encodeMessageFrame, decodeFrame };
})();
