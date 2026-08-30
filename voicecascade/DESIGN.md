# Cascade Voice — streaming STT → routed brain → streaming TTS

Branch: `cascade-voice` (checkpoint tag: `pre-cascade-voice`). Replaces
VoiceChat 11B as Sparky's voice IF it wins the bench; VoiceChat stays intact
and restorable (`models start voicechat -p voice`).

## Decision record (2026-08-30, with K)

- Rejected: tool-call consult inside VoiceChat (dislikes wait-for-full-answer),
  Qwen3-Omni (turn-based anyway, bleeding-edge on GB10, second resident 30B).
- Chosen: cascaded pipeline — the only architecture where speech starts after
  the FIRST phrase of the brain's streaming answer.
- Brain is dynamic per turn: router → gemma4-2b (fast/social) | default qwen
  (hard questions) | later OpenClaw agent endpoint. Routing must cost ~0
  (rules or single gemma token). Escalation replaces the speaker mid-stream.

## Components (all local, all proven-on-GB10 or design-proven)

| Stage | Choice | Why |
|---|---|---|
| STT | kyutai/stt-1b-en_fr (moshi>=0.2.6, streaming, 0.5s delay) | native streaming + built-in SEMANTIC VAD (turn detection for free); worker2 venv ~/cascade-stt/.venv |
| Brain | sparkstation gateway (gemma4-2b / default / per-session alias) | streams; tools native; memory/persona = text Sparky |
| TTS | martinb78/faster-qwen3-tts-dgx-spark:streaming (Qwen3-TTS-12Hz-1.7B) | GB10 CUDA-graph image, OpenAI-compatible, streams WAV <1s first audio, voice cloning for Sparky's voice |
| Orchestration | Pipecat (same pinned rev), same /ws-client protobuf contract | bridge contract v2 == v1.2 transport-wise; tool calls become native OpenAI tool-calls via brain |

## Latency budget (target ≤1.5s speech-end → first audio)

STT finalization ~0.5s + brain TTFT (gemma 26ms / qwen ~0.3-1s) + first
sentence gen (~0.3s gemma) + TTS first chunk (<1s, overlapped) — stretch: ~1.2s.

## First-light results

- STT (kyutai/stt-1b-en_fr, worker2, cu130): **35.4 ms/step vs 80 ms budget**
  (2.3x real-time headroom) after warmup; first-ever run pays ~20 s of model
  load + compile. Transcription solid given espeak test audio. Streaming STT
  on GB10: PROVEN.

### TTS serving notes (worker2)

- Image `martinb78/faster-qwen3-tts-dgx-spark:streaming` (:8023→8000). Two server
  modes: `run_server.py` + Base model = VOICE-CLONE (needs reference wavs in
  /config/speakers — dev voice only until K picks); `run_customvoice_server.py`
  + CustomVoice model = STOCK speakers (Ryan/Aiden/... — the no-cloning default;
  model downloading). Formats: wav|pcm|mp3, streamed (wav header sized 0xFFFFFFFF).
- Trap: don't mount an HF snapshot dir into containers (symlinks dangle) —
  dereference with `cp -rL` (plain copy at ~/cascade-tts/model).

- TTS (CustomVoice server, stock speaker "Ryan", worker2:8024, CUDA graphs):
  **first audio chunk 0.46–0.56 s**, streams while generating (7.8 s of audio
  delivered in 4.4 s wall). TTS on GB10: PROVEN. All three cascade components
  now individually proven.
- Open issue: the VOICE-CLONE server path 500s ("ref_text is required", ICL
  mode) even with ref_text supplied via file and CLI — a bug in the image's
  plumbing; not blocking (CustomVoice is the no-cloning default; revisit when
  K picks Sparky's voice — sample of "Ryan" sent to K 2026-08-30).

## Placement

Trial on worker2 (free). Production intent: STT+TTS are small (~6GB total) →
primary; brain already lives wherever the daily driver lives → worker2 freed
for the 2-worker big-model plan.

## Ownership boundaries (agreed with OpenClaw integration, 2026-08-30)

- **Persona & memory injection is OpenClaw-owned, opaque to this pipeline.**
  The session config carries `system_instruction` + memory digest verbatim from
  the bridge (source of truth: Sparky's workspace / MEMORY.md distillate).
  This repo never authors Sparky-persona content — no parallel persona on
  worker2. Playground/dev sessions use a minimal prompt explicitly labeled
  "dev voice, not Sparky".
- **Sparky's voice is K's deliberate choice, not a default.** The TTS voice is
  a config slot (stock voice `DEV0` until decided). Cloning path: K supplies a
  ~10s clean sample when ready; we generate the voice embedding from it. No
  cloning of anyone's voice without K saying so.

## Definition of done (K, 2026-08-30: "no legacy stuff")

If the cascade wins the bench, the merge to main INCLUDES the cleanup:
- Remove the voicechat backend (launcher, enums, gateway/health special-cases,
  autoload bucket), the `voice` profile entry, models.yaml voicechat block.
- worker2: retire ~/nemotron-voicechat checkout + 65GB bootstrap cache + images
  (the patched runtime survives in the private GitHub backup + tags only).
- Dashboard Voice row re-pointed at cascade metrics; dead VoiceChat-only
  metrics removed.
- homecloud runbook rewritten for the cascade (VoiceChat section reduced to a
  pointer at the backup repo); memory files updated.
- Bridge contract v2 published; v1.x VoiceChat-specific rules retired.
No parallel legacy path kept "just in case" — restoration path is git history.

## Working rules

- All sparkstation changes on `cascade-voice` only; merge via PR after bench.
- NOTE: the tree carries UNRELATED uncommitted WIP (dspark launcher placeholder
  hooks, gemma max_model_len 32K bump — another session's work). Never stage
  supervisor/main.py, health_check.py, launchers/dspark_launcher.py, models.yaml
  hunks that aren't ours.
- worker2 assets: ~/cascade-stt (kyutai repo+venv), ~/cascade-tts (recipe),
  TTS docker image, models via cluster sync-cache.
- Bench gate vs VoiceChat numbers: voice-to-voice latency, barge-in reaction,
  answer quality (the DC test), long-answer completeness, session stability.
