# Cascade Voice — streaming STT → routed brain → streaming TTS

Branch: `cascade-voice` (checkpoint tag: `pre-cascade-voice`). **WON —
2026-08-30**: K's call after live testing ("works really well"). VoiceChat
legacy fully removed the same day (backend/launcher/profile/worker2
artifacts ~130 GB); recovery path documented in
homecloud-infra/docs/cascade-voice-runbook.md. The cascade is
supervisor-managed as backend `voicecascade` (voice profile), with
cascade_* prometheus metrics and the Grafana "Voice — cascade" row.

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

## Latency: from 2.6 s to 0.67 s (2026-09-01)

Bench = `scripts/latency_bench.sh` (3 turns/lane, speech-end → first bot
audio, medians). Baseline → after: **fast 2.60 → 0.67 s, think 3.15 → 1.08 s,
tool turn 4.28 → 2.85 s**; tool calls 3/3 throughout; transcripts intact.
What moved it, in order of effect:

1. **Semantic-VAD endpointing + Kyutai flush trick** (−0.9 s): the PyTorch
   checkpoint has no VAD heads, but `kyutai/stt-1b-en_fr-candle` carries the
   same base tensors plus 4 `extra_heads`; loading its weights with
   `lm_kwargs_overrides={"extra_heads_num_heads": 4, "extra_heads_dim": 6}`
   makes `step_with_extra_heads` return them. Head 2 (`p(done)`, class 0)
   sits ~1.0 in pre-speech silence (so it is gated on speech seen), 0.0 while
   talking, crosses 0.6 ~0.2 s after speech ends. Then 7 zero frames are
   stepped back-to-back so the 0.5 s delayed text emerges in ~0.25 s;
   `TranscriptionFrame(finalized=True)`. Energy-silence (400 ms) is the
   fallback trigger, the 1.0 s text-gap the last resort.
2. **Adaptive TTS first chunk** (−0.8 s): chunk_size 16 (needed for 1.4×
   realtime) meant 1.0 s to first audio. `voicecascade/patches/` yields the
   first chunk after 4 codec steps and doubles up to chunk_size: first audio
   0.35 s, RTF unchanged (1.40×). Launcher re-applies it on every start.
3. **First-clause TTS** (`FirstClauseAggregator`): the first TTS request goes
   out at the first clause boundary (≥4 words) or 9 words, not the first
   sentence.
4. **Thinking on tool turns** — decision request only: with tools registered
   the think lane runs `enable_thinking: true, reasoning_effort: low`
   (0/15 lost calls vs 3/15 without); the post-tool continuation does not
   think (it just verbalizes; thinking there cost 1–3 s).

Remaining budget (fast lane 0.67 s): endpoint ~0.2 + flush 0.25 + brain TTFT
0.1–0.3 + TTS 0.35, partly overlapped. Next lever if wanted: speculative brain
start at VAD-stop (hides brain TTFT and the think-lane thinking).

## Latency budget (original target ≤1.5s speech-end → first audio)

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
- ~~Open issue: VOICE-CLONE path 500s ("ref_text is required")~~ SOLVED
  2026-08-30: not an image bug — the server reads the reference transcript
  only from voices.json (`ref_text` per voice); request/CLI supplies are
  ignored. With a registry entry carrying ref_audio + exact ref_text, ICL
  cloning works first try.

## Voice decision (K, 2026-08-30)

All three Qwen3-TTS variants now run side by side on worker2 (each ~2-4 GB,
same streaming image, per-request voice — no model reload to swap voices):

- :8023 VoiceClone (Base) — ~10-30 s reference clip + transcript per voice
- :8024 CustomVoice — 9 stock speakers (all vetoed by K; "hesitant laughter"
  is the TTS model freestyling prosody — tame with the `instruct` style field)
- :8025 VoiceDesign (1.7B) — voice invented from a prose description;
  fresh sample per request, pin via the recipe's seed workflow when chosen

## Tool calling (built + verified 2026-08-30)

Contract v1.1 shapes, unchanged for the bridge: RTVI client-message
`{"t":"register-tools","d":{"tools":[OpenAI fn specs],"system_instruction":...}}`
(total replacement) -> ack `tools-registered`; brain tool calls forward as
`llm-function-call` (via a catch-all register_function(None) that defers —
never calls result_callback); client answers `llm-function-call-result`,
which the RTVIProcessor turns into a FunctionCallResultFrame that completes
the call and re-runs the brain. Verified: battery question -> gemma (fast
lane!) called get_battery_level -> spoken answer in 4.55 s end-to-end.

## Clone-TTS choppiness (K report 2026-08-30, fixed)

Symptom: browser session "extremely choppy, keeps cutting". Cause: clone
generation at 1.05x realtime with the 30 s reference — every request
re-processes the full ICL reference (the server's embedding-precompute cache
path is broken in this image: `_load_voice_clone_prompt` missing). Fixes
measured: trim ref to 12.5 s (1.28x) + chunk_size 16 (1.45x). Registry now:
`K` = K_ref12.wav + chunk 16 (prod), `K_full` = 30 s ref kept for fidelity
comparisons. Sentence-gap pauses shrink with the same headroom.

**K chose the cloned-K voice as Sparky's voice** (registry entry `K`:
K_ref.wav 30 s + Kyutai-transcribed ref_text). Bot defaults now point at
:8023 / voice "K"; per-session override stays available via the `configure`
message. In-pipeline: TTFB 0.64 s, 2.15 s voice-to-voice. Audition assets:
worker2 ~/voice-audition (page on loopback :8090), refs in ~/voice-refs.

## End-to-end first light (2026-08-30)

Full pipeline works over /ws-client on worker2:7860: live STT → router →
gateway brain → streaming TTS → bot audio. Measured (smoke client, espeak wavs):

- fast lane ("how many eggs..."): **2.24 s** speech-end → first bot audio,
  gemma TTFB 0.27–0.36 s; multi-turn (3 turns/session) works, warm TTFB ~0.05 s.
- think lane ("why do seasons change"): **3.02 s**, router escalates to
  `default` correctly, answer quality good.

Root cause of the day-long "100% PAD" bug: a freshly-reset Kyutai stream
needs ~1 s of lead-in audio before it can decode speech — an utterance that
starts within ~100 ms of session start transcribes as nothing. Fix: prime
each session reset with 2 s of silence (`buf16 = zeros(CTX16 + 32000)`);
costs ~0.5 s of compute once per session, before anyone speaks. (The multi-day
red herring: the forensic capture file was appended across sessions and
re-created late, corrupting the offline "pacing breaks decode" experiments —
pacing was never a factor.)

Latency knobs learned:
- `CASCADE_STT_FINAL_GAP` (default 0.7 s) — text-gap finalization after last
  decoded token; the biggest remaining chunk of the budget (~1.0–1.2 s
  incl. Kyutai's 0.5 s text delay). Next win: Kyutai semantic-VAD heads.
- Think-lane reasoning suppression: dsv4-flash ignores
  `chat_template_kwargs.thinking=false`; the working knob is
  **`enable_thinking: false`** (or `thinking_token_budget`). Router sends it
  via `extra_body` on think-lane turns only — reasoning burned 2+ s of TTFB.
- Router regex must not fire on bare "how"/"write" — casual questions
  ("how many eggs") belong in the fast lane.

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

## Definition of done (K, 2026-08-30: "no legacy stuff") — ✅ DONE 2026-08-30

Every item below executed the day of the win:
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
