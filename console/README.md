# Sparkstation Console

A static single-page app served by the supervisor at **`/console/`**
(`http://127.0.0.1:9001/console/` on the primary). No build step: plain
HTML/CSS/JS, same-origin API, nothing here knows a hostname. Exposing it
(e.g. `console.<your-domain>`) is a reverse-proxy / zero-trust concern that
lives outside this repo.

The CLI stays the authoritative ops tool; the console is a control panel over
the same supervisor API. Sections are built one at a time:

| Section | Status | Backend |
|---|---|---|
| **Voice Studio** | built | `/voice/*` (supervisor/voice.py) |
| Cluster & models | planned | existing `/models*` |
| Playground | planned | gateway `/v1/*` |
| Clients & usage | planned | gateway client config |
| Logs | planned | log-tail endpoint (to add) |
| Metrics | link-out | set `CONSOLE_GRAFANA_URL` to show the link |

Settings: `CONSOLE_ENABLED` (default true), `CONSOLE_GRAFANA_URL` (optional).
If the supervisor has `API_KEY` set, the sidebar shows a "set API key" button;
the key is kept in the browser's localStorage and sent as `X-API-Key`.

## Voice Studio

Sparky's voice is produced by the cascade stack (`voicecascade` backend, see
`voicecascade/DESIGN.md`): three Qwen3-TTS servers side by side on the voice
role, each with its own registry file, plus the Pipecat bot.

| engine | server | registry file | applies edits |
|---|---|---|---|
| `clone` | VoiceClone (:8023) | `voices.json` + `speakers/*.wav` | container restart (~40 s, automatic) |
| `stock` | CustomVoice (:8024) | `customvoice_voices.json` | container restart (~35 s, automatic) |
| `design` | VoiceDesign (:8025) | `voicedesign_voices.json` | hot reload |

All registry files live in `extra_args.tts_config_dir` of the voicecascade
spec (`~/cascade-tts/config` on the role host, bind-mounted at `/config` in
the containers). **They are never committed anywhere** — the clone references
are biometric data. Back that directory up out-of-band.

### Talk tab

Browser mic ↔ `ws(s)://<console-origin>/voice/talk` ↔ bot `/ws-client`.
The supervisor relays bytes; the wire format is Pipecat's protobuf frame
serializer (`console/talk.js` carries a 60-line codec: `Frame{audio=2,
message=4,…}`, `AudioRawFrame{audio=3, sample_rate=4, num_channels=5}`), 16 kHz
mono PCM16 up, 24 kHz down, RTVI JSON for control. Before `client-ready` the
tab sends `configure` with the chosen voice+engine, brain and optional
per-session system instruction. The mic streams continuously (silence too);
the bot does VAD/turn-taking. One session at a time (busy → close 4429).

WebSocket rather than WebRTC on purpose: WebRTC media needs a TURN path that a
Cloudflare-style tunnel doesn't provide; a plain WS proxies anywhere.

### Voices tab

- **▶ play** — `POST /voice/speak` with the sample text (proxied to the right engine).
- **✎ edit** — per-voice style instruct (`PATCH /voice/voices/{engine}/{id}`); for
  designed voices the instruct *is* the identity and can't be blank.
- **★ default** — `POST …/default` writes `console.json` `{"default": {"voice", "engine"}}`
  next to the registries; the bot reads it at every session start
  (`CASCADE_VOICE_CONFIG_DIR`), falling back to `CASCADE_VOICE`.
- **Design from description** — preview unregistered (`engine=design` + `instruct`)
  as often as you like, then save. Every take differs a little; the server's
  `instruct` is combined identity-first with per-line direction.
- **Clone from recording** — record in-browser (MediaRecorder) or upload; the
  supervisor normalizes to 24 kHz mono WAV with ffmpeg, rejects clips > 20 s
  (keep 8–12 s: ICL cloning re-processes the reference per request; long refs
  make speech choppy), stores `speakers/<id>.wav`, registers `ref_text` (the
  cloner needs the transcript in the registry) and restarts the clone engine.
- Deleting a clone removes its clip; stock speakers can't be deleted; the
  default voice can't be deleted; the clone server keeps ≥1 voice.

CLI over the same API: `sparkstation voice status|list|default|design|clone|instruct|delete|sample`.

### API summary

```
GET    /voice/status                              stack + engine health, default voice
GET    /voice/voices                              merged registry (default first)
POST   /voice/speak            {voice|engine+instruct, text, instruct?, format}
POST   /voice/voices/design    {id, instruct, language}
POST   /voice/voices/clone     multipart: id, ref_text, language, instruct, chunk_size, file
PATCH  /voice/voices/{engine}/{id}   {instruct?, language?, ref_text?, chunk_size?}
DELETE /voice/voices/{engine}/{id}
POST   /voice/voices/{engine}/{id}/default
POST   /voice/engines/{engine}/apply              force a registry re-read (restart)
WS     /voice/talk                                relay to the bot's /ws-client
```

Mutations require `X-API-Key` when the supervisor enforces one. Responses
carry `applying: true` when an engine restart was scheduled; `/voice/status`
reports `engines.<name>.apply` while it runs.
