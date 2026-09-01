# Runtime patches applied to the Qwen3-TTS containers (worker2)

## Adaptive first chunk — `patch_tts_adaptive_chunk.py` (2026-09-01)

`martinb78/faster-qwen3-tts-dgx-spark:streaming` yields audio every
`chunk_size` codec steps (12 Hz). Voice "K" runs chunk_size 16 for smooth
playback (1.4x realtime), which made time-to-first-audio ~1.0 s. The patch
makes the FIRST chunk yield after `TTS_FIRST_CHUNK` (default 4) steps and
doubles each chunk until `chunk_size` — first audio 0.35 s, steady-state RTF
unchanged (measured 1.40x). The codec decoder already handles arbitrary chunk
sizes (accumulated decode until calibration), so no decoder changes.

Apply (on the voice host; the original is kept as `streaming.py.orig`):

    docker cp qwen3-tts-clone:/app/faster_qwen3_tts/streaming.py ~/cascade-tts/patches/streaming.py.orig
    python3 patch_tts_adaptive_chunk.py            # writes ~/cascade-tts/patches/streaming.py
    for c in qwen3-tts-clone qwen3-tts-vd qwen3-tts-cv; do
      docker cp ~/cascade-tts/patches/streaming.py $c:/app/faster_qwen3_tts/streaming.py; done
    docker restart qwen3-tts-clone qwen3-tts-vd qwen3-tts-cv

The patch lives in the containers' writable layer: it survives `docker
start/stop/restart` (what the voicecascade launcher does) but NOT a
`docker rm` + recreate — re-apply after recreating containers. Proper home
is the recipe fork's Dockerfile (~/cascade-tts) when the image is rebuilt.
