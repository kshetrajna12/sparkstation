import os, sys
src = os.path.expanduser("~/cascade-tts/patches/streaming.py.orig")
dst = os.path.expanduser("~/cascade-tts/patches/streaming.py")
s = open(src).read()
old = "        if len(chunk_buffer) >= chunk_size:"
n = s.count(old); assert n >= 1, n
s = s.replace(old, "        if len(chunk_buffer) >= _adaptive_target(chunk_size, chunk_count):")
helper = '''
import os as _os
# Sparkstation patch (2026-09-01): adaptive first chunk. Yield the first chunk
# after TTS_FIRST_CHUNK codec steps (4 = ~0.33 s of audio -> first audio in
# ~0.35 s instead of ~1.0 s at chunk_size 16), doubling each chunk until the
# configured chunk_size is reached (steady-state RTF unchanged).
# TTS_FIRST_CHUNK=0 disables.
_FIRST_CHUNK = int(_os.environ.get("TTS_FIRST_CHUNK", "4"))


def _adaptive_target(chunk_size, chunk_count):
    if _FIRST_CHUNK <= 0:
        return chunk_size
    return min(chunk_size, _FIRST_CHUNK * (2 ** chunk_count))

'''
import re
m = re.search(r"^from __future__ import .*$", s, re.M)
if m:
    s = s[:m.end()] + "\n" + helper + s[m.end():]
else:
    s = helper + s
open(dst, "w").write(s)
import py_compile; py_compile.compile(dst, doraise=True)
print("patched", n, "yield site(s)")
