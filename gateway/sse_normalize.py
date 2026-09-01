"""
Streaming-delta normalization for the Sparkstation gateway.

Problem this solves (2026-08-31): vLLM's `qwen3` reasoning parser, when the
`</think>` tag closes inside a decode step, packs the TAIL of the reasoning
(usually a lone "\\n") and the FIRST content token into one SSE delta:

    data: {"choices":[{"delta":{"reasoning":"\\n","content":"\\n\\nNO"}}]}
    data: {"choices":[{"delta":{"content":"_REPLY"}}]}

Clients that render reasoning as a separate block (OpenClaw / pi-ai) process
the content first, then open a thinking block for the reasoning field, so the
next content delta has to start a NEW text block. Result: the answer's first
word is split from the rest ("Str\\nuck", "Ag\\nreed" — 15 of foxhole's 54
sends in a week), and when the answer is the silent-reply sentinel the split
"NO\\n_REPLY" no longer matches and gets delivered literally to WhatsApp.

Fix: split such a delta into two SSE events — reasoning first, then content —
so the thinking block lands BEFORE the text block and consecutive content
deltas stay one block. Every other byte of the stream passes through
unchanged; any parse failure passes the line through as-is.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

REASONING_KEYS = ("reasoning", "reasoning_content")


def split_delta_line(line: bytes) -> list:
    """Return the list of SSE lines (bytes, no trailing newline) that `line`
    should be forwarded as. Usually [line] itself; two lines when a delta
    carries both reasoning and content."""
    if not line.startswith(b"data:"):
        return [line]
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return [line]
    # Cheap pre-filter before paying for a JSON parse on every token.
    if b"reasoning" not in payload or b"content" not in payload:
        return [line]
    try:
        obj = json.loads(payload)
        choices = obj.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            return [line]
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return [line]
        content = delta.get("content")
        rkeys = [k for k in REASONING_KEYS if isinstance(delta.get(k), str) and delta[k] != ""]
        if not (isinstance(content, str) and content != "" and rkeys):
            return [line]
        # Event 1: reasoning only (keep role / everything else, drop content).
        first = json.loads(payload)
        first["choices"][0]["delta"].pop("content", None)
        first["choices"][0]["finish_reason"] = None
        # Event 2: content only (drop the reasoning fields).
        second = json.loads(payload)
        for k in rkeys:
            second["choices"][0]["delta"].pop(k, None)
        second["choices"][0]["delta"].pop("role", None)
        enc = lambda o: b"data: " + json.dumps(o, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return [enc(first), enc(second)]
    except Exception:
        return [line]


async def normalize_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Line-buffer a raw SSE byte stream and apply split_delta_line to each
    complete line. Partial lines are held until their newline arrives; whatever
    remains at EOF is flushed untouched."""
    buf = b""
    async for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        out = []
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line, buf = buf[:nl], buf[nl + 1:]
            cr = line.endswith(b"\r")
            body = line[:-1] if cr else line
            nl = b"\r\n" if cr else b"\n"
            parts = split_delta_line(body)
            # SSE frames events with a BLANK line; two `data:` lines separated
            # by a single newline would be ONE event whose payload is both
            # JSON objects joined with "\n" (that broke OpenClaw's parser on
            # the first deploy, 2026-08-31 22:31). So every part except the
            # last gets its own event terminator; the original blank line
            # that follows in the stream closes the last one.
            for part in parts[:-1]:
                out.append(part + nl + nl)
            out.append(parts[-1] + nl)
        if out:
            yield b"".join(out)
    if buf:
        yield buf
