"""Unit tests for gateway/sse_normalize.py (no network)."""
import asyncio
import json

from gateway.sse_normalize import normalize_sse, split_delta_line


def _ev(delta, finish=None):
    return b"data: " + json.dumps(
        {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    ).encode()


def _delta(line):
    return json.loads(line[5:])["choices"][0]["delta"]


def test_mixed_delta_is_split_reasoning_first():
    out = split_delta_line(_ev({"role": "assistant", "reasoning": "\n", "content": "\n\nNO"}))
    assert len(out) == 2
    d1, d2 = _delta(out[0]), _delta(out[1])
    assert d1 == {"role": "assistant", "reasoning": "\n"}
    assert d2 == {"content": "\n\nNO"}
    assert json.loads(out[0][5:])["choices"][0]["finish_reason"] is None


def test_reasoning_content_key_also_split():
    out = split_delta_line(_ev({"reasoning_content": "tail", "content": "Hi"}, finish="stop"))
    assert len(out) == 2
    assert _delta(out[0]) == {"reasoning_content": "tail"}
    assert _delta(out[1]) == {"content": "Hi"}
    # finish_reason stays on the LAST event only
    assert json.loads(out[1][5:])["choices"][0]["finish_reason"] == "stop"


def test_pure_deltas_pass_through_unchanged():
    for d in ({"reasoning": "The"}, {"content": "hello"}, {"content": "", "role": "assistant"},
              {"reasoning": "", "content": "x"}, {"reasoning": None, "content": "x"}, {}):
        line = _ev(d)
        assert split_delta_line(line) == [line]


def test_non_data_and_done_lines_untouched():
    for line in (b"", b": keep-alive", b"event: ping", b"data: [DONE]", b"data: not json {", b"data:"):
        assert split_delta_line(line) == [line]


def test_stream_reassembles_partial_lines():
    mixed = _ev({"reasoning": "\n", "content": "\n\nNO"})
    raw = mixed + b"\n\n" + _ev({"content": "_REPLY"}) + b"\n\n" + b"data: [DONE]\n\n"
    # feed in awkward 7-byte chunks to exercise the line buffer
    chunks = [raw[i:i + 7] for i in range(0, len(raw), 7)]

    async def gen():
        for c in chunks:
            yield c

    async def run():
        return b"".join([c async for c in normalize_sse(gen())])

    out = asyncio.run(run())
    # SSE framing: every event must be its own blank-line-terminated block,
    # and no block may carry two data lines.
    events = [e for e in out.split(b"\n\n") if e.strip()]
    assert len(events) == 4, events
    assert all(e.count(b"data:") == 1 for e in events), events
    lines = [l for l in out.split(b"\n") if l.startswith(b"data:")]
    assert len(lines) == 4  # split mixed (2) + _REPLY + [DONE]
    assert _delta(lines[0]) == {"reasoning": "\n"}
    assert _delta(lines[1]) == {"content": "\n\nNO"}
    assert _delta(lines[2]) == {"content": "_REPLY"}
    assert lines[3] == b"data: [DONE]"
    # blank separator lines preserved
    assert b"\n\n" in out


def test_crlf_preserved():
    line = _ev({"reasoning": "r", "content": "c"}) + b"\r\n"

    async def gen():
        yield line

    async def run():
        return b"".join([c async for c in normalize_sse(gen())])

    out = asyncio.run(run())
    # part1 + CRLF CRLF (event end) + part2 + CRLF
    assert out.count(b"\r\n") == 3
    assert out.split(b"\r\n\r\n")[0].count(b"data:") == 1
