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


# --- stream facts --------------------------------------------------------
# One content-free record per finalized stream, on the "gateway.sse_facts"
# logger. Tests read it through their own handler so a missing record is an
# assertion failure, not an import error.
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest

import gateway.sse_normalize as sse_normalize

FACTS_LOGGER = "gateway.sse_facts"
REPO = Path(__file__).resolve().parents[1]

MIXED = (b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"\\n",'
         b'"content":"\\n\\nNO"},"finish_reason":null}]}')
FORWARD_RAW = (
    b": keep-alive\r\n\r\n"
    + MIXED + b"\r\n\r\n"
    + b"data: not json {\n\n"
    + b"event: ping\n"
    + b'data: {"choices":[{"index":0,"delta":{"content":"_REPLY"},"finish_reason":"stop"}]}\n\n'
    + b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
    + b"data: [DONE]\n\n"
    + b"data: tail-without-newline"
)
# Output of the pre-facts normalizer (39474fe) for FORWARD_RAW, any chunking.
FORWARD_GOLDEN = (
    b': keep-alive\r\n\r\ndata: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","rea'
    b'soning":"\\n"},"finish_reason":null}]}\r\n\r\ndata: {"id":"x","choices":[{"index":0,"delta'
    b'":{"content":"\\n\\nNO"},"finish_reason":null}]}\r\n\r\ndata: not json {\n\nevent: ping\nd'
    b'ata: {"choices":[{"index":0,"delta":{"content":"_REPLY"},"finish_reason":"stop"}]}\n\ndata'
    b': {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\ndat'
    b'a: [DONE]\n\ndata: tail-without-newline'
)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture
def cap():
    logger = logging.getLogger(FACTS_LOGGER)
    handler = _Capture()
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)


def _records(cap):
    out = []
    for line in cap.lines:
        _, sep, payload = line.partition("sse_stream_facts ")
        if sep:
            out.append(json.loads(payload))
    return out


def _one(cap):
    recs = _records(cap)
    assert len(recs) == 1, f"expected exactly one stream-facts record, got {len(recs)}"
    return recs[0]


def _chunks(raw, size):
    return [raw[i:i + size] for i in range(0, len(raw), size)]


def _run(chunks):
    async def gen():
        for c in chunks:
            yield c

    async def run():
        return b"".join([c async for c in normalize_sse(gen())])

    return asyncio.run(run())


def _line(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _untimed(rec):
    return {k: v for k, v in rec.items() if k not in ("ms", "elapsed_ms", "started_at")}


def test_forwarding_matches_pre_facts_bytes_for_any_chunking(cap):
    for size in (len(FORWARD_RAW), 7, 1):
        assert _run(_chunks(FORWARD_RAW, size)) == FORWARD_GOLDEN, f"forwarded bytes changed (chunk size {size})"


def test_facts_are_chunking_and_crlf_invariant(cap):
    _run([FORWARD_RAW])
    _run(_chunks(FORWARD_RAW, 1))
    crlf = FORWARD_RAW.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    _run(_chunks(crlf, 3))
    recs = _records(cap)
    assert len(recs) == 3, f"expected one stream-facts record per stream, got {len(recs)}"
    whole, bytewise, crlf_rec = (_untimed(r) for r in recs)
    assert bytewise == whole, "byte-split input changed the facts"
    assert {k: v for k, v in crlf_rec.items() if k != "raw_bytes"} == \
        {k: v for k, v in whole.items() if k != "raw_bytes"}, "CRLF framing changed the facts"
    assert whole["lines"] == 13 and whole["data_lines"] == 5 and whole["comment_lines"] == 1
    assert whole["other_lines"] == 1 and whole["done"] == 1 and whole["unparsed"] == 1
    assert whole["unterminated_bytes"] == len(b"data: tail-without-newline")


def test_text_reasoning_tool_usage_facts(cap):
    raw = b"".join([
        _line({"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "think", "reasoning_content": "think"}}]}),
        _line({"choices": [{"index": 0, "delta": {"reasoning": "more"}}]}),
        _line({"choices": [{"index": 0, "delta": {"content": "héllo"}}]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c0", "type": "function", "function": {"name": "f", "arguments": '{"a"'}}]}}]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}}]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 1, "id": "c1", "function": {"name": "g", "arguments": ""}}]}, "finish_reason": "tool_calls"}]}),
        _line({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33,
                                        "prompt_tokens_details": {"cached_tokens": 8},
                                        "completion_tokens_details": {"reasoning_tokens": 9}}}),
        b"data: [DONE]\n\n",
    ])
    _run([raw])
    rec = _one(cap)
    assert rec["outcome"] == "eof" and rec["error"] is None and rec["coverage"] == "full"
    assert rec["chars"] == {"content": 5, "reasoning": 9, "reasoning_content": 5, "tool_arguments": 7}
    # Mirrored aliases are reported per field and flagged, not merged into one estimate.
    assert rec["deltas"]["reasoning_both"] == 1
    assert rec["tool"] == {"items": 3, "named": 2, "indices": 2}
    assert rec["finish"] == {"tool_calls": 1} and rec["finish_last"] == "tool_calls"
    assert rec["usage"] == {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33,
                            "cached_tokens": 8, "reasoning_tokens": 9}
    assert rec["usage_after_finish"] is True
    for mark in ("first_byte", "first_reasoning", "first_content", "first_tool", "first_finish", "first_usage", "done"):
        assert isinstance(rec["ms"].get(mark), (int, float)), mark


def test_mixed_original_line_counted_once(cap):
    mixed = _line({"choices": [{"index": 0, "delta": {"reasoning": "ab", "content": "cde"}}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})
    out = _run([mixed])
    # The split forwards usage in BOTH synthetic events ...
    assert out.count(b'"usage"') == 2
    rec = _one(cap)
    # ... but the facts describe the single original upstream line.
    assert rec["split_lines"] == 1 and rec["parsed"] == 1 and rec["usage_snapshots"] == 1
    assert rec["chars"]["reasoning"] == 2 and rec["chars"]["content"] == 3
    assert rec["deltas"]["reasoning"] == 1 and rec["deltas"]["content"] == 1


def test_usage_missing_is_not_zero_and_cumulative_is_latest(cap):
    raw = b"".join([
        _line({"choices": [{"index": 0, "delta": {"content": "a"}}], "usage": None}),
        _line({"choices": [{"index": 0, "delta": {"content": "b"}}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}),
        _line({"choices": [{"index": 0, "delta": {"content": "c"}}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25}}),
        _line({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": "5", "completion_tokens": 30, "total_tokens": -1,
                         "prompt_tokens_details": {"cached_tokens": True},
                         "completion_tokens_details": {"reasoning_tokens": 0}}}),
    ])
    _run([raw])
    rec = _one(cap)
    assert rec["usage_snapshots"] == 3, "usage: null must not count as a snapshot"
    assert rec["usage"]["completion_tokens"] == 30, "cumulative usage must be the latest snapshot, not a sum"
    assert rec["usage"]["reasoning_tokens"] == 0, "a reported zero must stay zero"
    assert rec["usage"]["prompt_tokens"] is None and rec["usage"]["total_tokens"] is None
    assert rec["usage"]["cached_tokens"] is None, "invalid or missing usage must stay null"


def test_record_waits_for_trailing_usage_after_finish(cap):
    events = [
        _line({"choices": [{"index": 0, "delta": {"reasoning": "r" * 40}}]}),
        _line({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}),
        _line({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 40, "total_tokens": 47,
                                        "completion_tokens_details": {"reasoning_tokens": 40}}}),
        b"data: [DONE]\n\n",
    ]

    async def run():
        async def gen():
            for e in events:
                yield e

        agen = normalize_sse(gen())
        await agen.__anext__()
        await agen.__anext__()  # the finish event has been forwarded
        assert _records(cap) == [], "a finish event must not finalize the record"
        rest = [c async for c in agen]
        assert len(rest) == 2

    asyncio.run(run())
    rec = _one(cap)
    assert rec["finish"] == {"length": 1} and rec["finish_last"] == "length"
    assert rec["usage"]["reasoning_tokens"] == 40 and rec["usage_after_finish"] is True
    assert rec["chars"]["content"] == 0 and rec["chars"]["reasoning"] == 40


def test_partial_eof_is_not_a_model_finish(cap):
    tail = b'data: {"choices":[{"delta":{"cont'
    raw = _line({"choices": [{"index": 0, "delta": {"content": "abc"}}]}) + tail
    out = _run([raw])
    assert out.endswith(tail)
    rec = _one(cap)
    assert rec["outcome"] == "eof" and rec["done"] == 0
    assert rec["finish"] == {} and rec["finish_last"] is None
    assert rec["unterminated_bytes"] == len(tail)


class _UpstreamBroke(Exception):
    pass


def test_upstream_error_propagates_unchanged_with_safe_facts(cap):
    boom = _UpstreamBroke("PRIVATE-CANARY upstream detail https://canary.invalid/")
    boom.__cause__ = ConnectionResetError(104, "PRIVATE-CANARY reset")

    async def gen():
        yield _line({"choices": [{"index": 0, "delta": {"content": "x"}}]})
        raise boom

    async def run():
        got = []
        with pytest.raises(_UpstreamBroke) as info:
            async for c in normalize_sse(gen()):
                got.append(c)
        assert info.value is boom
        assert len(got) == 1

    asyncio.run(run())
    rec = _one(cap)
    assert rec["outcome"] == "error" and rec["stage"] == "upstream"
    assert rec["error"] == {"type": "_UpstreamBroke", "cause_type": "ConnectionResetError", "cause_errno": 104}
    assert "CANARY" not in "".join(cap.lines)


def test_cancellation_while_awaiting_upstream(cap):
    async def run():
        never = asyncio.Event()

        async def gen():
            yield _line({"choices": [{"index": 0, "delta": {"content": "x"}}]})
            await never.wait()
            yield b"unreachable"

        got = []

        async def consume():
            async for c in normalize_sse(gen()):
                got.append(c)

        task = asyncio.create_task(consume())
        while not got:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    rec = _one(cap)
    assert rec["outcome"] == "cancelled" and rec["stage"] == "upstream" and rec["error"] is None
    assert rec["chars"]["content"] == 1


def test_consumer_paused_at_yield_has_no_record_until_closed(cap):
    async def run():
        async def gen():
            for i in range(3):
                yield _line({"choices": [{"index": 0, "delta": {"content": str(i)}}]})

        never_started = normalize_sse(gen())
        await never_started.aclose()
        assert _records(cap) == [], "an unstarted normalizer must not emit"

        agen = normalize_sse(gen())
        await agen.__anext__()
        await asyncio.sleep(0)
        # Known limit: a consumer that stops at a yield without closing the
        # generator gets no prompt record.
        assert _records(cap) == []
        await agen.aclose()
        assert len(_records(cap)) == 1
        await agen.aclose()

    asyncio.run(run())
    rec = _one(cap)
    assert rec["outcome"] == "closed" and rec["stage"] == "yield"
    assert rec["chars"]["content"] == 1


def test_private_values_never_reach_the_log(cap):
    canaries = ["PRIVATE-CANARY-text", "PRIVATE-CANARY-think", "PRIVATE_CANARY_tool", "call_PRIVATE_CANARY",
                "PRIVATE-CANARY-arg", "PRIVATE-CANARY-model", "chatcmpl-PRIVATE-CANARY", "sk-PRIVATE-CANARY",
                "PRIVATE-CANARY-finish", "PRIVATE_CANARY_key", "PRIVATE-CANARY-stop", "https://canary.invalid/x"]
    raw = b"".join([
        _line({"id": "chatcmpl-PRIVATE-CANARY", "model": "PRIVATE-CANARY-model",
               "choices": [{"index": 0, "delta": {"reasoning": "PRIVATE-CANARY-think", "content": "PRIVATE-CANARY-text"}}]}),
        _line({"choices": [{"index": 0, "delta": {"PRIVATE_CANARY_key": "v", "tool_calls": [
            {"index": 0, "id": "call_PRIVATE_CANARY",
             "function": {"name": "PRIVATE_CANARY_tool", "arguments": '{"q":"PRIVATE-CANARY-arg"}'}}]}}]}),
        _line({"error": {"message": "PRIVATE-CANARY-text https://canary.invalid/x?key=sk-PRIVATE-CANARY"}}),
        _line({"choices": [{"index": 0, "delta": {}, "finish_reason": "PRIVATE-CANARY-finish",
                            "stop_reason": "PRIVATE-CANARY-stop"}]}),
    ])
    _run([raw])
    rec = _one(cap)
    assert rec["finish"] == {"other": 1} and rec["error_events"] == 1 and rec["deltas"]["unknown_keys"] == 1
    logged = "\n".join(cap.lines)
    for canary in canaries:
        assert canary not in logged, canary
        for digest in (hashlib.sha256, hashlib.md5, hashlib.sha1):
            assert digest(canary.encode()).hexdigest()[:12] not in logged, canary
    assert "CANARY" not in logged


def test_malformed_unsupported_and_over_bound_input_passes_through(cap):
    calls = [{"index": i, "function": {"arguments": "x"}} for i in range(40)]
    raw = b"".join([
        b"data: not json {\n\n",
        b"data: [1, 2]\n\n",
        b'data: "just a string"\n\n',
        b"data: " + b"[" * 100_000 + b"\n\n",                    # nesting beyond the decoder's recursion limit
        b'data: {"choices": "nope", "usage": [1]}\n\n',
        b"data: \xff\xfe broken utf-8\n\n",
        _line({"choices": [{"index": 0, "delta": {"content": "x" * 300_000}}]}),  # over the line bound
        _line({"choices": [{"index": i, "delta": {"content": "y"}} for i in range(20)]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": calls}}]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1000 + i} for i in range(16)]}}]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 2000 + i} for i in range(16)]}}]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 3000 + i} for i in range(16)]}}]}),
        _line({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 4000 + i} for i in range(16)]}}]}),
    ])
    assert _run(_chunks(raw, 4096)) == raw, "no line here is a mixed delta, so forwarding must be identity"
    rec = _one(cap)
    assert rec["unparsed"] == 3 and rec["non_object"] == 2 and rec["oversize"] == 1
    assert rec["invalid_choices"] == 1 and rec["usage_invalid"] == 1 and rec["usage"] is None
    assert rec["chars"]["content"] == 8, "only the inspected choices are counted; the oversize line is not parsed"
    assert rec["max_choices"] == 20 and rec["nonzero_choice_index"] is True
    assert rec["tool"]["indices"] == 64
    assert rec["coverage"] == "partial"
    assert rec["limits"] == ["choices", "line_bytes", "tool_indices", "tool_items"]
    assert max(len(line) for line in cap.lines) < 4096, "record must stay bounded"


def test_parse_budget_limits_diagnostics_only(cap, monkeypatch):
    monkeypatch.setattr(sse_normalize, "MAX_DIAG_PARSED_LINES", 2, raising=False)
    raw = b"".join(_line({"choices": [{"index": 0, "delta": {"content": "z"}}]}) for _ in range(5)) + b"data: [DONE]\n\n"
    assert _run([raw]) == raw
    rec = _one(cap)
    assert rec["parsed"] == 2 and rec["chars"]["content"] == 2 and rec["done"] == 1
    assert rec["data_lines"] == 6 and rec["limits"] == ["parse_budget"]


def test_diagnostic_and_logging_failures_do_not_touch_the_stream(cap, monkeypatch):
    class Explodes(logging.Handler):
        def emit(self, record):
            raise RuntimeError("handler failure")

    class BrokenJson:
        dumps = staticmethod(json.dumps)

        @staticmethod
        def loads(payload):
            raise RuntimeError("parser failure")

    logger = logging.getLogger(FACTS_LOGGER)
    bad = Explodes()
    logger.addHandler(bad)
    try:
        assert _run(_chunks(FORWARD_RAW, 5)) == FORWARD_GOLDEN
    finally:
        logger.removeHandler(bad)
    assert len(_records(cap)) == 1, "a failing sibling handler must not stop the record"

    # A collector failure leaves forwarding of non-mixed lines intact and is reported.
    monkeypatch.setattr(sse_normalize, "json", BrokenJson)
    plain = _line({"choices": [{"index": 0, "delta": {"content": "q"}}]}) + b"data: [DONE]\n\n"
    assert _run([plain]) == plain
    rec = _records(cap)[-1]
    assert rec["coverage"] == "partial" and "internal" in rec["limits"]


def test_record_visible_under_uvicorn_warning_logging_without_raising_other_loggers():
    script = r'''
import asyncio, importlib, logging, sys
import uvicorn
uvicorn.Config("gateway.proxy:app", log_level="warning")  # same logging setup as the launched proxy
import gateway.sse_normalize as m
importlib.reload(m)
logging.getLogger("gateway.proxy").info("proxy-info-must-stay-hidden")
async def up():
    yield b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
async def main():
    return [c async for c in m.normalize_sse(up())]
asyncio.run(main())
print(logging.getLogger().getEffectiveLevel(), logging.getLogger("gateway.proxy").getEffectiveLevel(),
      len(logging.getLogger().handlers), len(logging.getLogger("gateway.sse_facts").handlers))
'''
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    facts_lines = [l for l in proc.stderr.splitlines() if "sse_stream_facts " in l]
    assert len(facts_lines) == 1, f"expected one visible record on stderr, got {len(facts_lines)}"
    assert '"finish":{"stop":1}' in facts_lines[0]
    assert "proxy-info-must-stay-hidden" not in proc.stderr
    root_level, proxy_level, root_handlers, facts_handlers = map(int, proc.stdout.split())
    assert root_level == logging.WARNING and proxy_level == logging.WARNING and root_handlers == 0
    assert facts_handlers == 1, "module reload must not duplicate the facts handler"
