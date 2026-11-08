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

Stream facts (2026-11-07): when a normalized stream finalizes, one
content-free `sse_stream_facts` record goes to stderr (gateway-proxy.log)
describing what the UPSTREAM sent, observed per original line before any
split: the latest usage snapshot (missing stays null, never 0) plus each
field's last supplied value with its snapshot number, finish
reasons from a fixed set, character counts per channel, line/parse coverage
and timings. It never holds text, arguments, tool names, IDs or other
upstream strings, and diagnostic failures or bounds only make the record
incomplete; they never change the forwarded bytes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import AsyncIterator

REASONING_KEYS = ("reasoning", "reasoning_content")

# The proxy runs under `uvicorn --log-level warning`, which leaves gateway
# INFO records with no handler. The facts logger owns one stderr handler and
# does not propagate, so the record is visible without raising any other
# logger's verbosity or duplicating output if the root is configured later.
facts_logger = logging.getLogger("gateway.sse_facts")


class _FactsHandler(logging.StreamHandler):
    def handleError(self, record):  # diagnostics must never disturb the stream
        pass


# By class name so a module reload does not add a second handler.
if not any(type(h).__name__ == "_FactsHandler" for h in facts_logger.handlers):
    _handler = _FactsHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    facts_logger.addHandler(_handler)
facts_logger.setLevel(logging.INFO)
facts_logger.propagate = False

# Diagnostic bounds. Exceeding one marks the record partial; forwarding is unaffected.
MAX_DIAG_LINE_BYTES = 256 * 1024     # larger data lines are counted, not parsed
MAX_DIAG_PARSED_LINES = 500_000      # ~15x a 32k-token stream
MAX_DIAG_PARSED_BYTES = 64 * 1024 * 1024
MAX_DIAG_CHOICES = 8                 # choices inspected per event
MAX_DIAG_TOOL_ITEMS = 16             # tool_calls items inspected per delta
MAX_DIAG_TOOL_INDICES = 64           # distinct tool-call indices remembered
_MAX_COUNT = 2 ** 53                 # usage values outside [0, 2**53) are invalid

FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "function_call", "abort", "error")
_KNOWN_DELTA_KEYS = frozenset(("role", "content", "reasoning", "reasoning_content", "tool_calls"))


def _count(value):
    return value if type(value) is int and 0 <= value < _MAX_COUNT else None


class _StreamFacts:
    """Bounded, content-free observations of one upstream SSE stream. Character
    counts are Unicode code points per field, not tokens; `reasoning` and
    `reasoning_content` are counted separately because some builds mirror the
    same text into both."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.started_at = time.time()
        self.emitted = False
        self.broken = False
        self.parsing = True
        self.limits = set()
        self.ms = {}
        self.raw_bytes = self.lines = self.data_lines = self.comment_lines = self.other_lines = 0
        self.done = self.parsed = self.unparsed = self.non_object = self.oversize = self.parsed_bytes = 0
        self.error_events = self.invalid_choices = self.split_lines = 0
        self.usage = None
        self.usage_known = {}
        self.usage_snapshots = self.usage_invalid = 0
        self.usage_after_finish = None
        self.finish = {}
        self.finish_last = None
        self.chars = {"content": 0, "reasoning": 0, "reasoning_content": 0, "tool_arguments": 0}
        self.deltas = {"content": 0, "reasoning": 0, "reasoning_content": 0, "reasoning_both": 0, "unknown_keys": 0}
        self.tool_items = self.tool_named = 0
        self.tool_indices = set()
        self.max_choices = 0
        self.nonzero_choice_index = False

    def _mark(self, name):
        self.ms.setdefault(name, round((time.monotonic() - self.t0) * 1000, 1))

    def observe_chunk(self, chunk):
        if not self.raw_bytes:
            self._mark("first_byte")
        self.raw_bytes += len(chunk)

    def observe_line(self, body):
        self.lines += 1
        if body.startswith(b"data:"):
            self.data_lines += 1
            self._data(body[5:].strip())
        elif body.startswith(b":"):
            self.comment_lines += 1
        elif body:
            self.other_lines += 1

    def _data(self, payload):
        if payload == b"[DONE]":
            self.done += 1
            self._mark("done")
            return
        if not self.parsing:
            return
        if len(payload) > MAX_DIAG_LINE_BYTES:
            self.oversize += 1
            self.limits.add("line_bytes")
            return
        if self.parsed + self.unparsed >= MAX_DIAG_PARSED_LINES or self.parsed_bytes + len(payload) > MAX_DIAG_PARSED_BYTES:
            self.limits.add("parse_budget")
            self.parsing = False
            return
        self.parsed_bytes += len(payload)
        try:
            obj = json.loads(payload)
        except (ValueError, RecursionError):
            self.unparsed += 1
            return
        self.parsed += 1
        if not isinstance(obj, dict):
            self.non_object += 1
            return
        self.ms["last_event"] = round((time.monotonic() - self.t0) * 1000, 1)
        if obj.get("error") is not None:
            self.error_events += 1
        choices = obj.get("choices")
        if isinstance(choices, list):
            self.max_choices = max(self.max_choices, len(choices))
            if len(choices) > MAX_DIAG_CHOICES:
                self.limits.add("choices")
            for choice in choices[:MAX_DIAG_CHOICES]:
                if isinstance(choice, dict):
                    self._choice(choice)
        elif choices is not None:
            self.invalid_choices += 1
        # After choices, so usage carried on the finish event counts as after it.
        usage = obj.get("usage")
        if isinstance(usage, dict):
            # Usage is cumulative: keep the latest snapshot whole, never sum.
            prompt = usage.get("prompt_tokens_details")
            completion = usage.get("completion_tokens_details")
            self.usage = {
                "prompt_tokens": _count(usage.get("prompt_tokens")),
                "completion_tokens": _count(usage.get("completion_tokens")),
                "total_tokens": _count(usage.get("total_tokens")),
                "cached_tokens": _count(prompt.get("cached_tokens")) if isinstance(prompt, dict) else None,
                "reasoning_tokens": _count(completion.get("reasoning_tokens")) if isinstance(completion, dict) else None,
            }
            self.usage_snapshots += 1
            # A later snapshot may omit a detail an earlier one supplied; keep
            # each field's last supplied value with the snapshot it came from.
            for field, value in self.usage.items():
                if value is not None:
                    self.usage_known[field] = {"value": value, "snapshot": self.usage_snapshots}
            self.usage_after_finish = self.finish_last is not None
            self._mark("first_usage")
        elif usage is not None:
            self.usage_invalid += 1

    def _choice(self, choice):
        index = choice.get("index")
        if type(index) is int and index != 0:
            self.nonzero_choice_index = True
        reason = choice.get("finish_reason")
        if reason is not None:
            key = reason if isinstance(reason, str) and reason in FINISH_REASONS else "other"
            self.finish[key] = self.finish.get(key, 0) + 1
            self.finish_last = key
            self._mark("first_finish")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return
        seen = 0
        for key in ("content", "reasoning", "reasoning_content"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                self.chars[key] += len(value)
                self.deltas[key] += 1
                self._mark("first_content" if key == "content" else "first_reasoning")
                seen += key != "content"
        if seen == 2:
            self.deltas["reasoning_both"] += 1
        if delta.keys() - _KNOWN_DELTA_KEYS:
            self.deltas["unknown_keys"] += 1
        calls = delta.get("tool_calls")
        if isinstance(calls, list) and calls:
            self._mark("first_tool")
            if len(calls) > MAX_DIAG_TOOL_ITEMS:
                self.limits.add("tool_items")
            for call in calls[:MAX_DIAG_TOOL_ITEMS]:
                if not isinstance(call, dict):
                    continue
                self.tool_items += 1
                index = call.get("index")
                if type(index) is int and index not in self.tool_indices:
                    if len(self.tool_indices) < MAX_DIAG_TOOL_INDICES:
                        self.tool_indices.add(index)
                    else:
                        self.limits.add("tool_indices")
                function = call.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        self.tool_named += 1
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        self.chars["tool_arguments"] += len(arguments)

    def record(self, outcome, stage, error, unterminated):
        if self.broken:
            self.limits.add("internal")
        facts = None
        if error is not None:
            cause = error.__cause__
            facts = {"type": type(error).__name__, "cause_type": type(cause).__name__ if cause is not None else None}
            for label, exc in (("errno", error), ("cause_errno", cause)):
                value = getattr(exc, "errno", None)
                if type(value) is int:
                    facts[label] = value
        return {
            "event": "sse_stream_facts",
            "outcome": outcome,
            "stage": stage,
            "error": facts,
            "started_at": round(self.started_at, 3),
            "elapsed_ms": round((time.monotonic() - self.t0) * 1000, 1),
            "ms": self.ms,
            "raw_bytes": self.raw_bytes,
            "unterminated_bytes": unterminated,
            "lines": self.lines,
            "data_lines": self.data_lines,
            "comment_lines": self.comment_lines,
            "other_lines": self.other_lines,
            "done": self.done,
            "parsed": self.parsed,
            "unparsed": self.unparsed,
            "non_object": self.non_object,
            "oversize": self.oversize,
            "invalid_choices": self.invalid_choices,
            "error_events": self.error_events,
            "split_lines": self.split_lines,
            "finish": self.finish,
            "finish_last": self.finish_last,
            "usage": self.usage,
            "usage_known": self.usage_known,
            "usage_snapshots": self.usage_snapshots,
            "usage_invalid": self.usage_invalid,
            "usage_after_finish": self.usage_after_finish,
            "chars": self.chars,
            "deltas": self.deltas,
            "tool": {"items": self.tool_items, "named": self.tool_named, "indices": len(self.tool_indices)},
            "max_choices": self.max_choices,
            "nonzero_choice_index": self.nonzero_choice_index,
            "coverage": "partial" if self.limits else "full",
            "limits": sorted(self.limits),
        }


def _observe(facts, method, value):
    if facts.broken:
        return
    try:
        method(value)
    except Exception:
        facts.broken = True


def _emit(facts, outcome, stage, error, unterminated):
    if facts.emitted:
        return
    facts.emitted = True
    try:
        record = facts.record(outcome, stage, error, unterminated)
        facts_logger.info("sse_stream_facts %s", json.dumps(record, separators=(",", ":")))
    except Exception:
        pass


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
    remains at EOF is flushed untouched.

    Emits at most one stream-facts record, when this generator finalizes:
    upstream EOF, an upstream exception, cancellation while awaiting upstream,
    or a close while suspended at a yield. A consumer that stops iterating
    without closing the generator gets its record only when the generator is
    eventually finalized (e.g. by the event loop's async-generator cleanup)."""
    facts = _StreamFacts()
    outcome, stage, error = "eof", "upstream", None
    buf = b""
    try:
        async for chunk in chunks:
            stage = "normalize"
            if chunk:
                _observe(facts, facts.observe_chunk, chunk)
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
                    # Observe the ORIGINAL line so a split delta counts once.
                    _observe(facts, facts.observe_line, body)
                    parts = split_delta_line(body)
                    if len(parts) > 1:
                        facts.split_lines += 1
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
                    stage = "yield"
                    yield b"".join(out)
            stage = "upstream"
        if buf:
            stage = "yield"
            yield buf
        stage = "end"
    except GeneratorExit:
        outcome = "closed"
        raise
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException as exc:
        outcome, error = "error", exc
        raise
    finally:
        _emit(facts, outcome, stage, error, len(buf))
