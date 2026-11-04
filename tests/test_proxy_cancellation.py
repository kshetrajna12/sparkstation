"""Exact handler over fictional loopback HTTP; no gateway startup/config/keys.

The module's real startup reads client/supervisor configuration. Extract the
handler and its response class instead, with explicit fictional globals. These
tests exercise actual ASGI/httpx socket cancellation, not a mock disconnect flag.
"""
import ast
import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from typing import Optional

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import pytest
import uvicorn

from gateway.sse_normalize import normalize_sse


class Metric:
    def __init__(self):
        self.current = {}
        self.increments = []

    def labels(self, **labels):
        self.current = labels
        return self

    def inc(self, *args):
        self.increments.append(dict(self.current))

    def set(self, *args):
        pass

    def observe(self, *args):
        pass


class Policy:
    name = "fictional"
    reasoning = None

    def __init__(self, deny=False):
        self.inflight = self.admissions = self.releases = 0
        self.deny = deny

    def allows_model(self, alias):
        return alias == "fictional"

    def admit(self, *args):
        if self.deny:
            return False, "concurrency_limit", 1
        self.inflight += 1
        self.admissions += 1
        return True, None, None

    def release(self):
        self.inflight -= 1
        self.releases += 1


def handler(client, policy, *, unknown_key=False):
    tree = ast.parse((Path(__file__).parents[1] / "gateway/proxy.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HOP_BY_HOP" for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_DisconnectResponse":
            nodes.append(node)
        elif isinstance(node, ast.AsyncFunctionDef) and node.name == "forward":
            node.decorator_list = []
            nodes.append(node)

    async def available(alias):
        return None

    env = dict(asyncio=asyncio, anyio=anyio, json=json, time=time, Optional=Optional,
               Request=Request, Response=Response, JSONResponse=JSONResponse,
               StreamingResponse=StreamingResponse, client=client, direct_client=None,
               logger=logging.getLogger("gateway.proxy"), normalize_sse=normalize_sse,
               clients=SimpleNamespace(resolve=lambda key: None if unknown_key else policy),
               extract_key=lambda headers: "fictional", DENY_UNKNOWN_KEY="unknown",
               reasoning=SimpleNamespace(policy_for=lambda alias: ("passthrough", None, None)),
               _ensure_available=available, SYSTEMONE_PATH="v1/systemone")
    for name in ["CLIENT_DENIED", "REQUESTS_TOTAL", "CLIENT_REQUESTS", "CLIENT_INFLIGHT", "TTFT", "REQUEST_DURATION"]:
        env[name] = Metric()
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "gateway/proxy.py", "exec"), env)
    env["forward"].fixture_metrics = env
    return env["forward"]


CHUNK = b'data: {"choices":[{"delta":{"content":"Fictional"},"finish_reason":null}]}\n\n'


def frame(body):
    return f"{len(body):x}\r\n".encode() + body + b"\r\n"


async def seen(event):
    try:
        await asyncio.wait_for(event.wait(), 1)
    except asyncio.TimeoutError:
        pass
    return event.is_set()


@asynccontextmanager
async def world(mode="held_headers", *, unknown_key=False, deny=False):
    accepted, disconnected, resume = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tasks, writers = set(), []

    async def upstream(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        writers.append(writer)
        eof = None
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            size = next((int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:")), 0)
            await reader.readexactly(size)
            accepted.set()
            if mode == "failure":
                return
            if mode == "json":
                body = b'{"fictional":true}'
                writer.write(b"HTTP/1.1 418 Fictional\r\nContent-Type: application/json\r\nX-Fictional: kept\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                await writer.drain()
                return

            async def watch_eof():
                await reader.read()
                disconnected.set()

            eof = asyncio.create_task(watch_eof())
            response_headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            if mode != "held_headers":
                writer.write(response_headers + frame(CHUNK))
                await writer.drain()
            if mode != "normal":
                await resume.wait()
            if not disconnected.is_set():
                if mode == "held_headers":
                    writer.write(response_headers + frame(CHUNK))
                writer.write(frame(b"data: [DONE]\n\n") + b"0\r\n\r\n")
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if eof:
                eof.cancel()
                await asyncio.gather(eof, return_exceptions=True)
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    backend = await asyncio.start_server(upstream, "127.0.0.1", 0)
    port = backend.sockets[0].getsockname()[1]
    policy = Policy(deny=deny)
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False,
                              timeout=httpx.Timeout(connect=1, read=None, write=1, pool=1))
    app = FastAPI()
    app.add_api_route("/{path:path}", handler(client, policy, unknown_key=unknown_key), methods=["POST"])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    sock.setblocking(False)
    gateway_port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, log_level="critical", access_log=False, lifespan="off"))
    server.install_signal_handlers = lambda: None
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(.01)
        assert server.started, "fictional gateway started"
        yield SimpleNamespace(url=f"http://127.0.0.1:{gateway_port}", port=gateway_port, policy=policy,
                              accepted=accepted, disconnected=disconnected)
    finally:
        resume.set()
        await client.aclose()
        for writer in writers:
            writer.close()
        server.should_exit = True
        await asyncio.wait_for(serving, 3)
        backend.close()
        await backend.wait_closed()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*list(tasks), return_exceptions=True), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["held_headers", "held_body"])
async def test_client_disconnect_closes_upstream_and_releases_once(mode, caplog):
    caplog.set_level(logging.WARNING, logger="gateway.proxy")
    async with world(mode) as w:
        reader, writer = await asyncio.open_connection("127.0.0.1", w.port)
        body = b'{"model":"fictional","stream":true}'
        writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: fictional\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        await writer.drain()
        assert await seen(w.accepted), "fictional upstream admitted"
        if mode == "held_body":
            await asyncio.wait_for(reader.readuntil(b"Fictional"), 1)
        writer.close()
        await writer.wait_closed()
        assert await seen(w.disconnected), f"{mode}: client disconnect closes upstream before it resumes"
        for _ in range(100):
            if w.policy.releases:
                break
            await asyncio.sleep(.01)
        assert (w.policy.inflight, w.policy.admissions, w.policy.releases) == (0, 1, 1), "disconnect releases its admission exactly once"
    terminal = [record for record in caplog.records if record.getMessage().startswith("Forwarding ended")]
    assert len(terminal) == 1 and terminal[0].levelno == logging.WARNING, "cancellation facts survive the default WARNING threshold"
    assert f"stage={'upstream_headers' if mode == 'held_headers' else 'upstream_body'} outcome=cancelled" in terminal[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [logging.WARNING, logging.INFO])
async def test_normal_stream_keeps_bytes_and_releases_once(threshold, caplog):
    caplog.set_level(threshold, logger="gateway.proxy")
    async with world("normal") as w, httpx.AsyncClient(trust_env=False) as downstream:
        response = await downstream.post(w.url + "/v1/chat/completions", json={"model": "fictional", "stream": True})
        assert response.status_code == 200
        assert response.content == CHUNK + b"data: [DONE]\n\n"
        assert (w.policy.inflight, w.policy.releases) == (0, 1)
    terminal = [record for record in caplog.records if record.getMessage().startswith("Forwarding ended")]
    if threshold == logging.WARNING:
        assert terminal == [], "completed forwarding stays below the default WARNING threshold"
    else:
        assert len(terminal) == 1 and terminal[0].levelno == logging.INFO
        assert "outcome=completed upstream_status=200" in terminal[0].getMessage()


@pytest.mark.asyncio
async def test_nonstream_upstream_status_headers_and_bytes_unchanged():
    async with world("json") as w, httpx.AsyncClient(trust_env=False) as downstream:
        response = await downstream.post(w.url + "/v1/chat/completions", json={"model": "fictional", "stream": False})
        assert response.status_code == 418
        assert response.headers["x-fictional"] == "kept"
        assert response.content == b'{"fictional":true}'
        assert (w.policy.inflight, w.policy.releases) == (0, 1)


@pytest.mark.asyncio
async def test_upstream_transport_failure_keeps_502_and_releases_once(caplog):
    caplog.set_level(logging.WARNING, logger="gateway.proxy")
    async with world("failure") as w, httpx.AsyncClient(trust_env=False) as downstream:
        response = await downstream.post(w.url + "/v1/chat/completions", json={"model": "fictional"})
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "bad_gateway"
        assert (w.policy.inflight, w.policy.releases) == (0, 1)
        assert "stage=upstream_headers outcome=failed" in caplog.text
        assert '"type": "RemoteProtocolError"' in caplog.text
        terminal = [record for record in caplog.records if record.getMessage().startswith("Forwarding ended")]
        assert len(terminal) == 1 and terminal[0].levelno == logging.WARNING


@pytest.mark.asyncio
@pytest.mark.parametrize("options,model,status", [({"unknown_key": True}, "fictional", 401), ({}, "unpermitted", 403), ({"deny": True}, "fictional", 429)])
async def test_initial_refusal_never_forwards_or_releases_unadmitted(options, model, status):
    async with world(**options) as w, httpx.AsyncClient(trust_env=False) as downstream:
        response = await downstream.post(w.url + "/v1/chat/completions", json={"model": model})
        assert response.status_code == status
        assert not w.accepted.is_set(), "refusal never sends upstream"
        assert (w.policy.inflight, w.policy.admissions, w.policy.releases) == (0, 0, 0)


@pytest.mark.asyncio
async def test_request_task_cancel_before_headers_releases_admission():
    accepted = asyncio.Event()

    async def held(request):
        accepted.set()
        await asyncio.Event().wait()

    policy = Policy()
    async with httpx.AsyncClient(transport=httpx.MockTransport(held)) as client:
        forward = handler(client, policy)
        queue = asyncio.Queue()
        await queue.put({"type": "http.request", "body": b'{"model":"fictional"}', "more_body": False})
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "query_string": b"", "headers": []}

        async def send(message):
            pass

        async def call():
            response = await forward(Request(scope, queue.get), "v1/chat/completions")
            await response(scope, queue.get, send)

        task = asyncio.create_task(call())
        assert await seen(accepted)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert (policy.inflight, policy.releases) == (0, 1), "request-task cancellation releases admission"


@pytest.mark.asyncio
async def test_header_disconnect_race_closes_acquired_response_once():
    queue = asyncio.Queue()
    await queue.put({"type": "http.request", "body": b'{"model":"fictional"}', "more_body": False})
    closed = []

    class Upstream:
        status_code = 200
        headers = {"content-type": "application/octet-stream"}

        async def aiter_raw(self):
            await asyncio.Event().wait()
            yield b"never"

        async def aclose(self):
            closed.append(True)

    class Client:
        def build_request(self, *args, **kwargs):
            return None

        async def send(self, *args, **kwargs):
            await queue.put({"type": "http.disconnect"})
            return Upstream()

    policy = Policy()
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "query_string": b"", "headers": []}

    async def send(message):
        pass

    response = await handler(Client(), policy)(Request(scope, queue.get), "v1/chat/completions")
    await asyncio.wait_for(response(scope, queue.get, send), 1)
    assert (len(closed), policy.inflight, policy.releases) == (1, 0, 1), "header/disconnect race closes and releases once"


@pytest.mark.asyncio
async def test_cleanup_failure_still_releases_and_keeps_safe_cause(caplog):
    caplog.set_level(logging.WARNING, logger="gateway.proxy")
    reached = asyncio.Event()

    class Upstream:
        status_code = 200
        headers = {"content-type": "application/octet-stream"}

        async def aiter_raw(self):
            reached.set()
            await asyncio.Event().wait()
            yield b"never"

        async def aclose(self):
            try:
                raise ConnectionResetError(104, "FICTIONAL_PRIVATE_CAUSE")
            except ConnectionResetError as cause:
                raise OSError(32, "FICTIONAL_PRIVATE_ERROR") from cause

    class Client:
        def build_request(self, *args, **kwargs):
            return None

        async def send(self, *args, **kwargs):
            return Upstream()

    policy = Policy()
    queue = asyncio.Queue()
    await queue.put({"type": "http.request", "body": b'{"model":"fictional"}', "more_body": False})
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "query_string": b"", "headers": []}

    async def send(message):
        pass

    response = await handler(Client(), policy)(Request(scope, queue.get), "v1/chat/completions")
    task = asyncio.create_task(response(scope, queue.get, send))
    assert await seen(reached)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert (policy.inflight, policy.releases) == (0, 1), "failed cleanup cannot strand admission"
    assert '"errno": 32' in caplog.text and '"cause_errno": 104' in caplog.text
    assert "FICTIONAL_PRIVATE" not in caplog.text, "ordinary diagnostics omit exception messages"
    terminal = [record for record in caplog.records if record.getMessage().startswith("Forwarding ended")]
    assert len(terminal) == 1 and terminal[0].levelno == logging.WARNING


@pytest.mark.asyncio
async def test_request_build_failure_cannot_strand_admitted_slot():
    class Client:
        def build_request(self, *args, **kwargs):
            raise ValueError("fictional request construction failed")

    policy = Policy()
    queue = asyncio.Queue()
    await queue.put({"type": "http.request", "body": b'{"model":"fictional"}', "more_body": False})
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "query_string": b"", "headers": []}

    async def send(message):
        pass

    async def call():
        response = await handler(Client(), policy)(Request(scope, queue.get), "v1/chat/completions")
        await response(scope, queue.get, send)

    result = await asyncio.gather(call(), return_exceptions=True)
    assert isinstance(result[0], BaseException), "construction failure stays a failure"
    assert (policy.inflight, policy.releases) == (0, 1), "request construction failure releases admission"


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_fails", [False, True])
async def test_failed_final_send_is_accounted_after_release(upstream_fails, caplog):
    caplog.set_level(logging.WARNING, logger="gateway.proxy")
    closed = []

    class Upstream:
        status_code = 200
        headers = {"content-type": "application/octet-stream"}

        async def aiter_raw(self):
            yield b"Fictional"

        async def aclose(self):
            closed.append(True)

    class Client:
        def build_request(self, *args, **kwargs):
            return None

        async def send(self, *args, **kwargs):
            if upstream_fails:
                try:
                    raise ConnectionRefusedError(111, "FICTIONAL_PRIVATE_UPSTREAM_CAUSE")
                except ConnectionRefusedError as cause:
                    raise httpx.ConnectError("FICTIONAL_PRIVATE_UPSTREAM_ERROR") from cause
            return Upstream()

    policy = Policy()
    queue = asyncio.Queue()
    await queue.put({"type": "http.request", "body": b'{"model":"fictional"}', "more_body": False})
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "query_string": b"", "headers": []}
    forward = handler(Client(), policy)

    async def send(message):
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            assert (policy.inflight, policy.releases) == (0, 1), "resources released before final send"
            raise BrokenPipeError(32, "FICTIONAL_PRIVATE_SEND_ERROR")

    response = await forward(Request(scope, queue.get), "v1/chat/completions")
    result = await asyncio.gather(response(scope, queue.get, send), return_exceptions=True)
    assert isinstance(result[0], BaseException), "final send failure remains a failure"
    assert (policy.inflight, policy.releases) == (0, 1)
    assert len(closed) == (0 if upstream_fails else 1)
    metrics = forward.fixture_metrics["CLIENT_REQUESTS"].increments
    assert len(metrics) == 1 and metrics[0]["code"] == "499", "failed final send is not accounted as successful upstream status"
    terminal = [line for line in caplog.messages if line.startswith("Forwarding ended")]
    assert len(terminal) == 1 and "stage=downstream_send outcome=cancelled" in terminal[0]
    assert '"type": "BrokenPipeError"' in terminal[0]
    if upstream_fails:
        assert '"type": "ConnectError"' in terminal[0] and '"cause_errno": 111' in terminal[0], "upstream failure survives a later failed error response"
    assert "FICTIONAL_PRIVATE" not in terminal[0]
