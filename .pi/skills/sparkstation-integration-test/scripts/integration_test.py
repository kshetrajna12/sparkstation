#!/usr/bin/env python3
"""
Sparkstation Integration Tests

End-to-end tests for the full pipeline: Gateway → Supervisor → Backends.
"""

import argparse
import base64
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Optional

# Config
GATEWAY_URL = os.environ.get("SPARKSTATION_GATEWAY_URL", "http://127.0.0.1:8000")
SUPERVISOR_URL = os.environ.get("SPARKSTATION_SUPERVISOR_URL", "http://127.0.0.1:9001")
API_KEY = os.environ.get("SPARKSTATION_API_KEY", "dummy-key")

# ANSI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

VERBOSE = False


def _make_test_png() -> str:
    """Generate a 224x224 solid-color PNG as base64."""
    width, height = 224, 224
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + b"\x80\x40\x20" * width
    def _chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _chunk(b"IDAT", zlib.compress(raw_data))
    png += _chunk(b"IEND", b"")
    return base64.b64encode(png).decode()

TEST_IMAGE_B64 = _make_test_png()


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: Optional[dict] = None


@dataclass
class TestSuite:
    results: list = field(default_factory=list)

    @property
    def passed(self):
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self):
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self):
        return len(self.results)


def http_request(url, method="GET", data=None, headers=None, timeout=30):
    """Make an HTTP request and return (body_dict, status_code, elapsed_ms)."""
    hdrs = headers or {}
    start = time.perf_counter()
    try:
        body = json.dumps(data).encode() if data else None
        if body:
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed = (time.perf_counter() - start) * 1000
        result = json.loads(resp.read())
        return result, resp.status, elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - start) * 1000
        try:
            body = json.loads(e.read())
        except Exception:
            body = {"error": str(e)}
        return body, e.code, elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"error": str(e)}, 0, elapsed


def run_test(name, fn) -> TestResult:
    """Run a single test function."""
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000
        if result is True:
            return TestResult(name=name, passed=True, duration_ms=elapsed)
        elif isinstance(result, str):
            return TestResult(name=name, passed=False, duration_ms=elapsed, message=result)
        elif isinstance(result, tuple):
            return TestResult(name=name, passed=result[0], duration_ms=elapsed, message=result[1] if len(result) > 1 else "")
        return TestResult(name=name, passed=bool(result), duration_ms=elapsed)
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TestResult(name=name, passed=False, duration_ms=elapsed, message=str(e))


# ─── Supervisor Tests ───

def test_supervisor_health():
    data, status, _ = http_request(f"{SUPERVISOR_URL}/health")
    if status != 200:
        return f"Expected 200, got {status}"
    if data.get("status") != "healthy":
        return f"Status is {data.get('status')}, expected 'healthy'"
    return True


def test_supervisor_models():
    data, status, _ = http_request(f"{SUPERVISOR_URL}/models")
    if status != 200:
        return f"Expected 200, got {status}"
    if not isinstance(data, list):
        return f"Expected list, got {type(data)}"
    if len(data) == 0:
        return "No models registered"
    if VERBOSE:
        print(f"    Models: {[m.get('model_name') for m in data]}")
    return True


def test_supervisor_detailed():
    data, status, _ = http_request(f"{SUPERVISOR_URL}/models/detailed")
    if status != 200:
        return f"Expected 200, got {status}"
    models = data.get("models", [])
    if not models:
        return "No models in detailed response"
    for m in models:
        if "status" not in m:
            return f"Model {m.get('alias', '?')} missing status field"
        if "memory_gb" not in m:
            return f"Model {m.get('alias', '?')} missing memory_gb field"
    return True


def test_supervisor_resources():
    data, status, _ = http_request(f"{SUPERVISOR_URL}/resources")
    if status != 200:
        return f"Expected 200, got {status}"
    required = ["unified_memory_used_gb", "unified_memory_limit_gb", "gpu_temperature_c"]
    for key in required:
        if key not in data:
            return f"Missing field: {key}"
    if VERBOSE:
        print(f"    Memory: {data['unified_memory_used_gb']:.1f}/{data['unified_memory_limit_gb']:.1f} GiB")
    return True


# ─── Gateway Tests ───

def test_gateway_models():
    data, status, _ = http_request(
        f"{GATEWAY_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    if status != 200:
        return f"Expected 200, got {status}"
    models = data.get("data", [])
    if not models:
        return "No models in gateway"
    model_ids = [m["id"] for m in models]
    if VERBOSE:
        print(f"    Gateway models: {model_ids}")
    return True


def test_gateway_unknown_model():
    data, status, _ = http_request(
        f"{GATEWAY_URL}/v1/chat/completions",
        method="POST",
        data={
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    # Should return an error, not crash
    if status == 0:
        return "Gateway not reachable"
    if status == 200:
        return "Expected error for unknown model, got 200"
    return True


# ─── Chat Tests ───

def _discover_chat_model():
    data, status, _ = http_request(
        f"{GATEWAY_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    if status == 200:
        for m in data.get("data", []):
            mid = m["id"]
            if mid != "default" and "bge" not in mid and "clip" not in mid:
                return mid
    return None


def test_chat_non_streaming():
    model = _discover_chat_model()
    if not model:
        return "No chat model found"

    data, status, elapsed = http_request(
        f"{GATEWAY_URL}/v1/chat/completions",
        method="POST",
        data={
            "model": model,
            "messages": [{"role": "user", "content": "Say exactly: hello world"}],
            "max_tokens": 32,
            "stream": False,
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )
    if status != 200:
        return f"HTTP {status}: {json.dumps(data)[:200]}"

    # Validate response structure
    choices = data.get("choices", [])
    if not choices:
        return "No choices in response"
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        return "Empty content in response"
    usage = data.get("usage", {})
    if "completion_tokens" not in usage:
        return "Missing completion_tokens in usage"
    if VERBOSE:
        print(f"    Response: {content[:80]}...")
        print(f"    Tokens: {usage.get('completion_tokens', 0)}, {elapsed:.0f}ms")
    return True


def test_chat_streaming():
    model = _discover_chat_model()
    if not model:
        return "No chat model found"

    # Use raw urllib for streaming
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 64,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"

    chunks = 0
    content = ""
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    content += delta["content"]
                    chunks += 1
            except json.JSONDecodeError:
                pass

    if chunks == 0:
        return "No content chunks received"
    if VERBOSE:
        print(f"    Received {chunks} chunks: {content[:80]}...")
    return True


def test_chat_multi_turn():
    model = _discover_chat_model()
    if not model:
        return "No chat model found"

    data, status, _ = http_request(
        f"{GATEWAY_URL}/v1/chat/completions",
        method="POST",
        data={
            "model": model,
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 32,
            "stream": False,
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )
    if status != 200:
        return f"HTTP {status}"

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
    if "alice" not in content:
        return f"Model didn't recall name 'Alice'. Response: {content[:100]}"
    if VERBOSE:
        print(f"    Response: {content[:80]}")
    return True


# ─── Embedding Tests ───

def _discover_embedding_model():
    data, status, _ = http_request(
        f"{GATEWAY_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    if status == 200:
        for m in data.get("data", []):
            mid = m["id"]
            if "bge" in mid or "e5" in mid:
                return mid
    return None


def test_embedding_single():
    model = _discover_embedding_model()
    if not model:
        return "No embedding model found"

    data, status, elapsed = http_request(
        f"{GATEWAY_URL}/v1/embeddings",
        method="POST",
        data={"model": model, "input": "Hello world"},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    if status != 200:
        return f"HTTP {status}: {json.dumps(data)[:200]}"

    emb_data = data.get("data", [])
    if not emb_data:
        return "No embedding data"
    embedding = emb_data[0].get("embedding", [])
    if len(embedding) < 100:
        return f"Embedding too short: {len(embedding)} dims"
    if VERBOSE:
        print(f"    Dimensions: {len(embedding)}, {elapsed:.0f}ms")
    return True


def test_embedding_batch():
    model = _discover_embedding_model()
    if not model:
        return "No embedding model found"

    texts = ["First document", "Second document", "Third document"]
    data, status, _ = http_request(
        f"{GATEWAY_URL}/v1/embeddings",
        method="POST",
        data={"model": model, "input": texts},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    if status != 200:
        return f"HTTP {status}"

    emb_data = data.get("data", [])
    if len(emb_data) != 3:
        return f"Expected 3 embeddings, got {len(emb_data)}"
    # Check indices
    indices = sorted([e["index"] for e in emb_data])
    if indices != [0, 1, 2]:
        return f"Wrong indices: {indices}"
    return True


# ─── CLIP Tests ───

def _discover_clip_url():
    """Get CLIP backend direct URL."""
    try:
        data, status, _ = http_request(f"{SUPERVISOR_URL}/models")
        if status == 200:
            for m in data:
                if m.get("model_name") == "clip-vit":
                    return m["api_base"]
    except Exception:
        pass
    return "http://127.0.0.1:8003/v1"


def test_clip_text():
    data, status, elapsed = http_request(
        f"{GATEWAY_URL}/v1/embeddings",
        method="POST",
        data={"model": "clip-vit", "input": "a photo of a cat"},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    if status != 200:
        return f"HTTP {status}: {json.dumps(data)[:200]}"

    emb = data.get("data", [{}])[0].get("embedding", [])
    if len(emb) != 768:
        return f"Expected 768 dims, got {len(emb)}"
    if VERBOSE:
        print(f"    CLIP text embedding: {len(emb)} dims, {elapsed:.0f}ms")
    return True


def test_clip_image():
    clip_url = _discover_clip_url()

    data, status, elapsed = http_request(
        f"{clip_url}/embeddings",
        method="POST",
        data={
            "model": "openai/clip-vit-large-patch14",
            "input": [{"image": TEST_IMAGE_B64}],
        },
    )
    if status != 200:
        return f"HTTP {status}: {json.dumps(data)[:200]}"

    emb = data.get("data", [{}])[0].get("embedding", [])
    if len(emb) != 768:
        return f"Expected 768 dims, got {len(emb)}"
    if VERBOSE:
        print(f"    CLIP image embedding: {len(emb)} dims, {elapsed:.0f}ms")
    return True


# ─── Vision Tests ───

def test_vision_chat():
    model = _discover_chat_model()
    if not model:
        return "No vision model found"

    data, status, elapsed = http_request(
        f"{GATEWAY_URL}/v1/chat/completions",
        method="POST",
        data={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Answer in one word."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TEST_IMAGE_B64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 32,
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=120,
    )
    if status != 200:
        return f"HTTP {status}: {json.dumps(data)[:200]}"

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        return "Empty response"
    if VERBOSE:
        print(f"    Vision response: {content[:80]}, {elapsed:.0f}ms")
    return True


# ─── Test Runner ───

ALL_TESTS = {
    "supervisor": [
        ("supervisor_health", test_supervisor_health),
        ("supervisor_models", test_supervisor_models),
        ("supervisor_detailed", test_supervisor_detailed),
        ("supervisor_resources", test_supervisor_resources),
    ],
    "gateway": [
        ("gateway_models", test_gateway_models),
        ("gateway_unknown_model", test_gateway_unknown_model),
    ],
    "chat": [
        ("chat_non_streaming", test_chat_non_streaming),
        ("chat_streaming", test_chat_streaming),
        ("chat_multi_turn", test_chat_multi_turn),
    ],
    "embedding": [
        ("embedding_single", test_embedding_single),
        ("embedding_batch", test_embedding_batch),
    ],
    "clip": [
        ("clip_text", test_clip_text),
        ("clip_image", test_clip_image),
    ],
    "vision": [
        ("vision_chat", test_vision_chat),
    ],
}


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Sparkstation Integration Tests")
    parser.add_argument(
        "--test",
        choices=list(ALL_TESTS.keys()),
        help="Run specific test group",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON report")

    args = parser.parse_args()
    VERBOSE = args.verbose

    suite = TestSuite()

    if args.test:
        groups = {args.test: ALL_TESTS[args.test]}
    else:
        groups = ALL_TESTS

    if not args.json:
        print(f"\n{BOLD}═══ SPARKSTATION INTEGRATION TESTS ═══{RESET}\n")

    for group_name, tests in groups.items():
        if not args.json:
            print(f"{BOLD}{group_name.upper()}{RESET}")

        for test_name, test_fn in tests:
            result = run_test(test_name, test_fn)
            suite.results.append(result)

            if not args.json:
                if result.passed:
                    print(f"  {GREEN}✅ {test_name}{RESET} ({result.duration_ms:.0f}ms)")
                else:
                    print(f"  {RED}❌ {test_name}{RESET} ({result.duration_ms:.0f}ms)")
                    if result.message:
                        print(f"     {RED}{result.message}{RESET}")

        if not args.json:
            print()

    if args.json:
        report = {
            "total": suite.total,
            "passed": suite.passed,
            "failed": suite.failed,
            "tests": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": round(r.duration_ms, 1),
                    "message": r.message,
                }
                for r in suite.results
            ],
        }
        print(json.dumps(report, indent=2))
    else:
        print(f"{BOLD}{'─' * 40}{RESET}")
        if suite.failed == 0:
            print(f"{GREEN}{BOLD}  ALL {suite.total} TESTS PASSED ✅{RESET}")
        else:
            print(f"{RED}{BOLD}  {suite.failed}/{suite.total} TESTS FAILED ❌{RESET}")
        print()

    sys.exit(0 if suite.failed == 0 else 1)


if __name__ == "__main__":
    main()
