#!/usr/bin/env python3
"""Exercise the 384-patch GLM vision case and text through an OpenAI endpoint.

GATEWAY_LOCAL_KEY=... python3 scripts/glm53_vision_smoke.py
Use --fixture-dir for the original square.png, triangle.png, circle.png fixtures.
Without it, draw the same 320x200 RGBA pixels using the standard library.
"""

import argparse
import base64
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path

PHOTOS = (
    ("square", (30, 70, 220), "7", ("11111", "00001", "00010", "00100", "01000", "01000", "01000")),
    ("triangle", (20, 160, 60), "3", ("11110", "00001", "00001", "01110", "00001", "00001", "11110")),
    ("circle", (220, 30, 30), "5", ("11111", "10000", "11110", "00001", "00001", "10001", "01110")),
)


def draw_photo(shape, colour, number, glyph):
    pixels = bytearray([255]) * (320 * 200 * 4)

    def paint(x, y, rgb):
        at = (y * 320 + x) * 4
        pixels[at:at + 3] = bytes(rgb)

    for y in range(200):
        for x in range(160):
            dx, dy = x - 80, y - 100
            inside = (
                abs(dx) <= 60 and abs(dy) <= 60 if shape == "square" else
                dx * dx + dy * dy <= 65 * 65 if shape == "circle" else
                -60 <= dy <= 60 and abs(dx) <= (dy + 60) / 2
            )
            if inside:
                paint(x, y, colour)
    for row, bits in enumerate(glyph):
        for column, bit in enumerate(bits):
            if bit == "1":
                for y in range(30 + row * 20, 50 + row * 20):
                    for x in range(190 + column * 20, 210 + column * 20):
                        paint(x, y, (0, 0, 0))

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    rows = b"".join(b"\0" + pixels[y * 1280:(y + 1) * 1280] for y in range(200))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", 320, 200, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--api-key-env", default="GATEWAY_LOCAL_KEY")
    args = parser.parse_args()
    photos = [(args.fixture_dir / (p[0] + ".png")).read_bytes() if args.fixture_dir else draw_photo(*p)
              for p in PHOTOS]
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get(args.api_key_env):
        headers["Authorization"] = "Bearer " + key
    passed = True
    # A unique system prefix forces image encoding instead of a kept prompt hit.
    for index, count in enumerate((0, 1, 3, 1, 3, 0)):
        prompt = ("What is 17 plus 25? Reply with the number only." if not count else
                  "For each image, name the shape, its colour, and the digit. Be brief.")
        parts = [{"type": "text", "text": prompt}]
        parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                   base64.b64encode(photo).decode()}} for photo in photos[:count]]
        body = {"model": args.model, "messages": [
            {"role": "system", "content": "Vision regression " + uuid.uuid4().hex},
            {"role": "user", "content": parts}], "max_tokens": 160, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
        started = time.monotonic()
        try:
            request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                             json.dumps(body).encode(), headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.load(response)
                status = response.status
            answer = data["choices"][0]["message"].get("content") or ""
            required = [r"\b42\b"] if not count else [r"\bblue\b", r"\bsquare\b", r"\b(7|seven)\b"]
            if count == 3:
                required += [r"\bgreen\b", r"\btriangle\b", r"\b(3|three)\b",
                             r"\bred\b", r"\bcircle\b", r"\b(5|five)\b"]
            ok = all(re.search(pattern, answer, re.I) for pattern in required)
            result = {"status": status, "model": data.get("model"), "ok": ok,
                      "answer": answer, "usage": data.get("usage")}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as error:
            result = {"ok": False, "error": str(error)}
        result.update(case=index, images=count, seconds=round(time.monotonic() - started, 3))
        print(json.dumps(result), flush=True)
        passed &= result["ok"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
