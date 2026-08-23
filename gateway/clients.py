"""
Per-client access control for the Sparkstation gateway.

Every request to :8000 carries an API key (Authorization: Bearer <key>, or
x-api-key). This module maps that key to a *client* — a named policy with a
model allow-list and rate/concurrency limits — so we can:

  - attribute load per client (pi vs foxhole vs the scout background agent),
  - stop one noisy client (e.g. scout) from saturating a single-stream GPU
    without touching an interactive one (foxhole), and
  - lock the gateway to known keys once every client has one.

Design constraints (deliberate):
  - NO database, NO Redis. The proxy is one long-lived asyncio process, so all
    limiter state is in-memory and per-process — the right amount of machinery
    for a single-box gateway on a unified-memory GB10 (a Postgres/Prisma stack
    would be a permanent ~1GB tax on the same pool the models live in).
  - Config is a plain YAML file, hot-reloaded on mtime change, so editing
    limits or adding a client never needs a restart.

Config shape (gateway/clients.yaml):

    enforce_auth: false        # true -> a key matching no client is rejected 401
    clients:
      - name: pi
        keys: ["sk-spark-pi-..."]
        allow: ["*"]           # model-name globs; ["*"] = all
        rpm: 0                 # requests/minute, 0 = unlimited
        concurrency: 0         # max simultaneous in-flight, 0 = unlimited
      - name: scout
        keys: ["sk-spark-scout-..."]
        allow: ["default", "dsv4-flash", "bge-m3"]
        rpm: 120
        concurrency: 2
    default:                   # policy for any key not matched above
      name: anonymous
      allow: ["*"]
      rpm: 0
      concurrency: 0
"""
from __future__ import annotations

import fnmatch
import logging
import math
import os
import time
from collections import deque
from typing import Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

logger = logging.getLogger("gateway.clients")

# reasons returned by Client.admit() / Registry.resolve() — also metric labels.
DENY_UNKNOWN_KEY = "unknown_key"
DENY_MODEL_NOT_ALLOWED = "model_not_allowed"
DENY_RATE_LIMITED = "rate_limited"
DENY_CONCURRENCY = "concurrency_limited"


class Client:
    """One named policy + its live limiter state (rpm window + in-flight count)."""

    __slots__ = ("name", "allow", "rpm", "concurrency", "reasoning", "_window", "_inflight")

    def __init__(self, name: str, allow, rpm: int, concurrency: int, reasoning=None):
        self.name = name
        # normalise allow-list to a list of globs; empty or missing -> allow all
        if not allow:
            allow = ["*"]
        self.allow = list(allow)
        self.rpm = int(rpm or 0)
        self.concurrency = int(concurrency or 0)
        # Default reasoning intent to inject when a request carries none:
        # 'off' | 'low' | 'medium' | 'high' | 'xhigh' | None. Applied by the
        # proxy before dialect normalization (see gateway/reasoning.py). Lets a
        # client whose framework marks the model non-reasoning (openclaw sends no
        # thinking control, so qwen falls to its xhigh template default) get a
        # sane cap without touching that framework's config.
        self.reasoning = (str(reasoning).strip().lower() or None) if reasoning else None
        self._window: "deque[float]" = deque()
        self._inflight = 0

    @property
    def inflight(self) -> int:
        return self._inflight

    def allows_model(self, alias: str) -> bool:
        return any(fnmatch.fnmatchcase(alias, pat) for pat in self.allow)

    def admit(self, alias: str, now: float):
        """Try to admit one request for `alias`.

        Returns (ok, reason, retry_after). On ok=True the request is counted:
        an rpm slot is consumed and the in-flight counter incremented — the
        caller MUST call release() when the request finishes. On ok=False
        nothing is consumed.

        Runs entirely between awaits in the single-threaded proxy loop, so the
        check-then-commit is atomic (no lock needed).
        """
        if alias != "none" and not self.allows_model(alias):
            return False, DENY_MODEL_NOT_ALLOWED, None

        # rpm: sliding 60s window
        if self.rpm > 0:
            cutoff = now - 60.0
            w = self._window
            while w and w[0] < cutoff:
                w.popleft()
            if len(w) >= self.rpm:
                retry = max(1, math.ceil(w[0] + 60.0 - now))
                return False, DENY_RATE_LIMITED, retry

        # concurrency: cap simultaneous in-flight requests
        if self.concurrency > 0 and self._inflight >= self.concurrency:
            return False, DENY_CONCURRENCY, 1

        # admit — commit both counters
        if self.rpm > 0:
            self._window.append(now)
        self._inflight += 1
        return True, None, None

    def release(self) -> None:
        if self._inflight > 0:
            self._inflight -= 1


class Registry:
    """Loads clients.yaml, resolves a key to a Client, hot-reloads on change.

    Clients are keyed by API key. State (limiter windows / in-flight counts)
    is preserved across reloads for clients whose name is unchanged, so editing
    the file never resets live counters or drops in-flight bookkeeping.
    """

    def __init__(self, path: str):
        self.path = path
        self.enforce_auth = False
        self._by_key: dict[str, Client] = {}
        self._by_name: dict[str, Client] = {}
        self._default = Client("anonymous", ["*"], 0, 0)
        self._mtime = 0.0
        self.load(force=True)

    # ── config loading ──────────────────────────────────────────────────────
    def maybe_reload(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        if mtime != self._mtime:
            self.load()

    def load(self, force: bool = False) -> None:
        if yaml is None:
            logger.warning("PyYAML unavailable; per-client control disabled (allow-all)")
            return
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            if force:
                logger.info(f"No clients config at {self.path}; allowing all (anonymous)")
            return
        try:
            with open(self.path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to parse {self.path}: {e}; keeping previous config")
            return

        old_by_name = self._by_name
        by_key: dict[str, Client] = {}
        by_name: dict[str, Client] = {}

        def build(spec: dict, fallback_name: str) -> Client:
            name = str(spec.get("name") or fallback_name)
            prev = old_by_name.get(name)
            c = Client(name, spec.get("allow"), spec.get("rpm", 0), spec.get("concurrency", 0),
                       reasoning=spec.get("reasoning"))
            if prev is not None:  # preserve live limiter state across reloads
                c._window = prev._window
                c._inflight = prev._inflight
            return c

        for spec in (cfg.get("clients") or []):
            if not isinstance(spec, dict):
                continue
            c = build(spec, fallback_name="client")
            by_name[c.name] = c
            for k in (spec.get("keys") or []):
                by_key[str(k)] = c

        default_spec = cfg.get("default") or {}
        self._default = build(default_spec, fallback_name="anonymous")
        by_name.setdefault(self._default.name, self._default)

        self.enforce_auth = bool(cfg.get("enforce_auth", False))
        self._by_key = by_key
        self._by_name = by_name
        self._mtime = mtime
        logger.info(
            f"Loaded {len(by_name)} clients from {self.path} "
            f"(enforce_auth={self.enforce_auth}, keys={len(by_key)})"
        )

    # ── resolution ──────────────────────────────────────────────────────────
    def resolve(self, key: Optional[str]) -> Optional[Client]:
        """Map an API key to a Client. Returns None only when enforce_auth is on
        and the key is unknown (caller should 401). Otherwise falls back to the
        default 'anonymous' client so existing callers never break."""
        if key and key in self._by_key:
            return self._by_key[key]
        if self.enforce_auth:
            return None
        return self._default


def extract_key(headers) -> Optional[str]:
    """Pull the API key from an OpenAI-style request. Case-insensitive headers."""
    auth = headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key") or headers.get("api-key")
