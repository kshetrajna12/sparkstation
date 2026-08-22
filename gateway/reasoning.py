"""
Reasoning-control normalization for the Sparkstation gateway.

Problem this solves: every reasoning-capable backend spells "control the
thinking" differently, and the *same* model family changes dialect between
serving builds. Concretely, on Qwen3.8-27B:
  - the SGLang NVFP4 build honors `chat_template_kwargs.enable_thinking` (bool)
    for on/off and TOP-LEVEL `reasoning_effort` for the level (inside
    chat_template_kwargs it 400s);
  - the earlier DFlash2 build honored `chat_template_kwargs.thinking` (bool)
    plus `reasoning_effort` INSIDE chat_template_kwargs, and `thinking_token_budget`.

So a client that sends `thinking:false` (pi/openclaw's old config, the CLAUDE.md
examples) is silently ignored by the NVFP4 build → the model reasons on every
call → agentic clients crawl. Rather than reconfigure pi, foxhole, desire-foundry
and re-break them on the next model swap, the gateway translates whatever the
client sends into the dialect the *currently loaded* backend actually wants.

Dialect per served model is declared explicitly in gateway/reasoning.yaml
(matched by substring on the litellm target model), because "it's Qwen" is NOT
enough to pick the dialect — DFlash2 and NVFP4 are both Qwen and disagree.

A request that carries NO thinking signal is left untouched (the model's own
default applies) — we only rewrite when the client expressed an intent.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

logger = logging.getLogger("gateway.reasoning")

# Keys a client might use to express the on/off intent (first bool wins).
_ENABLE_KEYS_CTK = ("enable_thinking", "thinking")
_ENABLE_KEYS_TOP = ("enable_thinking", "thinking", "reasoning")
# Keys a client might use to express the effort level.
_EFFORT_OFF = {"none", "off", "false", "disable", "disabled"}


def _extract_intent(body: dict):
    """Pull (enabled, effort) intent from anywhere a client might have put it.

    enabled: True/False/None (None = client said nothing about on/off).
    effort:  str level or None. A None/off effort is treated as "no level".
    """
    ctk = body.get("chat_template_kwargs")
    ctk = ctk if isinstance(ctk, dict) else {}

    enabled: Optional[bool] = None
    for k in _ENABLE_KEYS_CTK:
        v = ctk.get(k)
        if isinstance(v, bool):
            enabled = v
            break
    if enabled is None:
        for k in _ENABLE_KEYS_TOP:
            v = body.get(k)
            if isinstance(v, bool):
                enabled = v
                break

    effort = body.get("reasoning_effort")
    if effort is None:
        effort = ctk.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip().lower() in _EFFORT_OFF:
        effort = None
        if enabled is None:
            enabled = False  # "reasoning_effort: none" is an off signal

    return enabled, effort


def _strip_reasoning(body: dict) -> dict:
    """Remove every reasoning-control key so a dialect can re-add its own."""
    for k in ("reasoning_effort", "thinking", "enable_thinking", "reasoning",
              "thinking_token_budget"):
        body.pop(k, None)
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        for k in ("reasoning_effort", "thinking", "enable_thinking",
                  "thinking_token_budget"):
            ctk.pop(k, None)
        if not ctk:
            body.pop("chat_template_kwargs", None)
    return body


def _has_signal(body: dict) -> bool:
    ctk = body.get("chat_template_kwargs")
    ctk = ctk if isinstance(ctk, dict) else {}
    keys = ("reasoning_effort", "thinking", "enable_thinking", "reasoning",
            "thinking_token_budget")
    return any(k in body for k in keys) or any(k in ctk for k in keys)


def _apply_qwen_enable_thinking(body: dict, enabled, effort) -> None:
    """SGLang NVFP4 build: enable_thinking (ctk bool) + top-level reasoning_effort."""
    ctk = body.get("chat_template_kwargs")
    ctk = ctk if isinstance(ctk, dict) else {}
    if enabled is not None:
        ctk["enable_thinking"] = enabled
    if ctk:
        body["chat_template_kwargs"] = ctk
    # effort only matters when thinking is on; inside ctk it 400s -> top level.
    if effort is not None and enabled is not False:
        body["reasoning_effort"] = effort


def _apply_qwen_thinking(body: dict, enabled, effort) -> None:
    """DFlash2 build: thinking (ctk bool) + reasoning_effort INSIDE ctk."""
    ctk = body.get("chat_template_kwargs")
    ctk = ctk if isinstance(ctk, dict) else {}
    if enabled is not None:
        ctk["thinking"] = enabled
    if effort is not None and enabled is not False:
        ctk["reasoning_effort"] = effort
    if ctk:
        body["chat_template_kwargs"] = ctk


_DIALECTS = {
    "qwen_enable_thinking": _apply_qwen_enable_thinking,
    "qwen_thinking": _apply_qwen_thinking,
    "passthrough": None,  # leave the request exactly as the client sent it
}


def normalize(body: dict, dialect: str) -> dict:
    """Rewrite a chat request's reasoning controls into `dialect`.

    No-op when the dialect is passthrough/unknown or the client sent no
    thinking signal at all.
    """
    apply = _DIALECTS.get(dialect)
    if apply is None or not _has_signal(body):
        return body
    enabled, effort = _extract_intent(body)
    _strip_reasoning(body)
    apply(body, enabled, effort)
    return body


class DialectResolver:
    """Maps a request's model alias -> its backend's reasoning dialect.

    Two small inputs, both hot-reloaded on mtime change:
      - gateway/reasoning.yaml: substring rules (served-model -> dialect) + default
      - gateway/litellm.yaml:   alias -> litellm target model string (authoritative
                                 map of what an alias currently resolves to)
    Keying on the *served model* (not the alias) is what makes this survive
    model swaps: `default` can point at NVFP4 today and DFlash2 tomorrow, and
    each is matched to its own dialect.
    """

    def __init__(self, config_path: Optional[str] = None, litellm_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            "SPARKSTATION_REASONING_FILE", "gateway/reasoning.yaml")
        self.litellm_path = litellm_path or os.environ.get(
            "SPARKSTATION_LITELLM_CONFIG", "gateway/litellm.yaml")
        self._rules: list[tuple[str, str]] = []
        self._default = "passthrough"
        self._alias_to_model: dict[str, str] = {}
        self._cfg_mtime = 0.0
        self._litellm_mtime = 0.0
        self._load_config(force=True)
        self._load_litellm(force=True)

    def maybe_reload(self) -> None:
        self._load_config()
        self._load_litellm()

    def _load_config(self, force: bool = False) -> None:
        if yaml is None:
            return
        try:
            mtime = os.path.getmtime(self.config_path)
        except OSError:
            return
        if not force and mtime == self._cfg_mtime:
            return
        try:
            cfg = yaml.safe_load(open(self.config_path)) or {}
            self._rules = [(str(r["match"]), str(r["dialect"]))
                           for r in (cfg.get("dialects") or []) if r.get("match")]
            self._default = str(cfg.get("default", "passthrough"))
            self._cfg_mtime = mtime
            logger.info(f"Loaded {len(self._rules)} reasoning dialect rule(s) "
                        f"from {self.config_path} (default={self._default})")
        except Exception as e:
            logger.error(f"Failed to load {self.config_path}: {e}")

    def _load_litellm(self, force: bool = False) -> None:
        if yaml is None:
            return
        try:
            mtime = os.path.getmtime(self.litellm_path)
        except OSError:
            return
        if not force and mtime == self._litellm_mtime:
            return
        try:
            cfg = yaml.safe_load(open(self.litellm_path)) or {}
            m = {}
            for entry in (cfg.get("model_list") or []):
                name = entry.get("model_name")
                target = (entry.get("litellm_params") or {}).get("model", "")
                if name:
                    m[name] = target
            self._alias_to_model = m
            self._litellm_mtime = mtime
        except Exception as e:
            logger.error(f"Failed to load {self.litellm_path}: {e}")

    def dialect_for(self, alias: str) -> str:
        served = self._alias_to_model.get(alias, alias) or alias
        for match, dialect in self._rules:
            if match in served:
                return dialect
        return self._default
