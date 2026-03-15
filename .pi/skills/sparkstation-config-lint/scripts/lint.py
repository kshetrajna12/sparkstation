#!/usr/bin/env python3
"""
Sparkstation Config Linter

Validates models.yaml for correctness, consistency, and resource feasibility.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Try to import yaml, fallback to manual parsing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Default limits from .env
DEFAULT_HARD_LIMIT_GB = 113
DEFAULT_MAX_MODELS = 5
DEFAULT_PORT_START = 8001
DEFAULT_PORT_END = 8100

VALID_BACKENDS = {"vllm", "sglang", "clip", "flux", "species"}
VALID_MODEL_TYPES = {"chat", "embedding", "image", "detection"}
CHAT_ONLY_ARGS = {"tool_call_parser", "reasoning_parser", "reasoning_parser_plugin"}


class LintResult:
    def __init__(self):
        self.errors = []    # Must fix
        self.warnings = []  # Should fix
        self.info = []      # FYI

    def error(self, msg, profile=None, model=None):
        prefix = ""
        if profile:
            prefix += f"[{profile}] "
        if model:
            prefix += f"{model}: "
        self.errors.append(f"{prefix}{msg}")

    def warn(self, msg, profile=None, model=None):
        prefix = ""
        if profile:
            prefix += f"[{profile}] "
        if model:
            prefix += f"{model}: "
        self.warnings.append(f"{prefix}{msg}")

    def note(self, msg, profile=None, model=None):
        prefix = ""
        if profile:
            prefix += f"[{profile}] "
        if model:
            prefix += f"{model}: "
        self.info.append(f"{prefix}{msg}")

    @property
    def ok(self):
        return len(self.errors) == 0


def load_config():
    """Load models.yaml."""
    yaml_path = PROJECT_ROOT / "models.yaml"
    if not yaml_path.exists():
        print(f"ERROR: models.yaml not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)

    if HAS_YAML:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    else:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)


def load_env_limits():
    """Load memory limits from .env."""
    env_path = PROJECT_ROOT / ".env"
    limits = {
        "hard_limit_gb": DEFAULT_HARD_LIMIT_GB,
        "max_models": DEFAULT_MAX_MODELS,
        "port_start": DEFAULT_PORT_START,
        "port_end": DEFAULT_PORT_END,
    }

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" not in line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.split("#")[0].strip()  # Remove inline comments

                if key == "MEMORY_HARD_LIMIT_GB":
                    limits["hard_limit_gb"] = float(value)
                elif key == "MAX_RESIDENT_MODELS":
                    limits["max_models"] = int(value)
                elif key == "MODEL_PORT_RANGE_START":
                    limits["port_start"] = int(value)
                elif key == "MODEL_PORT_RANGE_END":
                    limits["port_end"] = int(value)

    return limits


def lint_model(model, profile_name, result):
    """Lint a single model entry."""
    name = model.get("name", "")
    alias = model.get("alias", name)
    backend = model.get("backend", "")
    model_type = model.get("model_type", "")
    memory_gb = model.get("memory_gb")
    extra_args = model.get("extra_args", {}) or {}

    # Required: name
    if not name:
        result.error("Missing 'name' field", profile=profile_name)
        return

    # Required: backend
    if not backend:
        result.error("Missing 'backend' field", profile=profile_name, model=alias)
    elif backend not in VALID_BACKENDS:
        result.error(f"Invalid backend '{backend}'. Valid: {VALID_BACKENDS}", profile=profile_name, model=alias)

    # model_type validation
    if model_type and model_type not in VALID_MODEL_TYPES:
        result.error(f"Invalid model_type '{model_type}'. Valid: {VALID_MODEL_TYPES}", profile=profile_name, model=alias)

    # Memory declaration
    if memory_gb is None:
        if backend in ("vllm", "sglang"):
            result.warn("No memory_gb declared — supervisor will estimate (may be inaccurate)", profile=profile_name, model=alias)
    elif memory_gb <= 0:
        result.error(f"memory_gb must be positive, got {memory_gb}", profile=profile_name, model=alias)

    # Backend-specific checks
    if backend in ("vllm", "sglang"):
        # Chat-only args on embedding models
        if model_type == "embedding":
            for arg in CHAT_ONLY_ARGS:
                if arg in extra_args:
                    result.warn(f"'{arg}' in extra_args is only relevant for chat models", profile=profile_name, model=alias)

        # max_model_len sanity
        max_len = extra_args.get("max_model_len")
        if max_len and max_len > 131072:
            result.warn(f"max_model_len={max_len} is very large, may cause OOM", profile=profile_name, model=alias)

    elif backend == "clip":
        if model_type and model_type != "embedding":
            result.warn(f"CLIP backend should have model_type='embedding', got '{model_type}'", profile=profile_name, model=alias)

    elif backend == "flux":
        if model_type and model_type != "image":
            result.warn(f"FLUX backend should have model_type='image', got '{model_type}'", profile=profile_name, model=alias)

    # Docker image validation
    docker_image = model.get("docker_image")
    if docker_image:
        result.note(f"Custom docker image: {docker_image}", profile=profile_name, model=alias)

    # Quantization
    quant = model.get("quantization", "none")
    if quant not in ("none", "awq", "gptq", "fp8", "int4", "int8"):
        result.warn(f"Unusual quantization: '{quant}'", profile=profile_name, model=alias)


def lint_profile(profile_name, models, limits, result):
    """Lint a complete profile."""

    if not models:
        result.warn("Empty profile", profile=profile_name)
        return

    # Check total memory
    total_mem = 0
    has_mem = True
    for m in models:
        mem = m.get("memory_gb")
        if mem:
            total_mem += mem
        else:
            has_mem = False

    if has_mem:
        if total_mem > limits["hard_limit_gb"]:
            result.error(
                f"Total memory {total_mem:.1f} GB exceeds hard limit {limits['hard_limit_gb']:.1f} GB",
                profile=profile_name,
            )
        elif total_mem > limits["hard_limit_gb"] * 0.9:
            result.warn(
                f"Total memory {total_mem:.1f} GB is >90% of limit {limits['hard_limit_gb']:.1f} GB",
                profile=profile_name,
            )
        else:
            result.note(
                f"Total memory: {total_mem:.1f} / {limits['hard_limit_gb']:.1f} GB ({total_mem/limits['hard_limit_gb']*100:.0f}%)",
                profile=profile_name,
            )

    # Check model count
    if len(models) > limits["max_models"]:
        result.warn(
            f"{len(models)} models exceeds MAX_RESIDENT_MODELS={limits['max_models']}",
            profile=profile_name,
        )

    # Check port range
    available_ports = limits["port_end"] - limits["port_start"]
    if len(models) > available_ports:
        result.error(
            f"{len(models)} models exceeds available ports ({available_ports})",
            profile=profile_name,
        )

    # Check alias collisions
    aliases = []
    for m in models:
        alias = m.get("alias", m.get("name", ""))
        if alias in aliases:
            result.error(f"Duplicate alias '{alias}'", profile=profile_name)
        aliases.append(alias)

    # Check for default model
    defaults = [m for m in models if m.get("default")]
    if len(defaults) > 1:
        result.warn("Multiple models marked as default", profile=profile_name)

    # Lint each model
    for m in models:
        lint_model(m, profile_name, result)


def lint_cross_profile(config, result):
    """Check cross-profile consistency."""
    profiles = config.get("profiles", {})
    
    # Track alias → backend across profiles
    alias_backends = {}
    for pname, models in profiles.items():
        for m in models:
            alias = m.get("alias", m.get("name", ""))
            backend = m.get("backend", "")
            if alias in alias_backends:
                prev_backend, prev_profile = alias_backends[alias]
                if backend != prev_backend:
                    result.warn(
                        f"Alias '{alias}' has backend '{backend}' in [{pname}] but '{prev_backend}' in [{prev_profile}]"
                    )
            else:
                alias_backends[alias] = (backend, pname)


def main():
    parser = argparse.ArgumentParser(description="Sparkstation Config Linter")
    parser.add_argument("--profile", help="Lint specific profile only")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    config = load_config()
    limits = load_env_limits()
    result = LintResult()

    if args.profile:
        profiles_to_check = {args.profile: config.get("profiles", {}).get(args.profile, [])}
        if not profiles_to_check[args.profile]:
            print(f"ERROR: Profile '{args.profile}' not found", file=sys.stderr)
            sys.exit(1)
    else:
        profiles_to_check = config.get("profiles", {})
        # Also check autoload
        autoload = config.get("autoload", {}).get("models", [])
        if autoload:
            profiles_to_check = {"autoload": autoload, **profiles_to_check}

    for pname, models in profiles_to_check.items():
        lint_profile(pname, models, limits, result)

    # Cross-profile checks
    if not args.profile:
        lint_cross_profile(config, result)

    if args.json:
        output = {
            "ok": result.ok,
            "errors": result.errors,
            "warnings": result.warnings,
            "info": result.info,
        }
        print(json.dumps(output, indent=2))
        return

    # Display
    print(f"\n{BOLD}═══ SPARKSTATION CONFIG LINT ═══{RESET}\n")
    print(f"  Limits: {limits['hard_limit_gb']:.0f} GB hard limit, {limits['max_models']} max models, ports {limits['port_start']}-{limits['port_end']}")
    print()

    if result.errors:
        print(f"{RED}{BOLD}ERRORS ({len(result.errors)}){RESET}")
        for e in result.errors:
            print(f"  {RED}❌ {e}{RESET}")
        print()

    if result.warnings:
        print(f"{YELLOW}{BOLD}WARNINGS ({len(result.warnings)}){RESET}")
        for w in result.warnings:
            print(f"  {YELLOW}⚠️  {w}{RESET}")
        print()

    if result.info:
        print(f"{DIM}INFO ({len(result.info)}){RESET}")
        for i in result.info:
            print(f"  {DIM}ℹ️  {i}{RESET}")
        print()

    print(f"{BOLD}{'─' * 40}{RESET}")
    if result.ok:
        if result.warnings:
            print(f"{YELLOW}{BOLD}  PASSED with {len(result.warnings)} warning(s) ⚠️{RESET}")
        else:
            print(f"{GREEN}{BOLD}  ALL CHECKS PASSED ✅{RESET}")
    else:
        print(f"{RED}{BOLD}  {len(result.errors)} ERROR(S) FOUND ❌{RESET}")
    print()

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
