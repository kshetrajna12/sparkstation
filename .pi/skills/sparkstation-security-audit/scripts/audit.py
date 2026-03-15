#!/usr/bin/env python3
"""
Sparkstation Security Audit

Reviews configuration, runtime, and deployment for security issues.
"""

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class AuditResult:
    def __init__(self):
        self.findings = []  # (severity, category, message)

    def critical(self, category, msg):
        self.findings.append(("CRITICAL", category, msg))

    def high(self, category, msg):
        self.findings.append(("HIGH", category, msg))

    def medium(self, category, msg):
        self.findings.append(("MEDIUM", category, msg))

    def low(self, category, msg):
        self.findings.append(("LOW", category, msg))

    def info(self, category, msg):
        self.findings.append(("INFO", category, msg))

    def ok(self, category, msg):
        self.findings.append(("OK", category, msg))

    @property
    def criticals(self):
        return [f for f in self.findings if f[0] == "CRITICAL"]

    @property
    def highs(self):
        return [f for f in self.findings if f[0] == "HIGH"]

    @property
    def mediums(self):
        return [f for f in self.findings if f[0] == "MEDIUM"]


def check_env_file(result):
    """Check .env file for security issues."""
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        result.info("ENV", ".env file not found")
        return

    # Check permissions
    st = os.stat(env_path)
    mode = stat.S_IMODE(st.st_mode)
    if mode & stat.S_IROTH:
        result.high("ENV", f".env is world-readable (mode: {oct(mode)}). Run: chmod 600 .env")
    elif mode & stat.S_IRGRP:
        result.medium("ENV", f".env is group-readable (mode: {oct(mode)}). Consider: chmod 600 .env")
    else:
        result.ok("ENV", f".env permissions OK ({oct(mode)})")

    # Check .gitignore
    gitignore = PROJECT_ROOT / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".env" in content:
            result.ok("ENV", ".env is in .gitignore")
        else:
            result.high("ENV", ".env is NOT in .gitignore — secrets may be committed!")
    else:
        result.medium("ENV", "No .gitignore found")

    # Parse .env for secrets
    with open(env_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.split("#")[0].strip()

            # Default/weak master key
            if key == "LITELLM_MASTER_KEY":
                if value in ("sk-sparkstation-admin", "sk-1234", "test", "admin"):
                    result.high("AUTH", f"Default LITELLM_MASTER_KEY: '{value}'. Change for production!")
                elif len(value) < 16:
                    result.medium("AUTH", f"LITELLM_MASTER_KEY is short ({len(value)} chars). Use 32+ chars.")
                else:
                    result.ok("AUTH", "LITELLM_MASTER_KEY appears non-default")

            # HF token exposure
            if key == "HF_TOKEN" and value and value.startswith("hf_"):
                result.medium("SECRETS", f"HF_TOKEN present in .env (starts with hf_...{value[-4:]})")

            # Check binding
            if key == "HOST":
                if value in ("0.0.0.0", ""):
                    result.critical("NETWORK", "HOST=0.0.0.0 — supervisor exposed to network! Use 127.0.0.1")
                elif value == "127.0.0.1":
                    result.ok("NETWORK", "Supervisor bound to localhost only")


def check_network_binding(result):
    """Check what ports are actually listening."""
    try:
        out = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if out.returncode != 0:
            result.info("NETWORK", "Cannot check listening ports (ss failed)")
            return

        # Check sparkstation-related ports
        for line in out.stdout.split("\n"):
            if not line.strip():
                continue

            # Look for ports 8000-8100, 9001
            port_match = re.search(r"(\*|0\.0\.0\.0|127\.0\.0\.1|::):(\d+)", line)
            if port_match:
                bind_addr = port_match.group(1)
                port = int(port_match.group(2))

                if port in range(8000, 8101) or port == 9001:
                    if bind_addr in ("*", "0.0.0.0", "::"):
                        result.high("NETWORK", f"Port {port} is bound to ALL interfaces ({bind_addr}:{port})")
                    else:
                        result.ok("NETWORK", f"Port {port} bound to {bind_addr} (localhost only)")

    except FileNotFoundError:
        result.info("NETWORK", "'ss' command not found, skipping port check")
    except Exception as e:
        result.info("NETWORK", f"Port check error: {e}")


def check_container_security(result):
    """Check Docker container security settings."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=sparkstation-"],
            capture_output=True, text=True, timeout=10,
        )

        if out.returncode != 0 or not out.stdout.strip():
            result.info("CONTAINERS", "No sparkstation containers running")
            return

        for cid in out.stdout.strip().split("\n"):
            cid = cid.strip()
            if not cid:
                continue

            inspect = subprocess.run(
                ["docker", "inspect", cid],
                capture_output=True, text=True, timeout=10,
            )
            if inspect.returncode != 0:
                continue

            try:
                info = json.loads(inspect.stdout)[0]
            except (json.JSONDecodeError, IndexError):
                continue

            name = info.get("Name", "").lstrip("/")
            config = info.get("HostConfig", {})

            # Privileged mode
            if config.get("Privileged"):
                result.high("CONTAINERS", f"{name}: Running in PRIVILEGED mode!")
            else:
                result.ok("CONTAINERS", f"{name}: Not privileged")

            # Network mode
            network = config.get("NetworkMode", "")
            if network == "host":
                result.medium("CONTAINERS", f"{name}: Using host network mode")

            # User
            user = info.get("Config", {}).get("User", "")
            if not user or user == "root" or user == "0":
                result.low("CONTAINERS", f"{name}: Running as root (consider non-root user)")

            # Capabilities
            cap_add = config.get("CapAdd") or []
            if "SYS_ADMIN" in cap_add:
                result.high("CONTAINERS", f"{name}: Has SYS_ADMIN capability")
            if "NET_ADMIN" in cap_add:
                result.medium("CONTAINERS", f"{name}: Has NET_ADMIN capability")

    except FileNotFoundError:
        result.info("CONTAINERS", "Docker not found")
    except Exception as e:
        result.info("CONTAINERS", f"Container check error: {e}")


def check_secrets_in_code(result):
    """Check for secrets in tracked files."""
    try:
        # Get list of tracked files
        tracked = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, timeout=10, cwd=PROJECT_ROOT,
        )
        if tracked.returncode != 0:
            result.info("SECRETS", "Not a git repo, skipping code scan")
            return

        secret_patterns = [
            (r'hf_[a-zA-Z0-9]{20,}', "HuggingFace token"),
            (r'sk-[a-zA-Z0-9]{20,}', "API key (sk-)"),
            (r'ghp_[a-zA-Z0-9]{20,}', "GitHub PAT"),
            (r'AKIA[0-9A-Z]{16}', "AWS access key"),
            (r'-----BEGIN (?:RSA )?PRIVATE KEY-----', "Private key"),
        ]

        for filepath in tracked.stdout.strip().split("\n"):
            if not filepath.strip():
                continue
            # Skip binary and large files
            if filepath.endswith((".lock", ".pyc", ".db", ".log", ".json")):
                continue
            if filepath.startswith(".venv/"):
                continue

            full_path = PROJECT_ROOT / filepath
            if not full_path.exists() or full_path.stat().st_size > 100000:
                continue

            try:
                content = full_path.read_text(errors="ignore")
                for pattern, desc in secret_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Don't flag .env.example or comments
                        if filepath.endswith(".example"):
                            continue
                        if filepath == ".env":
                            continue  # Already checked separately
                        result.high("SECRETS", f"{filepath}: Found {desc} ({len(matches)} occurrence(s))")
            except Exception:
                pass

        # Check if .env is tracked
        if ".env" in tracked.stdout:
            result.critical("SECRETS", ".env file is tracked by git! Run: git rm --cached .env")
        else:
            result.ok("SECRETS", ".env is not tracked by git")

    except Exception as e:
        result.info("SECRETS", f"Code scan error: {e}")


def check_file_permissions(result):
    """Check sensitive file permissions."""
    sensitive = [
        ("data/sparkstation.db", "Database"),
        ("data/sparkstation.log", "Log file"),
    ]

    for filepath, desc in sensitive:
        full_path = PROJECT_ROOT / filepath
        if full_path.exists():
            st = os.stat(full_path)
            mode = stat.S_IMODE(st.st_mode)
            if mode & stat.S_IROTH:
                result.low("PERMISSIONS", f"{filepath} ({desc}) is world-readable ({oct(mode)})")
            else:
                result.ok("PERMISSIONS", f"{filepath} permissions OK ({oct(mode)})")


def check_auth_config(result):
    """Check authentication configuration."""
    auth_file = PROJECT_ROOT / "supervisor" / "auth.py"
    if auth_file.exists():
        content = auth_file.read_text()
        if "api_key" in content.lower() or "require_api_key" in content.lower():
            result.ok("AUTH", "Supervisor has API key authentication")
        else:
            result.high("AUTH", "No API key auth found in supervisor/auth.py")
    else:
        result.high("AUTH", "supervisor/auth.py not found — no auth?")

    # Check if management endpoints require auth
    main_file = PROJECT_ROOT / "supervisor" / "main.py"
    if main_file.exists():
        content = main_file.read_text()
        # Check protected endpoints
        protected = content.count("Depends(require_api_key)")
        unprotected_mutations = 0
        for line in content.split("\n"):
            if ("@app.post" in line or "@app.delete" in line) and "start" in line or "stop" in line or "suspend" in line:
                # Check if next few lines have require_api_key
                pass

        if protected >= 3:
            result.ok("AUTH", f"Found {protected} endpoints protected with API key")
        else:
            result.medium("AUTH", f"Only {protected} endpoints have API key protection")


def main():
    parser = argparse.ArgumentParser(description="Sparkstation Security Audit")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    result = AuditResult()

    # Run all checks
    check_env_file(result)
    check_network_binding(result)
    check_container_security(result)
    check_secrets_in_code(result)
    check_file_permissions(result)
    check_auth_config(result)

    if args.json:
        output = {
            "findings": [
                {"severity": s, "category": c, "message": m}
                for s, c, m in result.findings
            ],
            "summary": {
                "critical": len(result.criticals),
                "high": len(result.highs),
                "medium": len(result.mediums),
            },
        }
        print(json.dumps(output, indent=2))
        return

    # Display
    print(f"\n{BOLD}═══ SPARKSTATION SECURITY AUDIT ═══{RESET}\n")

    severity_colors = {
        "CRITICAL": RED + BOLD,
        "HIGH": RED,
        "MEDIUM": YELLOW,
        "LOW": DIM,
        "INFO": DIM,
        "OK": GREEN,
    }

    severity_icons = {
        "CRITICAL": "🚨",
        "HIGH": "❌",
        "MEDIUM": "⚠️ ",
        "LOW": "📋",
        "INFO": "ℹ️ ",
        "OK": "✅",
    }

    # Group by category
    categories = {}
    for severity, category, msg in result.findings:
        if category not in categories:
            categories[category] = []
        categories[category].append((severity, msg))

    for category, findings in categories.items():
        print(f"{BOLD}{category}{RESET}")
        for severity, msg in findings:
            color = severity_colors.get(severity, "")
            icon = severity_icons.get(severity, "")
            print(f"  {icon} {color}[{severity}] {msg}{RESET}")
        print()

    # Summary
    print(f"{BOLD}{'─' * 50}{RESET}")
    crits = len(result.criticals)
    highs = len(result.highs)
    meds = len(result.mediums)

    if crits > 0:
        print(f"{RED}{BOLD}  🚨 {crits} CRITICAL, {highs} HIGH, {meds} MEDIUM findings{RESET}")
    elif highs > 0:
        print(f"{RED}{BOLD}  ❌ {highs} HIGH, {meds} MEDIUM findings{RESET}")
    elif meds > 0:
        print(f"{YELLOW}{BOLD}  ⚠️  {meds} MEDIUM findings{RESET}")
    else:
        print(f"{GREEN}{BOLD}  ✅ No significant security issues found{RESET}")
    print()

    sys.exit(1 if crits > 0 or highs > 0 else 0)


if __name__ == "__main__":
    main()
