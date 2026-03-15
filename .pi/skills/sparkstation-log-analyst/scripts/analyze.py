#!/usr/bin/env python3
"""
Sparkstation Log Analyst

Parses log files and extracts structured events, patterns, and anomalies.
"""

import argparse
import collections
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

LOG_DIR = Path(os.environ.get("SPARKSTATION_LOG_DIR", "data"))

# ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Log patterns
TS_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
LEVEL_PATTERN = re.compile(r" - \w[\w.]* - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - ")

# Event patterns
MODEL_LOADED = re.compile(r"Auto-loaded (?:model|CLIP model|FLUX model|Species model): (\S+)")
MODEL_START = re.compile(r"Starting model: (\S+)")
MODEL_STOPPED = re.compile(r"Stopping model: (\S+)")
MODEL_FAILED = re.compile(r"Failed to auto-load (?:model|CLIP model|FLUX model) (\S+): (.+)")
MODEL_READY = re.compile(r"All models ready \((\d+) running, (\d+) failed\)")
PHASE_START = re.compile(r"=== (PHASE \S+): (.+) ===")
PHASE_COMPLETE = re.compile(r"=== (PHASE \S+ COMPLETE): (.+) ===")
HEALTH_FAIL = re.compile(r"Health check failed for (\S+)")
HEALTH_RECOVERY = re.compile(r"Model (\S+) recovered")
GATEWAY_SYNC_FAIL = re.compile(r"(?:Gateway sync failed|Failed to sync models|Config reload failed): (.+)")
THERMAL_SUSPEND = re.compile(r"[Tt]hermal.*suspend.*(\S+)")
STARTUP_COMPLETE = re.compile(r"Supervisor started.*startup complete")
RECONCILE = re.compile(r"Database reconciliation found inconsistencies: (.+)")
WAIT_MODEL = re.compile(r"Waiting for (\S+) to be ready")
MODEL_RUNNING = re.compile(r"Model (\S+) is now RUNNING on port (\d+)")
RESTART_ATTEMPT = re.compile(r"[Rr]estart(?:ing)? model (\S+)")
CONTAINER_EXIT = re.compile(r"[Cc]ontainer.*(\S+).*exit")
OOM = re.compile(r"OOM|out of memory|OutOfMemoryError", re.IGNORECASE)
SUSPEND_EVENT = re.compile(r"[Ss]uspend(?:ing|ed) model (\S+)")
RESUME_EVENT = re.compile(r"[Rr]esum(?:ing|ed) model (\S+)")


def parse_timestamp(line):
    """Extract datetime from log line."""
    m = TS_PATTERN.match(line)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def parse_level(line):
    """Extract log level."""
    m = LEVEL_PATTERN.search(line)
    return m.group(1) if m else None


def read_logs(all_logs=False):
    """Read log files and return lines."""
    log_files = []
    main_log = LOG_DIR / "sparkstation.log"

    if main_log.exists():
        log_files.append(main_log)

    if all_logs:
        for i in range(1, 10):
            rotated = LOG_DIR / f"sparkstation.log.{i}"
            if rotated.exists():
                log_files.append(rotated)

    # Read in chronological order (rotated logs first)
    lines = []
    for lf in reversed(log_files):
        try:
            with open(lf) as f:
                lines.extend(f.readlines())
        except Exception as e:
            print(f"Warning: Cannot read {lf}: {e}", file=sys.stderr)

    return lines


def analyze(lines, cutoff=None):
    """Analyze log lines and extract events."""
    events = {
        "errors": [],
        "warnings": [],
        "model_starts": [],
        "model_stops": [],
        "model_failures": [],
        "health_failures": [],
        "gateway_sync_issues": [],
        "thermal_events": [],
        "restarts": [],
        "suspends": [],
        "resumes": [],
        "oom_events": [],
        "phases": [],
        "startup_complete": [],
        "reconciliations": [],
    }

    error_counts = collections.Counter()
    warning_counts = collections.Counter()
    model_event_timeline = collections.defaultdict(list)

    for line in lines:
        ts = parse_timestamp(line)
        if ts and cutoff and ts < cutoff:
            continue

        level = parse_level(line)
        ts_str = ts.strftime("%H:%M:%S") if ts else "??:??:??"

        # Errors
        if level in ("ERROR", "CRITICAL"):
            # Deduplicate by message pattern (strip timestamps and IDs)
            msg = line.strip()
            # Normalize for counting
            normalized = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', 'TS', msg)
            normalized = re.sub(r'[a-f0-9]{8}', 'ID', normalized)
            error_counts[normalized] += 1
            events["errors"].append({"ts": ts_str, "line": msg[:200], "timestamp": ts})

        if level == "WARNING":
            msg = line.strip()
            normalized = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+', 'TS', msg)
            normalized = re.sub(r'[a-f0-9]{8}', 'ID', normalized)
            warning_counts[normalized] += 1
            events["warnings"].append({"ts": ts_str, "line": msg[:200], "timestamp": ts})

        # Model events
        m = MODEL_LOADED.search(line)
        if m:
            events["model_starts"].append({"ts": ts_str, "model": m.group(1), "timestamp": ts})
            model_event_timeline[m.group(1)].append(("started", ts))

        m = MODEL_STOPPED.search(line)
        if m:
            events["model_stops"].append({"ts": ts_str, "model": m.group(1), "timestamp": ts})
            model_event_timeline[m.group(1)].append(("stopped", ts))

        m = MODEL_FAILED.search(line)
        if m:
            events["model_failures"].append({"ts": ts_str, "model": m.group(1), "error": m.group(2)[:100], "timestamp": ts})
            model_event_timeline[m.group(1)].append(("failed", ts))

        m = HEALTH_FAIL.search(line)
        if m:
            events["health_failures"].append({"ts": ts_str, "model": m.group(1), "timestamp": ts})

        m = GATEWAY_SYNC_FAIL.search(line)
        if m:
            events["gateway_sync_issues"].append({"ts": ts_str, "error": m.group(1)[:100], "timestamp": ts})

        m = THERMAL_SUSPEND.search(line)
        if m:
            events["thermal_events"].append({"ts": ts_str, "model": m.group(1), "timestamp": ts})

        m = RESTART_ATTEMPT.search(line)
        if m:
            events["restarts"].append({"ts": ts_str, "model": m.group(1), "timestamp": ts})
            model_event_timeline[m.group(1)].append(("restart", ts))

        m = SUSPEND_EVENT.search(line)
        if m:
            events["suspends"].append({"ts": ts_str, "model": m.group(1), "timestamp": ts})

        m = RESUME_EVENT.search(line)
        if m:
            events["resumes"].append({"ts": ts_str, "model": m.group(1), "timestamp": ts})

        if OOM.search(line):
            events["oom_events"].append({"ts": ts_str, "line": line.strip()[:200], "timestamp": ts})

        m = PHASE_START.search(line)
        if m:
            events["phases"].append({"ts": ts_str, "phase": m.group(1), "action": m.group(2), "type": "start", "timestamp": ts})

        m = PHASE_COMPLETE.search(line)
        if m:
            events["phases"].append({"ts": ts_str, "phase": m.group(1), "action": m.group(2), "type": "complete", "timestamp": ts})

        m = STARTUP_COMPLETE.search(line)
        if m:
            events["startup_complete"].append({"ts": ts_str, "timestamp": ts})

        m = RECONCILE.search(line)
        if m:
            events["reconciliations"].append({"ts": ts_str, "details": m.group(1)[:200], "timestamp": ts})

    return events, error_counts, warning_counts, model_event_timeline


def display_errors(events):
    """Display error analysis."""
    errors = events["errors"]
    if not errors:
        print(f"  {GREEN}No errors found ✅{RESET}")
        return

    print(f"  {RED}{len(errors)} error(s){RESET}")
    # Show last 15
    for e in errors[-15:]:
        line = e["line"]
        if len(line) > 120:
            line = line[:117] + "..."
        print(f"    {DIM}{e['ts']}{RESET} {RED}{line}{RESET}")


def display_model_lifecycle(events, timeline):
    """Display model lifecycle events."""
    starts = events["model_starts"]
    failures = events["model_failures"]
    restarts = events["restarts"]

    if not starts and not failures:
        print(f"  {DIM}No model lifecycle events{RESET}")
        return

    print(f"  Starts: {len(starts)}  Failures: {len(failures)}  Restarts: {len(restarts)}")

    if failures:
        print(f"\n  {RED}Failed models:{RESET}")
        for f in failures[-5:]:
            print(f"    {DIM}{f['ts']}{RESET} {RED}{f['model']}: {f['error']}{RESET}")

    if restarts:
        print(f"\n  {YELLOW}Restarts:{RESET}")
        # Count per model
        restart_counts = collections.Counter(r["model"] for r in restarts)
        for model, count in restart_counts.most_common(5):
            print(f"    {YELLOW}{model}: {count} restart(s){RESET}")


def display_health(events):
    """Display health check analysis."""
    failures = events["health_failures"]
    if not failures:
        print(f"  {GREEN}No health check failures ✅{RESET}")
        return

    counts = collections.Counter(f["model"] for f in failures)
    print(f"  {YELLOW}{len(failures)} health check failure(s){RESET}")
    for model, count in counts.most_common():
        print(f"    {YELLOW}{model}: {count} failure(s){RESET}")


def display_gateway_sync(events):
    """Display gateway sync issues."""
    issues = events["gateway_sync_issues"]
    if not issues:
        print(f"  {GREEN}No sync issues ✅{RESET}")
        return

    # Deduplicate by error type
    error_types = collections.Counter(i["error"] for i in issues)
    print(f"  {YELLOW}{len(issues)} sync issue(s){RESET}")
    for error, count in error_types.most_common(3):
        print(f"    {YELLOW}({count}x) {error}{RESET}")


def display_startup(events):
    """Display startup timeline."""
    phases = events["phases"]
    completions = events["startup_complete"]

    if not completions:
        print(f"  {DIM}No startup events found{RESET}")
        return

    for sc in completions[-3:]:
        print(f"  Startup completed at {sc['ts']}")

    if phases:
        print(f"\n  Phases:")
        for p in phases[-10:]:
            icon = "🔄" if p["type"] == "start" else "✅"
            print(f"    {DIM}{p['ts']}{RESET} {icon} {p['phase']}: {p['action']}")


def display_warning_patterns(warning_counts):
    """Show most common warnings."""
    if not warning_counts:
        print(f"  {GREEN}No warnings ✅{RESET}")
        return

    print(f"  {len(warning_counts)} unique warning pattern(s)")
    for pattern, count in warning_counts.most_common(5):
        # Extract just the message part
        msg = pattern
        level_match = LEVEL_PATTERN.search(msg)
        if level_match:
            msg = msg[level_match.end():]
        if len(msg) > 100:
            msg = msg[:97] + "..."
        print(f"    {YELLOW}({count}x) {msg}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Sparkstation Log Analyst")
    parser.add_argument("--hours", type=int, default=24, help="Analyze last N hours")
    parser.add_argument("--all-logs", action="store_true", help="Include rotated logs")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--focus",
        choices=["errors", "restarts", "startup", "health", "sync"],
        help="Focus on specific analysis",
    )
    args = parser.parse_args()

    cutoff = datetime.now() - timedelta(hours=args.hours)
    lines = read_logs(all_logs=args.all_logs)

    if not lines:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)

    events, error_counts, warning_counts, timeline = analyze(lines, cutoff)

    if args.json:
        # Serialize — remove datetime objects
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if k != "timestamp"}
            if isinstance(obj, list):
                return [clean(i) for i in obj]
            return obj
        output = {
            "period_hours": args.hours,
            "total_errors": len(events["errors"]),
            "total_warnings": len(events["warnings"]),
            "events": clean(events),
            "top_errors": dict(error_counts.most_common(10)),
            "top_warnings": dict(warning_counts.most_common(10)),
        }
        print(json.dumps(output, indent=2, default=str))
        return

    print(f"\n{BOLD}═══ SPARKSTATION LOG ANALYSIS ═══{RESET}")
    print(f"  Period: last {args.hours} hours ({cutoff.strftime('%Y-%m-%d %H:%M')} to now)")
    print(f"  Lines analyzed: {len(lines)}")
    print()

    sections = {
        "errors": ("ERRORS", display_errors),
        "restarts": ("MODEL LIFECYCLE", display_model_lifecycle),
        "startup": ("STARTUP TIMELINE", display_startup),
        "health": ("HEALTH CHECKS", display_health),
        "sync": ("GATEWAY SYNC", display_gateway_sync),
    }

    if args.focus:
        name, fn = sections[args.focus]
        print(f"{BOLD}{name}{RESET}")
        if args.focus == "restarts":
            fn(events, timeline)
        else:
            fn(events)
        print()
    else:
        for key, (name, fn) in sections.items():
            print(f"{BOLD}{name}{RESET}")
            if key == "restarts":
                fn(events, timeline)
            else:
                fn(events)
            print()

        # Warning patterns
        print(f"{BOLD}WARNING PATTERNS{RESET}")
        display_warning_patterns(warning_counts)
        print()

        # Special events
        oom = events["oom_events"]
        thermal = events["thermal_events"]
        if oom or thermal:
            print(f"{BOLD}CRITICAL EVENTS{RESET}")
            if oom:
                print(f"  {RED}🚨 {len(oom)} OOM event(s)!{RESET}")
                for o in oom[-3:]:
                    print(f"    {o['ts']}: {o['line'][:100]}")
            if thermal:
                print(f"  {RED}🌡️  {len(thermal)} thermal event(s){RESET}")
            print()

    # Summary
    print(f"{BOLD}{'─' * 40}{RESET}")
    total_issues = len(events["errors"]) + len(events["model_failures"]) + len(events["oom_events"])
    if total_issues == 0:
        print(f"{GREEN}{BOLD}  CLEAN: No errors or failures ✅{RESET}")
    elif events["oom_events"]:
        print(f"{RED}{BOLD}  CRITICAL: OOM events detected! ❌{RESET}")
    else:
        print(f"{YELLOW}{BOLD}  {total_issues} issue(s) found ⚠️{RESET}")
    print()


if __name__ == "__main__":
    main()
