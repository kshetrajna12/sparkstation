---
name: sparkstation-log-analyst
description: Analyze Sparkstation logs to find patterns, errors, model restarts, startup times, and anomalies. Use when asked to "analyze logs", "what errors happened", "why did X restart", "show startup times", or "log report".
---

# Sparkstation Log Analyst

Parses sparkstation logs and extracts structured events, patterns, and anomalies.

## Usage

```bash
# Full log analysis (last 24 hours)
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py

# Last N hours
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py --hours 4

# Specific analysis
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py --focus errors
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py --focus restarts
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py --focus startup
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py --focus health

# Include rotated logs
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py --all-logs

# JSON output
python3 .pi/skills/sparkstation-log-analyst/scripts/analyze.py --json
```

## What It Extracts

- **Error timeline**: All errors/criticals with timestamps and context
- **Model lifecycle**: Start/stop/suspend/resume/fail events per model
- **Restart patterns**: Models that restarted, frequency, causes
- **Startup times**: How long each model took to become RUNNING
- **Health check failures**: Failed health probes and their frequency
- **Gateway sync issues**: Sync failures and patterns
- **Thermal events**: Temperature-related suspensions
