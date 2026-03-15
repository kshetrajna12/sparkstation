---
name: sparkstation-security-audit
description: Security audit for Sparkstation. Checks API auth, exposed ports, env file permissions, container privileges, network binding, and secrets. Use when asked to "audit security", "check for vulnerabilities", "is this secure", or "review security settings".
---

# Sparkstation Security Audit

Reviews Sparkstation configuration and runtime for security issues.

## Usage

```bash
# Full security audit
python3 .pi/skills/sparkstation-security-audit/scripts/audit.py

# JSON output
python3 .pi/skills/sparkstation-security-audit/scripts/audit.py --json
```

## What It Checks

1. **API Authentication**: Master key strength, default keys, auth bypass
2. **Network Binding**: Services bound to localhost vs 0.0.0.0
3. **Environment File**: .env permissions, exposed secrets, .gitignore
4. **Container Security**: Privileged mode, capabilities, user namespacing
5. **Port Exposure**: Unexpected ports open externally
6. **Secrets in Code**: HF tokens, API keys in tracked files
7. **File Permissions**: Data directory, log files, database
8. **Default Credentials**: Check for default/placeholder API keys
