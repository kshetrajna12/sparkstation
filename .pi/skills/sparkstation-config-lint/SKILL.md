---
name: sparkstation-config-lint
description: Validate models.yaml profiles for correctness, consistency, and resource feasibility. Checks memory limits, alias collisions, required args, backend compatibility. Use when asked to "lint config", "validate models.yaml", "check config", or before deploying a new profile.
---

# Sparkstation Config Lint

Validates `models.yaml` for correctness before deployment.

## Usage

```bash
# Lint all profiles
python3 .pi/skills/sparkstation-config-lint/scripts/lint.py

# Lint specific profile
python3 .pi/skills/sparkstation-config-lint/scripts/lint.py --profile image-indexing

# JSON output
python3 .pi/skills/sparkstation-config-lint/scripts/lint.py --json
```

## What It Checks

1. **Memory feasibility**: Total memory per profile fits within MEMORY_HARD_LIMIT_GB
2. **Alias collisions**: No duplicate aliases within a profile
3. **Required fields**: Each model has backend, name at minimum
4. **Backend compatibility**: Extra args match the backend type (e.g., tool_call_parser only for chat models)
5. **Cross-profile consistency**: Same model alias doesn't have conflicting backends across profiles
6. **Port range**: Model count doesn't exceed available port range
7. **Docker images**: Validates docker_image references exist for models that need them
8. **Memory declarations**: Warns on missing memory_gb for non-trivial models
