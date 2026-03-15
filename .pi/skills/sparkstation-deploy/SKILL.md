---
name: sparkstation-deploy
description: Safe deployment operations for Sparkstation. Profile switching with drain/health-check, rolling model updates, Docker image rebuilds, full stop/start, and rollback. Use when asked to "deploy", "switch profile", "restart sparkstation", "update model", "rebuild container", or "rollback".
---

# Sparkstation Deploy

Safe, verified deployment operations for Sparkstation.

## Usage

```bash
# Switch to a different profile (drains, stops, restarts with new profile)
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py switch-profile image-indexing

# Full restart (stop everything, start fresh)
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py restart

# Full restart with profile
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py restart --profile openclaw

# Stop everything gracefully
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py stop

# Start (uses current/default profile or specify one)
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py start --profile dev

# Rebuild a Docker container (clip, flux, species)
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py rebuild clip

# Health check after deploy (waits for all models to be RUNNING)
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py verify

# Dry run — show what would happen without doing it
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py switch-profile openclaw --dry-run

# Full deploy pipeline: stop → rebuild → start → verify
python3 .pi/skills/sparkstation-deploy/scripts/deploy.py full --profile image-indexing
```

## Operations

### `switch-profile <name>`
1. Pre-flight: validate profile exists, check memory feasibility
2. Stop all running models gracefully
3. Stop supervisor and gateway
4. Start supervisor with new profile
5. Start gateway
6. Wait for all models to be RUNNING
7. Run integration health check

### `restart [--profile name]`
Full stop/start cycle with optional profile change.

### `rebuild <backend>`
Rebuild Docker image for clip/flux/species backend.

### `stop`
Graceful shutdown: stop models, supervisor, gateway, verify containers are down.

### `start [--profile name]`
Start supervisor + gateway, optionally with a profile.

### `verify`
Check all expected models are RUNNING and responsive.

### `full --profile <name>`
Complete pipeline: stop → rebuild (if needed) → start → verify.

## Safety Features

- **Pre-flight checks**: Validates profile, memory, Docker images before any changes
- **Ordered shutdown**: Models stopped before supervisor
- **Health verification**: Waits for all models to be RUNNING after start
- **Timeout protection**: Fails gracefully if models don't start within 10 minutes
- **Dry run**: Preview operations without executing
