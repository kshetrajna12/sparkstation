---
name: sparkstation-profile-memory
description: Profile actual GPU memory usage per model and compare to declared memory_gb in models.yaml. Identifies waste, tight allocations, and optimization opportunities. Use when asked to "profile memory", "check memory usage", "optimize memory", or "how much memory is each model using".
---

# Sparkstation Memory Profiler

Measures actual memory usage per model and compares to declared `memory_gb` in `models.yaml`.

## Usage

```bash
# Profile all running models
python3 .pi/skills/sparkstation-profile-memory/scripts/profile_memory.py

# JSON output
python3 .pi/skills/sparkstation-profile-memory/scripts/profile_memory.py --json

# Include models.yaml analysis (declared vs actual)
python3 .pi/skills/sparkstation-profile-memory/scripts/profile_memory.py --compare
```

## What It Measures

For each running model:
- **Docker stats memory** — RSS from `docker stats` (actual process memory)
- **Declared memory_gb** — from `models.yaml` profile config
- **Delta** — over/under-allocation
- **Recommendations** — tighten or loosen allocations

## Output

```
═══ SPARKSTATION MEMORY PROFILE ═══

MODEL MEMORY USAGE
  qwen3-vl-30b    declared: 55.0 GB  actual: 48.2 GB  delta: -6.8 GB  ⚠️  over-allocated
  bge-m3           declared:  2.5 GB  actual:  2.1 GB  delta: -0.4 GB  ✅ OK
  clip-vit         declared:  5.0 GB  actual:  1.8 GB  delta: -3.2 GB  ⚠️  over-allocated
  species-detect   declared:  5.0 GB  actual:  1.7 GB  delta: -3.3 GB  ⚠️  over-allocated

SUMMARY
  Total declared:  67.5 GB
  Total actual:    53.8 GB
  Potential savings: 13.7 GB (20.3%)
```
