---
name: sparkstation-integration-test
description: End-to-end integration tests for Sparkstation. Tests the full pipeline from gateway through supervisor to model backends. Use when asked to "run integration tests", "test the full pipeline", "verify everything works", or "smoke test".
---

# Sparkstation Integration Test

Tests the complete request pipeline: Client → Gateway (LiteLLM) → Supervisor → Backend containers.

## Usage

```bash
# Run all integration tests
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py

# Test specific model type
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py --test chat
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py --test embedding
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py --test clip
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py --test vision
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py --test supervisor

# Verbose output (show response details)
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py -v

# JSON report
python3 .pi/skills/sparkstation-integration-test/scripts/integration_test.py --json
```

## What It Tests

1. **Supervisor API**: health, models list, detailed status, resources
2. **Chat completions**: streaming + non-streaming, basic + multi-turn
3. **Embeddings**: text embedding, batch embedding, dimension validation
4. **CLIP**: image embedding (direct backend), text embedding, dimension check
5. **Vision**: image+text chat completion
6. **Gateway**: model routing, error handling for unknown models

Each test validates response structure, status codes, and content correctness.
