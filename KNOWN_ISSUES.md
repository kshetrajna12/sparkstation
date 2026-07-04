# Known Issues

## LiteLLM sends `encoding_format=null` to vLLM embeddings (PATCHED)

**Status:** Workaround applied, waiting for upstream fix  
**Affects:** bge-m3 embedding endpoint when called via LiteLLM gateway  
**Upstream bug:** https://docs.litellm.ai/blog/vllm-embeddings-incident  
**Fix commit:** https://github.com/BerriAI/litellm/commit/55348dd (not yet in any release as of 1.82.4)

### Symptom

```
Error code: 400 - encoding_format: Input should be 'float', 'base64', 'bytes' or 'bytes_only', got None
```

### Root Cause

LiteLLM `main.py` explicitly sets `optional_params["encoding_format"] = None` when the client omits it. vLLM rejects `null` — it only accepts `"float"`, `"base64"`, or omission.

### Current Workaround

One-line patch applied to installed LiteLLM package:

```bash
# File: ~/.local/share/uv/tools/sparkstation/lib/python3.11/site-packages/litellm/main.py
# Line ~4915: replace
optional_params["encoding_format"] = None
# with
pass  # patched: do not send encoding_format=None (breaks vLLM)
```

### ⚠️ Re-apply after upgrades

This patch is lost on `uv tool upgrade sparkstation` or any LiteLLM update. Re-apply with:

```bash
LITELLM_MAIN=$(find ~/.local/share/uv/tools/sparkstation -name "main.py" -path "*/litellm/main.py")
sed -i 's/optional_params\["encoding_format"\] = None/pass  # patched: do not send encoding_format=None (breaks vLLM)/' "$LITELLM_MAIN"
```

### When to remove

Remove this workaround once LiteLLM releases a version containing commit `55348dd`. Check with:

```bash
grep 'optional_params\["encoding_format"\] = None' $(find ~/.local/share/uv/tools/sparkstation -name "main.py" -path "*/litellm/main.py")
# If no output → bug is fixed upstream, workaround no longer needed
```

### ✅ RESOLVED (2026-07-02)

LiteLLM 1.90.3 (current uv tool env) no longer contains the buggy
`optional_params["encoding_format"] = None` line — the upstream fix shipped.
The sed workaround is obsolete; verified bge-m3 embeddings work unpatched
through the gateway. This section is kept for history only.
