#!/usr/bin/env python3
"""CLAUDE.md generation for sparkstation init command."""
import os
import re
import sys
from pathlib import Path

import click

SPARKSTATION_START_MARKER = "<!-- SPARKSTATION-START -->"
SPARKSTATION_END_MARKER = "<!-- SPARKSTATION-END -->"


DECISION_SECTION_TEMPLATE = """
## Decision model (`__MODEL__`) — typed judgments, not text

`__MODEL__` is a "System One" decision model (reflex, an open re-creation of
TypeSafe's Jev). Give it a **state** (text, JSON, images) and typed
**questions**; it answers ALL of them in one forward pass and returns
**calibrated probabilities over the options you supplied**. It never generates
text, so it cannot invent an option, and nothing needs parsing. Use it wherever
a chat call would only be asked "which queue / is this spam / how urgent / does
this photo match its caption": ~100 ms for a handful of questions once the
state is cached, vs seconds of generation.

Endpoint: `POST http://localhost:8000/v1/systemone` (same gateway, same API
key). It is NOT an OpenAI route: `/v1/models` never lists it, and the OpenAI
SDK has no method for it — use `httpx`/`requests`/curl. `model` may be omitted
(or `"reflex-latest"`): it resolves to the loaded decision model.

| type | asks | you get back |
|---|---|---|
| `noul` | "is this true?" | `noul`: P(yes) |
| `choice` | "which one of these?" (≤26 options) | `choice`, `probabilities` per option, `confidence` |
| `score` | "how much, on this ordered scale?" (2–10 levels) | `score` (probability-weighted level), `probabilities`, `legend`, `confidence` |

```python
import httpx
r = httpx.post("http://localhost:8000/v1/systemone",
               headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120,
               json={
                   "state": {"ticket": "My payouts failed three times this week and nobody replied."},
                   "questions": {
                       "queue":    {"type": "choice", "instructions": "Which team should handle this?",
                                    "criteria": {"payments": "payouts, refunds", "account": "login, 2FA", "other": None}},
                       "escalate": {"type": "noul",   "instructions": "Should this be escalated to a manager?"},
                       "urgency":  {"type": "score",  "instructions": "How urgent is this?",
                                    "criteria": ["can wait a week", "handle today", "blocked right now"]},
                   },
               }).json()
a = r["answers"]
if a["queue"]["confidence"] > 0.6:        # code owns the policy, the model supplies the judgment
    route(a["queue"]["choice"])
if a["escalate"]["noul"] > 0.5 or a["urgency"]["score"] > 1.5:
    page_oncall()
```

- **Images**: put `{"type": "image", "source": "<file path | URL | data: URI>"}`
  anywhere in `state`; every question sees it (~1k tokens per megapixel).
- **Cost model**: the state is cached by content hash — ask many questions
  about one document for roughly the price of one; repeats over the same
  state only pay for the questions (`usage.state_cache_hit`).
- **Calibration**: served with a fitted temperature (and a LoRA when
  configured), so "85 %" is right about 85 % of the time — still check a
  handful of your own examples before trusting a threshold, and route
  low-`confidence` cases to a person or to `default`.
- **Errors**: 422 = bad question shape (e.g. a `choice` with one option);
  404 = no decision model loaded; 503 = model starting.
"""


def run_init(profile):
    """Add Sparkstation instructions to CLAUDE.md (creates, appends, or updates)."""
    claude_md_path = Path("CLAUDE.md")

    # Get available models from models.yaml
    models_info = []
    profiles_info = {}

    try:
        # Use the resolver so profile overrides + local overlay both apply,
        # rather than duplicating YAML-parsing logic here.
        from supervisor.models_config import load_models_config, get_profile_models

        cfg = load_models_config()

        # Build profile info for all profiles by resolving each one.
        for profile_name in cfg.profiles:
            resolved = get_profile_models(profile_name)
            profiles_info[profile_name] = [
                {
                    "name": m.alias or (m.name.split("/")[-1] if m.name else "unknown"),
                    "full_name": m.name,
                    "model_type": m.model_type,
                }
                for m in resolved
            ]

        # Choose which profile's models to document
        target_profile = profile or cfg.default_profile
        if target_profile:
            if target_profile not in profiles_info:
                click.secho(
                    f"Profile '{target_profile}' not found. Available: {', '.join(profiles_info.keys())}",
                    fg="red",
                )
                sys.exit(1)
            for m in profiles_info[target_profile]:
                models_info.append(
                    {"name": m["name"], "full_name": m["full_name"], "model_type": m["model_type"]}
                )
        else:
            # No profile hint anywhere — enumerate every defined model
            for alias, defn in cfg.models.items():
                models_info.append(
                    {"name": alias, "full_name": defn.name, "model_type": defn.model_type}
                )
    except Exception:
        # Fallback if models.yaml doesn't exist or can't be parsed
        models_info = [
            {"name": "gpt-oss-20b", "full_name": "openai/gpt-oss-20b"},
            {"name": "bge-large", "full_name": "BAAI/bge-large-en-v1.5"},
            {"name": "clip-vit", "full_name": "openai/clip-vit-large-patch14"},
            {"name": "qwen3-vl-4b", "full_name": "Qwen/Qwen3-VL-4B-Instruct-FP8"},
            {"name": "flux-dev", "full_name": "black-forest-labs/FLUX.1-dev"},
        ]

    # Generate model list for documentation
    def _model_line(m):
        line = f"- `{m['name']}` - {m['full_name']}"
        if m.get("model_type") == "decision":
            line += (" — DECISION model (typed judgments over `POST /v1/systemone`; "
                     "NOT chat, never in `/v1/models`). See \"Decision model\" below.")
        return line

    model_list_str = "\n".join(_model_line(m) for m in models_info)

    # Document the profile-following `default` alias so clients prefer it over
    # pinning a model name that goes stale on the next profile/model swap.
    try:
        from supervisor.models_config import get_default_model_alias
        _default_alias = get_default_model_alias(profile)
        if _default_alias:
            model_list_str += (
                f"\n- `default` - alias for the loaded profile's default chat model "
                f"(currently `{_default_alias}`). Prefer this unless you need a specific model."
            )
        from supervisor.models_config import get_vision_model_alias as _gva
        _vision_alias = _gva(profile)
        if _vision_alias:
            model_list_str += (
                f"\n- `vision` - alias for the loaded profile's vision model "
                f"(currently `{_vision_alias}`). Use this for any image-understanding request."
            )
    except Exception:
        pass

    # Generate profiles section
    profiles_section = ""
    if profiles_info:
        profile_lines = []
        for pname, pmodels in profiles_info.items():
            model_names = ", ".join([m["name"] for m in pmodels])
            profile_lines.append(f"- **{pname}**: {model_names}")
        profiles_section = f"""
## Available Profiles

Switch profiles with `sparkstation start -d --profile <name>`:

{chr(10).join(profile_lines)}
"""

    active_profile_note = f"\n**Active profile**: `{profile}`\n" if profile else ""

    # Determine which models are available for conditional sections
    model_aliases = {m["name"] for m in models_info}
    has_clip = "clip-vit" in model_aliases
    has_flux = "flux-dev" in model_aliases
    has_gpt_oss = "gpt-oss-20b" in model_aliases
    has_nemotron = "nemotron3-nano" in model_aliases

    # Pick the primary chat model for examples: first chat-type model in the
    # profile (profile order puts the daily driver first), falling back to the
    # legacy preference list for configs without model_type info.
    chat_model = "qwen3-vl-4b"
    for m in models_info:
        if m.get("model_type") == "chat":
            chat_model = m["name"]
            break
    else:
        for name in ["qwen3-vl-30b", "qwen3-vl-4b", "nemotron3-nano", "gpt-oss-20b"]:
            if name in model_aliases:
                chat_model = name
                break

    # Text-embedding model for the embeddings examples (clip-vit is
    # image-embedding and documented separately).
    text_embed_model = None
    for m in models_info:
        if m.get("model_type") == "embedding" and m["name"] != "clip-vit":
            text_embed_model = m["name"]
            break
    if text_embed_model is None and "bge-large" in model_aliases:
        text_embed_model = "bge-large"
    # The embeddings doc section always renders — never let it say "None"
    text_embed_model = text_embed_model or "bge-m3"

    # Vision model for examples: prefer the profile-following "vision"
    # gateway alias (published by gateway_sync from vision_default markers in
    # models.yaml) so generated docs never go stale on a model swap. Fall back
    # to name-sniffing for configs predating the alias.
    vision_model = None
    try:
        from supervisor.models_config import get_vision_model_alias
        if get_vision_model_alias(profile):
            vision_model = "vision"
    except Exception:
        pass
    if vision_model is None:
        for name in model_aliases:
            if name.startswith("qwen3-vl"):
                vision_model = name
                break

    # Pick the reasoning model for examples
    reasoning_model = None
    if has_nemotron:
        reasoning_model = "nemotron3-nano"
    elif has_gpt_oss:
        reasoning_model = "gpt-oss-20b"

    # Build reasoning section
    reasoning_section = ""
    if reasoning_model:
        reasoning_section = f"""
## Reasoning Models

The `{reasoning_model}` model is a reasoning model that shows its thinking process. Access both the reasoning and final response:

```python
response = client.chat.completions.create(
    model="{reasoning_model}",
    messages=[{{"role": "user", "content": "What is 2+2?"}}]
)

# Final answer
print(response.choices[0].message.content)

# Reasoning process (if available)
if hasattr(response.choices[0].message, 'reasoning_content'):
    print(response.choices[0].message.reasoning_content)
```
"""

    # Build CLIP section
    clip_section = ""
    if has_clip:
        clip_section = """
### Image Embeddings (CLIP)

The `clip-vit` model generates embeddings for images using OpenAI's CLIP.

**Important**: CLIP embeddings use a structured array format (different from standard OpenAI embeddings API).

#### With Image URL
```python
response = client.embeddings.create(
    model="clip-vit",
    input=[{"image": "https://example.com/image.jpg"}]
)

embedding = response.data[0].embedding  # 768 dimensions
```

#### With Base64 Encoded Image
```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.embeddings.create(
    model="clip-vit",
    input=[{"image": image_data}]
)

embedding = response.data[0].embedding  # 768 dimensions
```

**Note**: The input must be an array of objects with `"image"` keys, not flat strings.

### Cross-Modal Search with CLIP

CLIP embeddings enable searching images with text or finding similar images:

```python
# Embed text query
text_response = client.embeddings.create(
    model="clip-vit",
    input="a red car"
)
text_embedding = text_response.data[0].embedding

# Embed image
image_response = client.embeddings.create(
    model="clip-vit",
    input=[{"image": "https://example.com/car.jpg"}]
)
image_embedding = image_response.data[0].embedding

# Compare via cosine similarity (both in same 768-dim embedding space)
from numpy import dot
from numpy.linalg import norm

similarity = dot(text_embedding, image_embedding) / (norm(text_embedding) * norm(image_embedding))
print(f"Similarity: {similarity}")
```
"""

    # Build FLUX section
    flux_section = ""
    if has_flux:
        flux_section = """
## Image Generation

Sparkstation provides FLUX.1-dev for high-quality image generation via the OpenAI-compatible `/v1/images/generations` endpoint.

### Basic Image Generation

```python
import base64

response = client.images.generate(
    model="flux-dev",
    prompt="A photorealistic image of a red robot in a garden",
    n=1,
    size="512x512",
    response_format="b64_json"
)

image_data = base64.b64decode(response.data[0].b64_json)
with open("generated_image.png", "wb") as f:
    f.write(image_data)
```

### With curl

```bash
curl http://localhost:8000/v1/images/generations \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer dummy-key" \\
  -d '{
    "model": "flux-dev",
    "prompt": "A cyberpunk city at night with neon lights",
    "n": 1,
    "size": "512x512"
  }'
```

**Notes**:
- Image generation takes 20-60 seconds depending on size
- FLUX.1-dev produces high-quality photorealistic images
- First request may be slower (model warmup)
"""

    # Decision model (reflex): typed judgments with calibrated probabilities
    # over its own POST /v1/systemone route — documented as its own section
    # because it is not an OpenAI chat/embeddings API.
    decision_model = None
    for m in models_info:
        if m.get("model_type") == "decision":
            decision_model = m["name"]
            break
    decision_section = ""
    if decision_model:
        decision_section = DECISION_SECTION_TEMPLATE.replace("__MODEL__", decision_model)

    # Build model-specific details
    model_details_lines = []
    if vision_model:
        model_details_lines.append(f"""- **Vision Chat** (`{vision_model}`):
  - Profile-following alias — always routes to the loaded profile's vision model
  - Supports image analysis via URL or base64
  - Uses standard OpenAI vision format: `{{"type": "image_url", "image_url": {{"url": "..."}}}}`""")
    if has_nemotron:
        model_details_lines.append("""- **Reasoning + Tool Calling** (`nemotron3-nano`):
  - NVIDIA Nemotron 3 Nano 30B with NVFP4 quantization
  - 65k context window, includes reasoning traces in `reasoning_content` field
  - Supports tool calling via qwen3_coder parser""")
    if has_gpt_oss:
        model_details_lines.append("""- **Reasoning** (`gpt-oss-20b`):
  - Includes reasoning traces in `reasoning_content` field""")
    if text_embed_model:
        model_details_lines.append(f"""- **Text Embeddings** (`{text_embed_model}`):
  - Generates 1024-dim embeddings for text semantic tasks
  - Standard format: `input="text"` or `input=["text1", "text2"]`""")
    if has_clip:
        model_details_lines.append("""- **Image Embeddings** (`clip-vit`):
  - Generates 768-dim embeddings for images and cross-modal search
  - **Special format required**: Images must use `input=[{"image": "..."}]` (not flat strings)
  - Text queries use simple format: `input="text query"`""")
    if has_flux:
        model_details_lines.append("""- **Image Generation** (`flux-dev`):
  - Generates high-quality images from text prompts using FLUX.1-dev
  - Supports sizes: 512x512, 1024x1024
  - Takes 20-60 seconds per image""")
    if decision_model:
        model_details_lines.append(f"""- **Decision model** (`{decision_model}`):
  - `POST /v1/systemone` only (TypeSafe-Jev-compatible); NOT a chat model and not listed by `/v1/models`
  - Typed questions (`noul` / `choice` / `score`) over one state, answered in a single pass
  - Returns calibrated probabilities over the options YOU supply — never free text""")

    model_details_str = "\n\n".join(model_details_lines)

    # Build API capabilities list
    api_lines = []
    chat_models = [m["name"] for m in models_info if m.get("model_type", "chat") == "chat"]
    if chat_models:
        api_lines.append(f"  - Chat: `/v1/chat/completions` ({', '.join(chat_models)})")
    embed_models = [m["name"] for m in models_info if m.get("model_type") == "embedding" or m["name"] == "clip-vit"]
    if embed_models:
        api_lines.append(f"  - Embeddings: `/v1/embeddings` ({', '.join(embed_models)})")
    if has_flux:
        api_lines.append("  - Image Generation: `/v1/images/generations` (flux-dev)")
    if decision_model:
        api_lines.append(f"  - Decisions: `/v1/systemone` ({decision_model}) — typed judgments, Jev-compatible, not in `/v1/models`")
    api_capabilities_str = "\n".join(api_lines)

    sparkstation_section = f"""{SPARKSTATION_START_MARKER}
# Sparkstation Local LLM Gateway

This project has access to local LLM models through Sparkstation gateway.
{active_profile_note}
## Available Models

{model_list_str}
{profiles_section}
## API Endpoint

- **Base URL**: `http://localhost:8000/v1`
- **Protocol**: OpenAI-compatible API
- **Authentication**: Any string works until you enable per-client keys (gateway/clients.yaml `enforce_auth: true`); then use a registered key.

## Usage with OpenAI Python SDK

```python
from openai import OpenAI

# Initialize client pointing to local Sparkstation gateway
client = OpenAI(
    api_key="dummy-key",  # Any value works
    base_url="http://localhost:8000/v1"
)

# Make a request
response = client.chat.completions.create(
    model="{chat_model}",
    messages=[
        {{"role": "user", "content": "Hello!"}}
    ]
)

print(response.choices[0].message.content)
```

## Usage with curl

```bash
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer dummy-key" \\
  -d '{{
    "model": "{chat_model}",
    "messages": [{{"role": "user", "content": "Hello!"}}]
  }}'
```

## Streaming

```python
stream = client.chat.completions.create(
    model="{chat_model}",
    messages=[{{"role": "user", "content": "Tell me a story"}}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Vision (Image Analysis)

The `{vision_model}` model supports vision capabilities. You can pass images via URL or base64:

### With Image URL

```python
response = client.chat.completions.create(
    model="{vision_model}",
    messages=[
        {{
            "role": "user",
            "content": [
                {{"type": "text", "text": "What's in this image?"}},
                {{"type": "image_url", "image_url": {{"url": "https://example.com/image.jpg"}}}}
            ]
        }}
    ]
)
```

### With Base64 Encoded Image

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="{vision_model}",
    messages=[
        {{
            "role": "user",
            "content": [
                {{"type": "text", "text": "Describe this image"}},
                {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{image_data}}"}}}}
            ]
        }}
    ]
)
```

**Note**: Vision requests use more tokens (~5000+ tokens for image processing).
{reasoning_section}
## Reasoning Models & Token Budgets (IMPORTANT)

Chat models on this gateway may be reasoning models. **`max_tokens` caps
reasoning + final content COMBINED.** The serving stack pins a server-side
default of `reasoning_effort: low`, so a modest budget still leaves room for
the answer.

**You do not need to know a model's thinking dialect.** The gateway normalizes
reasoning controls for whichever backend is loaded (`gateway/reasoning.py` +
`gateway/reasoning.yaml`): express the intent any common way and it is
rewritten into what the current model actually honors — including effort
values the model would otherwise reject outright.

```python
# Off (fastest; simple or structured tasks). Equivalent through the gateway:
extra_body={{"chat_template_kwargs": {{"thinking": False}}}}
extra_body={{"chat_template_kwargs": {{"enable_thinking": False}}}}
extra_body={{"reasoning": False}}

# Level. Send the OpenAI spelling; the gateway maps it into the model's own
# vocabulary (current driver: low | medium | xhigh; high -> xhigh,
# unrecognized -> medium).
extra_body={{"reasoning_effort": "low"}}    # brief
extra_body={{"reasoning_effort": "high"}}   # deep — size max_tokens 16K+
```

Going **direct to a backend port** bypasses this translation. There you must
use the model's own spelling: Qwen3.8-Flash-Next reads `enable_thinking` and
`reasoning_effort` inside `chat_template_kwargs`, **ignores `thinking`
entirely** (it is not a variable in the chat template, so the request reasons
anyway), and raises on any effort outside `low|medium|xhigh`.

**Symptom of an under-sized budget**: `content` empty with
`finish_reason: "length"`. Diagnose with
`usage.completion_tokens_details.reasoning_tokens` — that field is reliable.
Do NOT key on `reasoning_content`: the field name varies by build (the current
driver streams `reasoning` deltas and leaves `reasoning_content` empty).

For structured output (`response_format` json_schema) leave ~250 completion
tokens of headroom beyond the JSON itself — reasoning is emitted first.
`thinking_token_budget` is a DFlash2-era knob and is **NOT** honored by the
current daily driver (measured 2026-09-14: budget 32 produced 147 reasoning
tokens); use `reasoning_effort` or turn thinking off instead.

## Embeddings

Sparkstation provides text embedding models for semantic search, RAG, and similarity tasks.

### Text Embeddings ({text_embed_model})

Generate embeddings for text using the `{text_embed_model}` model:

```python
# Generate text embeddings
response = client.embeddings.create(
    model="{text_embed_model}",
    input="Hello world"
)

# Get embedding vector (1024 dimensions)
embedding = response.data[0].embedding
print(f"Embedding dimensions: {{len(embedding)}}")
```

### Batch Embeddings

Generate embeddings for multiple inputs at once:

```python
response = client.embeddings.create(
    model="{text_embed_model}",
    input=["First document", "Second document", "Third document"]
)

for i, data in enumerate(response.data):
    print(f"Document {{i}}: {{len(data.embedding)}} dimensions")
```
{clip_section}
### Use Cases

- **Semantic Search**: Embed documents and queries, find similar content via cosine similarity
- **RAG (Retrieval Augmented Generation)**: Embed knowledge base for context retrieval
- **Classification**: Use embeddings as features for downstream ML tasks
{flux_section}{decision_section}
## Important Notes

- **Do not start/stop Sparkstation services** - they are managed by the system
- Models are already running and ready to use
- Use the gateway endpoint (`http://localhost:8000/v1`) for all requests
- All models support standard OpenAI APIs:
{api_capabilities_str}

### Model-Specific Details

{model_details_str}
{SPARKSTATION_END_MARKER}"""

    # Handle the three cases: create, append, or update
    if claude_md_path.exists():
        existing_content = claude_md_path.read_text()

        if SPARKSTATION_START_MARKER in existing_content and SPARKSTATION_END_MARKER in existing_content:
            # Case 3: Update existing Sparkstation section
            import re
            pattern = re.escape(SPARKSTATION_START_MARKER) + r".*?" + re.escape(SPARKSTATION_END_MARKER)
            new_content = re.sub(pattern, sparkstation_section, existing_content, flags=re.DOTALL)
            claude_md_path.write_text(new_content)
            click.secho(f"✓ Updated Sparkstation section in {claude_md_path}", fg="green")
        else:
            # Case 2: Append to existing file
            with open(claude_md_path, "a") as f:
                f.write("\n\n" + sparkstation_section)
            click.secho(f"✓ Added Sparkstation section to {claude_md_path}", fg="green")
    else:
        # Case 1: Create new file
        claude_md_path.write_text(sparkstation_section)
        click.secho(f"✓ Created {claude_md_path}", fg="green")

    click.echo("\nThis section provides instructions for AI assistants on how to use")
    click.echo("the Sparkstation gateway.")


