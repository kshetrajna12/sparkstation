#!/usr/bin/env python3
"""CLAUDE.md generation for sparkstation init command."""
import os
import re
import sys
from pathlib import Path

import click

SPARKSTATION_START_MARKER = "<!-- SPARKSTATION-START -->"
SPARKSTATION_END_MARKER = "<!-- SPARKSTATION-END -->"


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
    model_list_str = "\n".join([f"- `{m['name']}` - {m['full_name']}" for m in models_info])

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

    # Detect vision-capable model (any qwen3-vl variant)
    vision_model = None
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

    # Build model-specific details
    model_details_lines = []
    if vision_model:
        model_details_lines.append(f"""- **Vision Chat** (`{vision_model}`):
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
- **Authentication**: Use any string as API key (e.g., `"dummy-key"`)

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
{flux_section}
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


