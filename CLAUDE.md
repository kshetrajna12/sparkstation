<!-- SPARKSTATION-START -->
# Sparkstation Local LLM Gateway

This project has access to local LLM models through Sparkstation gateway.

## Available Models

- `qwen-flash-next` - daily-driver chat+vision (Flash-Next AutoRound, worker1)
- `gemma4-2b` - small fast chat (Gemma 4 E2B QAT, primary; also audio-capable)
- `bge-m3` - BAAI/bge-m3 text embeddings
- `clip-vit` - openai/clip-vit-large-patch14 image embeddings
- `face-detect` - face-recognition
- `default` - alias for the loaded profile's default chat model (currently `qwen-flash-next`). Prefer this unless you need a specific model.
- `vision` - alias for the loaded profile's vision model (currently `qwen-flash-next`). Use this for any image-understanding request.
- `voicecascade` - Sparky's voice stack (worker2, `voice` profile). NOT an
  OpenAI API — WebSocket/WebRTC audio directly on worker2:7860; never appears
  in `/v1/models`. See homecloud-infra/docs/cascade-voice-runbook.md.

## Available Profiles

Switch profiles with `sparkstation stop && sparkstation start -d --profile <name>` (see models.yaml `profiles:` for ground truth):

- **generic**: qwen-flash-next, bge-m3, clip-vit, face-detect, gemma4-2b — daily driver
- **voice**: generic + voicecascade on worker2 (also on demand: `sparkstation models start voicecascade -p voice`)
- **deep**: GLM-5.3-Flash 2-node reserve (workers) + aux on primary
- **image-indexing**: batch photo intake (vLLM+MTP concurrency recipe)

## Sparkstation Console

Web control panel served by the supervisor at `http://127.0.0.1:9001/console/`
(static SPA in `console/`, no build step; see `console/README.md`). Voice
Studio is live: Talk tab (WebSocket relay `/voice/talk` → cascade bot) and
Voices tab over the `/voice/*` API (registry of cloned/stock/designed voices,
samples, default voice, per-voice style). CLI twin: `sparkstation voice …`.
The voice registry (reference clips, transcripts) lives only on the voice
role host — never commit it.

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
    model="default",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Usage with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Streaming

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Vision (Image Analysis)

The `vision` model supports vision capabilities. You can pass images via URL or base64:

### With Image URL

```python
response = client.chat.completions.create(
    model="vision",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
```

### With Base64 Encoded Image

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="vision",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]
)
```

**Note**: Vision requests use more tokens (~5000+ tokens for image processing).

## Reasoning Models & Token Budgets (IMPORTANT)

Chat models on this gateway may be reasoning models (e.g. Qwen thinking
variants, DeepSeek-V4-Flash when the deep-era specs are revived). **`max_tokens` caps reasoning + final content
COMBINED.** The server's default reasoning effort is `low`; a complex prompt
can still spend most of a small budget on reasoning.

**Symptom of an under-sized budget**: `content` empty, `reasoning_content`
non-empty, `finish_reason: "length"` — the model ran out of tokens before
writing the answer. Raise `max_tokens`, or cap reasoning explicitly.

```python
# Bounded reasoning — guarantees room for the final answer:
response = client.chat.completions.create(
    model="default",
    messages=[...],
    max_tokens=4096,
    extra_body={"thinking_token_budget": 2048},   # reasoning hard-capped
)

# Deep reasoning on demand (size max_tokens generously — 16K+):
response = client.chat.completions.create(
    model="default",
    messages=[...],
    max_tokens=16384,
    extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
    # NB some models key on a different name: dsv4-flash ignores "thinking"
    # and obeys {"enable_thinking": False} — verify per model (2026-08-30).
)

# No reasoning at all (fastest, for simple/structured tasks):
#   extra_body={"chat_template_kwargs": {"thinking": False}}
```

These knobs pass through the gateway unchanged. For structured output
(`response_format` json_schema), prefer `thinking_token_budget` or
`thinking: False` — reasoning length is highly variable and can starve the
JSON answer on tight budgets.

## Embeddings

Sparkstation provides text embedding models for semantic search, RAG, and similarity tasks.

### Text Embeddings (bge-m3)

Generate embeddings for text using the `bge-m3` model:

```python
# Generate text embeddings
response = client.embeddings.create(
    model="bge-m3",
    input="Hello world"
)

# Get embedding vector (1024 dimensions)
embedding = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding)}")
```

### Batch Embeddings

Generate embeddings for multiple inputs at once:

```python
response = client.embeddings.create(
    model="bge-m3",
    input=["First document", "Second document", "Third document"]
)

for i, data in enumerate(response.data):
    print(f"Document {i}: {len(data.embedding)} dimensions")
```

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

### Use Cases

- **Semantic Search**: Embed documents and queries, find similar content via cosine similarity
- **RAG (Retrieval Augmented Generation)**: Embed knowledge base for context retrieval
- **Classification**: Use embeddings as features for downstream ML tasks

## Important Notes

- **Do not start/stop Sparkstation services** - they are managed by the system
- Models are already running and ready to use
- Use the gateway endpoint (`http://localhost:8000/v1`) for all requests
- All models support standard OpenAI APIs:
  - Chat: `/v1/chat/completions` (qwen-flash-next, gemma4-2b, `default`/`vision` aliases)
  - Embeddings: `/v1/embeddings` (bge-m3, clip-vit)
  - Voice is NOT here: the `voicecascade` stack speaks WebSocket audio directly on worker2:7860

### Model-Specific Details

- **Vision Chat** (`vision`):
  - Profile-following alias — always routes to the loaded profile's vision model (currently qwen-flash-next)
  - Supports image analysis via URL or base64
  - Uses standard OpenAI vision format: `{"type": "image_url", "image_url": {"url": "..."}}`

- **Text Embeddings** (`bge-m3`):
  - Generates 1024-dim embeddings for text semantic tasks
  - Standard format: `input="text"` or `input=["text1", "text2"]`

- **Image Embeddings** (`clip-vit`):
  - Generates 768-dim embeddings for images and cross-modal search
  - **Special format required**: Images must use `input=[{"image": "..."}]` (not flat strings)
  - Text queries use simple format: `input="text query"`
<!-- SPARKSTATION-END -->