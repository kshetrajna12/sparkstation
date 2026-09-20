<!-- SPARKSTATION-START -->
# Sparkstation Local LLM Gateway

This project has access to local LLM models through Sparkstation gateway.

## Available Models

- `qwen-flash-next` - daily-driver chat+vision (Flash-Next AutoRound, worker1)
- `gemma4-2b` - small fast chat (Gemma 4 E2B QAT, primary; also audio-capable)
- `bge-m3` - BAAI/bge-m3 text embeddings
- `clip-vit` - openai/clip-vit-large-patch14 image embeddings
- `face-detect` - face-recognition
- `reflex` - System One DECISION model (frozen Qwen3.5-4B + reflex's prompt, primary;
  auto-tracks the reflex repo's `stable` tag, rebuilt on launch when it moves). Typed
  judgments with calibrated probabilities over `POST /v1/systemone` — NOT a chat
  model, never appears in `/v1/models`. See "Decision model (reflex)" below.
- `default` - alias for the loaded profile's default chat model (currently `qwen-flash-next`). Prefer this unless you need a specific model.
- `vision` - alias for the loaded profile's vision model (currently `qwen-flash-next`). Use this for any image-understanding request.
- `voicecascade` - Sparky's voice stack (worker2, `voice` profile). NOT an
  OpenAI API — WebSocket/WebRTC audio directly on worker2:7860; never appears
  in `/v1/models`. See homecloud-infra/docs/cascade-voice-runbook.md.

## Available Profiles

Switch profiles with `sparkstation stop && sparkstation start -d --profile <name>` (see models.yaml `profiles:` for ground truth):

- **generic**: qwen-flash-next, bge-m3, clip-vit, face-detect, gemma4-2b, reflex — daily driver
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
- **Authentication**: the gateway enforces registered API keys (401 otherwise).
  For local work use the `local-ops` key from the repo's gitignored `.env`:
  `export SPARK_KEY=$(grep ^GATEWAY_LOCAL_KEY .env | cut -d= -f2)`

## Usage with OpenAI Python SDK

```python
from openai import OpenAI

# Initialize client pointing to local Sparkstation gateway
import os
client = OpenAI(
    api_key=os.environ["SPARK_KEY"],  # from .env GATEWAY_LOCAL_KEY (see above)
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
  -H "Authorization: Bearer $SPARK_KEY" \
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
extra_body={"chat_template_kwargs": {"thinking": False}}
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
extra_body={"reasoning": False}

# Level. Send the OpenAI spelling; the gateway maps it into the model's own
# vocabulary (current driver: low | medium | xhigh; high -> xhigh,
# unrecognized -> medium).
extra_body={"reasoning_effort": "low"}    # brief
extra_body={"reasoning_effort": "high"}   # deep — size max_tokens 16K+
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

## Decision model (reflex) — typed judgments, not text

`reflex` (github.com/kshetrajna12/reflex) is an open re-creation of TypeSafe's
Jev "System One" model on Qwen3.5-4B. Give it a **state** (text, JSON, images)
and typed **questions**; it answers ALL of them in one forward pass and returns
**calibrated probabilities over the options you supplied**. It never generates
text, so it cannot invent an option, and nothing needs parsing. Use it wherever
a chat call would only be asked "which queue / is this spam / how urgent / does
this photo match its caption": ~100 ms for a handful of questions once the
state is cached, vs seconds of generation.

Endpoint: `POST http://localhost:8000/v1/systemone` (same gateway, same API
key; the proxy forwards it straight to the reflex container — LiteLLM and
`/v1/models` never see it). Request/response shapes are TypeSafe-Jev
compatible, so Jev client code works with `base_url` pointed here. `model`
may be omitted (or `"reflex-latest"`): it resolves to the loaded decision model.

Three question types:

| type | asks | you get back |
|---|---|---|
| `noul` | "is this true?" | `noul`: P(yes) |
| `choice` | "which one of these?" (≤26 options) | `choice`, `probabilities` per option, `confidence` |
| `score` | "how much, on this ordered scale?" (2–10 levels) | `score` (probability-weighted level), `probabilities`, `legend`, `confidence` |

```bash
curl -s http://localhost:8000/v1/systemone -H "Authorization: Bearer $SPARK_KEY" \
  -H 'content-type: application/json' -d '{
  "state": {"ticket": "My payouts failed three times this week and nobody replied."},
  "questions": {
    "queue":    {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"payments": "payouts, refunds", "account": "login, 2FA", "other": null}},
    "escalate": {"type": "noul",   "instructions": "Should this be escalated to a manager?"},
    "urgency":  {"type": "score",  "instructions": "How urgent is this?",
                 "criteria": ["can wait a week", "handle today", "blocked right now"]}
  }}'
```

```python
import httpx, os
r = httpx.post("http://localhost:8000/v1/systemone",
               headers={"Authorization": f"Bearer {os.environ['SPARK_KEY']}"},
               json={"state": {...}, "questions": {...}}, timeout=120).json()
a = r["answers"]
if a["queue"]["confidence"] > 0.6:        # code owns the policy, the model supplies the judgment
    route(a["queue"]["choice"])
if a["escalate"]["noul"] > 0.5 or a["urgency"]["score"] > 1.5:
    page_oncall()
```

- **Images**: put `{"type": "image", "source": "<file path | URL | data: URI>"}`
  anywhere in `state`; it is encoded once with the state and every question
  sees it (~1k tokens per megapixel).
- **Calibration**: served with the reflex LoRA (public 8-dataset mix) + fitted
  temperature — held-out ECE 0.024, so "85 %" is right about 85 % of the time.
  Still check a handful of your own examples before trusting a threshold, and
  route low-`confidence` cases to a person or to `default`.
- **Cost model**: the state is cached by content hash (LRU 8) — ask many
  questions about one document for roughly the price of one; repeated calls
  over the same state only pay for the questions (`usage.state_cache_hit`).
- **Errors**: 422 = bad question shape (e.g. a `choice` with one option);
  404 = no decision model loaded; 503 = model starting.
- Serving knobs (adapter, calibration, token budgets) live in `models.yaml`
  under the `reflex` spec; the container is `docker/reflex`.

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
  - Decisions: `/v1/systemone` (reflex) — typed judgments, Jev-compatible, not in `/v1/models`
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