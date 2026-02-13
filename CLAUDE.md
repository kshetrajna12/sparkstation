<!-- SPARKSTATION-START -->
# Sparkstation Local LLM Gateway

This project has access to local LLM models through Sparkstation gateway.

**Active profile**: `openclaw`

## Available Models

- `nemotron3-nano` - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
- `bge-large` - BAAI/bge-large-en-v1.5
- `qwen3-vl-4b` - Qwen/Qwen3-VL-4B-Instruct-FP8

## Available Profiles

Switch profiles with `sparkstation start -d --profile <name>`:

- **dev**: qwen3-vl-4b
- **prod**: qwen3-vl-4b, gpt-oss-20b
- **inference**: qwen3-vl-4b
- **openclaw**: nemotron3-nano, bge-large, qwen3-vl-4b

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
    model="qwen3-vl-4b",
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
    "model": "qwen3-vl-4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Streaming

```python
stream = client.chat.completions.create(
    model="qwen3-vl-4b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Vision (Image Analysis)

The `qwen3-vl-4b` model supports vision capabilities. You can pass images via URL or base64:

### With Image URL

```python
response = client.chat.completions.create(
    model="qwen3-vl-4b",
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
    model="qwen3-vl-4b",
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

## Reasoning Models

The `nemotron3-nano` model is a reasoning model that shows its thinking process. Access both the reasoning and final response:

```python
response = client.chat.completions.create(
    model="nemotron3-nano",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

# Final answer
print(response.choices[0].message.content)

# Reasoning process (if available)
if hasattr(response.choices[0].message, 'reasoning_content'):
    print(response.choices[0].message.reasoning_content)
```

## Embeddings

Sparkstation provides text embedding models for semantic search, RAG, and similarity tasks.

### Text Embeddings (bge-large)

Generate embeddings for text using the `bge-large` model:

```python
# Generate text embeddings
response = client.embeddings.create(
    model="bge-large",
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
    model="bge-large",
    input=["First document", "Second document", "Third document"]
)

for i, data in enumerate(response.data):
    print(f"Document {i}: {len(data.embedding)} dimensions")
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
  - Chat: `/v1/chat/completions` (nemotron3-nano, qwen3-vl-4b)
  - Embeddings: `/v1/embeddings` (bge-large)

### Model-Specific Details

- **Vision Chat** (`qwen3-vl-4b`):
  - Supports image analysis via URL or base64
  - Uses standard OpenAI vision format: `{"type": "image_url", "image_url": {"url": "..."}}`

- **Reasoning + Tool Calling** (`nemotron3-nano`):
  - NVIDIA Nemotron 3 Nano 30B with NVFP4 quantization
  - 65k context window, includes reasoning traces in `reasoning_content` field
  - Supports tool calling via qwen3_coder parser

- **Text Embeddings** (`bge-large`):
  - Generates 1024-dim embeddings for text semantic tasks
  - Standard format: `input="text"` or `input=["text1", "text2"]`
<!-- SPARKSTATION-END -->