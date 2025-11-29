# FLUX.1-dev Image Generation Server

This directory contains the Docker image for FLUX.1-dev image generation using Diffusers.

## Building the Image

```bash
cd docker/flux
docker build --platform linux/arm64 -t flux-server:latest .
```

## Requirements

- NVIDIA PyTorch 25.11 base image
- HuggingFace token with access to FLUX.1-dev (gated model)
- ~35GB GPU memory
- ~20GB disk space for model weights

## Files

- `Dockerfile` - Docker image definition
- `server.py` - FastAPI server with OpenAI-compatible API
- `README.md` - This file

## API Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/images/generations` - Generate images (OpenAI-compatible)

## Configuration

The server is configured via environment variables:

- `HF_TOKEN` - HuggingFace token (required for gated model access)
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `MODEL_PATH` - Model path (default: black-forest-labs/FLUX.1-dev)

## Usage Examples

### Direct API Access (Port 8005)

```bash
# Health check
curl http://localhost:8005/health

# List models
curl http://localhost:8005/v1/models

# Generate image
curl -X POST http://localhost:8005/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photorealistic image of a red robot in a garden",
    "n": 1,
    "size": "512x512"
  }' | python3 -c "
import json, sys, base64
data = json.load(sys.stdin)
with open('output.png', 'wb') as f:
    f.write(base64.b64decode(data['data'][0]['b64_json']))
print('Saved to output.png')
"
```

### Via LiteLLM Gateway (Port 8000)

When running through Sparkstation, FLUX is accessible via the unified gateway:

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux-dev",
    "prompt": "A cyberpunk city at night with neon lights",
    "n": 1,
    "size": "512x512"
  }'
```

### Python Examples

```python
import openai
import base64

# Configure client for Sparkstation gateway
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:8000/v1"
)

# Generate an image
response = client.images.generate(
    model="flux-dev",
    prompt="A serene Japanese garden with cherry blossoms",
    n=1,
    size="512x512",
    response_format="b64_json"
)

# Save the image
image_data = base64.b64decode(response.data[0].b64_json)
with open("generated_image.png", "wb") as f:
    f.write(image_data)
print("Image saved to generated_image.png")
```

### Using requests directly

```python
import requests
import base64

response = requests.post(
    "http://localhost:8000/v1/images/generations",
    headers={
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json"
    },
    json={
        "model": "flux-dev",
        "prompt": "A watercolor painting of mountains at sunset",
        "n": 1,
        "size": "1024x1024"
    },
    timeout=120  # Image generation can take 20-60 seconds
)

if response.ok:
    data = response.json()
    image_b64 = data["data"][0]["b64_json"]
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(image_b64))
    print("Image saved to output.png")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | - | Model name (use `flux-dev` via gateway) |
| `prompt` | string | - | Text description of the image to generate |
| `n` | integer | 1 | Number of images to generate |
| `size` | string | "512x512" | Image dimensions (e.g., "512x512", "1024x1024") |
| `response_format` | string | "b64_json" | Response format: "b64_json" or "url" |

## Response Format

```json
{
  "created": 1234567890,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

## Performance Notes

- Image generation takes 20-60 seconds depending on size
- First request after startup may take longer (model warmup)
- FLUX.1-dev produces high-quality 512x512 to 1024x1024 images
- Memory usage: ~33-35GB GPU memory

## Integration

The FLUX launcher (`supervisor/launchers/flux_launcher.py`) handles:
- Docker container lifecycle
- Health checks via `/health` endpoint
- Auto-suspend/resume
- Port allocation

See `models.yaml` for model configuration.

## Troubleshooting

### Model fails to load
- Ensure `HF_TOKEN` is set and has access to FLUX.1-dev
- Check available GPU memory (~35GB required)
- View container logs: `docker logs sparkstation-flux.1-dev-*`

### Slow generation
- First request warms up the model
- Larger images take longer
- Check GPU utilization with `nvidia-smi`

### Gateway returns 400 "Invalid model name"
- Ensure `model_info: mode: image_generation` is in `gateway/litellm.yaml`
- Restart the LiteLLM gateway after config changes
