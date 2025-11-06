#!/usr/bin/env python3
"""
Sparkstation CLI - Unified interface for managing Sparkstation and models.
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
import httpx


DEFAULT_SUPERVISOR_URL = "http://localhost:9001"
DEFAULT_GATEWAY_URL = "http://localhost:8000"


@click.group()
@click.option("--supervisor-url", default=DEFAULT_SUPERVISOR_URL, help="Supervisor API URL")
@click.pass_context
def cli(ctx, supervisor_url):
    """Sparkstation - LLM orchestration CLI."""
    ctx.ensure_object(dict)
    ctx.obj["supervisor_url"] = supervisor_url


@cli.command()
@click.option("--detach", "-d", is_flag=True, help="Run in background")
@click.pass_context
def start(ctx, detach):
    """Start Sparkstation (supervisor + gateway) and wait for models to be ready."""
    click.echo("Starting Sparkstation...")

    if detach:
        # Start supervisor in background
        click.echo("  → Starting supervisor...")

        # Create logs directory
        log_dir = Path.home() / ".sparkstation" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        supervisor_log = log_dir / "supervisor.log"

        with open(supervisor_log, "w") as log_file:
            subprocess.Popen(
                ["uv", "run", "uvicorn", "supervisor.main:app", "--host", "127.0.0.1", "--port", "9001"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        click.echo(f"     Logs: {supervisor_log}")

        # Wait for supervisor to be ready (with models loaded)
        supervisor_url = ctx.obj['supervisor_url']
        for i in range(600):  # Try for 600 seconds (10 minutes max for model loading)
            try:
                response = httpx.get(f"{supervisor_url}/health", timeout=2)
                if response.status_code == 200:
                    click.echo("     Supervisor is running")
                    break
                elif response.status_code == 503:
                    # Supervisor is starting, continue waiting
                    pass
            except:
                pass
            time.sleep(1)
        else:
            click.secho("✗ Supervisor failed to start", fg="red")
            return

        # Wait for all models to be ready (RUNNING or FAILED)
        click.echo("  → Waiting for models to load...")
        max_wait = 180  # 3 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = httpx.get(f"{supervisor_url}/models/detailed", timeout=5)
                if response.status_code == 200:
                    models = response.json()["models"]

                    if not models:
                        # No models to wait for
                        break

                    # Check if all models are done loading (RUNNING or FAILED)
                    starting_models = [m for m in models if m["status"] == "starting"]

                    if not starting_models:
                        # All models are done
                        running = [m for m in models if m["status"] == "running"]
                        failed = [m for m in models if m["status"] == "failed"]

                        click.echo(f"     {len(running)} model(s) running, {len(failed)} failed")
                        break
                    else:
                        # Still loading
                        click.echo(f"     {len(starting_models)} model(s) still loading...", nl=False)
                        click.echo("\r", nl=False)

            except Exception as e:
                pass

            time.sleep(5)

        # Start gateway in background
        click.echo("  → Starting gateway...")
        # Remove SUPERVISOR_DATABASE_URL from environment - gateway runs as simple router without DB features
        # TODO: Add gateway database support for usage tracking, API key management, rate limiting
        gateway_env = os.environ.copy()
        gateway_env.pop("SUPERVISOR_DATABASE_URL", None)

        gateway_log = log_dir / "gateway.log"
        with open(gateway_log, "w") as log_file:
            subprocess.Popen(
                ["uv", "run", "litellm", "--config", "gateway/litellm.yaml",
                 "--host", "127.0.0.1", "--port", "8000"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=gateway_env,
            )

        click.echo(f"     Logs: {gateway_log}")

        # Wait for gateway to be ready
        gateway_url = "http://localhost:8000"
        for i in range(15):  # Try for 15 seconds
            try:
                response = httpx.get(f"{gateway_url}/health", timeout=2)
                if response.status_code == 200:
                    click.echo("     Gateway is running")
                    break
            except:
                pass
            time.sleep(1)
        else:
            click.secho("     Warning: Gateway may not have started", fg="yellow")

        # Final status
        click.echo("\n")
        try:
            response = httpx.get(f"{supervisor_url}/models/detailed", timeout=5)
            if response.status_code == 200:
                models = response.json()["models"]
                for model in models:
                    status_icon = {"running": "●", "starting": "◐", "stopped": "○", "failed": "✗"}.get(model["status"], "?")
                    status_color = {"running": "green", "starting": "yellow", "stopped": "white", "failed": "red"}.get(model["status"], "white")

                    click.echo(f"  {status_icon} ", nl=False)
                    click.secho(f"{model['alias'] or model['model_name']}: {model['status']}", fg=status_color)

                click.echo("")
                click.secho("✓ Sparkstation is running!", fg="green")
                click.echo(f"Gateway: http://localhost:8000 (OpenAI-compatible API)")
                click.echo(f"Supervisor: {supervisor_url} (Model management)")
        except:
            click.secho("✓ Supervisor started (could not verify models)", fg="yellow")
    else:
        # Start in foreground (supervisor only)
        click.echo("Starting supervisor in foreground...")
        click.echo("Note: Run with -d to also start gateway")
        subprocess.run(
            ["uv", "run", "uvicorn", "supervisor.main:app", "--host", "127.0.0.1", "--port", "9001"]
        )


@cli.command()
def stop():
    """Stop Sparkstation (gateway + supervisor) and all model containers."""
    click.echo("Stopping Sparkstation...")

    # Stop gateway
    click.echo("  → Stopping gateway...")
    result = subprocess.run(
        ["pkill", "-f", "litellm.*gateway/litellm.yaml"],
        capture_output=True,
    )

    if result.returncode == 0:
        click.echo("     Gateway stopped")
    else:
        click.echo("     No gateway process found")

    # Stop supervisor
    click.echo("  → Stopping supervisor...")
    result = subprocess.run(
        ["pkill", "-f", "uvicorn supervisor.main:app"],
        capture_output=True,
    )

    if result.returncode == 0:
        click.echo("     Supervisor stopped")
    else:
        click.echo("     No supervisor process found")

    # Stop all model containers
    click.echo("  → Stopping model containers...")
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=sparkstation-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        container_names = result.stdout.strip().split("\n")
        for container_name in container_names:
            if container_name:
                subprocess.run(["docker", "stop", container_name], capture_output=True)
                click.echo(f"     Stopping: {container_name}")

        # Wait for containers to actually stop
        click.echo("  → Waiting for containers to stop...")
        for i in range(30):  # Wait up to 30 seconds
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=sparkstation-", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                click.echo("     All containers stopped")
                break
            time.sleep(1)
        else:
            click.secho("     Warning: Some containers may still be stopping", fg="yellow")

        click.secho("\n✓ Sparkstation stopped", fg="green")
    else:
        click.echo("     No model containers running")
        click.secho("\n✓ Sparkstation stopped", fg="green")


@cli.command()
@click.pass_context
def restart(ctx, detach=True):
    """Restart Sparkstation supervisor."""
    ctx.invoke(stop)
    time.sleep(2)
    ctx.invoke(start, detach=detach)


@cli.command()
@click.pass_context
def status(ctx):
    """Show Sparkstation status."""
    supervisor_url = ctx.obj["supervisor_url"]

    # Check supervisor
    click.echo("Checking Sparkstation status...\n")

    try:
        response = httpx.get(f"{supervisor_url}/health", timeout=5)
        if response.status_code == 200:
            click.secho("✓ Supervisor: RUNNING", fg="green")
        else:
            click.secho("✗ Supervisor: UNHEALTHY", fg="red")
            return
    except Exception as e:
        click.secho(f"✗ Supervisor: NOT RUNNING ({e})", fg="red")
        click.echo("\nRun 'sparkstation start' to start the supervisor")
        return

    # Get models
    try:
        response = httpx.get(f"{supervisor_url}/models/detailed", timeout=10)
        if response.status_code == 200:
            models = response.json()["models"]

            if not models:
                click.echo("\nNo models loaded")
                return

            click.echo(f"\n{len(models)} model(s) loaded:\n")

            for model in models:
                status_icon = {
                    "running": "●",
                    "starting": "◐",
                    "stopped": "○",
                    "failed": "✗",
                }.get(model["status"], "?")

                status_color = {
                    "running": "green",
                    "starting": "yellow",
                    "stopped": "white",
                    "failed": "red",
                }.get(model["status"], "white")

                click.echo(f"  {status_icon} ", nl=False)
                click.secho(f"{model['alias'] or model['model_name']}", fg=status_color, bold=True)
                click.echo(f"     Model: {model['model_name']}")
                click.echo(f"     Status: {model['status']} | Port: {model['port']} | Backend: {model['backend']}")

                if model.get("memory_gb"):
                    click.echo(f"     Memory: {model['memory_gb']}GB")

                click.echo()

    except Exception as e:
        click.secho(f"Failed to get model status: {e}", fg="red")


@cli.group()
def models():
    """Manage models."""
    pass


@models.command("list")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def models_list(ctx, output_json):
    """List all models."""
    supervisor_url = ctx.obj["supervisor_url"]

    try:
        response = httpx.get(f"{supervisor_url}/models/detailed", timeout=10)
        response.raise_for_status()

        if output_json:
            click.echo(response.text)
            return

        models = response.json()["models"]

        if not models:
            click.echo("No models loaded")
            return

        click.echo(f"Found {len(models)} model(s):\n")

        for model in models:
            click.echo(f"ID: {model['id']}")
            click.echo(f"  Name: {model['model_name']}")
            click.echo(f"  Alias: {model['alias']}")
            click.echo(f"  Status: {model['status']}")
            click.echo(f"  Backend: {model['backend']}")
            click.echo(f"  Port: {model['port']}")
            click.echo()

    except Exception as e:
        click.secho(f"Failed to list models: {e}", fg="red")
        sys.exit(1)


@models.command("start")
@click.argument("model_name", required=False)
@click.option("--profile", help="Load models from named profile")
@click.option("--alias", help="Model alias")
@click.option("--backend", default="vllm", help="Backend to use (default: vllm)")
@click.option("--quantization", default="fp8", help="Quantization type (default: fp8)")
@click.pass_context
def models_start(ctx, model_name, profile, alias, backend, quantization):
    """Start a model (from config, profile, or by name)."""
    supervisor_url = ctx.obj["supervisor_url"]

    if profile:
        # Load models from profile
        import yaml
        try:
            with open("models.yaml") as f:
                config = yaml.safe_load(f)
                profile_models = config.get("profiles", {}).get(profile, [])

            if not profile_models:
                click.secho(f"✗ Profile not found: {profile}", fg="red")
                sys.exit(1)

            click.echo(f"Loading {len(profile_models)} model(s) from profile '{profile}'...")
            for model_config in profile_models:
                _start_model(supervisor_url, model_config)

        except Exception as e:
            click.secho(f"✗ Failed to load profile: {e}", fg="red")
            sys.exit(1)

    elif model_name:
        # Start specific model
        model_config = {
            "model_name": model_name,
            "backend": backend,
            "model_alias": alias,
            "quantization": quantization,
        }
        _start_model(supervisor_url, model_config)

    else:
        click.secho("✗ Specify either a model name or --profile", fg="red")
        sys.exit(1)


def _start_model(supervisor_url: str, model_config: dict):
    """Helper to start a single model."""
    try:
        response = httpx.post(
            f"{supervisor_url}/models/start",
            json=model_config,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        model_name = model_config.get("model_alias") or model_config["model_name"]
        click.secho(f"✓ Started model: {model_name}", fg="green")

    except httpx.HTTPStatusError as e:
        model_name = model_config.get("model_alias") or model_config["model_name"]
        click.secho(f"✗ Failed to start {model_name}: {e.response.text}", fg="red")
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red")


@models.command("stop")
@click.argument("model_id")
@click.pass_context
def models_stop(ctx, model_id):
    """Stop a running model."""
    supervisor_url = ctx.obj["supervisor_url"]

    try:
        response = httpx.post(f"{supervisor_url}/models/{model_id}/stop", timeout=30)
        response.raise_for_status()

        result = response.json()
        click.secho(f"✓ Stopped model: {result['model_id']}", fg="green")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            click.secho(f"✗ Model not found: {model_id}", fg="red")
        else:
            click.secho(f"✗ Failed to stop model: {e.response.text}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red")
        sys.exit(1)


@models.command("logs")
@click.argument("model_id")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--tail", default=50, help="Number of lines to show")
def models_logs(model_id, follow, tail):
    """Show model container logs."""
    # Find container name from model_id
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=sparkstation-{model_id}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        click.secho(f"✗ No container found for model: {model_id}", fg="red")
        sys.exit(1)

    container_name = result.stdout.strip().split("\n")[0]

    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.extend(["--tail", str(tail), container_name])

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Overwrite existing CLAUDE.md")
@click.pass_context
def init(ctx, force):
    """Create CLAUDE.md with instructions for using Sparkstation gateway."""
    claude_md_path = Path("CLAUDE.md")

    if claude_md_path.exists() and not force:
        click.echo(f"CLAUDE.md already exists in {Path.cwd()}")
        click.echo("Use --force to overwrite")
        return

    # Get available models from supervisor
    supervisor_url = ctx.obj['supervisor_url']
    models_info = []

    try:
        response = httpx.get(f"{supervisor_url}/models/detailed", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            for model in models:
                if model.get("status") == "running":
                    models_info.append({
                        "name": model.get("alias") or model.get("model_name", "unknown"),
                        "full_name": model.get("model_name", "unknown"),
                        "port": model.get("port", "unknown"),
                    })
    except Exception:
        # If can't connect to supervisor, use defaults
        models_info = [
            {"name": "qwen-vl-3b", "full_name": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ", "port": 8001},
            {"name": "gpt-oss-20b", "full_name": "openai/gpt-oss-20b", "port": 8002},
        ]

    # Generate model list for documentation
    model_list_str = "\n".join([f"- `{m['name']}` - {m['full_name']}" for m in models_info])

    claude_md_content = f"""# Sparkstation Local LLM Gateway

This project has access to local LLM models through Sparkstation gateway.

## Available Models

{model_list_str}

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
    model="qwen-vl-3b",  # or "gpt-oss-20b"
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
    "model": "qwen-vl-3b",
    "messages": [{{"role": "user", "content": "Hello!"}}]
  }}'
```

## Streaming

```python
stream = client.chat.completions.create(
    model="qwen-vl-3b",
    messages=[{{"role": "user", "content": "Tell me a story"}}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Vision (Image Analysis)

The `qwen-vl-3b` model supports vision capabilities. You can pass images via URL or base64:

### With Image URL

```python
response = client.chat.completions.create(
    model="qwen-vl-3b",
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
    model="qwen-vl-3b",
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

## Reasoning Models

The `gpt-oss-20b` model is a reasoning model that shows its thinking process. Access both the reasoning and final response:

```python
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{{"role": "user", "content": "What is 2+2?"}}]
)

# Final answer
print(response.choices[0].message.content)
# Output: "4"

# Reasoning process (if available)
if hasattr(response.choices[0].message, 'reasoning_content'):
    print(response.choices[0].message.reasoning_content)
    # Output: "We need to add 2 and 2. That equals 4."
```

## Embeddings

Sparkstation provides both text and image embedding models for semantic search, RAG, and similarity tasks.

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
print(f"Embedding dimensions: {{len(embedding)}}")
```

### Image Embeddings (CLIP)

The `clip-vit` model generates embeddings for images using OpenAI's CLIP. Supports both URLs and base64:

#### With Image URL
```python
response = client.embeddings.create(
    model="clip-vit",
    input="https://example.com/image.jpg"
)

embedding = response.data[0].embedding
```

#### With Base64 Encoded Image
```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.embeddings.create(
    model="clip-vit",
    input=f"data:image/jpeg;base64,{{image_data}}"
)

embedding = response.data[0].embedding
```

### Batch Embeddings

Generate embeddings for multiple inputs at once:

```python
response = client.embeddings.create(
    model="bge-large",
    input=["First document", "Second document", "Third document"]
)

for i, data in enumerate(response.data):
    print(f"Document {{i}}: {{len(data.embedding)}} dimensions")
```

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
    input="https://example.com/car.jpg"
)
image_embedding = image_response.data[0].embedding

# Compare via cosine similarity (both in same embedding space)
from numpy import dot
from numpy.linalg import norm

similarity = dot(text_embedding, image_embedding) / (norm(text_embedding) * norm(image_embedding))
print(f"Similarity: {{similarity}}")
```

### Use Cases

- **Semantic Search**: Embed documents and queries, find similar content via cosine similarity
- **RAG (Retrieval Augmented Generation)**: Embed knowledge base for context retrieval
- **Image Search**: Use CLIP to search images by text description or find similar images
- **Cross-Modal Retrieval**: Search images with text queries or text with image queries
- **Classification**: Use embeddings as features for downstream ML tasks

## Important Notes

- **Do not start/stop Sparkstation services** - they are managed by the system
- Models are already running and ready to use
- Use the gateway endpoint (`http://localhost:8000/v1`) for all requests
- All models support standard OpenAI APIs:
  - Chat: `/v1/chat/completions` (qwen-vl-3b, gpt-oss-20b)
  - Embeddings: `/v1/embeddings` (bge-large, clip-vit)
- **Vision Chat**: `qwen-vl-3b` supports image analysis (URL and base64)
- **Reasoning**: `gpt-oss-20b` includes reasoning traces in `reasoning_content` field
- **Text Embeddings**: `bge-large` generates 1024-dim embeddings for text semantic tasks
- **Image Embeddings**: `clip-vit` generates embeddings for images and cross-modal search (URL and base64)
"""

    # Write file
    with open(claude_md_path, "w") as f:
        f.write(claude_md_content)

    click.secho(f"\n✓ Created CLAUDE.md in {Path.cwd()}", fg="green")
    click.echo("\nThis file provides instructions for AI assistants on how to use")
    click.echo("the Sparkstation gateway without managing the services.")


@cli.command()
@click.option("--force", is_flag=True, help="Force cleanup without confirmation")
@click.pass_context
def cleanup(ctx, force):
    """Clean up database and orphaned containers."""
    if not force:
        click.confirm("This will stop all models, clean up containers, and reset the database. Continue?", abort=True)

    click.echo("Cleaning up...")

    # Stop gateway
    click.echo("  → Stopping gateway...")
    subprocess.run(["pkill", "-9", "-f", "litellm.*gateway/litellm.yaml"], capture_output=True)

    # Stop supervisor
    click.echo("  → Stopping supervisor...")
    subprocess.run(["pkill", "-9", "-f", "uvicorn supervisor.main:app"], capture_output=True)

    # Stop and remove ALL Sparkstation containers (running and stopped)
    click.echo("  → Stopping and removing Sparkstation containers...")
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=sparkstation-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        container_names = result.stdout.strip().split("\n")
        for container_name in container_names:
            if container_name:  # Skip empty lines
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                click.echo(f"     Removed: {container_name}")
    else:
        click.echo("     No Sparkstation containers found")

    # Clean database
    click.echo("  → Cleaning database...")
    db_path = Path("data/sparkstation.db")
    if db_path.exists():
        db_path.unlink()
        click.echo("     Database removed")

    for ext in ["-shm", "-wal"]:
        db_file = Path(f"data/sparkstation.db{ext}")
        if db_file.exists():
            db_file.unlink()

    click.secho("\n✓ Cleanup complete!", fg="green")
    click.echo("\nRun 'sparkstation start' to restart with a clean state")


if __name__ == "__main__":
    cli(obj={})
