# SparkStation

**Model fleet management for NVIDIA DGX Spark.**

SparkStation is a headless control plane for running and managing multiple AI models across one or more NVIDIA DGX Spark systems.

Define models once, group them into workload profiles, place them on logical Spark hosts, and expose them through a stable OpenAI-compatible endpoint. SparkStation manages model lifecycle, resource admission, health checks, restarts, routing synchronization, and DGX Spark-specific unified-memory constraints.

SparkStation coordinates existing infrastructure:

- vLLM and SGLang run language and embedding models
- LiteLLM provides the stable OpenAI-compatible gateway
- Docker isolates model services
- SparkStation coordinates configuration, placement, lifecycle, and health

Example workloads include local assistants, embedding services, image indexing, vision models, detection pipelines, image generation, and other persistent local AI applications.

**Version:** `0.4.0`

## Why SparkStation?

Running one model on a DGX Spark is straightforward. Running several model services reliably becomes operationally messy.

Without SparkStation, users often accumulate:

- separate Docker commands and shell scripts
- manually assigned ports
- model-specific backend flags
- duplicated health and restart logic
- inconsistent endpoint URLs
- no unified resource view
- manual placement across multiple Sparks
- fragile startup ordering
- application configuration tied directly to backend details

SparkStation turns that collection of scripts and services into a declared model fleet. The goal is not to hide the inference engines. The goal is to make a collection of DGX Spark model services behave like one manageable appliance.

SparkStation is profile-driven. A profile describes the set of model services required for a workload, where they should run, and any workload-specific overrides.

## Who It Is For

SparkStation is for someone who:

- owns one or more DGX Spark systems
- runs several models or inference backends
- integrates models into applications or pipelines
- wants deterministic aliases, placement, startup, and recovery
- wants a stable API surface across backend changes
- does not want to operate a full Kubernetes-based inference platform

### Who Probably Does Not Need It

SparkStation is probably unnecessary for:

- someone running only one model manually
- someone primarily looking for a graphical chat interface
- someone only experimenting occasionally with models
- someone already operating a mature cluster orchestration platform

## Key Features

1. **Profile-driven model fleets**: define workload-specific desired state in `models.yaml`.
2. **Multi-Spark logical host placement**: place model services on roles such as `primary` and `worker1`.
3. **Stable model aliases**: keep client configuration stable while changing model IDs, images, quantization, or runtime flags.
4. **LiteLLM gateway synchronization**: register running models with the OpenAI-compatible gateway.
5. **Multi-backend and multi-model-type support**: run chat, embeddings, image, detection, recognition, and custom services behind the same management plane.
6. **DGX Spark unified-memory admission control**: reserve configured memory before launching local services.
7. **Ordered model startup**: launch profile models in configuration order, with extra handling for large and image-generation services.
8. **Health monitoring**: periodic probes detect unresponsive services.
9. **Automatic restart**: failed services can restart with configured backoff and attempt limits.
10. **State reconciliation after supervisor restart**: persisted model state is compared with actual Docker containers.
11. **Suspend and resume support**: models can be manually or automatically suspended and resumed.
12. **Prometheus metrics**: expose supervisor, model, resource, and gateway proxy metrics.
13. **API-key-protected management endpoints**: protect lifecycle mutation endpoints with `API_KEY`.
14. **Docker-based backend isolation**: run model services in backend-specific containers.

## Architecture

```text
Applications
assistants | indexing pipelines | agents | automation | local tools
                         |
                         v
              Stable OpenAI-compatible API
                    LiteLLM Gateway
                         |
                         v
              SparkStation Supervisor
     profiles | placement | lifecycle | health | resources
                         |
             +-----------+-----------+
             v                       v
       DGX Spark primary        DGX Spark worker(s)
       vLLM / SGLang            vLLM / SGLang
       embeddings               large chat models
       vision backends          additional services
```

Clients call stable aliases through the gateway. They do not need to know which host runs the model, which backend serves it, which Docker image is used, which port it currently occupies, or whether the model has restarted.

### Components

| Component     | Responsibility                                                                       |
| ------------- | ------------------------------------------------------------------------------------ |
| vLLM / SGLang | Model inference                                                                      |
| LiteLLM       | OpenAI-compatible gateway and routing                                                |
| Docker        | Process and dependency isolation                                                     |
| Prometheus    | Metrics collection and querying                                                      |
| SparkStation  | Model definitions, profiles, placement, lifecycle, health, and resource coordination |

## Core Concepts

### Model

A model is a canonical model or backend service definition. It can include:

- backend
- model identifier
- model type
- memory budget
- Docker image
- runtime arguments
- environment variables
- lifecycle policy
- default placement

SparkStation supports more than chat models:

- chat and completion models
- text embeddings
- image embeddings
- image generation
- detection and recognition services
- custom model backends

Each model type uses the backend appropriate for that service. SparkStation does not imply that every model type runs through the same inference engine.

### Alias

An alias is a stable client-facing name.

The underlying Hugging Face model, quantization, engine version, Docker image, or runtime configuration may change without forcing clients to change their configuration.

### Profile

A profile is a named desired-state collection of models for a workload. Examples include:

- `dev`
- `openclaw`
- `image-indexing`
- `inference`

Profiles may override host placement, memory allocation, concurrency, context length, runtime arguments, environment variables, and lifecycle behavior.

Example:

```yaml
profiles:
  openclaw:
    qwen3.5-35b:
      extra_args:
        max_concurrent_requests: 8
    bge-m3: {}

  image-indexing:
    qwen3.5-35b:
      host: worker1
      memory_gb: 100
    bge-m3: {}
    clip-vit: {}
    species-detect: {}
    face-detect: {}
```

In this topology, `image-indexing` dedicates one Spark to the large chat model while auxiliary embedding and vision services remain on the primary Spark.

### Host Role

A host role is a logical machine name such as `primary` or `worker1`.

Public configuration contains the model topology. Private machine-specific configuration contains IP addresses, SSH users, and hostnames.

### Supervisor

The supervisor is the long-running service that reconciles configuration with running containers and coordinates:

- model startup
- model shutdown
- resource reservations
- health checks
- automatic restart
- state reconciliation
- gateway synchronization

## Quick Start

This is the smallest useful single-node flow. Multi-Spark configuration is shown after the local path.

### 1. Clone the Repository

```bash
git clone https://github.com/kshetrajna12/sparkstation.git
cd sparkstation
uv sync
```

Optional backend setup helpers:

```bash
./scripts/setup_backends.sh
./scripts/verify_backends.sh
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Set values that apply to your machine:

```env
USE_DOCKER=true
HF_TOKEN=hf_...
API_KEY=change-me
TOTAL_UNIFIED_MEMORY_GB=128
MEMORY_HARD_LIMIT_GB=110
```

`API_KEY` is optional, but if set it is required for model lifecycle mutation endpoints.

### 3. Define Host Roles

For one Spark, `primary` is enough:

```yaml
cluster:
  hosts:
    primary: {}
```

### 4. Define Canonical Models

Add model definitions to `models.yaml`:

```yaml
default_profile: dev

cluster:
  hosts:
    primary: {}

models:
  chat:
    name: Qwen/Qwen3-8B
    backend: vllm
    model_type: chat
    memory_gb: 40

  embeddings:
    name: BAAI/bge-m3
    backend: vllm
    model_type: embedding
    memory_gb: 3

profiles:
  dev:
    chat: {}
    embeddings: {}
```

Starting the `dev` profile launches both services and registers them with the gateway.

### 5. Start the Supervisor and Gateway

```bash
sparkstation start -d --profile dev
```

The detached start flow launches the supervisor, waits for configured models, writes `gateway/litellm.yaml`, starts LiteLLM, and starts the gateway proxy on `127.0.0.1:8000`.

### 6. Check Health and Model Status

```bash
sparkstation status
sparkstation models list
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/models/detailed
```

### 7. Call a Model Through the Stable Gateway Endpoint

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer dummy-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

For embeddings:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Authorization: Bearer dummy-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "embeddings",
    "input": "DGX Spark local model fleet"
  }'
```

## Model Configuration

`models.yaml` contains the public desired state:

- `cluster.hosts`: logical host roles
- `default_profile`: the profile used when no startup profile is provided
- `models`: canonical model definitions keyed by alias
- `profiles`: workload profiles keyed by profile name

Common model fields:

```yaml
models:
  qwen3.5-35b:
    name: nvidia/Qwen3.6-35B-A3B-NVFP4
    backend: vllm
    model_type: chat
    host: primary
    memory_gb: 40
    docker_image: vllm/vllm-openai:nightly-20260611-goodgb10
    env_vars:
      VLLM_FP8_MOE_BACKEND: flashinfer_cutlass
    extra_args:
      max_model_len: 65536
      max_concurrent_requests: 4
```

Supported fields are parsed by `supervisor.models_config.ModelDefinition` and include `name`, `backend`, `model_type`, `host`, `quantization`, `memory_gb`, `idle_timeout_minutes`, `auto_suspend_enabled`, speculative decoding fields, `default`, `extra_args`, `docker_image`, `env_vars`, and `volumes`.

## Profiles

Profiles describe desired state for a workload. A profile entry enables an alias and may override the canonical model definition:

```yaml
profiles:
  inference:
    qwen3.5-35b: {}

  openclaw:
    qwen3.5-35b:
      extra_args:
        max_concurrent_requests: 8
    bge-m3: {}
```

Profile overrides deep-merge dictionary fields such as `extra_args`, `env_vars`, and `speculative_extra`. Scalar fields replace the base value.

Useful commands:

```bash
sparkstation start -d --profile openclaw
sparkstation models start bge-m3 --profile openclaw
sparkstation models stop bge-m3
sparkstation models swap qwen3.5-35b --profile image-indexing
sparkstation gateway restart
```

## Multi-Spark Operation

SparkStation treats multiple DGX Sparks as one administratively managed fleet, not as one shared GPU pool. Each model service is assigned to a logical host, and SparkStation manages that service on the selected machine.

The supervisor runs on the primary Spark. For remote roles, it drives Docker over SSH by setting `DOCKER_HOST=ssh://<ssh-user>@<worker-ip>` for Docker commands. The gateway uses the configured host IP when routing to containers on workers.

### Public and Private Configuration

Keep role names and model placement in committed `models.yaml`:

```yaml
cluster:
  hosts:
    primary: {}
    worker1: {}
```

Keep real IPs and SSH users in gitignored `.sparkstation.local.yaml`:

```yaml
cluster:
  hosts:
    worker1:
      ip: <worker-ip>
      ssh_user: <ssh-user>
      label: lab-worker-1
```

Before assigning a model to a worker, configure passwordless SSH from `primary` to the worker, ensure the SSH user can run Docker without `sudo`, and make sure required Docker images and Hugging Face cache entries are available on the target host.

Useful cluster commands:

```bash
sparkstation cluster status
sparkstation cluster sync-cache worker1 --only models--nvidia--Qwen3.6-35B-A3B-NVFP4
sparkstation cluster sync-cache worker1 --dry-run
sparkstation cluster ncclbench
```

Custom local Docker images must be copied to the worker's Docker daemon before placement there, for example:

```bash
docker save vllm-qwen35-mxfp4:cu130 | ssh <ssh-user>@<worker-ip> 'docker load'
```

### Independent Model Placement

SparkStation supports independent placement of model services:

- chat model on `worker1`
- embeddings and vision services on `primary`
- each service runs independently
- SparkStation manages placement and lifecycle

### Distributed Execution of One Model

Distributed execution means one model is served by a distributed runtime, for example:

- tensor parallel inference
- multi-node inference
- distributed runtime execution

SparkStation does not implement distributed inference itself. If a backend supports distributed execution, that behavior belongs to the underlying inference runtime or external tooling. SparkStation can still manage the resulting service as a configured model backend.

## Lifecycle and Recovery

SparkStation's operational model is:

- configured models are loaded from the active profile
- resource reservations are created before launch
- models are started in profile order
- health checks move models into healthy or failed states
- failed services may be restarted with configured backoff
- the supervisor reconciles persisted state with actual containers after restart
- gateway routes are updated as services become available or disappear

The supervisor exposes lifecycle operations through API endpoints and the CLI. The gateway proxy also handles requests to suspended or starting models by asking the supervisor for current model state.

## Metrics and Operations

Supervisor endpoints:

```text
GET  /health
GET  /metrics
GET  /models
GET  /models/detailed
GET  /models/{model_id}/status
GET  /prometheus/targets
GET  /resources
POST /models/start
POST /models/{model_id}/stop
POST /models/{model_id}/suspend
POST /models/{model_id}/resume
```

Lifecycle mutation endpoints require `X-API-Key: <API_KEY>` when `API_KEY` is set.

Gateway endpoints:

```text
GET  http://127.0.0.1:8000/proxy/health
GET  http://127.0.0.1:8000/metrics
POST http://127.0.0.1:8000/v1/chat/completions
POST http://127.0.0.1:8000/v1/embeddings
POST http://127.0.0.1:8000/v1/images/generations
```

Logs in detached mode are written under `~/.sparkstation/logs/`:

```text
supervisor.log
gateway.log
gateway-proxy.log
```

Prometheus can scrape `/metrics` from the supervisor and gateway proxy. The supervisor also exposes `/prometheus/targets` for model backends that provide native metrics endpoints.

## Example Deployment

The current repository configuration is one possible topology:

- `worker1` runs the large Qwen chat model
- `primary` runs text embeddings, CLIP embeddings, species detection, and face detection
- applications use stable model aliases
- SparkStation handles startup order, placement, health, restart, and gateway registration

The active checked-in default profile is `image-indexing`. Users can define their own profiles and host roles for different local applications, pipelines, and services.

## API and CLI Reference

Common CLI commands:

```bash
sparkstation start -d --profile <profile>
sparkstation stop
sparkstation restart --profile <profile>
sparkstation status

sparkstation models list
sparkstation models start <alias> --profile <profile>
sparkstation models stop <alias>
sparkstation models swap <alias> --profile <profile>
sparkstation models logs <model_id> --tail 100

sparkstation gateway restart

sparkstation cluster status
sparkstation cluster sync-cache <host-role>
sparkstation cluster ncclbench
```

Example authenticated lifecycle call:

```bash
curl -X POST http://127.0.0.1:9001/models/<model_id>/suspend \
  -H "X-API-Key: $API_KEY"
```

## What SparkStation Manages

SparkStation manages:

- canonical model definitions
- workload profiles
- logical host placement
- lifecycle operations
- startup ordering
- unified-memory admission control
- model health monitoring
- automatic restart
- state reconciliation
- stable model aliases
- LiteLLM gateway synchronization
- operational metrics and status
- local and multi-Spark orchestration

## What SparkStation Does Not Do

SparkStation is not:

- an inference engine
- a chat application
- a model marketplace
- a training or fine-tuning system
- a replacement for LiteLLM
- Kubernetes
- intended to become a generic cloud inference platform

## Project Status

SparkStation is designed for persistent local services on DGX Spark systems. The implementation includes Docker launchers, profile resolution, local and Docker-over-SSH placement, LiteLLM route generation, health checks, restart handling, suspend/resume, Prometheus metrics, and a CLI.

Known boundaries:

- remote worker memory is configured through profile placement, but primary-side unified-memory admission control is the implemented local admission path
- distributed inference is delegated to the selected backend runtime
- some helper commands assume local repo layout or local infrastructure scripts and should be checked before automation

## License

No license file is currently present in this repository.
