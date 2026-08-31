"""
Data models for Sparkstation Supervisor.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ModelStatus(str, Enum):
    """Model lifecycle states."""

    STARTING = "starting"
    RUNNING = "running"
    SUSPENDED = "suspended"
    STOPPED = "stopped"
    FAILED = "failed"


class HealthStatus(str, Enum):
    """Model health states."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class Backend(str, Enum):
    """Supported LLM backends."""

    VLLM = "vllm"
    SGLANG = "sglang"
    FLUX = "flux"
    CLIP = "clip"
    SPECIES = "species"
    FACE = "face"
    TRT_LLM = "trt-llm"  # Future support
    DSPARK = "dspark"  # Anemll DSpark multi-node compose stack (2x GB10 TP=2)
    VOICECASCADE = "voicecascade"  # cascade voice stack: Kyutai STT -> routed brain -> Qwen3-TTS (pipecat)


class ModelType(str, Enum):
    """Model type for different inference tasks."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    DETECTION = "detection"
    VOICE = "voice"  # WebSocket/WebRTC audio — not an OpenAI-API model


# SQLAlchemy ORM Model
class ModelInstanceDB(Base):
    """Database model for model instances."""

    __tablename__ = "model_instances"

    id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_alias = Column(String, nullable=True)
    backend = Column(String, nullable=False)
    model_type = Column(String, nullable=False, default="chat")
    status = Column(String, nullable=False)
    health_status = Column(String, default="unknown")
    port = Column(Integer, nullable=False)
    gpu_ids = Column(JSON, nullable=False)  # List[int]
    base_url = Column(String, nullable=False)
    pid = Column(Integer, nullable=True)
    container_id = Column(String, nullable=True)
    systemd_service = Column(String, nullable=True)

    started_at = Column(DateTime, nullable=False)
    stopped_at = Column(DateTime, nullable=True)
    last_health_check = Column(DateTime, nullable=True)
    last_request_time = Column(DateTime, nullable=True)

    # Auto-suspend configuration
    auto_suspend_enabled = Column(Boolean, default=True)
    idle_timeout_minutes = Column(Integer, default=30)
    memory_gb = Column(Float, nullable=True)  # Estimated memory usage

    # Configuration saved for resume
    saved_config = Column(JSON, nullable=True)  # Dict saved for resume
    extra_args = Column(JSON, nullable=True)  # Additional launch args

    # Restart tracking
    restart_count = Column(Integer, default=0)  # Number of restart attempts
    last_restart_time = Column(DateTime, nullable=True)  # Last restart timestamp

    # Cluster role (from cluster.hosts in models.yaml). Nullable for legacy
    # rows that predate cluster support — the registry mapper defaults NULL
    # back to "primary" so pre-cluster deployments keep working unchanged.
    host = Column(String, nullable=True)


# Pydantic models for API
class ModelConfig(BaseModel):
    """Configuration for starting a new model."""

    model_name: str = Field(..., description="HuggingFace model name")
    backend: Backend = Field(..., description="LLM backend to use")
    model_type: ModelType = Field(ModelType.CHAT, description="Model type (chat or embedding)")
    model_alias: Optional[str] = Field(None, description="Alias for easier reference")
    host: str = Field("primary", description="Cluster role (key in cluster.hosts)")
    num_gpus: int = Field(1, description="Number of GPUs to allocate")
    quantization: Optional[str] = Field("fp8", description="Quantization type (fp8, int4, awq)")
    idle_timeout_minutes: int = Field(30, description="Minutes before auto-suspend")
    auto_suspend_enabled: bool = Field(True, description="Enable auto-suspend")
    use_subprocess: bool = Field(True, description="Use subprocess vs systemd/docker")
    # Speculative decoding (vLLM only)
    speculative_model: Optional[str] = Field(None, description="Draft model for speculative decoding")
    num_speculative_tokens: int = Field(5, description="Number of speculative tokens to generate")
    speculative_method: Optional[str] = Field(None, description="Speculative decoding method (auto-detected if None)")
    speculative_extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keys merged into --speculative-config JSON (e.g. {moe_backend: triton})",
    )
    extra_args: Dict[str, Any] = Field(
        default_factory=dict, description="Additional backend-specific args"
    )
    docker_image: Optional[str] = Field(None, description="Per-model Docker image override")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Extra container env vars")
    volumes: List[str] = Field(default_factory=list, description="Extra volume mounts (host:container)")

    @classmethod
    def from_saved_config(cls, saved: Dict[str, Any]) -> "ModelConfig":
        """Rebuild the launch config from a ModelInstance.saved_config snapshot.

        Every relaunch path (auto-restart, resume-from-suspend) must go through
        this so the replacement container is launched with the SAME host,
        docker image, model type, env vars, volumes, and speculative config as
        the original — hand-rolled partial reconstructions previously dropped
        `host`/`docker_image` and relaunched worker models on the primary with
        the default image.
        """
        return cls(
            model_name=saved["model_name"],
            backend=Backend(saved["backend"]),
            model_type=ModelType(saved.get("model_type", "chat")),
            model_alias=saved.get("model_alias"),
            host=saved.get("host") or "primary",
            num_gpus=len(saved.get("gpu_ids") or [0]),
            quantization=saved.get("quantization"),
            idle_timeout_minutes=saved.get("idle_timeout_minutes", 30),
            auto_suspend_enabled=saved.get("auto_suspend_enabled", True),
            speculative_model=saved.get("speculative_model"),
            num_speculative_tokens=saved.get("num_speculative_tokens", 5),
            speculative_method=saved.get("speculative_method"),
            speculative_extra=saved.get("speculative_extra") or {},
            extra_args=saved.get("extra_args") or {},
            docker_image=saved.get("docker_image"),
            env_vars=saved.get("env_vars") or {},
            volumes=saved.get("volumes") or [],
        )


def build_saved_config(
    config: ModelConfig,
    gpu_ids: List[int],
    port: int,
    memory_gb: Optional[float],
) -> Dict[str, Any]:
    """Snapshot of everything ModelConfig.from_saved_config needs to relaunch."""
    def _val(x):
        return x.value if hasattr(x, "value") else x

    return {
        "model_name": config.model_name,
        "backend": _val(config.backend),
        "model_type": _val(config.model_type),
        "model_alias": config.model_alias,
        "host": config.host,
        "gpu_ids": gpu_ids,
        "port": port,
        "memory_gb": memory_gb,
        "quantization": config.quantization,
        "auto_suspend_enabled": config.auto_suspend_enabled,
        "idle_timeout_minutes": config.idle_timeout_minutes,
        "speculative_model": config.speculative_model,
        "num_speculative_tokens": config.num_speculative_tokens,
        "speculative_method": config.speculative_method,
        "speculative_extra": config.speculative_extra,
        "extra_args": config.extra_args,
        "docker_image": config.docker_image,
        "env_vars": config.env_vars,
        "volumes": config.volumes,
    }


class ModelInstance(BaseModel):
    """Runtime model instance."""

    id: str
    model_name: str
    model_alias: Optional[str] = None
    backend: Backend
    model_type: ModelType = ModelType.CHAT
    # Cluster role this instance lives on. Persisted so reconcile / stop /
    # health-check know which docker daemon to talk to for this container.
    host: str = "primary"
    status: ModelStatus
    health_status: HealthStatus = HealthStatus.UNKNOWN
    port: int
    gpu_ids: List[int]
    base_url: str
    pid: Optional[int] = None
    container_id: Optional[str] = None
    systemd_service: Optional[str] = None

    started_at: datetime
    stopped_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    last_request_time: Optional[datetime] = None

    auto_suspend_enabled: bool = True
    idle_timeout_minutes: int = 30
    memory_gb: Optional[float] = None

    saved_config: Optional[Dict[str, Any]] = None
    extra_args: Optional[Dict[str, Any]] = None

    restart_count: int = 0
    last_restart_time: Optional[datetime] = None

    model_config = ConfigDict(use_enum_values=True)


class ModelStartRequest(BaseModel):
    """Request to start a new model."""

    model_name: str
    backend: Backend
    model_type: ModelType = ModelType.CHAT
    model_alias: Optional[str] = None
    host: str = "primary"  # Cluster role — must exist in models.yaml cluster.hosts
    num_gpus: int = 1
    quantization: Optional[str] = "fp8"
    idle_timeout_minutes: int = 30
    auto_suspend_enabled: bool = True
    # Speculative decoding (vLLM only)
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5
    speculative_method: Optional[str] = None
    speculative_extra: Dict[str, Any] = Field(default_factory=dict)
    extra_args: Dict[str, Any] = Field(default_factory=dict)
    # Explicit memory reservation in GB. When set, bypasses the built-in
    # estimate_model_memory() heuristic (which substring-matches model names
    # and mis-estimates MoE models like "Qwen3.6-35B-A3B" because "3b" in "a3b"
    # triggers the 3B branch).
    memory_gb: Optional[float] = None
    # Per-model image / env / volume overrides. Without these on the request
    # body, /models/start cannot reconstruct a config that uses a non-default
    # docker_image (e.g. vllm-qwen35-mxfp4:cu130 or vllm/vllm-openai:cu130-nightly)
    # and the autoload path's per-model image override is unreachable from the API.
    docker_image: Optional[str] = None
    env_vars: Dict[str, str] = Field(default_factory=dict)
    volumes: List[str] = Field(default_factory=list)


class ModelStartResponse(BaseModel):
    """Response from starting a model."""

    model_id: str
    model_name: str
    backend: str
    model_type: str
    status: str
    port: int
    gpu_ids: List[int]
    base_url: str
    started_at: datetime
    idle_timeout_minutes: int
    auto_suspend_enabled: bool


class ModelStatusResponse(BaseModel):
    """Detailed model status response."""

    model_id: str
    model_name: str
    model_type: str
    status: str
    health_status: str
    uptime_seconds: Optional[float] = None
    last_health_check: Optional[datetime] = None
    last_request_time: Optional[datetime] = None
    idle_seconds: Optional[float] = None
    idle_timeout_minutes: int
    auto_suspend_enabled: bool
    seconds_until_suspend: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


class ResourceStatus(BaseModel):
    """System resource status."""

    unified_memory_gb: float
    unified_memory_used_gb: float
    unified_memory_limit_gb: float
    gpu_temperature_c: float
    gpu_power_draw_w: float
    resident_models_count: int
    max_resident_models: int
    available_ports: int


class LiteLLMModelFormat(BaseModel):
    """LiteLLM-compatible model format (simplified)."""

    model_name: str
    litellm_provider: str = "openai"
    api_base: str
    api_key: str = "EMPTY"
