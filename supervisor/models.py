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
    TRT_LLM = "trt-llm"  # Future support


# SQLAlchemy ORM Model
class ModelInstanceDB(Base):
    """Database model for model instances."""

    __tablename__ = "model_instances"

    id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_alias = Column(String, nullable=True)
    backend = Column(String, nullable=False)
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


# Pydantic models for API
class ModelConfig(BaseModel):
    """Configuration for starting a new model."""

    model_name: str = Field(..., description="HuggingFace model name")
    backend: Backend = Field(..., description="LLM backend to use")
    model_alias: Optional[str] = Field(None, description="Alias for easier reference")
    num_gpus: int = Field(1, description="Number of GPUs to allocate")
    quantization: Optional[str] = Field("fp8", description="Quantization type (fp8, int4, awq)")
    idle_timeout_minutes: int = Field(30, description="Minutes before auto-suspend")
    auto_suspend_enabled: bool = Field(True, description="Enable auto-suspend")
    use_subprocess: bool = Field(True, description="Use subprocess vs systemd/docker")
    # Speculative decoding (vLLM only)
    speculative_model: Optional[str] = Field(None, description="Draft model for speculative decoding")
    num_speculative_tokens: int = Field(5, description="Number of speculative tokens to generate")
    speculative_method: Optional[str] = Field(None, description="Speculative decoding method (auto-detected if None)")
    extra_args: Dict[str, Any] = Field(
        default_factory=dict, description="Additional backend-specific args"
    )


class ModelInstance(BaseModel):
    """Runtime model instance."""

    id: str
    model_name: str
    model_alias: Optional[str] = None
    backend: Backend
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
    model_alias: Optional[str] = None
    num_gpus: int = 1
    quantization: Optional[str] = "fp8"
    idle_timeout_minutes: int = 30
    auto_suspend_enabled: bool = True
    # Speculative decoding (vLLM only)
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5
    speculative_method: Optional[str] = None
    extra_args: Dict[str, Any] = Field(default_factory=dict)


class ModelStartResponse(BaseModel):
    """Response from starting a model."""

    model_id: str
    model_name: str
    backend: str
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
