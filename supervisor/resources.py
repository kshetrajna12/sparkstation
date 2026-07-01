"""
Resource management for DGX Spark unified memory architecture.
"""
import subprocess
import logging
from typing import Dict, Set, Optional
from supervisor.config import settings

logger = logging.getLogger(__name__)


class ResourceError(Exception):
    """Raised when resource allocation fails."""

    pass


class ResourceManager:
    """
    DGX Spark unified memory resource manager.

    CRITICAL: DGX Spark has unified 128 GB memory shared by CPU+GPU.
    Unlike discrete GPUs with separate VRAM, we must track total system memory.
    """

    def __init__(self):
        # DGX Spark constraints
        self.total_unified_memory_gb = settings.total_unified_memory_gb
        self.hard_limit_gb = settings.memory_hard_limit_gb
        self.soft_limit_gb = settings.memory_soft_limit_gb
        self.max_resident_models = settings.max_resident_models

        # Port tracking
        self.allocated_ports: Dict[str, int] = {}  # model_id -> port
        self.port_range = (settings.model_port_range_start, settings.model_port_range_end)

        # Memory tracking
        self.model_memory_usage: Dict[str, float] = {}  # model_id -> estimated_gb

    def get_unified_memory_usage(self) -> float:
        """
        Get current unified memory usage in GB.

        CRITICAL: DGX Spark has unified memory shared by CPU+GPU.
        nvidia-smi only shows GPU-visible portion, not total system usage.
        Must track both GPU memory AND system memory.

        Returns:
            Estimated unified memory usage in GB (conservative estimate)
        """
        try:
            # Get GPU-reported memory usage
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            gpu_memory_str = gpu_result.stdout.strip()
            # Handle [N/A] case (happens during model startup)
            if gpu_memory_str == "[N/A]" or not gpu_memory_str:
                gpu_memory_mb = 0
            else:
                gpu_memory_mb = float(gpu_memory_str)

            # Get system memory usage from /proc/meminfo
            # MemTotal - MemAvailable = used memory
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    key, value = line.split(":")
                    meminfo[key.strip()] = int(value.strip().split()[0])  # Value in kB

            mem_total_kb = meminfo.get("MemTotal", 0)
            mem_available_kb = meminfo.get("MemAvailable", 0)
            system_used_mb = (mem_total_kb - mem_available_kb) / 1024  # Convert kB to MB

            # Conservative estimate: use MAX of GPU-reported or system-reported
            # Plus headroom buffer (16 GB safety margin)
            estimated_used_mb = max(gpu_memory_mb, system_used_mb) + (16 * 1024)  # +16 GB

            logger.debug(
                f"Memory usage: GPU={gpu_memory_mb/1024:.1f}GB, "
                f"System={system_used_mb/1024:.1f}GB, "
                f"Estimated={estimated_used_mb/1024:.1f}GB"
            )

            return estimated_used_mb / 1024  # Convert to GB

        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            # Fallback: sum of tracked model memory
            return sum(self.model_memory_usage.values())

    def get_gpu_temperature(self) -> float:
        """Get GPU temperature in Celsius."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get GPU temperature: {e}")
            return 0.0

    def get_gpu_power_draw(self) -> float:
        """Get GPU power draw in Watts."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get GPU power draw: {e}")
            return 0.0

    def can_allocate_model(self, estimated_memory_gb: float) -> bool:
        """
        Check if we can allocate a new model.

        Args:
            estimated_memory_gb: Expected memory usage for the model

        Returns:
            True if model can be allocated, False otherwise
        """
        current_usage = self.get_unified_memory_usage()
        projected_usage = current_usage + estimated_memory_gb

        # Check memory limit
        if projected_usage > self.hard_limit_gb:
            logger.warning(
                f"Cannot allocate model: would exceed hard limit "
                f"({projected_usage:.1f} GB > {self.hard_limit_gb} GB)"
            )
            return False

        # Check resident model count
        if len(self.model_memory_usage) >= self.max_resident_models:
            logger.warning(
                f"Cannot allocate model: max resident models "
                f"({self.max_resident_models}) reached"
            )
            return False

        # Warn on soft limit
        if projected_usage > self.soft_limit_gb:
            logger.warning(
                f"Approaching memory soft limit: "
                f"{projected_usage:.1f} GB / {self.soft_limit_gb} GB"
            )

        return True

    def allocate_port(self) -> int:
        """
        Find and allocate an available port.

        Returns:
            Available port number

        Raises:
            ResourceError: If no ports available
        """
        allocated_set = set(self.allocated_ports.values())
        for port in range(*self.port_range):
            if port not in allocated_set:
                return port
        raise ResourceError("No available ports in range")

    def allocate_model(self, model_id: str, estimated_memory_gb: float,
                       host: str = "primary") -> int:
        """
        Allocate resources for a model.

        Args:
            model_id: Unique model identifier
            estimated_memory_gb: Expected memory usage (on the target host)
            host: Cluster role the model runs on. When != "primary" the
                  memory limit isn't enforced here — the primary supervisor's
                  113 GB budget is only about the primary Spark. Remote hosts
                  have their own physical limits which we don't track from
                  here (yet); a bad allocation surfaces as a docker OOM on
                  the target when the container fails to start. Ports are
                  still tracked globally so we don't hand out the same one
                  twice across hosts.

        Returns:
            Allocated port number

        Raises:
            ResourceError: If allocation fails
        """
        if host == "primary" and not self.can_allocate_model(estimated_memory_gb):
            raise ResourceError(
                f"Cannot allocate model {model_id}: resource limits exceeded"
            )

        # Allocate port
        port = self.allocate_port()
        self.allocated_ports[model_id] = port
        # Only count memory against the primary Spark's budget. Remote-host
        # memory tracking is a follow-up; the memory-usage metrics + auto-
        # suspend heuristics all operate against the primary's unified memory.
        if host == "primary":
            self.model_memory_usage[model_id] = estimated_memory_gb

        logger.info(
            f"Allocated resources for {model_id} on host={host}: "
            f"port={port}, memory={estimated_memory_gb}GB"
        )

        return port

    def release_model(self, model_id: str, full_release: bool = False):
        """
        Release resources for a model.

        Args:
            model_id: Model identifier
            full_release: If True, release port as well (for stopped models).
                         If False, keep port reserved (for suspended models).
        """
        # Release memory
        if model_id in self.model_memory_usage:
            freed_memory = self.model_memory_usage.pop(model_id)
            logger.info(f"Released {freed_memory}GB memory for {model_id}")

        # Optionally release port
        if full_release and model_id in self.allocated_ports:
            freed_port = self.allocated_ports.pop(model_id)
            logger.info(f"Released port {freed_port} for {model_id}")

    def estimate_model_memory(
        self, model_name: str, quantization: Optional[str] = "fp8"
    ) -> float:
        """
        Estimate model memory usage in GB.

        Args:
            model_name: HuggingFace model name
            quantization: Quantization type (fp8, int4, awq, gptq)

        Returns:
            Estimated memory usage in GB
        """
        # Rough estimates for quantized models on DGX Spark
        # These are conservative estimates including KV cache overhead

        # Extract model size from name (e.g., "7B", "8B", "3B")
        model_lower = model_name.lower()

        if "70b" in model_lower or "72b" in model_lower:
            if quantization == "fp8":
                return 45  # ~45 GB for fp8 quantized 70B
            elif quantization in ["int4", "awq"]:
                return 25  # ~25 GB for int4 quantized 70B
            else:
                return 140  # fp16 not recommended for 70B on DGX Spark

        elif "13b" in model_lower or "14b" in model_lower:
            if quantization == "fp8":
                return 12  # ~12 GB for fp8 quantized 13B
            elif quantization in ["int4", "awq"]:
                return 7  # ~7 GB for int4 quantized 13B
            else:
                return 26  # fp16

        elif "7b" in model_lower or "8b" in model_lower:
            if quantization == "fp8":
                return 8  # ~8 GB for fp8 quantized 7B
            elif quantization in ["int4", "awq"]:
                return 5  # ~5 GB for int4 quantized 7B
            else:
                return 14  # ~14 GB for fp16 (not recommended)

        elif "3b" in model_lower:
            if quantization == "fp8":
                return 4  # ~4 GB for fp8 quantized 3B
            elif quantization in ["int4", "awq"]:
                return 2.5  # ~2.5 GB for int4
            else:
                return 7  # fp16

        else:
            # Default conservative estimate
            logger.warning(f"Could not determine model size for {model_name}, using default 10GB")
            return 10

    def get_resource_status(self) -> dict:
        """
        Get current resource status.

        Returns:
            Dictionary with resource metrics
        """
        current_memory = self.get_unified_memory_usage()
        return {
            "unified_memory_gb": self.total_unified_memory_gb,
            "unified_memory_used_gb": current_memory,
            "unified_memory_limit_gb": self.hard_limit_gb,
            "memory_utilization_percent": (current_memory / self.total_unified_memory_gb) * 100,
            "gpu_temperature_c": self.get_gpu_temperature(),
            "gpu_power_draw_w": self.get_gpu_power_draw(),
            "resident_models_count": len(self.model_memory_usage),
            "max_resident_models": self.max_resident_models,
            "allocated_ports": list(self.allocated_ports.values()),
            "available_ports": (self.port_range[1] - self.port_range[0])
            - len(self.allocated_ports),
        }
