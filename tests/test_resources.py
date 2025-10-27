"""
Tests for resource management.
"""
import pytest
from supervisor.resources import ResourceManager


def test_estimate_model_memory_7b():
    """Test memory estimation for 7B models."""
    rm = ResourceManager()

    # 7B model with different quantizations
    mem_fp8 = rm.estimate_model_memory("Qwen/Qwen2.5-7B-Instruct", "fp8")
    mem_int4 = rm.estimate_model_memory("Qwen/Qwen2.5-7B-Instruct", "int4")
    mem_fp16 = rm.estimate_model_memory("Qwen/Qwen2.5-7B-Instruct", "fp16")

    # fp8 should be ~8GB
    assert 6 <= mem_fp8 <= 10

    # int4 should be ~5GB
    assert 4 <= mem_int4 <= 6

    # fp16 should be ~14GB
    assert 12 <= mem_fp16 <= 16

    # int4 < fp8 < fp16
    assert mem_int4 < mem_fp8 < mem_fp16


def test_estimate_model_memory_70b():
    """Test memory estimation for 70B models."""
    rm = ResourceManager()

    mem_fp8 = rm.estimate_model_memory("meta-llama/Llama-3-70B", "fp8")
    mem_int4 = rm.estimate_model_memory("meta-llama/Llama-3-70B", "int4")

    # 70B fp8 should be ~45GB
    assert 40 <= mem_fp8 <= 50

    # 70B int4 should be ~25GB
    assert 20 <= mem_int4 <= 30

    # int4 should be roughly half of fp8
    ratio = mem_fp8 / mem_int4
    assert 1.5 <= ratio <= 2.0


def test_port_allocation():
    """Test port allocation within range."""
    rm = ResourceManager()

    # Allocate a port
    port = rm.allocate_port()

    # Should be within configured range
    assert rm.port_range[0] <= port < rm.port_range[1]


def test_port_allocation_tracking():
    """Test port allocation tracking."""
    rm = ResourceManager()

    # Allocate multiple models
    rm.allocated_ports["model1"] = rm.allocate_port()
    rm.allocated_ports["model2"] = rm.allocate_port()
    rm.allocated_ports["model3"] = rm.allocate_port()

    # All should be different
    ports = list(rm.allocated_ports.values())
    assert len(ports) == len(set(ports))  # All unique


def test_can_allocate_model_under_limit():
    """Test allocation check when under limits."""
    rm = ResourceManager()

    # 10GB model should be allocatable
    can_allocate = rm.can_allocate_model(10.0)
    assert can_allocate is True


def test_can_allocate_model_over_hard_limit():
    """Test allocation check when over hard limit."""
    rm = ResourceManager()

    # 120GB model exceeds hard limit (110GB)
    can_allocate = rm.can_allocate_model(120.0)
    assert can_allocate is False


def test_resource_status_format():
    """Test resource status dictionary format."""
    rm = ResourceManager()
    status = rm.get_resource_status()

    # Check required keys
    required_keys = [
        "unified_memory_gb",
        "unified_memory_used_gb",
        "unified_memory_limit_gb",
        "resident_models_count",
        "max_resident_models",
        "available_ports",
    ]

    for key in required_keys:
        assert key in status

    # Check value types
    assert isinstance(status["unified_memory_gb"], (int, float))
    assert isinstance(status["resident_models_count"], int)
    assert isinstance(status["available_ports"], int)

    # Check reasonable ranges
    assert 0 <= status["unified_memory_used_gb"] <= status["unified_memory_gb"]
    assert 0 <= status["resident_models_count"] <= status["max_resident_models"]
