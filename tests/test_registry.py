"""
Tests for model registry.
"""
import pytest
from supervisor.registry import ModelRegistry


def test_generate_id_format():
    """Test model ID generation format."""
    model_id = ModelRegistry.generate_id("Qwen/Qwen2.5-7B-Instruct")

    # Should be: model-name-uuid
    assert "-" in model_id
    parts = model_id.split("-")
    assert len(parts) >= 2

    # First part should be sanitized model name
    assert parts[0] in ["qwen2.5", "7b"]  # Could be different parts

    # UUID part should be 8 characters
    uuid_part = parts[-1]
    assert len(uuid_part) == 8
    assert uuid_part.isalnum()


def test_generate_id_unique():
    """Test that generated IDs are unique."""
    ids = set()
    model_name = "meta-llama/Llama-3-8B"

    for _ in range(100):
        model_id = ModelRegistry.generate_id(model_name)
        ids.add(model_id)

    # All 100 should be unique
    assert len(ids) == 100


def test_generate_id_sanitizes_name():
    """Test that model names are sanitized."""
    test_cases = [
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5"),
        ("meta-llama/Llama-3-8B", "llama"),
        ("mistralai/Mistral-7B-v0.1", "mistral"),
    ]

    for model_name, expected_prefix in test_cases:
        model_id = ModelRegistry.generate_id(model_name)
        # Should contain sanitized name somewhere
        assert expected_prefix in model_id.lower()
