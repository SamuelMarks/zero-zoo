"""Tests for Qwen 3 architectures and weight serialization."""

from __future__ import annotations

from zero_zoo.qwen import (
    get_qwen3_native,
    get_qwen3_zero,
    load_qwen3_weights,
    t_qwen3_edge_cases,
    t_qwen3_inference,
)


def test_qwen() -> None:
    """Test Qwen 3 native/zero models, checkpoint loading, inference, and edge cases."""
    assert get_qwen3_native()
    assert get_qwen3_zero()
    assert load_qwen3_weights()
    assert t_qwen3_inference()
    assert t_qwen3_edge_cases()
