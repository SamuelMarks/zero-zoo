"""Tests for Gemma 4 architectures and weight serialization."""

from __future__ import annotations

from zero_zoo.gemma import (
    get_gemma4_native,
    get_gemma4_zero,
    load_gemma4_weights,
    t_gemma4_edge_cases,
    t_gemma4_inference,
)


def test_gemma() -> None:
    """Test Gemma 4 models, checkpoint loading, inference, and edge cases."""
    assert get_gemma4_native()
    assert get_gemma4_zero()
    assert load_gemma4_weights()
    assert t_gemma4_inference()
    assert t_gemma4_edge_cases()
