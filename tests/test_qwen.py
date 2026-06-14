"""Tests for Qwen."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    pass

from zero_zoo.qwen import (
    get_qwen3_native,
    get_qwen3_zero,
    load_qwen3_weights,
    t_qwen3_edge_cases,
    t_qwen3_inference,
)

T = TypeVar("T")


def test_qwen() -> None:
    """Test Qwen functions."""
    assert get_qwen3_native()
    assert get_qwen3_zero()
    assert load_qwen3_weights()
    assert t_qwen3_inference()
    assert t_qwen3_edge_cases()
