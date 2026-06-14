"""Tests for Gemma."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    pass

from zero_zoo.gemma import (
    get_gemma4_native,
    get_gemma4_zero,
    load_gemma4_weights,
    t_gemma4_edge_cases,
    t_gemma4_inference,
)

T = TypeVar("T")


def test_gemma() -> None:
    """Test Gemma functions."""
    assert get_gemma4_native()
    assert get_gemma4_zero()
    assert load_gemma4_weights()
    assert t_gemma4_inference()
    assert t_gemma4_edge_cases()
