"""Tests for the model zoo."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    pass

from zero_zoo import (
    get_flax_cnn,
    get_flax_nanogpt,
    get_keras_cnn,
    get_keras_nanogpt,
    get_pytorch_cnn,
    get_pytorch_nanogpt,
    run_golden_seed_test,
    validate_in_browser,
)

T = TypeVar("T")


def test_cnn() -> None:
    """Test CNN creation."""
    assert get_pytorch_cnn() is not None
    assert get_flax_cnn() is not None
    assert get_keras_cnn() is not None


def test_nanogpt() -> None:
    """Test NanoGPT creation."""
    assert get_pytorch_nanogpt() is not None
    assert get_flax_nanogpt() is not None
    assert get_keras_nanogpt() is not None


def test_golden_seed() -> None:
    """Test golden seed execution."""
    from zero_zoo.mlp import get_native_pytorch_mlp

    res = run_golden_seed_test(get_native_pytorch_mlp, steps=2)
    assert len(res["loss"]) == 2
    assert "weights" in res


def test_headless_ci() -> None:
    """Test headless CI validation."""
    assert validate_in_browser(b"dummy", {}) is True
