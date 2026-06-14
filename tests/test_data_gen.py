"""Tests for data generation utilities."""

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    pass

import pytest

from zero_zoo.data_gen import generate_random_tensor

T = TypeVar("T")


def test_generate_random_tensor_basic() -> None:
    """Test basic tensor generation."""
    t1 = generate_random_tensor((2, 2), seed=42, backend="numpy")
    t2 = generate_random_tensor((2, 2), seed=42, backend="numpy")
    assert np.allclose(t1, t2)
    assert t1.shape == (2, 2)


def test_generate_random_tensor_backends() -> None:
    """Test backend validation."""
    for backend in ["numpy", "jax", "mlx", "cupy", "dusk", "torch"]:
        t = generate_random_tensor((2,), backend=backend)
        assert t is not None

    with pytest.raises(ValueError, match="Unsupported backend"):
        generate_random_tensor((2,), backend="invalid_backend")


def test_generate_random_tensor_edge_cases() -> None:
    """Test edge cases: scalars, empty tensors, massive dimensionality."""
    # Scalar (empty tuple shape)
    scalar = generate_random_tensor(())
    assert isinstance(scalar, float)

    # Empty Tensor (dimension of 0)
    empty = generate_random_tensor((0, 5))
    assert empty.shape == (0, 5)
    assert empty.size == 0

    # Massive dimensionality
    with pytest.raises(ValueError, match="Dimensionality exceeds safety limit of 10."):
        generate_random_tensor((2,) * 11)
