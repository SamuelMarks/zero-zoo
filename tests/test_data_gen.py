"""Tests for data generation utilities."""

from __future__ import annotations

import pytest
import zero_jax.numpy as jnp

from zero_zoo._types import scalar_float
from zero_zoo.data_gen import generate_random_tensor


def test_generate_random_tensor_backends() -> None:
    """Test tensor generation across all valid compiler backends."""
    valid_backends = ["numpy", "jax", "mlx", "cupy", "dusk", "torch"]
    for backend in valid_backends:
        tensor = generate_random_tensor((2, 2), seed=42, backend=backend)
        assert tensor.shape == (2, 2)

    t1 = generate_random_tensor((2, 2), seed=42)
    t2 = generate_random_tensor((2, 2), seed=42)
    assert bool(jnp.allclose(t1, t2))


def test_generate_random_tensor_edge_cases() -> None:
    """Test edge cases in tensor generation (invalid backend, scalar, empty, high-D)."""
    with pytest.raises(ValueError, match="Unsupported backend: invalid"):
        generate_random_tensor((2, 2), backend="invalid")

    scalar = generate_random_tensor(())
    assert scalar.shape == ()
    assert isinstance(scalar_float(scalar), float)

    empty = generate_random_tensor((0, 5))
    assert empty.shape == (0, 5)
    assert empty.size == 0

    with pytest.raises(ValueError, match="Dimensionality exceeds safety limit of 10."):
        generate_random_tensor((2,) * 11)
