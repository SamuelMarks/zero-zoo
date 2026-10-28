"""Data generation utilities utilizing the compiler and Zero JAX backends."""

from __future__ import annotations

import math

import zero_jax.numpy as jnp
from zero_jax import random

from ._types import VALID_BACKENDS, JaxArray, as_jax, ensure_eager_mode, scalar_float


def generate_random_tensor(
    shape: tuple[int, ...],
    seed: int = 42,
    backend: str = "numpy",
) -> JaxArray:
    """Generate a deterministic random tensor using the specified backend.

    Args:
        shape: Shape of the tensor to generate.
        seed: Random seed for reproducibility.
        backend: The target compiler backend name.

    Returns:
        Generated random tensor as a `JaxArray`.

    Raises:
        ValueError: If `backend` is unsupported or if `len(shape)` exceeds 10.
    """
    ensure_eager_mode()
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend}. Must be one of {VALID_BACKENDS}"
        )

    if len(shape) > 10:
        raise ValueError("Dimensionality exceeds safety limit of 10.")

    key = as_jax(random.PRNGKey(seed))
    key_seed = int(scalar_float(key[1]))

    if len(shape) == 0:
        scalar_val = ((key_seed * 1103515245 + 12345) & 0x7FFFFFFF) / 2147483648.0
        return as_jax(jnp.asarray(scalar_val))

    if any(dim == 0 for dim in shape):
        return as_jax(jnp.zeros(shape))

    total_elements = math.prod(shape)
    state = key_seed & 0x7FFFFFFF
    flat_vals: list[float] = []
    for _ in range(total_elements):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        flat_vals.append(state / 2147483648.0)

    return as_jax(jnp.reshape(jnp.asarray(flat_vals), shape))
