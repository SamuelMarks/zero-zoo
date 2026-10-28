"""Matrix testing harness and equivalence checking utilities."""

from __future__ import annotations

from collections.abc import Sequence

import zero_jax.numpy as jnp

from ._types import JaxArray, ZooTensor, ensure_eager_mode, to_jax_array
from .data_gen import generate_random_tensor


def check_equivalence(
    output_a: ZooTensor | Sequence[float] | Sequence[Sequence[float]],
    output_b: ZooTensor | Sequence[float] | Sequence[Sequence[float]],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """Check numerical equivalence between two tensors from any ecosystem backend.

    Handles `NaN` and `Inf` propagation parity as well as shape mismatches.

    Args:
        output_a: First tensor or sequence to compare.
        output_b: Second tensor or sequence to compare.
        rtol: Relative tolerance for floating-point comparison.
        atol: Absolute tolerance for floating-point comparison.

    Returns:
        `True` if tensors have identical shapes, matching `NaN`/`Inf` positions,
        and numerically close finite elements; `False` otherwise.
    """
    ensure_eager_mode()
    arr_a = to_jax_array(output_a)
    arr_b = to_jax_array(output_b)

    if arr_a.shape != arr_b.shape:
        return False

    nan_a = jnp.isnan(arr_a)
    nan_b = jnp.isnan(arr_b)
    if not bool(jnp.array_equal(nan_a, nan_b)):
        return False

    inf_a = jnp.isinf(arr_a)
    inf_b = jnp.isinf(arr_b)
    if not bool(jnp.array_equal(inf_a, inf_b)):
        return False

    invalid_mask = jnp.logical_or(nan_a, inf_a)
    if bool(jnp.all(invalid_mask)):
        return True

    clean_a = jnp.where(invalid_mask, jnp.zeros_like(arr_a), arr_a)
    clean_b = jnp.where(invalid_mask, jnp.zeros_like(arr_b), arr_b)
    return bool(jnp.allclose(clean_a, clean_b, rtol=rtol, atol=atol))


def simulate_stateful_run(seed: int = 42) -> JaxArray:
    """Simulate a stateful model run to verify random state isolation.

    Args:
        seed: Random seed for the isolated PRNG key.

    Returns:
        A deterministically generated `(5, 5)` `JaxArray` state matrix.
    """
    return generate_random_tensor((5, 5), seed=seed)
