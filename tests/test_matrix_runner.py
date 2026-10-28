"""Tests for matrix testing harness and tensor equivalence utilities."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_mlx as zmx
import zero_tensorflow as ztf
import zero_torch
from zero_chex.types import ZeroArray
from zero_pax.types import Tensor as PaxTensor

from zero_zoo._types import (
    as_jax,
    as_restored_jax,
    to_compiler_tensor,
    to_jax_array,
)
from zero_zoo.matrix_runner import check_equivalence, simulate_stateful_run


def test_to_jax_array_all_ecosystem_types() -> None:
    """Verify `to_jax_array` and `as_restored_jax` across all tensor types."""
    jax_in = as_jax(jnp.ones((2, 2)))
    sw_in = to_compiler_tensor(jax_in)
    arr_jax = to_jax_array(jax_in)
    arr_torch = to_jax_array(zero_torch.ones((2, 2)))
    arr_keras = to_jax_array(sw_in)
    arr_tf = to_jax_array(ztf.ones((2, 2)))
    arr_mlx = to_jax_array(zmx.ones((2, 2)))
    arr_pax = to_jax_array(PaxTensor([1.0, 1.0, 1.0, 1.0], (2, 2)))
    arr_chex = to_jax_array(ZeroArray([[1.0, 1.0], [1.0, 1.0]]))
    arr_seq = to_jax_array([[1.0, 1.0], [1.0, 1.0]])

    for converted in (
        arr_jax,
        arr_torch,
        arr_keras,
        arr_tf,
        arr_mlx,
        arr_pax,
        arr_chex,
        arr_seq,
    ):
        assert converted.shape == (2, 2)

    assert as_restored_jax(jax_in).shape == (2, 2)
    assert as_restored_jax(sw_in).shape == (2, 2)
    assert as_restored_jax(0.0).shape == ()


def test_check_equivalence() -> None:
    """Test numerical equivalence checks across normal, NaN, Inf, and shapes."""
    a = as_jax(jnp.asarray([1.0, 2.0, 3.0]))
    b = as_jax(jnp.asarray([1.0, 2.0, 3.0 + 1e-6]))
    c = as_jax(jnp.asarray([1.0, 2.0, 4.0]))

    assert check_equivalence(a, b) is True
    assert check_equivalence(a, c) is False

    d = as_jax(jnp.asarray([[1.0, 2.0, 3.0]]))
    assert check_equivalence(a, d) is False

    nan_a = as_jax(jnp.asarray([1.0, float("nan"), 3.0]))
    nan_b = as_jax(jnp.asarray([1.0, float("nan"), 3.0]))
    nan_c = as_jax(jnp.asarray([float("nan"), 1.0, 3.0]))
    assert check_equivalence(nan_a, nan_b) is True
    assert check_equivalence(nan_a, nan_c) is False
    assert check_equivalence(a, nan_a) is False

    inf_a = as_jax(jnp.asarray([1.0, float("inf"), 3.0]))
    inf_b = as_jax(jnp.asarray([1.0, float("inf"), 3.0]))
    inf_c = as_jax(jnp.asarray([float("inf"), 1.0, 3.0]))
    assert check_equivalence(inf_a, inf_b) is True
    assert check_equivalence(inf_a, inf_c) is False
    assert check_equivalence(a, inf_a) is False

    all_nan_a = as_jax(jnp.asarray([float("nan"), float("nan")]))
    all_nan_b = as_jax(jnp.asarray([float("nan"), float("nan")]))
    assert check_equivalence(all_nan_a, all_nan_b) is True

    all_inf_a = as_jax(jnp.asarray([float("inf"), float("inf")]))
    all_inf_b = as_jax(jnp.asarray([float("inf"), float("inf")]))
    assert check_equivalence(all_inf_a, all_inf_b) is True


def test_simulate_stateful_run() -> None:
    """Test that stateful simulation guarantees random state isolation."""
    run1 = simulate_stateful_run(42)
    run2 = simulate_stateful_run(42)
    run3 = simulate_stateful_run(99)

    assert bool(jnp.allclose(run1, run2))
    assert not bool(jnp.allclose(run1, run3))
