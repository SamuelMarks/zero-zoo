"""Tests for NanoGPT models across all `zero-*` frameworks."""

from __future__ import annotations

import pytest
import zero_chex
import zero_jax.numpy as jnp
import zero_keras
import zero_mlx as zmx
import zero_torch
from zero_flax import nnx
from zero_jax import random
from zero_pax.types import Tensor as PaxTensor

from zero_zoo._types import as_jax
from zero_zoo.matrix_runner import check_equivalence
from zero_zoo.nanogpt import (
    ZeroPaxNanoGPT,
    get_flax_nanogpt,
    get_keras_nanogpt,
    get_native_flax_linen_nanogpt,
    get_native_flax_nnx_nanogpt,
    get_native_jax_nanogpt,
    get_native_keras_nanogpt,
    get_native_mlx_nanogpt,
    get_native_pax_nanogpt,
    get_native_pytorch_nanogpt,
    get_pytorch_nanogpt,
    get_zero_flax_linen_nanogpt,
    get_zero_flax_nnx_nanogpt,
    get_zero_jax_nanogpt,
    get_zero_keras_nanogpt,
    get_zero_mlx_nanogpt,
    get_zero_pax_nanogpt,
    get_zero_pytorch_nanogpt,
)


@pytest.mark.torch
def test_pytorch_nanogpt() -> None:
    """Verify Zero PyTorch and reference PyTorch NanoGPT forward passes."""
    pt_native = get_native_pytorch_nanogpt()
    pt_zero = get_zero_pytorch_nanogpt()
    pt_alias = get_pytorch_nanogpt()

    x = zero_torch.zeros((2, 5), dtype=zero_torch.int32)
    out_native = pt_native(x)
    out_zero = pt_zero(x)
    out_alias = pt_alias(x)

    assert out_native.shape == (2, 5, 10)
    assert out_zero.shape == (2, 5, 10)
    assert out_alias.shape == (2, 5, 10)


@pytest.mark.tf
def test_keras_nanogpt() -> None:
    """Verify Zero Keras and reference Keras NanoGPT forward passes."""
    k_native = get_native_keras_nanogpt()
    k_zero = get_zero_keras_nanogpt()
    k_alias = get_keras_nanogpt()

    x = zero_keras.ops.zeros((2, 5), dtype="int32")
    out_native = k_native(x)
    out_zero = k_zero(x)
    out_alias = k_alias(x)

    assert zero_keras.ops.shape(out_native) == (2, 5, 10)
    assert zero_keras.ops.shape(out_zero) == (2, 5, 10)
    assert zero_keras.ops.shape(out_alias) == (2, 5, 10)


@pytest.mark.jax
def test_jax_nanogpt() -> None:
    """Verify Zero JAX and reference JAX NanoGPT initialization and forward passes."""
    init_native, fwd_native = get_native_jax_nanogpt()
    init_zero, fwd_zero = get_zero_jax_nanogpt()

    key = as_jax(random.PRNGKey(0))
    params_native = init_native(key)
    params_zero = init_zero(key)

    x = as_jax(jnp.zeros((2, 5), dtype=jnp.int32))
    out_train = fwd_native(params_native, x, is_training=True)
    out_eval = fwd_zero(params_zero, x, is_training=False)

    assert out_train.shape == (2, 5, 10)
    assert out_eval.shape == (2, 5, 10)
    assert check_equivalence(out_train, out_eval)


@pytest.mark.jax
def test_flax_nanogpt() -> None:
    """Verify Zero Flax NNX and Linen-adapter NanoGPT forward passes."""
    flax_native = get_native_flax_linen_nanogpt()
    flax_zero = get_zero_flax_linen_nanogpt()
    flax_alias = get_flax_nanogpt()

    x = as_jax(jnp.zeros((2, 5), dtype=jnp.int32))
    key = as_jax(random.PRNGKey(0))
    params = flax_native.init(key, x)
    out_apply = flax_native.apply(params, x, use_running_average=True)
    out_direct = flax_zero(x, use_running_average=False)
    out_alias = flax_alias(x, use_running_average=True)

    assert out_apply.shape == (2, 5, 10)
    assert out_direct.shape == (2, 5, 10)
    assert out_alias.shape == (2, 5, 10)

    nnx_native_cls = get_native_flax_nnx_nanogpt()
    nnx_zero_cls = get_zero_flax_nnx_nanogpt()
    nnx_native = nnx_native_cls(rngs=nnx.Rngs(0))
    nnx_zero = nnx_zero_cls(rngs=nnx.Rngs(0))

    out_nnx_native = nnx_native(x)
    out_nnx_zero = nnx_zero(x)
    assert out_nnx_native.shape == (2, 5, 10)
    assert check_equivalence(out_nnx_native, out_nnx_zero)


@pytest.mark.mlx
def test_mlx_nanogpt() -> None:
    """Verify Zero MLX and reference MLX NanoGPT forward passes."""
    mlx_native = get_native_mlx_nanogpt()
    mlx_zero = get_zero_mlx_nanogpt()

    x = zmx.zeros((2, 5), dtype=zmx.int32)
    out_native = mlx_native(x)
    out_zero = mlx_zero(x)

    assert out_native.shape == (2, 5, 10)
    assert out_zero.shape == (2, 5, 10)


@pytest.mark.pax
def test_pax_nanogpt() -> None:
    """Verify Zero Pax and reference Pax NanoGPT forward passes including lazy setup."""
    pax_native = get_native_pax_nanogpt()
    pax_zero = get_zero_pax_nanogpt()
    pax_lazy = ZeroPaxNanoGPT()

    x = PaxTensor([0.0] * 10, (2, 5))
    out_native = pax_native(x)
    out_zero = pax_zero(x)
    out_lazy = pax_lazy(x)

    assert out_native.shape == (2, 5, 10)
    assert out_zero.shape == (2, 5, 10)
    assert out_lazy.shape == (2, 5, 10)
    assert check_equivalence(out_native, out_zero)


def test_nanogpt_edge_cases() -> None:
    """Verify NanoGPT edge cases: `seq_len=1` and finite outputs."""
    nnx_cls = get_zero_flax_nnx_nanogpt()
    model = nnx_cls(rngs=nnx.Rngs(0))
    single_token = as_jax(jnp.zeros((1, 1), dtype=jnp.int32))
    out_single = model(single_token)
    zero_chex.assert_tree_all_finite(out_single)
    assert out_single.shape == (1, 1, 10)
