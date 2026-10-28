"""Tests for MLP models across all `zero-*` frameworks."""

from __future__ import annotations

import pytest
import zero_jax.numpy as jnp
import zero_keras
import zero_mlx as zmx
import zero_tensorflow as ztf
import zero_torch
from zero_flax import nnx
from zero_jax import random
from zero_pax.types import Tensor as PaxTensor

from zero_zoo._types import as_jax, ensure_eager_mode
from zero_zoo.matrix_runner import check_equivalence
from zero_zoo.mlp import (
    ZeroPaxMLP,
    get_flax_mlp,
    get_keras_mlp,
    get_native_flax_linen_mlp,
    get_native_flax_nnx_mlp,
    get_native_jax_mlp,
    get_native_keras_mlp,
    get_native_mlx_mlp,
    get_native_pax_mlp,
    get_native_pytorch_mlp,
    get_pytorch_mlp,
    get_zero_flax_linen_mlp,
    get_zero_flax_nnx_mlp,
    get_zero_jax_mlp,
    get_zero_keras_mlp,
    get_zero_mlx_mlp,
    get_zero_pax_mlp,
    get_zero_pytorch_mlp,
    get_zero_tensorflow_mlp,
)


@pytest.mark.torch
def test_pytorch_mlp() -> None:
    """Verify Zero PyTorch and reference PyTorch MLP forward passes."""
    ensure_eager_mode()
    pt_native = get_native_pytorch_mlp()
    pt_zero = get_zero_pytorch_mlp()
    pt_alias = get_pytorch_mlp()

    x = zero_torch.ones((4, 2))
    out_native = pt_native(x)
    out_zero = pt_zero(x)
    out_alias = pt_alias(x)

    assert out_native.shape == (4, 2)
    assert out_zero.shape == (4, 2)
    assert out_alias.shape == (4, 2)
    assert check_equivalence(out_native, out_native)


@pytest.mark.tf
def test_keras_and_tensorflow_mlp() -> None:
    """Verify Zero Keras and Zero TensorFlow MLP forward passes."""
    k_native = get_native_keras_mlp()
    k_zero = get_zero_keras_mlp()
    k_alias = get_keras_mlp()

    x_keras = zero_keras.ops.ones((4, 2))
    out_k_native = k_native(x_keras)
    out_k_zero = k_zero(x_keras)
    out_k_alias = k_alias(x_keras)

    assert zero_keras.ops.shape(out_k_native) == (4, 2)
    assert zero_keras.ops.shape(out_k_zero) == (4, 2)
    assert zero_keras.ops.shape(out_k_alias) == (4, 2)

    tf_mlp = get_zero_tensorflow_mlp()
    x_tf = ztf.ones((4, 2))
    out_tf = tf_mlp(x_tf)
    assert out_tf.shape == (4, 2)


@pytest.mark.jax
def test_jax_mlp() -> None:
    """Verify Zero JAX and reference JAX MLP training and eval forward passes."""
    init_native, fwd_native = get_native_jax_mlp()
    init_zero, fwd_zero = get_zero_jax_mlp()

    key = as_jax(random.PRNGKey(0))
    params_native = init_native(key)
    params_zero = init_zero(key)

    x = as_jax(jnp.ones((4, 2)))
    out_train_native = fwd_native(params_native, x, is_training=True)
    out_eval_native = fwd_native(params_native, x, is_training=False)
    out_train_zero = fwd_zero(params_zero, x, is_training=True)
    out_eval_zero = fwd_zero(params_zero, x, is_training=False)

    assert out_train_native.shape == (4, 2)
    assert out_eval_native.shape == (4, 2)
    assert check_equivalence(out_train_native, out_train_zero)
    assert check_equivalence(out_eval_native, out_eval_zero)


@pytest.mark.jax
def test_flax_mlp() -> None:
    """Verify Zero Flax NNX and Linen-adapter MLP forward passes."""
    flax_native = get_native_flax_linen_mlp()
    flax_zero = get_zero_flax_linen_mlp()
    flax_alias = get_flax_mlp()

    x = as_jax(jnp.ones((4, 2)))
    key = as_jax(random.PRNGKey(0))
    params = flax_native.init(key, x)
    out_apply = flax_native.apply(params, x, use_running_average=True)
    out_direct = flax_zero(x, use_running_average=False)
    out_alias = flax_alias(x, use_running_average=True)

    assert out_apply.shape == (4, 2)
    assert out_direct.shape == (4, 2)
    assert out_alias.shape == (4, 2)

    nnx_native_cls = get_native_flax_nnx_mlp()
    nnx_zero_cls = get_zero_flax_nnx_mlp()
    nnx_native = nnx_native_cls(rngs=nnx.Rngs(0))
    nnx_zero = nnx_zero_cls(rngs=nnx.Rngs(0))

    out_nnx_native = nnx_native(x, use_running_average=False)
    out_nnx_zero = nnx_zero(x, use_running_average=False)
    assert out_nnx_native.shape == (4, 2)
    assert check_equivalence(out_nnx_native, out_nnx_zero)


@pytest.mark.mlx
def test_mlx_mlp() -> None:
    """Verify Zero MLX and reference MLX MLP forward passes."""
    mlx_native = get_native_mlx_mlp()
    mlx_zero = get_zero_mlx_mlp()

    x = zmx.ones((4, 2))
    out_native = mlx_native(x)
    out_zero = mlx_zero(x)

    assert out_native.shape == (4, 2)
    assert out_zero.shape == (4, 2)


@pytest.mark.pax
def test_pax_mlp() -> None:
    """Verify Zero Pax and reference Pax MLP forward passes including lazy setup."""
    pax_native = get_native_pax_mlp()
    pax_zero = get_zero_pax_mlp()
    pax_lazy = ZeroPaxMLP()

    x = PaxTensor([1.0] * 8, (4, 2))
    out_native = pax_native(x)
    out_zero = pax_zero(x)
    out_lazy = pax_lazy(x)

    assert out_native.shape == (4, 2)
    assert out_zero.shape == (4, 2)
    assert out_lazy.shape == (4, 2)
    assert check_equivalence(out_native, out_zero)
