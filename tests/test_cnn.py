"""Tests for CNN models across all `zero-*` frameworks."""

from __future__ import annotations

import pytest
import zero_jax.numpy as jnp
import zero_keras
import zero_mlx as zmx
import zero_tensorflow as ztf
import zero_torch
from zero_flax import nnx
from zero_jax import random
from zero_pax.praxis.layers import Conv2D as PaxConv2D
from zero_pax.types import Tensor as PaxTensor

from zero_zoo._types import as_jax
from zero_zoo.cnn import (
    ZeroPaxCNN,
    get_flax_cnn,
    get_keras_cnn,
    get_native_flax_linen_cnn,
    get_native_flax_nnx_cnn,
    get_native_jax_cnn,
    get_native_keras_cnn,
    get_native_mlx_cnn,
    get_native_pax_cnn,
    get_native_pytorch_cnn,
    get_pytorch_cnn,
    get_zero_flax_linen_cnn,
    get_zero_flax_nnx_cnn,
    get_zero_jax_cnn,
    get_zero_keras_cnn,
    get_zero_mlx_cnn,
    get_zero_pax_cnn,
    get_zero_pytorch_cnn,
    get_zero_tensorflow_cnn,
)
from zero_zoo.matrix_runner import check_equivalence


@pytest.mark.torch
def test_pytorch_cnn() -> None:
    """Verify Zero PyTorch and reference PyTorch CNN forward passes."""
    pt_native = get_native_pytorch_cnn()
    pt_zero = get_zero_pytorch_cnn()
    pt_alias = get_pytorch_cnn()

    x = zero_torch.ones((4, 1, 4, 4))
    out_native = pt_native(x)
    out_zero = pt_zero(x)
    out_alias = pt_alias(x)

    assert out_native.shape == (4, 2)
    assert out_zero.shape == (4, 2)
    assert out_alias.shape == (4, 2)


@pytest.mark.tf
def test_keras_and_tensorflow_cnn() -> None:
    """Verify Zero Keras and Zero TensorFlow CNN forward passes."""
    k_native = get_native_keras_cnn()
    k_zero = get_zero_keras_cnn()
    k_alias = get_keras_cnn()

    x_keras = zero_keras.ops.ones((4, 4, 4, 1))
    out_k_native = k_native(x_keras)
    out_k_zero = k_zero(x_keras)
    out_k_alias = k_alias(x_keras)

    assert zero_keras.ops.shape(out_k_native) == (4, 2)
    assert zero_keras.ops.shape(out_k_zero) == (4, 2)
    assert zero_keras.ops.shape(out_k_alias) == (4, 2)

    tf_cnn = get_zero_tensorflow_cnn()
    x_tf = ztf.ones((4, 4, 4, 1))
    out_tf = tf_cnn(x_tf)
    assert out_tf.shape == (4, 2)


@pytest.mark.jax
def test_jax_cnn() -> None:
    """Verify Zero JAX and reference JAX CNN training and eval forward passes."""
    init_native, fwd_native = get_native_jax_cnn()
    init_zero, fwd_zero = get_zero_jax_cnn()

    key = as_jax(random.PRNGKey(0))
    params_native = init_native(key)
    params_zero = init_zero(key)

    x = as_jax(jnp.ones((4, 4, 4, 1)))
    out_train_native = fwd_native(params_native, x, is_training=True)
    out_eval_native = fwd_native(params_native, x, is_training=False)
    out_train_zero = fwd_zero(params_zero, x, is_training=True)
    out_eval_zero = fwd_zero(params_zero, x, is_training=False)

    assert out_train_native.shape == (4, 2)
    assert out_eval_native.shape == (4, 2)
    assert check_equivalence(out_train_native, out_train_zero)
    assert check_equivalence(out_eval_native, out_eval_zero)


@pytest.mark.jax
def test_flax_cnn() -> None:
    """Verify Zero Flax NNX and Linen-adapter CNN forward passes."""
    flax_native = get_native_flax_linen_cnn()
    flax_zero = get_zero_flax_linen_cnn()
    flax_alias = get_flax_cnn()

    x = as_jax(jnp.ones((4, 4, 4, 1)))
    key = as_jax(random.PRNGKey(0))
    params = flax_native.init(key, x)
    out_apply = flax_native.apply(params, x, use_running_average=True)
    out_direct = flax_zero(x, use_running_average=False)
    out_alias = flax_alias(x, use_running_average=True)

    assert out_apply.shape == (4, 2)
    assert out_direct.shape == (4, 2)
    assert out_alias.shape == (4, 2)

    nnx_native_cls = get_native_flax_nnx_cnn()
    nnx_zero_cls = get_zero_flax_nnx_cnn()
    nnx_native = nnx_native_cls(rngs=nnx.Rngs(0))
    nnx_zero = nnx_zero_cls(rngs=nnx.Rngs(0))

    out_nnx_native = nnx_native(x, use_running_average=False)
    out_nnx_zero = nnx_zero(x, use_running_average=False)
    assert out_nnx_native.shape == (4, 2)
    assert check_equivalence(out_nnx_native, out_nnx_zero)


@pytest.mark.mlx
def test_mlx_cnn() -> None:
    """Verify Zero MLX and reference MLX CNN forward passes."""
    mlx_native = get_native_mlx_cnn()
    mlx_zero = get_zero_mlx_cnn()

    x = zmx.ones((4, 4, 4, 1))
    out_native = mlx_native(x)
    out_zero = mlx_zero(x)

    assert out_native.shape == (4, 2)
    assert out_zero.shape == (4, 2)


@pytest.mark.pax
def test_pax_cnn() -> None:
    """Verify Zero Pax and reference Pax CNN forward passes including lazy setup."""
    pax_native = get_native_pax_cnn()
    pax_zero = get_zero_pax_cnn()
    pax_lazy = ZeroPaxCNN()

    x = PaxTensor([1.0] * 64, (4, 4, 4, 1))
    out_native = pax_native(x)
    out_zero = pax_zero(x)
    out_lazy = pax_lazy(x)

    assert out_native.shape == (4, 2)
    assert out_zero.shape == (4, 2)
    assert out_lazy.shape == (4, 2)
    assert check_equivalence(out_native, out_zero)


def test_cnn_edge_cases() -> None:
    """Verify CNN edge cases: `1x1`, strided, and transposed convolutions."""
    rngs = nnx.Rngs(0)
    conv_1x1 = nnx.Conv(1, 3, kernel_size=(1, 1), rngs=rngs)
    conv_strided = nnx.Conv(1, 2, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
    conv_transpose = nnx.ConvTranspose(
        2,
        1,
        kernel_size=(2, 2),
        strides=(2, 2),
        rngs=rngs,
    )

    x = as_jax(jnp.ones((2, 4, 4, 1)))
    out_1x1 = conv_1x1(x)
    out_strided = conv_strided(x)
    out_transposed = conv_transpose(out_strided)

    assert out_1x1.shape == (2, 4, 4, 3)
    assert out_strided.shape == (2, 2, 2, 2)
    assert out_transposed.shape == (2, 4, 4, 1)

    pax_valid_conv = PaxConv2D(
        filter_shape=(1, 1, 1, 2),
        filter_stride=(1, 1),
        padding="VALID",
    )
    pax_out = pax_valid_conv(PaxTensor([1.0] * 32, (2, 4, 4, 1)))
    assert pax_out.shape == (2, 4, 4, 2)
