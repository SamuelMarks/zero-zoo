"""Tests for MLP models."""

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    import typing
from unittest import mock

import pytest

from zero_zoo.mlp import (
    get_native_flax_linen_mlp,
    get_native_flax_nnx_mlp,
    get_native_jax_mlp,
    get_native_keras_mlp,
    get_native_mlx_mlp,
    get_native_pax_mlp,
    get_native_pytorch_mlp,
    get_zero_flax_linen_mlp,
    get_zero_flax_nnx_mlp,
    get_zero_jax_mlp,
    get_zero_keras_mlp,
    get_zero_mlx_mlp,
    get_zero_pax_mlp,
    get_zero_pytorch_mlp,
)

T = TypeVar("T")


@pytest.mark.torch
def test_pytorch_mlps() -> None:
    """Test PyTorch MLPs."""
    import torch

    native_mlp = get_native_pytorch_mlp()
    x = torch.randn(4, 2)
    assert native_mlp(x).shape == (4, 2)

    x_single = torch.randn(1, 2)
    native_mlp.eval()
    assert native_mlp(x_single).shape == (1, 2)

    x_nan = torch.tensor([[float("nan"), 1.0]])
    assert torch.isnan(native_mlp(x_nan)).any()
    native_mlp.bfloat16()
    assert native_mlp(x.bfloat16()).dtype == torch.bfloat16

    with (
        mock.patch("zero_torch.nn.Linear") as mock_lin,
        mock.patch("zero_torch.nn.BatchNorm1d") as mock_bn,
        mock.patch("zero_torch.nn.ReLU") as mock_relu,
    ):
        mock_lin.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.return_value.side_effect = lambda x: x
        zero_mlp = get_zero_pytorch_mlp()
        assert zero_mlp(x) is not None


@pytest.mark.tf
def test_keras_mlps() -> None:
    """Test Keras MLPs."""
    import tensorflow as tf

    native_mlp = get_native_keras_mlp()
    x = tf.random.normal((4, 2))
    assert native_mlp(x).shape == (4, 2)

    x_single = tf.random.normal((1, 2))
    assert native_mlp(x_single, training=False).shape == (1, 2)
    x_nan = tf.constant([[float("nan"), 1.0]])
    assert tf.math.is_nan(native_mlp(x_nan, training=False)[0, 0])

    with (
        mock.patch("zero_keras.Input"),
        mock.patch("zero_keras.layers.Dense") as mock_dense,
        mock.patch("zero_keras.layers.BatchNormalization") as mock_bn,
        mock.patch("zero_keras.layers.ReLU") as mock_relu,
        mock.patch("zero_keras.Model") as mock_model,
    ):
        mock_dense.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.return_value.side_effect = lambda x: x
        mock_model.return_value = mock.MagicMock()
        zero_mlp = get_zero_keras_mlp()
        assert zero_mlp is not None


@pytest.mark.jax
def test_jax_mlps() -> None:
    """Test pure JAX MLPs."""
    import jax
    import jax.numpy as jnp

    n_init, n_fwd = get_native_jax_mlp()
    key = jax.random.PRNGKey(0)
    n_params = n_init(key)

    x = jax.random.normal(key, (4, 2))
    assert n_fwd(n_params, x).shape == (4, 2)
    x_single = jax.random.normal(key, (1, 2))
    assert n_fwd(n_params, x_single, is_training=False).shape == (1, 2)
    x_nan = jnp.array([[jnp.nan, 1.0]])
    assert jnp.isnan(n_fwd(n_params, x_nan, is_training=False)).any()

    with (
        mock.patch("zero_jax.random.split") as m_split,
        mock.patch("zero_jax.random.normal") as m_norm,
        mock.patch("zero_jax.numpy.zeros") as m_zeros,
        mock.patch("zero_jax.numpy.ones") as m_ones,
        mock.patch("zero_jax.numpy.dot") as m_dot,
        mock.patch("zero_jax.numpy.mean") as m_mean,
        mock.patch("zero_jax.numpy.var") as m_var,
        mock.patch("zero_jax.numpy.sqrt") as m_sqrt,
        mock.patch("zero_jax.nn.relu") as m_relu,
    ):
        m_split.return_value = (1, 2)
        m_norm.return_value = 1
        m_zeros.return_value = 0
        m_ones.return_value = 1
        m_dot.return_value = x
        m_mean.return_value = 0
        m_var.return_value = 1
        m_sqrt.return_value = 1
        m_relu.return_value = x

        z_init, z_fwd = get_zero_jax_mlp()
        z_params = z_init(key)
        assert z_fwd(z_params, x) is not None
        assert z_fwd(z_params, x_single, is_training=False) is not None


@pytest.mark.jax
def test_flax_mlps() -> None:
    """Test Flax Linen and NNX MLPs."""
    import jax
    import jax.numpy as jnp

    # Linen
    n_linen = get_native_flax_linen_mlp()
    key = jax.random.PRNGKey(0)
    x = jnp.ones((4, 2))
    n_params = n_linen.init(key, x)
    assert n_linen.apply(n_params, x, use_running_average=True).shape == (4, 2)  # type: ignore[union-attr]

    with mock.patch("zero_flax.nnx.Module"), mock.patch("zero_flax.nnx.Param"):
        # Test fallback logic.
        z_linen = get_zero_flax_linen_mlp()
        z_linen(x)
    # NNX
    from flax import nnx

    n_nnx = get_native_flax_nnx_mlp()
    n_model = n_nnx(rngs=nnx.Rngs(0))  # type: ignore[operator]
    assert n_model(x).shape == (4, 2)

    with (
        mock.patch("zero_flax.nnx.Linear") as mock_lin,
        mock.patch("zero_flax.nnx.BatchNorm") as mock_bn,
        mock.patch("zero_jax.nn.relu") as mock_relu,
    ):
        mock_lin.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.side_effect = lambda x: x
        z_nnx_mlp = get_zero_flax_nnx_mlp()
        z_model = z_nnx_mlp(rngs=None)  # type: ignore[operator]
        assert z_model(x) is not None


@pytest.mark.mlx
def test_mlx_mlps() -> None:
    """Test MLX MLPs."""
    import mlx.core as mx

    native_mlp = get_native_mlx_mlp()
    x = mx.random.normal((4, 2))
    assert native_mlp(x).shape == (4, 2)  # type: ignore[operator]

    with (
        mock.patch("zero_mlx.nn.Linear") as mock_lin,
        mock.patch("zero_mlx.nn.BatchNorm") as mock_bn,
        mock.patch("zero_mlx.nn.relu") as mock_relu,
    ):
        mock_lin.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.side_effect = lambda x: x

        zero_mlp = get_zero_mlx_mlp()
        assert zero_mlp(x) is not None  # type: ignore[operator]


@pytest.mark.pax
def test_pax_mlps() -> None:
    """Test Pax MLPs."""
    native_mlp = get_native_pax_mlp()
    x = np.ones((4, 2))
    assert native_mlp(x).shape == (4, 2)

    with (
        mock.patch("zero_pax.Linear") as mock_lin,
        mock.patch("zero_pax.BatchNorm") as mock_bn,
        mock.patch("zero_pax.ReLU") as mock_relu,
    ):
        mock_lin.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.return_value.side_effect = lambda x: x

        zero_mlp = get_zero_pax_mlp()
        assert zero_mlp(x) is not None


@pytest.mark.jax
def test_flax_mlps_linen_fallback() -> None:
    """Test flax linen mock paths.

    Returns:
        None
    """
    from zero_zoo.mlp import get_zero_flax_linen_mlp

    # Test the try path (pretend it's installed)
    class MockNN:
        """Mock flax.linen module for testing."""

        class Module:
            """Mock Module class."""

            pass

        @staticmethod
        def compact(fn: T) -> T:
            """Mock compact decorator.

            Args:
                fn: The function to decorate.

            Returns:
                The unmodified function.
            """
            return fn

        @staticmethod
        def Dense(x: T) -> "typing.Callable[[T], T]":
            """Mock Dense layer.

            Args:
                x: The layer features/input.

            Returns:
                A callable returning its input.
            """
            return lambda y: y

        @staticmethod
        def BatchNorm(use_running_average: bool = False) -> "typing.Callable[[T], T]":
            """Mock BatchNorm layer.

            Args:
                use_running_average: Whether to use running average.

            Returns:
                A callable returning its input.
            """
            return lambda y: y

        @staticmethod
        def relu(x: T) -> T:
            """Mock ReLU activation.

            Args:
                x: The input tensor.

            Returns:
                The unmodified input tensor.
            """
            return x

    with mock.patch.dict("sys.modules", {"zero_flax.linen": MockNN}):
        linen_mock = get_zero_flax_linen_mlp()
        # Call the __call__ method
        linen_mock(np.ones((4, 2)))

    # Test the except path (already installed/not installed)
    with mock.patch.dict("sys.modules", {"zero_flax.linen": None}):
        fallback = get_zero_flax_linen_mlp()
        fallback(np.ones((4, 2)))


@pytest.mark.pax
def test_pax_mlps_fallback() -> None:
    """Test Pax mlps mock paths."""
    from zero_zoo.mlp import get_native_pax_mlp

    # Test the try path
    class MockBaseLayer:
        """Mock BaseLayer for Pax testing."""

        def create_child(self, name: str, hparams: object) -> None:
            """Mock create_child method.

            Args:
                name: The name of the child.
                hparams: The hyperparameters for the child.
            """
            setattr(self, name, lambda x: x)

    class MockLayers:
        """Mock layers module for Pax testing."""

        class Linear:
            """Mock Linear layer."""

            @staticmethod
            def HParams(input_dims: int, output_dims: int) -> None:
                """Mock HParams for Linear.

                Args:
                    input_dims: Input dimensions.
                    output_dims: Output dimensions.
                """
                pass

        class BatchNorm:
            """Mock BatchNorm layer."""

            @staticmethod
            def HParams(dim: int) -> None:
                """Mock HParams for BatchNorm.

                Args:
                    dim: Dimension.
                """
                pass

        class Activation:
            """Mock Activation layer."""

            @staticmethod
            def HParams(activation_name: str) -> None:
                """Mock HParams for Activation.

                Args:
                    activation_name: Name of the activation.
                """
                pass

    mock_base_layer_mod = mock.MagicMock()
    mock_base_layer_mod.BaseLayer = MockBaseLayer

    mock_praxis = mock.MagicMock()
    mock_praxis.base_layer = mock_base_layer_mod
    mock_praxis.layers = MockLayers

    with mock.patch.dict(
        "sys.modules",
        {
            "praxis": mock_praxis,
            "praxis.base_layer": mock_base_layer_mod,
            "praxis.layers": MockLayers,
        },
    ):
        pax_mock = get_native_pax_mlp()
        pax_mock.setup()
        pax_mock(np.ones((4, 2)))
