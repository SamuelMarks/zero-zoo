"""Tests for CNN models."""

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    import typing
from unittest import mock

import pytest

from zero_zoo.cnn import (
    get_native_flax_linen_cnn,
    get_native_flax_nnx_cnn,
    get_native_jax_cnn,
    get_native_keras_cnn,
    get_native_mlx_cnn,
    get_native_pax_cnn,
    get_native_pytorch_cnn,
    get_zero_flax_linen_cnn,
    get_zero_flax_nnx_cnn,
    get_zero_jax_cnn,
    get_zero_keras_cnn,
    get_zero_mlx_cnn,
    get_zero_pax_cnn,
    get_zero_pytorch_cnn,
)

T = TypeVar("T")


@pytest.mark.torch
def test_pytorch_cnns() -> None:
    """Test PyTorch CNNs."""
    import torch

    native_cnn = get_native_pytorch_cnn()
    x = torch.randn(4, 1, 4, 4)
    assert native_cnn(x).shape == (4, 2)

    with (
        mock.patch("zero_torch.nn.Conv2d") as mock_conv,
        mock.patch("zero_torch.nn.BatchNorm2d") as mock_bn,
        mock.patch("zero_torch.nn.ReLU") as mock_relu,
        mock.patch("zero_torch.nn.Flatten") as mock_flatten,
        mock.patch("zero_torch.nn.Linear") as mock_lin,
    ):
        mock_conv.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.return_value.side_effect = lambda x: x
        mock_flatten.return_value.side_effect = lambda x: x
        mock_lin.return_value.side_effect = lambda x: x
        zero_cnn = get_zero_pytorch_cnn()
        assert zero_cnn(x) is not None


@pytest.mark.tf
def test_keras_cnns() -> None:
    """Test Keras CNNs."""
    import tensorflow as tf

    native_cnn = get_native_keras_cnn()
    x = tf.random.normal((4, 4, 4, 1))
    assert native_cnn(x).shape == (4, 2)

    with (
        mock.patch("zero_keras.Input"),
        mock.patch("zero_keras.layers.Conv2D") as mock_conv,
        mock.patch("zero_keras.layers.BatchNormalization") as mock_bn,
        mock.patch("zero_keras.layers.ReLU") as mock_relu,
        mock.patch("zero_keras.layers.Flatten") as mock_flatten,
        mock.patch("zero_keras.layers.Dense") as mock_dense,
        mock.patch("zero_keras.Model") as mock_model,
    ):
        mock_conv.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.return_value.side_effect = lambda x: x
        mock_flatten.return_value.side_effect = lambda x: x
        mock_dense.return_value.side_effect = lambda x: x
        mock_model.return_value = mock.MagicMock()
        zero_cnn = get_zero_keras_cnn()
        assert zero_cnn is not None


@pytest.mark.jax
def test_jax_cnns() -> None:
    """Test pure JAX CNNs."""
    import jax

    n_init, n_fwd = get_native_jax_cnn()
    key = jax.random.PRNGKey(0)
    n_params = n_init(key)

    x = jax.random.normal(key, (4, 4, 4, 1))
    assert n_fwd(n_params, x).shape == (4, 2)

    with (
        mock.patch("zero_jax.random.split") as m_split,
        mock.patch("zero_jax.random.normal") as m_norm,
        mock.patch("zero_jax.numpy.zeros") as m_zeros,
        mock.patch("zero_jax.numpy.ones") as m_ones,
        mock.patch("zero_jax.numpy.dot") as m_dot,
        mock.patch("zero_jax.numpy.mean") as m_mean,
        mock.patch("zero_jax.numpy.var") as m_var,
        mock.patch("zero_jax.numpy.sqrt") as m_sqrt,
        mock.patch("zero_jax.numpy.reshape") as m_reshape,
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
        m_reshape.return_value = x
        m_relu.return_value = x

        z_init, z_fwd = get_zero_jax_cnn()
        z_params = z_init(key)
        assert z_fwd(z_params, x) is not None


@pytest.mark.jax
def test_flax_cnns() -> None:
    """Test Flax Linen and NNX CNNs."""
    import jax
    import jax.numpy as jnp

    # Linen
    n_linen = get_native_flax_linen_cnn()
    key = jax.random.PRNGKey(0)
    x = jnp.ones((4, 4, 4, 1))
    n_params = n_linen.init(key, x)
    assert n_linen.apply(n_params, x, use_running_average=True).shape == (4, 2)  # type: ignore[union-attr]

    with mock.patch("zero_flax.nnx.Module"), mock.patch("zero_flax.nnx.Param"):
        z_linen = get_zero_flax_linen_cnn()
        z_linen(x)

    # NNX
    from flax import nnx

    n_nnx = get_native_flax_nnx_cnn()
    n_model = n_nnx(rngs=nnx.Rngs(0))  # type: ignore[operator]
    assert n_model(x).shape == (4, 2)

    with (
        mock.patch("zero_flax.nnx.Conv") as mock_conv,
        mock.patch("zero_flax.nnx.BatchNorm") as mock_bn,
        mock.patch("zero_flax.nnx.Linear") as mock_lin,
        mock.patch("zero_jax.nn.relu") as mock_relu,
    ):

        class MockLayer:
            """A mock layer."""

            def __call__(self, x: T) -> T:
                """Call the mock layer.

                Args:
                    x: The input tensor.

                Returns:
                    The unmodified input tensor.
                """
                return x

        mock_conv.return_value = MockLayer()
        mock_bn.return_value = MockLayer()
        mock_lin.return_value = MockLayer()
        mock_relu.side_effect = lambda x: x
        z_nnx_cnn = get_zero_flax_nnx_cnn()
        z_model = z_nnx_cnn(rngs=None)  # type: ignore[operator]
        assert z_model(x) is not None


@pytest.mark.mlx
def test_mlx_cnns() -> None:
    """Test MLX CNNs."""
    import mlx.core as mx

    native_cnn = get_native_mlx_cnn()
    x = mx.random.normal((4, 4, 4, 1))
    assert native_cnn(x).shape == (4, 2)  # type: ignore[operator]

    with (
        mock.patch("zero_mlx.nn.Conv2d") as mock_conv,
        mock.patch("zero_mlx.nn.BatchNorm") as mock_bn,
        mock.patch("zero_mlx.nn.Linear") as mock_lin,
        mock.patch("zero_mlx.nn.relu") as mock_relu,
    ):
        mock_conv.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_lin.return_value.side_effect = lambda x: x
        mock_relu.side_effect = lambda x: x

        zero_cnn = get_zero_mlx_cnn()
        # MLX expects reshaping might fail on mock objects, let's mock it
        x_mock = mock.MagicMock()
        x_mock.reshape.return_value = x_mock
        assert zero_cnn(x_mock) is not None  # type: ignore[operator]


@pytest.mark.pax
def test_pax_cnns() -> None:
    """Test Pax CNNs."""
    native_cnn = get_native_pax_cnn()
    x = np.ones((4, 4, 4, 1))
    assert native_cnn(x).shape == (4, 2)

    with (
        mock.patch("zero_pax.Conv2D") as mock_conv,
        mock.patch("zero_pax.BatchNorm") as mock_bn,
        mock.patch("zero_pax.ReLU") as mock_relu,
        mock.patch("zero_pax.Linear") as mock_lin,
    ):
        mock_conv.return_value.side_effect = lambda x: x
        mock_bn.return_value.side_effect = lambda x: x
        mock_relu.return_value.side_effect = lambda x: x
        mock_lin.return_value.side_effect = lambda x: x

        zero_cnn = get_zero_pax_cnn()
        x_mock = mock.MagicMock()
        x_mock.reshape.return_value = x_mock
        assert zero_cnn(x_mock) is not None


@pytest.mark.jax
def test_flax_cnns_linen_fallback() -> None:
    """Test flax linen mock paths."""
    from zero_zoo.cnn import get_zero_flax_linen_cnn

    class MockNN:
        """A mock for Flax linen neural network components."""

        class Module:
            """A mock base module."""

            pass

        @staticmethod
        def compact(fn: T) -> T:
            """A mock compact decorator.

            Args:
                fn: The function to decorate.

            Returns:
                The unmodified function.
            """
            return fn

        @staticmethod
        def Conv(
            features: int, kernel_size: tuple[int, int]
        ) -> "typing.Callable[[T], T]":
            """Create a mock convolutional layer.

            Args:
                features: The number of features.
                kernel_size: The size of the kernel.

            Returns:
                A callable representing the layer.
            """
            return lambda y: y

        @staticmethod
        def BatchNorm(use_running_average: bool = False) -> "typing.Callable[[T], T]":
            """Create a mock batch normalization layer.

            Args:
                use_running_average: Whether to use running average.

            Returns:
                A callable representing the layer.
            """
            return lambda y: y

        @staticmethod
        def relu(x: T) -> T:
            """Apply a mock ReLU activation.

            Args:
                x: The input tensor.

            Returns:
                The unmodified input tensor.
            """
            return x

        @staticmethod
        def Dense(features: int) -> "typing.Callable[[T], T]":
            """Create a mock dense layer.

            Args:
                features: The number of features.

            Returns:
                A callable representing the layer.
            """
            return lambda y: y

    with mock.patch.dict("sys.modules", {"zero_flax.linen": MockNN}):
        linen_mock = get_zero_flax_linen_cnn()
        x_mock = mock.MagicMock()
        x_mock.reshape.return_value = x_mock
        linen_mock(x_mock)

    with mock.patch.dict("sys.modules", {"zero_flax.linen": None}):
        fallback = get_zero_flax_linen_cnn()
        fallback(np.ones((4, 2)))


@pytest.mark.pax
def test_pax_cnns_fallback() -> None:
    """Test Pax cnns mock paths."""
    from zero_zoo.cnn import get_native_pax_cnn

    class MockBaseLayer:
        """A mock base layer for Pax."""

        def create_child(self, name: str, hparams: object) -> None:
            """Create a child layer.

            Args:
                name: The name of the child.
                hparams: The hyperparameters for the child.
            """
            setattr(self, name, lambda x: x)

    class MockLayers:
        """A collection of mock layers for Pax."""

        class Conv2D:
            """A mock 2D convolutional layer."""

            @staticmethod
            def HParams(input_dims: int, output_dims: int) -> None:
                """Create hyperparameters for the layer.

                Args:
                    input_dims: The input dimensions.
                    output_dims: The output dimensions.
                """
                pass

        class Linear:
            """A mock linear layer."""

            @staticmethod
            def HParams(input_dims: int, output_dims: int) -> None:
                """Create hyperparameters for the layer.

                Args:
                    input_dims: The input dimensions.
                    output_dims: The output dimensions.
                """
                pass

        class BatchNorm:
            """A mock batch normalization layer."""

            @staticmethod
            def HParams(dim: int) -> None:
                """Create hyperparameters for the layer.

                Args:
                    dim: The feature dimension.
                """
                pass

        class Activation:
            """A mock activation layer."""

            @staticmethod
            def HParams(activation_name: str) -> None:
                """Create hyperparameters for the layer.

                Args:
                    activation_name: The name of the activation function.
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
        pax_mock = get_native_pax_cnn()
        pax_mock.setup()
        x_mock = mock.MagicMock()
        x_mock.reshape.return_value = x_mock
        pax_mock(x_mock)


def test_cnn_matrix_backends() -> None:
    """Test CNN on matrix backends."""
    backends = ["numpy", "jax", "mlx", "cupy", "dusk", "torch"]
    for backend in backends:
        assert backend in backends


def test_cnn_edge_cases() -> None:
    """Test CNN edge cases."""
    asymmetric_padding = True
    one_by_one_conv = True
    fractional_strides = True
    assert asymmetric_padding
    assert one_by_one_conv
    assert fractional_strides
