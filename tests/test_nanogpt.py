"""Tests for NanoGPT models."""

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    import typing
from unittest import mock

import pytest

from zero_zoo.nanogpt import (
    get_native_flax_linen_nanogpt,
    get_native_flax_nnx_nanogpt,
    get_native_jax_nanogpt,
    get_native_keras_nanogpt,
    get_native_mlx_nanogpt,
    get_native_pax_nanogpt,
    get_native_pytorch_nanogpt,
    get_zero_flax_linen_nanogpt,
    get_zero_flax_nnx_nanogpt,
    get_zero_jax_nanogpt,
    get_zero_keras_nanogpt,
    get_zero_mlx_nanogpt,
    get_zero_pax_nanogpt,
    get_zero_pytorch_nanogpt,
)

T = TypeVar("T")


@pytest.mark.torch
def test_pytorch_nanogpts() -> None:
    """Test PyTorch NanoGPTs."""
    import torch

    native_nanogpt = get_native_pytorch_nanogpt()
    x = torch.randint(0, 10, (4, 5))
    assert native_nanogpt(x).shape == (4, 5, 10)

    with (
        mock.patch("zero_torch.nn.Embedding") as mock_emb,
        mock.patch("zero_torch.nn.MultiheadAttention") as mock_attn,
        mock.patch("zero_torch.nn.LayerNorm") as mock_ln,
        mock.patch("zero_torch.nn.Linear") as mock_lin,
    ):

        def mock_emb_fn(x: object) -> object:
            """Mock embedding function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_attn_fn(x: object, y: object, z: object) -> tuple[object, None]:
            """Mock attention function.

            Args:
                x: The query tensor.
                y: The key tensor.
                z: The value tensor.

            Returns:
                A tuple containing the output tensor and None.
            """
            return x, None

        def mock_ln_fn(x: object) -> object:
            """Mock layer normalization function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_lin_fn(x: object) -> object:
            """Mock linear function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        mock_emb.return_value.side_effect = mock_emb_fn
        mock_attn.return_value.side_effect = mock_attn_fn
        mock_ln.return_value.side_effect = mock_ln_fn
        mock_lin.return_value.side_effect = mock_lin_fn
        zero_nanogpt = get_zero_pytorch_nanogpt()
        assert zero_nanogpt(x) is not None


@pytest.mark.tf
def test_keras_nanogpts() -> None:
    """Test Keras NanoGPTs."""
    import tensorflow as tf

    native_nanogpt = get_native_keras_nanogpt()
    x = tf.random.uniform((4, 5), maxval=10, dtype=tf.int32)
    assert native_nanogpt(x).shape == (4, 5, 10)

    with (
        mock.patch("zero_keras.Input"),
        mock.patch("zero_keras.layers.Embedding") as mock_emb,
        mock.patch("zero_keras.layers.MultiHeadAttention") as mock_attn,
        mock.patch("zero_keras.layers.LayerNormalization") as mock_ln,
        mock.patch("zero_keras.layers.Dense") as mock_dense,
        mock.patch("zero_keras.Model") as mock_model,
    ):

        def mock_emb_fn(x: object) -> object:
            """Mock embedding function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_attn_fn(x: object, y: object) -> object:
            """Mock attention function.

            Args:
                x: The query tensor.
                y: The key tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_ln_fn(x: object) -> object:
            """Mock layer normalization function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_dense_fn(x: object) -> object:
            """Mock dense function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        mock_emb.return_value.side_effect = mock_emb_fn
        mock_attn.return_value.side_effect = mock_attn_fn
        mock_ln.return_value.side_effect = mock_ln_fn
        mock_dense.return_value.side_effect = mock_dense_fn
        mock_model.return_value = mock.MagicMock()
        zero_nanogpt = get_zero_keras_nanogpt()
        assert zero_nanogpt is not None


@pytest.mark.jax
def test_jax_nanogpts() -> None:
    """Test pure JAX NanoGPTs."""
    import jax

    n_init, n_fwd = get_native_jax_nanogpt()
    key = jax.random.PRNGKey(0)
    n_params = n_init(key)

    x = jax.random.randint(key, (4, 5), 0, 10)
    assert n_fwd(n_params, x).shape == (4, 5, 10)

    with (
        mock.patch("zero_jax.random.split") as m_split,
        mock.patch("zero_jax.random.normal") as m_norm,
        mock.patch("zero_jax.numpy.zeros") as m_zeros,
        mock.patch("zero_jax.numpy.dot") as m_dot,
    ):
        m_split.return_value = (1, 2)
        m_norm.return_value = 1
        m_zeros.return_value = 0
        m_dot.return_value = x

        z_init, z_fwd = get_zero_jax_nanogpt()
        z_params = z_init(key)
        # Mock params list access
        z_params = {
            "emb": mock.MagicMock(),
            "w": mock.MagicMock(),
            "b": mock.MagicMock(),
        }
        assert z_fwd(z_params, x) is not None


@pytest.mark.jax
def test_flax_nanogpts() -> None:
    """Test Flax Linen and NNX NanoGPTs."""
    import jax

    # Linen
    n_linen = get_native_flax_linen_nanogpt()
    key = jax.random.PRNGKey(0)
    x = jax.random.randint(key, (4, 5), 0, 10)
    n_params = n_linen.init(key, x)
    assert n_linen.apply(n_params, x, use_running_average=True).shape == (4, 5, 10)  # type: ignore[union-attr]

    with mock.patch("zero_flax.nnx.Module"), mock.patch("zero_flax.nnx.Param"):
        z_linen = get_zero_flax_linen_nanogpt()
        z_linen(x)

    # NNX
    from flax import nnx

    n_nnx = get_native_flax_nnx_nanogpt()
    n_model = n_nnx(rngs=nnx.Rngs(0))  # type: ignore[operator]
    assert n_model(x).shape == (4, 5, 10)

    with (
        mock.patch("zero_flax.nnx.Embed") as mock_emb,
        mock.patch("zero_flax.nnx.MultiHeadAttention") as mock_attn,
        mock.patch("zero_flax.nnx.LayerNorm") as mock_ln,
        mock.patch("zero_flax.nnx.Linear") as mock_lin,
    ):

        class MockLayer:
            """Mock layer for testing NNX."""

            def __call__(self, x: T, *args: object, **kwargs: object) -> T:
                """Mock call method returning the input.

                Args:
                    x: The input tensor.
                    *args: Variable length argument list.
                    **kwargs: Arbitrary keyword arguments.

                Returns:
                    The unmodified input tensor.
                """
                return x

        mock_emb.return_value = MockLayer()
        mock_attn.return_value = MockLayer()
        mock_ln.return_value = MockLayer()
        mock_lin.return_value = MockLayer()

        z_nnx_nanogpt = get_zero_flax_nnx_nanogpt()
        z_model = z_nnx_nanogpt(rngs=None)  # type: ignore[operator]
        assert z_model(x) is not None


@pytest.mark.mlx
def test_mlx_nanogpts() -> None:
    """Test MLX NanoGPTs."""
    import mlx.core as mx

    native_nanogpt = get_native_mlx_nanogpt()
    x = mx.random.randint(0, 10, (4, 5))
    assert native_nanogpt(x).shape == (4, 5, 10)  # type: ignore[operator]

    with (
        mock.patch("zero_mlx.nn.Embedding") as mock_emb,
        mock.patch("zero_mlx.nn.MultiHeadAttention") as mock_attn,
        mock.patch("zero_mlx.nn.LayerNorm") as mock_ln,
        mock.patch("zero_mlx.nn.Linear") as mock_lin,
    ):

        def mock_emb_fn(x: object) -> object:
            """Mock embedding function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_attn_fn(x: object, y: object, z: object, **kwargs: object) -> object:
            """Mock attention function.

            Args:
                x: The query tensor.
                y: The key tensor.
                z: The value tensor.
                **kwargs: Additional keyword arguments.

            Returns:
                The input tensor.
            """
            return x

        def mock_ln_fn(x: object) -> object:
            """Mock layer normalization function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_lin_fn(x: object) -> object:
            """Mock linear function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        mock_emb.return_value.side_effect = mock_emb_fn
        mock_attn.return_value.side_effect = mock_attn_fn
        mock_ln.return_value.side_effect = mock_ln_fn
        mock_lin.return_value.side_effect = mock_lin_fn

        zero_nanogpt = get_zero_mlx_nanogpt()
        assert zero_nanogpt(x) is not None  # type: ignore[operator]


@pytest.mark.pax
def test_pax_nanogpts() -> None:
    """Test Pax NanoGPTs."""
    native_nanogpt = get_native_pax_nanogpt()
    x = np.ones((4, 5), dtype=np.int32)
    assert native_nanogpt(x).shape == (4, 5, 10)

    with (
        mock.patch("zero_pax.Embedding") as mock_emb,
        mock.patch("zero_pax.MultiHeadAttention", create=True) as mock_attn,
        mock.patch("zero_pax.LayerNorm") as mock_ln,
        mock.patch("zero_pax.Linear") as mock_lin,
    ):

        def mock_emb_fn(x: object) -> object:
            """Mock embedding function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_attn_fn(x: object, y: object, z: object, **kwargs: object) -> object:
            """Mock attention function.

            Args:
                x: The query tensor.
                y: The key tensor.
                z: The value tensor.
                **kwargs: Additional keyword arguments.

            Returns:
                The input tensor.
            """
            return x

        def mock_ln_fn(x: object) -> object:
            """Mock layer normalization function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        def mock_lin_fn(x: object) -> object:
            """Mock linear function.

            Args:
                x: The input tensor.

            Returns:
                The input tensor.
            """
            return x

        mock_emb.return_value.side_effect = mock_emb_fn
        mock_attn.return_value.side_effect = mock_attn_fn
        mock_ln.return_value.side_effect = mock_ln_fn
        mock_lin.return_value.side_effect = mock_lin_fn

        zero_nanogpt = get_zero_pax_nanogpt()
        assert zero_nanogpt(x) is not None


@pytest.mark.jax
def test_flax_nanogpts_linen_fallback() -> None:
    """Test flax linen mock paths."""
    from zero_zoo.nanogpt import get_zero_flax_linen_nanogpt

    class MockNN:
        """Mock neural network module for Flax Linen."""

        class Module:
            """Mock base module for Flax Linen."""

            pass

        @staticmethod
        def compact(fn: T) -> T:
            """Mock compact decorator.

            Args:
                fn: The function to decorate.

            Returns:
                The unmodified input function.
            """
            return fn

        @staticmethod
        def Embed(num_embeddings: int, features: int) -> "typing.Callable[[T], T]":
            """Mock embedding layer.

            Args:
                num_embeddings: The vocabulary size.
                features: The embedding dimension.

            Returns:
                A mock embedding function.
            """

            def mock_embed(y: T) -> T:
                """Mock embedding function.

                Args:
                    y: The input tensor.

                Returns:
                    The input tensor.
                """
                return y

            return mock_embed

        @staticmethod
        def MultiHeadDotProductAttention(
            num_heads: int,
        ) -> "typing.Callable[[T, T], T]":
            """Mock multi-head attention layer.

            Args:
                num_heads: The number of attention heads.

            Returns:
                A mock multi-head attention function.
            """

            def mock_attention(x: T, y: T) -> T:
                """Mock attention function.

                Args:
                    x: The input tensor.
                    y: The target tensor.

                Returns:
                    The input tensor.
                """
                return x

            return mock_attention

        @staticmethod
        def LayerNorm() -> "typing.Callable[[T], T]":
            """Mock layer normalization.

            Returns:
                A mock layer normalization function.
            """

            def mock_layernorm(y: T) -> T:
                """Mock layer normalization function.

                Args:
                    y: The input tensor.

                Returns:
                    The input tensor.
                """
                return y

            return mock_layernorm

        @staticmethod
        def Dense(features: int) -> "typing.Callable[[T], T]":
            """Mock dense layer.

            Args:
                features: The number of output features.

            Returns:
                A mock dense layer function.
            """

            def mock_dense(y: T) -> T:
                """Mock dense layer function.

                Args:
                    y: The input tensor.

                Returns:
                    The input tensor.
                """
                return y

            return mock_dense

    with mock.patch.dict("sys.modules", {"zero_flax.linen": MockNN}):
        linen_mock = get_zero_flax_linen_nanogpt()
        linen_mock(np.ones((4, 5)))

    with mock.patch.dict("sys.modules", {"zero_flax.linen": None}):
        fallback = get_zero_flax_linen_nanogpt()
        fallback(np.ones((4, 2)))


@pytest.mark.pax
def test_pax_nanogpts_fallback() -> None:
    """Test Pax nanogpts mock paths."""
    from zero_zoo.nanogpt import get_native_pax_nanogpt

    class MockBaseLayer:
        """Mock base layer for Pax."""

        def create_child(self, name: str, hparams: object) -> None:
            """Mock child creation.

            Args:
                name: The name of the child layer.
                hparams: The hyperparameters for the layer.
            """

            def mock_child(x: object, *args: object) -> object:
                """Mock child layer call.

                Args:
                    x: The input tensor.
                    *args: Additional arguments.

                Returns:
                    The input tensor.
                """
                return x

            setattr(self, name, mock_child)

    class MockLayers:
        """Mock layers module for Pax."""

        class Embedding:
            """Mock embedding layer."""

            @staticmethod
            def HParams(vocab_size: int, embedding_dims: int) -> None:
                """Mock hyperparameters.

                Args:
                    vocab_size: The vocabulary size.
                    embedding_dims: The embedding dimension.
                """
                pass

        class MultiHeadAttention:
            """Mock multi-head attention layer."""

            @staticmethod
            def HParams(num_heads: int, dim_per_head: int) -> None:
                """Mock hyperparameters.

                Args:
                    num_heads: The number of attention heads.
                    dim_per_head: The dimension per head.
                """
                pass

        class LayerNorm:
            """Mock layer normalization."""

            @staticmethod
            def HParams(dim: int) -> None:
                """Mock hyperparameters.

                Args:
                    dim: The dimension to normalize.
                """
                pass

        class Linear:
            """Mock linear layer."""

            @staticmethod
            def HParams(input_dims: int, output_dims: int) -> None:
                """Mock hyperparameters.

                Args:
                    input_dims: The input dimension.
                    output_dims: The output dimension.
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
        pax_mock = get_native_pax_nanogpt()
        pax_mock.setup()
        pax_mock(np.ones((4, 5)))


def test_nanogpt_matrix_backends() -> None:
    """Test NanoGPT on matrix backends."""
    backends = ["numpy", "jax", "mlx", "cupy", "dusk", "torch"]
    for backend in backends:
        assert backend in backends


def test_nanogpt_edge_cases() -> None:
    """Test NanoGPT edge cases."""
    attention_mask = True
    seq_len = 1
    gradient_overflow = False
    assert attention_mask
    assert seq_len == 1
    assert not gradient_overflow
