"""NanoGPT architectures across frameworks."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typing

    import flax.linen as nn
    import keras
    import numpy.typing as npt
    import torch
    import zero_flax.linen as zero_nn
    import zero_keras
    import zero_pax
    import zero_torch
    from praxis import base_layer

    NDArray = npt.NDArray["typing.Any"]
    JAX_INIT_FN = typing.Callable[..., dict[str, NDArray]]
    JAX_FWD_FN = typing.Callable[..., NDArray]
    JAX_RET = tuple[JAX_INIT_FN, JAX_FWD_FN]

    ZJAX_INIT_FN = typing.Callable[..., dict[str, NDArray]]
    ZJAX_FWD_FN = typing.Callable[..., NDArray]
    ZJAX_RET = tuple[ZJAX_INIT_FN, ZJAX_FWD_FN]


# -----------------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------------
def get_native_pytorch_nanogpt() -> "torch.nn.Module":
    """Return native PyTorch NanoGPT.

    Returns:
        The native PyTorch NanoGPT module.
    """
    from torch import nn

    class NanoGPT(nn.Module):
        """Native PyTorch NanoGPT."""

        def __init__(self) -> None:
            """Initialize PyTorch NanoGPT."""
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.attn = nn.MultiheadAttention(4, 2, batch_first=True)
            self.ln = nn.LayerNorm(4)
            self.fc = nn.Linear(4, 10)

        def forward(self, x: "NDArray") -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                The output array.
            """
            x = self.emb(x)
            attn_out, _ = self.attn(x, x, x)
            x = self.ln(x + attn_out)
            return self.fc(x)  # type: ignore[no-any-return]

    return NanoGPT()


def get_zero_pytorch_nanogpt() -> "zero_torch.nn.Module":
    """Return Zero PyTorch NanoGPT.

    Returns:
        The Zero PyTorch NanoGPT module.
    """
    import zero_torch
    from zero_torch import nn

    class NanoGPT(nn.Module):  # type: ignore[misc]
        """Zero PyTorch NanoGPT."""

        def __init__(self) -> None:
            """Initialize Zero PyTorch NanoGPT."""
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.attn = nn.MultiheadAttention(4, 2, batch_first=True)
            self.ln = nn.LayerNorm(4)
            self.fc = nn.Linear(4, 10)

        def forward(self, x: zero_torch.Tensor) -> zero_torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor.

            Returns:
                The output tensor.
            """
            x = self.emb(x)
            attn_out, _ = self.attn(x, x, x)
            x = self.ln(x + attn_out)
            return self.fc(x)

    return NanoGPT()


# -----------------------------------------------------------------------------
# Keras / TensorFlow
# -----------------------------------------------------------------------------
def get_native_keras_nanogpt() -> "keras.Model":
    """Return native Keras NanoGPT.

    Returns:
        The native Keras NanoGPT model.
    """
    import keras
    from keras import layers

    inputs = keras.Input(shape=(5,), dtype="int32")
    x = layers.Embedding(10, 4)(inputs)
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = layers.LayerNormalization()(x + attn_out)
    outputs = layers.Dense(10)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def get_zero_keras_nanogpt() -> "zero_keras.Model":
    """Return Zero Keras NanoGPT.

    Returns:
        The Zero Keras NanoGPT model.
    """
    import zero_keras as keras
    from zero_keras import layers

    inputs = keras.Input(shape=(5,), dtype="int32")
    x = layers.Embedding(10, 4)(inputs)
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = layers.LayerNormalization()(x + attn_out)
    outputs = layers.Dense(10)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# -----------------------------------------------------------------------------
# JAX (Pure)
# -----------------------------------------------------------------------------
def get_native_jax_nanogpt() -> "JAX_RET":
    """Return native JAX NanoGPT parameters and forward function.

    Returns:
        A tuple containing the initialization and forward functions.
    """
    import jax.numpy as jnp
    from jax import random

    def init_params(key: "NDArray") -> dict[str, "NDArray"]:
        """Initialize parameters.

        Args:
            key: PRNG key.

        Returns:
            Dictionary of parameters.
        """
        k1, k2 = random.split(key)
        return {
            "emb": random.normal(k1, (10, 4)),  # type: ignore[dict-item]
            "w": random.normal(k2, (4, 10)),  # type: ignore[dict-item]
            "b": jnp.zeros((10,)),  # type: ignore[dict-item]
        }

    def forward(
        params: dict[str, "NDArray"],
        x: "NDArray",
        is_training: bool = True,
    ) -> "NDArray":
        """Forward pass.

        Args:
            params: Dictionary of parameters.
            x: Input array.
            is_training: Whether in training mode.

        Returns:
            The output array.
        """
        # Simple mock
        x = params["emb"][x]
        return jnp.dot(x, params["w"]) + params["b"]  # type: ignore[return-value]

    return init_params, forward


def get_zero_jax_nanogpt() -> "ZJAX_RET":
    """Return Zero JAX NanoGPT parameters and forward function.

    Returns:
        A tuple containing the initialization and forward functions.
    """
    import zero_jax.numpy as jnp
    from zero_jax import random

    def init_params(key: "NDArray") -> dict[str, "NDArray"]:
        """Initialize parameters.

        Args:
            key: PRNG key.

        Returns:
            Dictionary of parameters.
        """
        k1, k2 = random.split(key)
        return {
            "emb": random.normal(k1, (10, 4)),
            "w": random.normal(k2, (4, 10)),
            "b": jnp.zeros((10,)),
        }

    def forward(
        params: dict[str, "NDArray"],
        x: "NDArray",
        is_training: bool = True,
    ) -> "NDArray":
        """Forward pass.

        Args:
            params: Dictionary of parameters.
            x: Input array.
            is_training: Whether in training mode.

        Returns:
            The output array.
        """
        x = params["emb"][x]
        return jnp.dot(x, params["w"]) + params["b"]  # type: ignore[no-any-return]

    return init_params, forward


# -----------------------------------------------------------------------------
# Flax
# -----------------------------------------------------------------------------
def get_native_flax_linen_nanogpt() -> "nn.Module":
    """Return native Flax Linen NanoGPT.

    Returns:
        The native Flax Linen NanoGPT module.
    """
    import flax.linen as nn

    class NanoGPT(nn.Module):
        """Native Flax Linen NanoGPT."""

        @nn.compact
        def __call__(
            self,
            x: "NDArray",
            use_running_average: bool = False,
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.
                use_running_average: Whether to use running average.

            Returns:
                The output array.
            """
            x = nn.Embed(10, 4)(x)  # type: ignore[assignment]
            attn_out = nn.MultiHeadDotProductAttention(num_heads=2)(x, x)
            x = nn.LayerNorm()(x + attn_out)
            return nn.Dense(10)(x)  # type: ignore[return-value]

    return NanoGPT()


def get_zero_flax_linen_nanogpt() -> "zero_nn.Module":
    """Return Zero Flax Linen NanoGPT.

    Returns:
        The Zero Flax Linen NanoGPT module.
    """
    try:
        from zero_flax import linen as nn

        class NanoGPT(nn.Module):  # type: ignore[misc]
            """Zero Flax Linen NanoGPT."""

            @nn.compact  # type: ignore[untyped-decorator]
            def __call__(
                self,
                x: "NDArray",
                use_running_average: bool = False,
            ) -> "NDArray":
                """Forward pass.

                Args:
                    x: Input array.
                    use_running_average: Whether to use running average.

                Returns:
                    The output array.
                """
                x = nn.Embed(10, 4)(x)
                attn_out = nn.MultiHeadDotProductAttention(num_heads=2)(x, x)
                x = nn.LayerNorm()(x + attn_out)
                return nn.Dense(10)(x)  # type: ignore[no-any-return]

        return NanoGPT()
    except ImportError:
        from zero_flax.nnx import Module, Param

        class NanoWrapper(Module):  # type: ignore[misc]
            """Zero Flax Linen NanoWrapper."""

            def __init__(self) -> None:
                """Initialize NanoWrapper."""
                super().__init__()
                self.p = Param(1.0)
                super().__setattr__("_is_initializing", False)

            def __call__(
                self,
                x: "NDArray",
                use_running_average: bool = False,
            ) -> "NDArray":
                """Forward pass.

                Args:
                    x: Input array.
                    use_running_average: Whether to use running average.

                Returns:
                    The output array.
                """
                return x

        return NanoWrapper()


def get_native_flax_nnx_nanogpt() -> "NDArray":
    """Return native Flax NNX NanoGPT.

    Returns:
        The native Flax NNX NanoGPT module class.
    """
    from flax import nnx

    class NanoGPT(nnx.Module):
        """Native Flax NNX NanoGPT."""

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize native Flax NNX NanoGPT.

            Args:
                rngs: Random number generators.
            """
            self.emb = nnx.Embed(10, 4, rngs=rngs)
            self.attn = nnx.MultiHeadAttention(2, 4, rngs=rngs)
            self.ln = nnx.LayerNorm(4, rngs=rngs)
            self.fc = nnx.Linear(4, 10, rngs=rngs)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                The output array.
            """
            x = self.emb(x)  # type: ignore[arg-type,assignment]
            attn_out = self.attn(x, x, x, decode=False)  # type: ignore[arg-type]
            x = self.ln(x + attn_out)
            return self.fc(x)  # type: ignore[arg-type,return-value]

    return NanoGPT  # type: ignore[return-value]


def get_zero_flax_nnx_nanogpt() -> "NDArray":
    """Return Zero Flax NNX NanoGPT.

    Returns:
        The Zero Flax NNX NanoGPT module class.
    """
    from zero_flax import nnx

    class NanoGPT(nnx.Module):  # type: ignore[misc]
        """Zero Flax NNX NanoGPT."""

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize Zero Flax NNX NanoGPT.

            Args:
                rngs: Random number generators.
            """
            self.emb = nnx.Embed(10, 4, rngs=rngs)
            self.attn = nnx.MultiHeadAttention(2, 4, rngs=rngs)
            self.ln = nnx.LayerNorm(4, rngs=rngs)
            self.fc = nnx.Linear(4, 10, rngs=rngs)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                The output array.
            """
            x = self.emb(x)
            attn_out = self.attn(x, x, x, decode=False)
            x = self.ln(x + attn_out)
            return self.fc(x)  # type: ignore[no-any-return]

    return NanoGPT  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# MLX
# -----------------------------------------------------------------------------
def get_native_mlx_nanogpt() -> "NDArray":
    """Return native MLX NanoGPT.

    Returns:
        The native MLX NanoGPT module.
    """
    import mlx.nn as nn

    class NanoGPT(nn.Module):  # type: ignore[name-defined, misc]
        """Native MLX NanoGPT."""

        def __init__(self) -> None:
            """Initialize MLX NanoGPT."""
            super().__init__()
            self.emb = nn.Embedding(10, 4)  # type: ignore[attr-defined]
            self.attn = nn.MultiHeadAttention(4, 2)  # type: ignore[attr-defined]
            self.ln = nn.LayerNorm(4)  # type: ignore[attr-defined]
            self.fc = nn.Linear(4, 10)  # type: ignore[attr-defined]

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                The output array.
            """
            x = self.emb(x)
            attn_out = self.attn(x, x, x)
            x = self.ln(x + attn_out)
            return self.fc(x)  # type: ignore[no-any-return]

    return NanoGPT()


def get_zero_mlx_nanogpt() -> "NDArray":
    """Return Zero MLX NanoGPT.

    Returns:
        The Zero MLX NanoGPT module.
    """
    import zero_mlx.nn as nn

    class NanoGPT(nn.Module):  # type: ignore[misc]
        """Zero MLX NanoGPT."""

        def __init__(self) -> None:
            """Initialize Zero MLX NanoGPT."""
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.attn = nn.MultiHeadAttention(4, 2)
            self.ln = nn.LayerNorm(4)
            self.fc = nn.Linear(4, 10)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                The output array.
            """
            x = self.emb(x)
            attn_out = self.attn(x, x, x)
            x = self.ln(x + attn_out)
            return self.fc(x)  # type: ignore[no-any-return]

    return NanoGPT()


# -----------------------------------------------------------------------------
# Pax
# -----------------------------------------------------------------------------
def get_native_pax_nanogpt() -> "base_layer.BaseLayer":
    """Return native Pax NanoGPT.

    Returns:
        The native Pax NanoGPT layer.
    """
    try:
        from praxis import base_layer, layers

        class NanoGPT(base_layer.BaseLayer):  # type: ignore[misc]
            """Native Pax NanoGPT."""

            def setup(self) -> None:
                """Setup Pax NanoGPT."""
                self.create_child(
                    "emb", layers.Embedding.HParams(vocab_size=10, embedding_dims=4)
                )
                self.create_child(
                    "attn",
                    layers.MultiHeadAttention.HParams(num_heads=2, dim_per_head=2),
                )
                self.create_child("ln", layers.LayerNorm.HParams(dim=4))
                self.create_child(
                    "fc", layers.Linear.HParams(input_dims=4, output_dims=10)
                )

            def __call__(
                self,
                x: "NDArray",
            ) -> "NDArray":
                """Forward pass.

                Args:
                    x: Input array.

                Returns:
                    The output array.
                """
                x = self.emb(x)
                attn_out = self.attn(x, x, x)
                x = self.ln(x + attn_out)
                return self.fc(x)  # type: ignore[no-any-return]

        return NanoGPT()
    except ImportError:

        class DummyPax:
            """Dummy Pax layer for fallback."""

            def __call__(
                self,
                x: "NDArray",
            ) -> "NDArray":
                """Forward pass.

                Args:
                    x: Input array.

                Returns:
                    The output array.
                """
                import numpy as np

                return np.zeros((x.shape[0], x.shape[1], 10))

        return DummyPax()


def get_zero_pax_nanogpt() -> "zero_pax.Layer":
    """Return Zero Pax NanoGPT.

    Returns:
        The Zero Pax NanoGPT layer.
    """
    import zero_pax
    from zero_pax import Layer

    class NanoGPT(Layer):  # type: ignore[misc]
        """Zero Pax NanoGPT."""

        def __init__(self) -> None:
            """Initialize Zero Pax NanoGPT."""
            super().__init__()
            self.emb = zero_pax.Embedding(vocab_size=10, embedding_dims=4)
            self.attn = zero_pax.MultiHeadAttention(num_heads=2, dim_per_head=2)
            self.ln = zero_pax.LayerNorm(dim=4)
            self.fc = zero_pax.Linear(input_dims=4, output_dims=10)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                The output array.
            """
            x = self.emb(x)
            attn_out = self.attn(x, x, x)
            x = self.ln(x + attn_out)
            return self.fc(x)  # type: ignore[no-any-return]

    return NanoGPT()


get_pytorch_nanogpt = get_native_pytorch_nanogpt
get_flax_nanogpt = get_native_flax_linen_nanogpt
get_keras_nanogpt = get_native_keras_nanogpt
