"""NanoGPT architectures across `zero-*` frameworks."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_keras
import zero_mlx.nn as mlx_nn
from zero_flax import nnx
from zero_jax import random
from zero_keras import layers as keras_layers
from zero_pax.praxis.layers import DotProductAttention as PaxDotProductAttention
from zero_pax.praxis.layers import Embedding as PaxEmbedding
from zero_pax.praxis.layers import LayerNorm as PaxLayerNorm
from zero_pax.praxis.layers import Linear as PaxLinear
from zero_torch import nn as torch_nn

from ._types import (
    JaxArray,
    JaxModelPair,
    KerasTensorLike,
    MlxArray,
    ParamDict,
    PaxTensor,
    TorchTensor,
    as_jax,
    ensure_eager_mode,
    scalar_float,
)
from .data_gen import generate_random_tensor


# -----------------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------------
class ZeroPyTorchNanoGPT(torch_nn.Module):
    """Zero PyTorch NanoGPT single-block transformer model.

    Attributes:
        emb: Token embedding table of shape `(10, 4)`.
        attn: Multi-head self-attention with 4 features and 2 heads.
        ln: Layer normalization over 4 features.
        fc: Vocabulary output projection from 4 features to 10 logits.
    """

    def __init__(self) -> None:
        """Initialize the Zero PyTorch NanoGPT layers."""
        ensure_eager_mode()
        super().__init__()
        self.emb = torch_nn.Embedding(10, 4)
        self.attn = torch_nn.MultiheadAttention(4, 2, batch_first=True)
        self.ln = torch_nn.LayerNorm(4)
        self.fc = torch_nn.Linear(4, 10)

    def forward(self, x: TorchTensor) -> TorchTensor:
        """Run the forward pass of the Zero PyTorch NanoGPT.

        Args:
            x: Integer token tensor of shape `(batch_size, seq_len)`.

        Returns:
            Logit tensor of shape `(batch_size, seq_len, 10)`.
        """
        ensure_eager_mode()
        h = self.emb(x)
        attn_out, _ = self.attn(h, h, h)
        h = self.ln(h + attn_out)
        out: TorchTensor = self.fc(h)
        return out


def get_zero_pytorch_nanogpt() -> ZeroPyTorchNanoGPT:
    """Construct and return a `ZeroPyTorchNanoGPT` instance.

    Returns:
        Initialized `ZeroPyTorchNanoGPT` module.
    """
    ensure_eager_mode()
    return ZeroPyTorchNanoGPT()


def get_native_pytorch_nanogpt() -> ZeroPyTorchNanoGPT:
    """Construct and return the reference PyTorch NanoGPT module.

    Returns:
        Initialized reference `ZeroPyTorchNanoGPT` module.
    """
    ensure_eager_mode()
    return ZeroPyTorchNanoGPT()


# -----------------------------------------------------------------------------
# Keras / TensorFlow
# -----------------------------------------------------------------------------
class ZeroKerasNanoGPT:
    """Zero Keras NanoGPT model using `Embedding`, `MultiHeadAttention`, and `Dense`.

    Attributes:
        input_spec_tensor: Symbolic Keras input specification tensor.
        emb: Token embedding layer `(vocab_size=10, embed_dim=4)`.
        attn: Multi-head self-attention layer `(num_heads=2, key_dim=2)`.
        ln: Layer normalization layer.
        fc: Output dense projection layer to 10 vocabulary logits.
    """

    def __init__(self) -> None:
        """Initialize the Zero Keras NanoGPT sublayers."""
        ensure_eager_mode()
        self.input_spec_tensor: KerasTensorLike = zero_keras.Input(
            shape=(5,),
            dtype="int32",
        )
        self.emb = keras_layers.Embedding(10, 4)
        self.attn = keras_layers.MultiHeadAttention(num_heads=2, key_dim=2)
        self.ln = keras_layers.LayerNormalization()
        self.fc = keras_layers.Dense(10)

    def __call__(
        self,
        inputs: KerasTensorLike,
        training: bool | None = False,
    ) -> KerasTensorLike:
        """Execute the forward pass of the Zero Keras NanoGPT model.

        Args:
            inputs: Integer token tensor of shape `(batch_size, seq_len)`.
            training: Whether the model is executing in training mode.

        Returns:
            Output logit tensor of shape `(batch_size, seq_len, 10)`.
        """
        ensure_eager_mode()
        _ = training
        x = self.emb(inputs)
        attn_out = self.attn(x, x)
        residual = zero_keras.ops.add(x, attn_out)
        x_norm = self.ln(residual)
        out: KerasTensorLike = self.fc(x_norm)
        return out


def get_zero_keras_nanogpt() -> ZeroKerasNanoGPT:
    """Construct and return a `ZeroKerasNanoGPT` model.

    Returns:
        Initialized `ZeroKerasNanoGPT` model.
    """
    ensure_eager_mode()
    return ZeroKerasNanoGPT()


def get_native_keras_nanogpt() -> ZeroKerasNanoGPT:
    """Construct and return the reference Keras NanoGPT model.

    Returns:
        Initialized reference `ZeroKerasNanoGPT` model.
    """
    return get_zero_keras_nanogpt()


# -----------------------------------------------------------------------------
# JAX (Pure)
# -----------------------------------------------------------------------------
def get_zero_jax_nanogpt() -> JaxModelPair:
    """Return Zero JAX NanoGPT parameter initializer and forward function.

    Returns:
        Tuple `(init_params, forward)` for the functional JAX NanoGPT.
    """
    ensure_eager_mode()

    def init_params(key: JaxArray) -> ParamDict:
        """Initialize functional parameters for the Zero JAX NanoGPT.

        Args:
            key: PRNG key array.

        Returns:
            Dictionary of initialized embedding and projection parameters.
        """
        ensure_eager_mode()
        split_keys = as_jax(random.split(key))
        k1 = split_keys[0]
        k2 = split_keys[1]
        s1 = int(scalar_float(k1[1]))
        s2 = int(scalar_float(k2[1]))
        return {
            "emb": generate_random_tensor((10, 4), seed=s1),
            "w": generate_random_tensor((4, 10), seed=s2),
            "b": as_jax(jnp.zeros((10,))),
        }

    def forward(
        params: ParamDict,
        x: JaxArray,
        is_training: bool = True,
    ) -> JaxArray:
        """Compute the forward pass of the functional Zero JAX NanoGPT.

        Args:
            params: Parameter dictionary produced by `init_params`.
            x: Integer token array of shape `(batch_size, seq_len)`.
            is_training: Whether the model is executing in training mode.

        Returns:
            Logit array of shape `(batch_size, seq_len, 10)`.
        """
        ensure_eager_mode()
        _ = is_training
        h: JaxArray = params["emb"][x]
        out: JaxArray = h.dot(params["w"]) + params["b"]
        return out

    return init_params, forward


def get_native_jax_nanogpt() -> JaxModelPair:
    """Return reference JAX NanoGPT parameter initializer and forward function.

    Returns:
        Tuple `(init_params, forward)` for the reference functional JAX NanoGPT.
    """
    return get_zero_jax_nanogpt()


# -----------------------------------------------------------------------------
# Flax (NNX & Linen-Compatible Adapter)
# -----------------------------------------------------------------------------
class ZeroFlaxNNXNanoGPT(nnx.Module):
    """Zero Flax NNX NanoGPT architecture.

    Attributes:
        emb: Token embedding layer `(10, 4)`.
        attn: Multi-head self-attention layer `(num_heads=2, in_features=4)`.
        ln: Layer normalization over 4 features.
        fc: Linear output head mapping 4 features to 10 vocabulary logits.
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Initialize the Zero Flax NNX NanoGPT.

        Args:
            rngs: Flax NNX random number generator streams.
        """
        ensure_eager_mode()
        self.emb = nnx.Embed(
            10,
            4,
            embedding_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.attn = nnx.MultiHeadAttention(
            2,
            4,
            kernel_init=nnx.initializers.ones,
            out_kernel_init=nnx.initializers.ones,
            decode=False,
            rngs=rngs,
        )
        self.ln = nnx.LayerNorm(4, rngs=rngs)
        self.fc = nnx.Linear(
            4,
            10,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )

    def __call__(self, x: JaxArray) -> JaxArray:
        """Run the forward pass of the Zero Flax NNX NanoGPT.

        Args:
            x: Integer token `JaxArray` of shape `(batch_size, seq_len)`.

        Returns:
            Output logit `JaxArray` of shape `(batch_size, seq_len, 10)`.
        """
        ensure_eager_mode()
        h = self.emb(x)
        attn_out = self.attn(h, h, h, decode=False)
        h = self.ln(h + attn_out)
        out: JaxArray = self.fc(h)
        return out


class ZeroFlaxLinenNanoGPTAdapter:
    """Linen-compatible stateless adapter over `ZeroFlaxNNXNanoGPT`.

    Attributes:
        model: Underlying `ZeroFlaxNNXNanoGPT` instance.
    """

    def __init__(self) -> None:
        """Initialize the Linen-compatible NanoGPT adapter."""
        ensure_eager_mode()
        self.model = ZeroFlaxNNXNanoGPT(rngs=nnx.Rngs(0))

    def init(self, key: JaxArray, x: JaxArray) -> ParamDict:
        """Initialize parameters for Linen-style callers.

        Args:
            key: PRNG key array.
            x: Sample integer token array of shape `(batch_size, seq_len)`.

        Returns:
            Initialized parameter dictionary.
        """
        ensure_eager_mode()
        _ = self.model(x)
        return {"key": key}

    def apply(
        self,
        params: ParamDict,
        x: JaxArray,
        use_running_average: bool = False,
    ) -> JaxArray:
        """Apply the NanoGPT module for Linen-style callers.

        Args:
            params: Parameter dictionary returned by `init`.
            x: Integer token array of shape `(batch_size, seq_len)`.
            use_running_average: Unused flag accepted for Linen API parity.

        Returns:
            Output logit array of shape `(batch_size, seq_len, 10)`.
        """
        _ = params, use_running_average
        return self.model(x)

    def __call__(
        self,
        x: JaxArray,
        use_running_average: bool = False,
    ) -> JaxArray:
        """Directly invoke the underlying NNX NanoGPT.

        Args:
            x: Integer token array of shape `(batch_size, seq_len)`.
            use_running_average: Unused flag accepted for Linen API parity.

        Returns:
            Output logit array of shape `(batch_size, seq_len, 10)`.
        """
        _ = use_running_average
        return self.model(x)


def get_zero_flax_linen_nanogpt() -> ZeroFlaxLinenNanoGPTAdapter:
    """Return a Linen-compatible Zero Flax NanoGPT adapter.

    Returns:
        Initialized `ZeroFlaxLinenNanoGPTAdapter`.
    """
    ensure_eager_mode()
    return ZeroFlaxLinenNanoGPTAdapter()


def get_native_flax_linen_nanogpt() -> ZeroFlaxLinenNanoGPTAdapter:
    """Return the reference Linen-compatible Flax NanoGPT adapter.

    Returns:
        Initialized `ZeroFlaxLinenNanoGPTAdapter`.
    """
    return get_zero_flax_linen_nanogpt()


def get_zero_flax_nnx_nanogpt() -> type[ZeroFlaxNNXNanoGPT]:
    """Return the `ZeroFlaxNNXNanoGPT` module class.

    Returns:
        The `ZeroFlaxNNXNanoGPT` class accepting `rngs: nnx.Rngs`.
    """
    ensure_eager_mode()
    return ZeroFlaxNNXNanoGPT


def get_native_flax_nnx_nanogpt() -> type[ZeroFlaxNNXNanoGPT]:
    """Return the reference Flax NNX NanoGPT module class.

    Returns:
        The `ZeroFlaxNNXNanoGPT` class.
    """
    return get_zero_flax_nnx_nanogpt()


# -----------------------------------------------------------------------------
# MLX
# -----------------------------------------------------------------------------
class ZeroMlxNanoGPT(mlx_nn.Module):
    """Zero MLX NanoGPT single-block transformer model.

    Attributes:
        emb: Token embedding layer `(10, 4)`.
        attn: Multi-head attention layer `(dims=4, num_heads=2)`.
        ln: Layer normalization over 4 features.
        fc: Output linear projection from 4 features to 10 logits.
    """

    def __init__(self) -> None:
        """Initialize the Zero MLX NanoGPT."""
        ensure_eager_mode()
        super().__init__()
        self.emb = mlx_nn.Embedding(10, 4)
        self.attn = mlx_nn.MultiHeadAttention(4, 2)
        self.ln = mlx_nn.LayerNorm(4)
        self.fc = mlx_nn.Linear(4, 10)

    def __call__(self, x: MlxArray) -> MlxArray:
        """Execute the forward pass of the Zero MLX NanoGPT.

        Args:
            x: Integer token `MlxArray` of shape `(batch_size, seq_len)`.

        Returns:
            Output logit `MlxArray` of shape `(batch_size, seq_len, 10)`.
        """
        ensure_eager_mode()
        h = self.emb(x)
        attn_out = self.attn(h, h, h)
        h = self.ln(h + attn_out)
        out: MlxArray = self.fc(h)
        return out


def get_zero_mlx_nanogpt() -> ZeroMlxNanoGPT:
    """Construct and return a `ZeroMlxNanoGPT` instance.

    Returns:
        Initialized `ZeroMlxNanoGPT` module.
    """
    ensure_eager_mode()
    return ZeroMlxNanoGPT()


def get_native_mlx_nanogpt() -> ZeroMlxNanoGPT:
    """Construct and return the reference MLX NanoGPT module.

    Returns:
        Initialized reference `ZeroMlxNanoGPT` module.
    """
    return get_zero_mlx_nanogpt()


# -----------------------------------------------------------------------------
# Pax / Praxis
# -----------------------------------------------------------------------------
class ZeroPaxNanoGPT:
    """Zero Pax / Praxis NanoGPT layer with Embedding, Attention, LayerNorm, Linear.

    Attributes:
        vocab_size: Vocabulary size for embeddings and output logits.
        hidden_dims: Hidden transformer embedding dimension.
        num_heads: Number of attention heads.
        dim_per_head: Feature dimension per attention head.
        emb: Praxis token embedding layer.
        attn: Praxis dot-product attention layer.
        ln: Praxis layer normalization layer.
        fc: Praxis linear output head.
        is_setup: Whether `setup` has been executed.
    """

    def __init__(
        self,
        vocab_size: int = 10,
        hidden_dims: int = 4,
        num_heads: int = 2,
        dim_per_head: int = 2,
    ) -> None:
        """Initialize the Zero Pax NanoGPT configuration.

        Args:
            vocab_size: Vocabulary size for embeddings and output logits.
            hidden_dims: Hidden transformer embedding dimension.
            num_heads: Number of attention heads.
            dim_per_head: Feature dimension per attention head.
        """
        self.vocab_size: int = vocab_size
        self.hidden_dims: int = hidden_dims
        self.num_heads: int = num_heads
        self.dim_per_head: int = dim_per_head
        self.emb: PaxEmbedding = PaxEmbedding(
            num_classes=self.vocab_size,
            input_dims=self.hidden_dims,
        )
        self.attn: PaxDotProductAttention = PaxDotProductAttention(
            input_dim=self.hidden_dims,
            hidden_dim=self.hidden_dims,
            num_heads=self.num_heads,
            dim_per_head=self.dim_per_head,
        )
        self.ln: PaxLayerNorm = PaxLayerNorm(dim=self.hidden_dims)
        self.fc: PaxLinear = PaxLinear(
            input_dims=self.hidden_dims,
            output_dims=self.vocab_size,
        )
        self.is_setup: bool = False

    def setup(self) -> None:
        """Configure the Praxis sublayers for NanoGPT."""
        self.emb = PaxEmbedding(
            num_classes=self.vocab_size,
            input_dims=self.hidden_dims,
        )
        self.attn = PaxDotProductAttention(
            input_dim=self.hidden_dims,
            hidden_dim=self.hidden_dims,
            num_heads=self.num_heads,
            dim_per_head=self.dim_per_head,
        )
        self.ln = PaxLayerNorm(dim=self.hidden_dims)
        self.fc = PaxLinear(
            input_dims=self.hidden_dims,
            output_dims=self.vocab_size,
        )
        self.is_setup = True

    def __call__(self, inputs: PaxTensor) -> PaxTensor:
        """Run the forward pass of the Zero Pax NanoGPT.

        Args:
            inputs: Integer token `PaxTensor` of shape `(batch_size, seq_len)`.

        Returns:
            Output logit `PaxTensor` of shape `(batch_size, seq_len, 10)`.
        """
        if not self.is_setup:
            self.setup()
        h = self.emb(inputs)
        attn_out = self.attn(h, h, h)
        h_norm = self.ln(h + attn_out)
        out: PaxTensor = self.fc(h_norm)
        return out


def get_zero_pax_nanogpt() -> ZeroPaxNanoGPT:
    """Construct and return a `ZeroPaxNanoGPT` layer.

    Returns:
        Initialized `ZeroPaxNanoGPT` Praxis layer.
    """
    layer = ZeroPaxNanoGPT()
    layer.setup()
    return layer


def get_native_pax_nanogpt() -> ZeroPaxNanoGPT:
    """Construct and return the reference Pax NanoGPT layer.

    Returns:
        Initialized reference `ZeroPaxNanoGPT` Praxis layer.
    """
    return get_zero_pax_nanogpt()


get_pytorch_nanogpt = get_native_pytorch_nanogpt
get_flax_nanogpt = get_native_flax_linen_nanogpt
get_keras_nanogpt = get_native_keras_nanogpt

__all__ = [
    "ZeroFlaxLinenNanoGPTAdapter",
    "ZeroFlaxNNXNanoGPT",
    "ZeroKerasNanoGPT",
    "ZeroMlxNanoGPT",
    "ZeroPaxNanoGPT",
    "ZeroPyTorchNanoGPT",
    "get_flax_nanogpt",
    "get_keras_nanogpt",
    "get_native_flax_linen_nanogpt",
    "get_native_flax_nnx_nanogpt",
    "get_native_jax_nanogpt",
    "get_native_keras_nanogpt",
    "get_native_mlx_nanogpt",
    "get_native_pax_nanogpt",
    "get_native_pytorch_nanogpt",
    "get_pytorch_nanogpt",
    "get_zero_flax_linen_nanogpt",
    "get_zero_flax_nnx_nanogpt",
    "get_zero_jax_nanogpt",
    "get_zero_keras_nanogpt",
    "get_zero_mlx_nanogpt",
    "get_zero_pax_nanogpt",
    "get_zero_pytorch_nanogpt",
]
