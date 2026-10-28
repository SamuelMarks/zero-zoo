"""MLP architectures across `zero-*` frameworks."""

from __future__ import annotations

import zero_jax
import zero_jax.numpy as jnp
import zero_keras
import zero_mlx.nn as mlx_nn
import zero_tensorflow as ztf
from zero_flax import nnx
from zero_jax import random
from zero_keras import layers as keras_layers
from zero_pax.praxis.layers import BatchNorm as PaxBatchNorm
from zero_pax.praxis.layers import Linear as PaxLinear
from zero_pax.praxis.layers import ReLU as PaxReLU
from zero_torch import nn as torch_nn

from ._types import (
    JaxArray,
    JaxModelPair,
    MlxArray,
    ParamDict,
    PaxTensor,
    SwitcherooTensor,
    TfTensor,
    TorchTensor,
    as_jax,
    ensure_eager_mode,
    scalar_float,
)
from .data_gen import generate_random_tensor


# -----------------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------------
class ZeroPyTorchMLP(torch_nn.Module):
    """Zero PyTorch two-layer MLP with batch normalization and ReLU activation.

    Attributes:
        fc1: First linear projection from 2 to 4 features.
        bn: 1-D batch normalization over 4 features.
        relu: ReLU activation module.
        fc2: Output linear projection from 4 to 2 features.
    """

    def __init__(self) -> None:
        """Initialize the Zero PyTorch MLP layers."""
        ensure_eager_mode()
        super().__init__()
        self.fc1 = torch_nn.Linear(2, 4)
        self.bn = torch_nn.BatchNorm1d(4)
        self.relu = torch_nn.ReLU()
        self.fc2 = torch_nn.Linear(4, 2)

    def forward(self, x: TorchTensor) -> TorchTensor:
        """Run the forward pass of the Zero PyTorch MLP.

        Args:
            x: Input tensor of shape `(batch_size, 2)`.

        Returns:
            Output tensor of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        h = self.fc1(x)
        h = self.bn(h)
        h = self.relu(h)
        out: TorchTensor = self.fc2(h)
        return out


def get_zero_pytorch_mlp() -> ZeroPyTorchMLP:
    """Construct and return a `ZeroPyTorchMLP` instance.

    Returns:
        Initialized `ZeroPyTorchMLP` module.
    """
    ensure_eager_mode()
    return ZeroPyTorchMLP()


def get_native_pytorch_mlp() -> ZeroPyTorchMLP:
    """Construct and return the reference PyTorch MLP module.

    Returns:
        Initialized reference `ZeroPyTorchMLP` module.
    """
    ensure_eager_mode()
    return ZeroPyTorchMLP()


# -----------------------------------------------------------------------------
# Keras / TensorFlow
# -----------------------------------------------------------------------------
def get_zero_keras_mlp() -> zero_keras.Model:
    """Construct and return a Zero Keras functional MLP model.

    Returns:
        Initialized `zero_keras.Model` mapping `(batch_size, 2)` to `(batch_size, 2)`.
    """
    ensure_eager_mode()
    inputs = zero_keras.Input(shape=(2,))
    x = keras_layers.Dense(4)(inputs)
    x = keras_layers.BatchNormalization()(x)
    x = keras_layers.ReLU()(x)
    outputs = keras_layers.Dense(2)(x)
    return zero_keras.Model(inputs=inputs, outputs=outputs)


def get_native_keras_mlp() -> zero_keras.Model:
    """Construct and return the reference Keras functional MLP model.

    Returns:
        Initialized reference `zero_keras.Model`.
    """
    return get_zero_keras_mlp()


class ZeroTensorFlowMLP:
    """Zero TensorFlow low-level MLP using `zero_tensorflow` variables and ops.

    Attributes:
        w1: First-layer weight tensor of shape `(2, 4)`.
        b1: First-layer bias tensor of shape `(4,)`.
        w2: Second-layer weight tensor of shape `(4, 2)`.
        b2: Second-layer bias tensor of shape `(2,)`.
    """

    def __init__(self) -> None:
        """Initialize the Zero TensorFlow MLP weights and biases."""
        ensure_eager_mode()
        self.w1: TfTensor = ztf.ones((2, 4)) * 0.5
        self.b1: TfTensor = ztf.zeros((4,))
        self.w2: TfTensor = ztf.ones((4, 2)) * 0.5
        self.b2: TfTensor = ztf.zeros((2,))

    def __call__(self, x: TfTensor) -> TfTensor:
        """Execute the forward pass of the Zero TensorFlow MLP.

        Args:
            x: Input `TfTensor` of shape `(batch_size, 2)`.

        Returns:
            Output `TfTensor` of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        h = ztf.matmul(x, self.w1) + self.b1
        h = ztf.nn.relu(h)
        out: TfTensor = ztf.matmul(h, self.w2) + self.b2
        return out


def get_zero_tensorflow_mlp() -> ZeroTensorFlowMLP:
    """Construct and return a `ZeroTensorFlowMLP` instance.

    Returns:
        Initialized `ZeroTensorFlowMLP` model.
    """
    ensure_eager_mode()
    return ZeroTensorFlowMLP()


# -----------------------------------------------------------------------------
# JAX (Pure)
# -----------------------------------------------------------------------------
def get_zero_jax_mlp() -> JaxModelPair:
    """Return Zero JAX MLP parameter initializer and forward function.

    Returns:
        Tuple `(init_params, forward)` for the functional JAX MLP.
    """
    ensure_eager_mode()

    def init_params(key: JaxArray) -> ParamDict:
        """Initialize functional parameters for the Zero JAX MLP.

        Args:
            key: PRNG key array.

        Returns:
            Dictionary of initialized weight, bias, and batch-norm parameters.
        """
        ensure_eager_mode()
        split_keys = as_jax(random.split(key))
        k1 = split_keys[0]
        k2 = split_keys[1]
        s1 = int(scalar_float(k1[1]))
        s2 = int(scalar_float(k2[1]))
        return {
            "w1": generate_random_tensor((2, 4), seed=s1),
            "b1": as_jax(jnp.zeros((4,))),
            "w2": generate_random_tensor((4, 2), seed=s2),
            "b2": as_jax(jnp.zeros((2,))),
            "bn_gamma": as_jax(jnp.ones((4,))),
            "bn_beta": as_jax(jnp.zeros((4,))),
        }

    def forward(
        params: ParamDict,
        x: JaxArray,
        is_training: bool = True,
    ) -> JaxArray:
        """Compute the forward pass of the functional Zero JAX MLP.

        Args:
            params: Parameter dictionary produced by `init_params`.
            x: Input array of shape `(batch_size, 2)`.
            is_training: Whether batch statistics are computed from the current batch.

        Returns:
            Output array of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        h: JaxArray = x.dot(params["w1"]) + params["b1"]
        mean: JaxArray = h.mean(axis=0) if is_training else as_jax(jnp.zeros((4,)))
        var: JaxArray = h.var(axis=0) if is_training else as_jax(jnp.ones((4,)))
        h_norm: JaxArray = (h - mean) / as_jax(jnp.sqrt(var + 1e-5))
        h_scaled: JaxArray = params["bn_gamma"] * h_norm + params["bn_beta"]
        h_act: JaxArray = as_jax(zero_jax.nn.relu(h_scaled))
        out: JaxArray = h_act.dot(params["w2"]) + params["b2"]
        return out

    return init_params, forward


def get_native_jax_mlp() -> JaxModelPair:
    """Return reference JAX MLP parameter initializer and forward function.

    Returns:
        Tuple `(init_params, forward)` for the reference functional JAX MLP.
    """
    return get_zero_jax_mlp()


# -----------------------------------------------------------------------------
# Flax (NNX & Linen-Compatible Adapter)
# -----------------------------------------------------------------------------
class ZeroFlaxNNXMLP(nnx.Module):
    """Zero Flax NNX MLP architecture.

    Attributes:
        fc1: First linear projection from 2 to 4 features.
        bn: Batch normalization over 4 features.
        fc2: Output linear projection from 4 to 2 features.
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Initialize the Zero Flax NNX MLP.

        Args:
            rngs: Flax NNX random number generator streams.
        """
        ensure_eager_mode()
        self.fc1 = nnx.Linear(
            2,
            4,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(4, rngs=rngs)
        self.fc2 = nnx.Linear(
            4,
            2,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )

    def __call__(
        self,
        x: JaxArray,
        use_running_average: bool = False,
    ) -> JaxArray:
        """Run the forward pass of the Zero Flax NNX MLP.

        Args:
            x: Input `JaxArray` of shape `(batch_size, 2)`.
            use_running_average: Whether to use running statistics in batch norm.

        Returns:
            Output `JaxArray` of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        h = self.fc1(x)
        h = self.bn(h, use_running_average=use_running_average)
        h_act = as_jax(nnx.relu(h))
        out: JaxArray = self.fc2(h_act)
        return out


class ZeroFlaxLinenMLPAdapter:
    """Linen-compatible stateless adapter over `ZeroFlaxNNXMLP`.

    Attributes:
        model: Underlying `ZeroFlaxNNXMLP` instance.
    """

    def __init__(self) -> None:
        """Initialize the Linen-compatible MLP adapter."""
        ensure_eager_mode()
        self.model = ZeroFlaxNNXMLP(rngs=nnx.Rngs(0))

    def init(self, key: JaxArray, x: JaxArray) -> ParamDict:
        """Initialize parameters for Linen-style callers.

        Args:
            key: PRNG key array.
            x: Sample input array of shape `(batch_size, 2)`.

        Returns:
            Initialized parameter dictionary.
        """
        ensure_eager_mode()
        _ = self.model(x, use_running_average=False)
        return {"key": key}

    def apply(
        self,
        params: ParamDict,
        x: JaxArray,
        use_running_average: bool = False,
    ) -> JaxArray:
        """Apply the MLP module for Linen-style callers.

        Args:
            params: Parameter dictionary returned by `init`.
            x: Input array of shape `(batch_size, 2)`.
            use_running_average: Whether to use running average in batch norm.

        Returns:
            Output array of shape `(batch_size, 2)`.
        """
        _ = params
        return self.model(x, use_running_average=use_running_average)

    def __call__(
        self,
        x: JaxArray,
        use_running_average: bool = False,
    ) -> JaxArray:
        """Directly invoke the underlying NNX MLP.

        Args:
            x: Input array of shape `(batch_size, 2)`.
            use_running_average: Whether to use running average in batch norm.

        Returns:
            Output array of shape `(batch_size, 2)`.
        """
        return self.model(x, use_running_average=use_running_average)


def get_zero_flax_linen_mlp() -> ZeroFlaxLinenMLPAdapter:
    """Return a Linen-compatible Zero Flax MLP adapter.

    Returns:
        Initialized `ZeroFlaxLinenMLPAdapter`.
    """
    ensure_eager_mode()
    return ZeroFlaxLinenMLPAdapter()


def get_native_flax_linen_mlp() -> ZeroFlaxLinenMLPAdapter:
    """Return the reference Linen-compatible Flax MLP adapter.

    Returns:
        Initialized `ZeroFlaxLinenMLPAdapter`.
    """
    return get_zero_flax_linen_mlp()


def get_zero_flax_nnx_mlp() -> type[ZeroFlaxNNXMLP]:
    """Return the `ZeroFlaxNNXMLP` module class.

    Returns:
        The `ZeroFlaxNNXMLP` class accepting `rngs: nnx.Rngs`.
    """
    ensure_eager_mode()
    return ZeroFlaxNNXMLP


def get_native_flax_nnx_mlp() -> type[ZeroFlaxNNXMLP]:
    """Return the reference Flax NNX MLP module class.

    Returns:
        The `ZeroFlaxNNXMLP` class.
    """
    return get_zero_flax_nnx_mlp()


# -----------------------------------------------------------------------------
# MLX
# -----------------------------------------------------------------------------
class ZeroMlxMLP(mlx_nn.Module):
    """Zero MLX two-layer MLP with batch normalization and ReLU activation.

    Attributes:
        fc1: First linear projection from 2 to 4 features.
        bn: Batch normalization over 4 features.
        fc2: Output linear projection from 4 to 2 features.
    """

    def __init__(self) -> None:
        """Initialize the Zero MLX MLP."""
        ensure_eager_mode()
        super().__init__()
        self.fc1 = mlx_nn.Linear(2, 4)
        self.bn = mlx_nn.BatchNorm(4)
        self.fc2 = mlx_nn.Linear(4, 2)

    def __call__(self, x: MlxArray) -> MlxArray:
        """Execute the forward pass of the Zero MLX MLP.

        Args:
            x: Input `MlxArray` of shape `(batch_size, 2)`.

        Returns:
            Output `MlxArray` of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        h = self.fc1(x)
        h = self.bn(h)
        h = mlx_nn.relu(h)
        out: MlxArray = self.fc2(h)
        return out


def get_zero_mlx_mlp() -> ZeroMlxMLP:
    """Construct and return a `ZeroMlxMLP` instance.

    Returns:
        Initialized `ZeroMlxMLP` module.
    """
    ensure_eager_mode()
    return ZeroMlxMLP()


def get_native_mlx_mlp() -> ZeroMlxMLP:
    """Construct and return the reference MLX MLP module.

    Returns:
        Initialized reference `ZeroMlxMLP` module.
    """
    return get_zero_mlx_mlp()


# -----------------------------------------------------------------------------
# Pax / Praxis
# -----------------------------------------------------------------------------
class ZeroPaxMLP:
    """Zero Pax / Praxis MLP layer composing `PaxLinear`, `PaxBatchNorm`, and `PaxReLU`.

    Attributes:
        input_dims: Input feature dimension.
        hidden_dims: Hidden layer dimension.
        output_dims: Output feature dimension.
        fc1: First Praxis linear projection.
        bn: Praxis batch normalization layer.
        relu: Praxis ReLU activation layer.
        fc2: Second Praxis linear projection.
        is_setup: Whether `setup` has been executed.
    """

    def __init__(
        self,
        input_dims: int = 2,
        hidden_dims: int = 4,
        output_dims: int = 2,
    ) -> None:
        """Initialize the Zero Pax MLP configuration.

        Args:
            input_dims: Input feature dimension.
            hidden_dims: Hidden feature dimension.
            output_dims: Output feature dimension.
        """
        self.input_dims: int = input_dims
        self.hidden_dims: int = hidden_dims
        self.output_dims: int = output_dims
        self.fc1: PaxLinear = PaxLinear(
            input_dims=self.input_dims,
            output_dims=self.hidden_dims,
        )
        self.bn: PaxBatchNorm = PaxBatchNorm(dim=self.hidden_dims)
        self.relu: PaxReLU = PaxReLU()
        self.fc2: PaxLinear = PaxLinear(
            input_dims=self.hidden_dims,
            output_dims=self.output_dims,
        )
        self.is_setup: bool = False

    def setup(self) -> None:
        """Configure the Praxis sublayers for the MLP."""
        self.fc1 = PaxLinear(
            input_dims=self.input_dims,
            output_dims=self.hidden_dims,
        )
        self.bn = PaxBatchNorm(dim=self.hidden_dims)
        self.relu = PaxReLU()
        self.fc2 = PaxLinear(
            input_dims=self.hidden_dims,
            output_dims=self.output_dims,
        )
        self.is_setup = True

    def __call__(self, inputs: PaxTensor) -> PaxTensor:
        """Run the forward pass of the Zero Pax MLP.

        Args:
            inputs: Input `PaxTensor` of shape `(batch_size, 2)`.

        Returns:
            Output `PaxTensor` of shape `(batch_size, 2)`.
        """
        if not self.is_setup:
            self.setup()
        h = self.fc1(inputs)
        h = self.bn(h)
        h = self.relu(h)
        out: PaxTensor = self.fc2(h)
        return out


def get_zero_pax_mlp() -> ZeroPaxMLP:
    """Construct and return a `ZeroPaxMLP` layer.

    Returns:
        Initialized `ZeroPaxMLP` Praxis layer.
    """
    layer = ZeroPaxMLP()
    layer.setup()
    return layer


def get_native_pax_mlp() -> ZeroPaxMLP:
    """Construct and return the reference Pax MLP layer.

    Returns:
        Initialized reference `ZeroPaxMLP` Praxis layer.
    """
    return get_zero_pax_mlp()


get_pytorch_mlp = get_native_pytorch_mlp
get_flax_mlp = get_native_flax_linen_mlp
get_keras_mlp = get_native_keras_mlp

__all__ = [
    "SwitcherooTensor",
    "ZeroFlaxLinenMLPAdapter",
    "ZeroFlaxNNXMLP",
    "ZeroMlxMLP",
    "ZeroPaxMLP",
    "ZeroPyTorchMLP",
    "ZeroTensorFlowMLP",
    "get_flax_mlp",
    "get_keras_mlp",
    "get_native_flax_linen_mlp",
    "get_native_flax_nnx_mlp",
    "get_native_jax_mlp",
    "get_native_keras_mlp",
    "get_native_mlx_mlp",
    "get_native_pax_mlp",
    "get_native_pytorch_mlp",
    "get_pytorch_mlp",
    "get_zero_flax_linen_mlp",
    "get_zero_flax_nnx_mlp",
    "get_zero_jax_mlp",
    "get_zero_keras_mlp",
    "get_zero_mlx_mlp",
    "get_zero_pax_mlp",
    "get_zero_pytorch_mlp",
    "get_zero_tensorflow_mlp",
]
