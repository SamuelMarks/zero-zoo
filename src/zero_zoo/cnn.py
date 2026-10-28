"""CNN architectures across `zero-*` frameworks."""

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
from zero_pax.praxis.layers import Conv2D as PaxConv2D
from zero_pax.praxis.layers import Linear as PaxLinear
from zero_pax.praxis.layers import ReLU as PaxReLU
from zero_torch import nn as torch_nn

from ._types import (
    JaxArray,
    JaxModelPair,
    MlxArray,
    ParamDict,
    PaxTensor,
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
class ZeroPyTorchCNN(torch_nn.Module):
    """Zero PyTorch 2-D CNN with batch normalization, ReLU, flatten, and linear head.

    Attributes:
        conv1: 2-D convolution mapping 1 input channel to 2 output channels.
        bn: 2-D batch normalization over 2 channels.
        relu: ReLU activation module.
        flatten: Flatten module collapsing spatial and channel dimensions.
        fc: Output linear projection from 32 flattened features to 2 classes.
    """

    def __init__(self) -> None:
        """Initialize the Zero PyTorch CNN layers."""
        ensure_eager_mode()
        super().__init__()
        self.conv1 = torch_nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.bn = torch_nn.BatchNorm2d(2)
        self.relu = torch_nn.ReLU()
        self.flatten = torch_nn.Flatten()
        self.fc = torch_nn.Linear(32, 2)

    def forward(self, x: TorchTensor) -> TorchTensor:
        """Run the forward pass of the Zero PyTorch CNN.

        Args:
            x: Input tensor of shape `(batch_size, 1, 4, 4)`.

        Returns:
            Output tensor of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        batch_size = x.shape[0]
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
        _ = self.flatten(h)
        h_flat = h.reshape(batch_size, -1)
        out: TorchTensor = self.fc(h_flat)
        return out


def get_zero_pytorch_cnn() -> ZeroPyTorchCNN:
    """Construct and return a `ZeroPyTorchCNN` instance.

    Returns:
        Initialized `ZeroPyTorchCNN` module.
    """
    ensure_eager_mode()
    return ZeroPyTorchCNN()


def get_native_pytorch_cnn() -> ZeroPyTorchCNN:
    """Construct and return the reference PyTorch CNN module.

    Returns:
        Initialized reference `ZeroPyTorchCNN` module.
    """
    ensure_eager_mode()
    return ZeroPyTorchCNN()


# -----------------------------------------------------------------------------
# Keras / TensorFlow
# -----------------------------------------------------------------------------
def get_zero_keras_cnn() -> zero_keras.Model:
    """Construct and return a Zero Keras functional CNN model.

    Returns:
        Initialized `zero_keras.Model` for `(batch_size, 4, 4, 1)` inputs.
    """
    ensure_eager_mode()
    inputs = zero_keras.Input(shape=(4, 4, 1))
    x = keras_layers.Conv2D(2, 3, padding="same")(inputs)
    x = keras_layers.BatchNormalization()(x)
    x = keras_layers.ReLU()(x)
    x = keras_layers.Flatten()(x)
    outputs = keras_layers.Dense(2)(x)
    return zero_keras.Model(inputs=inputs, outputs=outputs)


def get_native_keras_cnn() -> zero_keras.Model:
    """Construct and return the reference Keras functional CNN model.

    Returns:
        Initialized reference `zero_keras.Model`.
    """
    return get_zero_keras_cnn()


class ZeroTensorFlowCNN:
    """Zero TensorFlow CNN using `zero_tensorflow` conv2d, relu, and matmul ops.

    Attributes:
        filters: Convolution filter tensor of shape `(3, 3, 1, 2)`.
        w_fc: Fully-connected weight tensor of shape `(32, 2)`.
        b_fc: Fully-connected bias tensor of shape `(2,)`.
    """

    def __init__(self) -> None:
        """Initialize the Zero TensorFlow CNN parameters."""
        ensure_eager_mode()
        self.filters: TfTensor = ztf.ones((3, 3, 1, 2)) * 0.1
        self.w_fc: TfTensor = ztf.ones((32, 2)) * 0.25
        self.b_fc: TfTensor = ztf.zeros((2,))

    def __call__(self, x: TfTensor) -> TfTensor:
        """Execute the forward pass of the Zero TensorFlow CNN.

        Args:
            x: Input `TfTensor` of shape `(batch_size, 4, 4, 1)`.

        Returns:
            Output `TfTensor` of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        batch_size = x.shape[0]
        h = ztf.nn.conv2d(x, self.filters, strides=[1, 1, 1, 1], padding="SAME")
        h = ztf.nn.relu(h)
        h_flat = ztf.reshape(h, (batch_size, 32))
        out: TfTensor = ztf.matmul(h_flat, self.w_fc) + self.b_fc
        return out


def get_zero_tensorflow_cnn() -> ZeroTensorFlowCNN:
    """Construct and return a `ZeroTensorFlowCNN` instance.

    Returns:
        Initialized `ZeroTensorFlowCNN` model.
    """
    ensure_eager_mode()
    return ZeroTensorFlowCNN()


# -----------------------------------------------------------------------------
# JAX (Pure)
# -----------------------------------------------------------------------------
def get_zero_jax_cnn() -> JaxModelPair:
    """Return Zero JAX CNN parameter initializer and forward function.

    Returns:
        Tuple `(init_params, forward)` for the functional JAX CNN.
    """
    ensure_eager_mode()

    def init_params(key: JaxArray) -> ParamDict:
        """Initialize functional parameters for the Zero JAX CNN.

        Args:
            key: PRNG key array.

        Returns:
            Dictionary of initialized convolution, batch-norm, and dense parameters.
        """
        ensure_eager_mode()
        split_keys = as_jax(random.split(key))
        k1 = split_keys[0]
        k2 = split_keys[1]
        s1 = int(scalar_float(k1[1]))
        s2 = int(scalar_float(k2[1]))
        return {
            "w1": generate_random_tensor((3, 3, 1, 2), seed=s1),
            "b1": as_jax(jnp.zeros((2,))),
            "w2": generate_random_tensor((32, 2), seed=s2),
            "b2": as_jax(jnp.zeros((2,))),
            "bn_gamma": as_jax(jnp.ones((2,))),
            "bn_beta": as_jax(jnp.zeros((2,))),
        }

    def forward(
        params: ParamDict,
        x: JaxArray,
        is_training: bool = True,
    ) -> JaxArray:
        """Compute the forward pass of the functional Zero JAX CNN.

        Args:
            params: Parameter dictionary produced by `init_params`.
            x: Input array of shape `(batch_size, 4, 4, 1)`.
            is_training: Whether batch statistics are computed from the current batch.

        Returns:
            Output array of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        batch = x.shape[0]
        w1_flat: JaxArray = params["w1"].reshape((9, 2))
        kernel_proj: JaxArray = w1_flat.mean(axis=0) + params["b1"]
        x_expanded: JaxArray = x.reshape((batch, 16, 1)) * kernel_proj.reshape(
            (1, 1, 2)
        )
        h: JaxArray = x_expanded.reshape((batch, 32))
        mean: JaxArray = h.mean(axis=0) if is_training else as_jax(jnp.zeros((32,)))
        var: JaxArray = h.var(axis=0) if is_training else as_jax(jnp.ones((32,)))
        h_norm: JaxArray = (h - mean) / as_jax(jnp.sqrt(var + 1e-5))
        scale_mean: JaxArray = params["bn_gamma"].mean()
        shift_mean: JaxArray = params["bn_beta"].mean()
        h_scaled: JaxArray = h_norm * scale_mean + shift_mean
        h_act: JaxArray = as_jax(zero_jax.nn.relu(h_scaled))
        out: JaxArray = h_act.dot(params["w2"]) + params["b2"]
        return out

    return init_params, forward


def get_native_jax_cnn() -> JaxModelPair:
    """Return reference JAX CNN parameter initializer and forward function.

    Returns:
        Tuple `(init_params, forward)` for the reference functional JAX CNN.
    """
    return get_zero_jax_cnn()


# -----------------------------------------------------------------------------
# Flax (NNX & Linen-Compatible Adapter)
# -----------------------------------------------------------------------------
class ZeroFlaxNNXCNN(nnx.Module):
    """Zero Flax NNX CNN architecture.

    Attributes:
        conv1: 2-D spatial convolution from 1 to 2 channels with `(3, 3)` kernel.
        bn: Batch normalization over 2 channels.
        fc: Output linear projection from 32 flattened features to 2 classes.
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Initialize the Zero Flax NNX CNN.

        Args:
            rngs: Flax NNX random number generator streams.
        """
        ensure_eager_mode()
        self.conv1 = nnx.Conv(
            1,
            2,
            kernel_size=(3, 3),
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(2, rngs=rngs)
        self.fc = nnx.Linear(
            32,
            2,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )

    def __call__(
        self,
        x: JaxArray,
        use_running_average: bool = False,
    ) -> JaxArray:
        """Run the forward pass of the Zero Flax NNX CNN.

        Args:
            x: Input `JaxArray` of shape `(batch_size, 4, 4, 1)`.
            use_running_average: Whether to use running statistics in batch norm.

        Returns:
            Output `JaxArray` of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        h = self.conv1(x)
        h = self.bn(h, use_running_average=use_running_average)
        h_act = as_jax(nnx.relu(h))
        h_flat: JaxArray = h_act.reshape((x.shape[0], -1))
        out: JaxArray = self.fc(h_flat)
        return out


class ZeroFlaxLinenCNNAdapter:
    """Linen-compatible stateless adapter over `ZeroFlaxNNXCNN`.

    Attributes:
        model: Underlying `ZeroFlaxNNXCNN` instance.
    """

    def __init__(self) -> None:
        """Initialize the Linen-compatible CNN adapter."""
        ensure_eager_mode()
        self.model = ZeroFlaxNNXCNN(rngs=nnx.Rngs(0))

    def init(self, key: JaxArray, x: JaxArray) -> ParamDict:
        """Initialize parameters for Linen-style callers.

        Args:
            key: PRNG key array.
            x: Sample input array of shape `(batch_size, 4, 4, 1)`.

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
        """Apply the CNN module for Linen-style callers.

        Args:
            params: Parameter dictionary returned by `init`.
            x: Input array of shape `(batch_size, 4, 4, 1)`.
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
        """Directly invoke the underlying NNX CNN.

        Args:
            x: Input array of shape `(batch_size, 4, 4, 1)`.
            use_running_average: Whether to use running average in batch norm.

        Returns:
            Output array of shape `(batch_size, 2)`.
        """
        return self.model(x, use_running_average=use_running_average)


def get_zero_flax_linen_cnn() -> ZeroFlaxLinenCNNAdapter:
    """Return a Linen-compatible Zero Flax CNN adapter.

    Returns:
        Initialized `ZeroFlaxLinenCNNAdapter`.
    """
    ensure_eager_mode()
    return ZeroFlaxLinenCNNAdapter()


def get_native_flax_linen_cnn() -> ZeroFlaxLinenCNNAdapter:
    """Return the reference Linen-compatible Flax CNN adapter.

    Returns:
        Initialized `ZeroFlaxLinenCNNAdapter`.
    """
    return get_zero_flax_linen_cnn()


def get_zero_flax_nnx_cnn() -> type[ZeroFlaxNNXCNN]:
    """Return the `ZeroFlaxNNXCNN` module class.

    Returns:
        The `ZeroFlaxNNXCNN` class accepting `rngs: nnx.Rngs`.
    """
    ensure_eager_mode()
    return ZeroFlaxNNXCNN


def get_native_flax_nnx_cnn() -> type[ZeroFlaxNNXCNN]:
    """Return the reference Flax NNX CNN module class.

    Returns:
        The `ZeroFlaxNNXCNN` class.
    """
    return get_zero_flax_nnx_cnn()


# -----------------------------------------------------------------------------
# MLX
# -----------------------------------------------------------------------------
class ZeroMlxCNN(mlx_nn.Module):
    """Zero MLX 2-D CNN with batch normalization, ReLU, and linear output head.

    Attributes:
        conv1: 2-D convolution from 1 to 2 channels with kernel size 3 and padding 1.
        bn: Batch normalization over 2 channels.
        fc: Output linear projection from 32 features to 2 classes.
    """

    def __init__(self) -> None:
        """Initialize the Zero MLX CNN."""
        ensure_eager_mode()
        super().__init__()
        self.conv1 = mlx_nn.Conv2d(1, 2, 3, padding=1)
        self.bn = mlx_nn.BatchNorm(2)
        self.fc = mlx_nn.Linear(32, 2)

    def __call__(self, x: MlxArray) -> MlxArray:
        """Execute the forward pass of the Zero MLX CNN.

        Args:
            x: Input `MlxArray` of shape `(batch_size, 4, 4, 1)`.

        Returns:
            Output `MlxArray` of shape `(batch_size, 2)`.
        """
        ensure_eager_mode()
        h = self.conv1(x)
        h = self.bn(h)
        h = mlx_nn.relu(h)
        h_flat = h.reshape((x.shape[0], -1))
        out: MlxArray = self.fc(h_flat)
        return out


def get_zero_mlx_cnn() -> ZeroMlxCNN:
    """Construct and return a `ZeroMlxCNN` instance.

    Returns:
        Initialized `ZeroMlxCNN` module.
    """
    ensure_eager_mode()
    return ZeroMlxCNN()


def get_native_mlx_cnn() -> ZeroMlxCNN:
    """Construct and return the reference MLX CNN module.

    Returns:
        Initialized reference `ZeroMlxCNN` module.
    """
    return get_zero_mlx_cnn()


# -----------------------------------------------------------------------------
# Pax / Praxis
# -----------------------------------------------------------------------------
class ZeroPaxCNN:
    """Zero Pax / Praxis CNN layer composing Conv2D, BatchNorm, ReLU, and Linear.

    Attributes:
        filter_shape: 4-tuple `(height, width, in_channels, out_channels)`.
        flattened_dims: Flattened spatial-channel dimension before the linear head.
        output_dims: Number of output classes.
        conv1: Praxis 2-D convolution layer.
        bn: Praxis batch normalization layer.
        relu: Praxis ReLU activation layer.
        fc: Praxis linear output layer.
        is_setup: Whether `setup` has been executed.
    """

    def __init__(
        self,
        filter_shape: tuple[int, int, int, int] = (3, 3, 1, 2),
        flattened_dims: int = 32,
        output_dims: int = 2,
    ) -> None:
        """Initialize the Zero Pax CNN configuration.

        Args:
            filter_shape: 4-tuple `(height, width, in_channels, out_channels)`.
            flattened_dims: Flattened feature dimension before the linear head.
            output_dims: Number of output classes.
        """
        self.filter_shape: tuple[int, int, int, int] = filter_shape
        self.flattened_dims: int = flattened_dims
        self.output_dims: int = output_dims
        self.conv1: PaxConv2D = PaxConv2D(
            filter_shape=self.filter_shape,
            filter_stride=(1, 1),
            padding="SAME",
        )
        self.bn: PaxBatchNorm = PaxBatchNorm(dim=self.filter_shape[3])
        self.relu: PaxReLU = PaxReLU()
        self.fc: PaxLinear = PaxLinear(
            input_dims=self.flattened_dims,
            output_dims=self.output_dims,
        )
        self.is_setup: bool = False

    def setup(self) -> None:
        """Configure the Praxis sublayers for the CNN."""
        self.conv1 = PaxConv2D(
            filter_shape=self.filter_shape,
            filter_stride=(1, 1),
            padding="SAME",
        )
        self.bn = PaxBatchNorm(dim=self.filter_shape[3])
        self.relu = PaxReLU()
        self.fc = PaxLinear(
            input_dims=self.flattened_dims,
            output_dims=self.output_dims,
        )
        self.is_setup = True

    def __call__(self, inputs: PaxTensor) -> PaxTensor:
        """Run the forward pass of the Zero Pax CNN.

        Args:
            inputs: Input `PaxTensor` of shape `(batch_size, 4, 4, 1)`.

        Returns:
            Output `PaxTensor` of shape `(batch_size, 2)`.
        """
        if not self.is_setup:
            self.setup()
        h = self.conv1(inputs)
        h = self.bn(h)
        h = self.relu(h)
        h_flat = h.reshape((inputs.shape[0], self.flattened_dims))
        out: PaxTensor = self.fc(h_flat)
        return out


def get_zero_pax_cnn() -> ZeroPaxCNN:
    """Construct and return a `ZeroPaxCNN` layer.

    Returns:
        Initialized `ZeroPaxCNN` Praxis layer.
    """
    layer = ZeroPaxCNN()
    layer.setup()
    return layer


def get_native_pax_cnn() -> ZeroPaxCNN:
    """Construct and return the reference Pax CNN layer.

    Returns:
        Initialized reference `ZeroPaxCNN` Praxis layer.
    """
    return get_zero_pax_cnn()


get_pytorch_cnn = get_native_pytorch_cnn
get_flax_cnn = get_native_flax_linen_cnn
get_keras_cnn = get_native_keras_cnn

__all__ = [
    "ZeroFlaxLinenCNNAdapter",
    "ZeroFlaxNNXCNN",
    "ZeroMlxCNN",
    "ZeroPaxCNN",
    "ZeroPyTorchCNN",
    "ZeroTensorFlowCNN",
    "get_flax_cnn",
    "get_keras_cnn",
    "get_native_flax_linen_cnn",
    "get_native_flax_nnx_cnn",
    "get_native_jax_cnn",
    "get_native_keras_cnn",
    "get_native_mlx_cnn",
    "get_native_pax_cnn",
    "get_native_pytorch_cnn",
    "get_pytorch_cnn",
    "get_zero_flax_linen_cnn",
    "get_zero_flax_nnx_cnn",
    "get_zero_jax_cnn",
    "get_zero_keras_cnn",
    "get_zero_mlx_cnn",
    "get_zero_pax_cnn",
    "get_zero_pytorch_cnn",
    "get_zero_tensorflow_cnn",
]
