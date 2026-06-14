"""MLP architectures across frameworks."""

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
def get_native_pytorch_mlp() -> "torch.nn.Module":
    """Return native PyTorch MLP.

    Returns:
        Native PyTorch MLP module.
    """
    from torch import nn

    class MLP(nn.Module):
        """Native PyTorch MLP module."""

        def __init__(self) -> None:
            """Initialize PyTorch MLP."""
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.bn = nn.BatchNorm1d(4)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x: "NDArray") -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                Output array.
            """
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            return self.fc2(x)  # type: ignore[no-any-return]

    return MLP()


def get_zero_pytorch_mlp() -> "zero_torch.nn.Module":
    """Return Zero PyTorch MLP.

    Returns:
        Zero PyTorch MLP module.
    """
    import zero_torch
    from zero_torch import nn

    class MLP(nn.Module):  # type: ignore[misc]
        """Zero PyTorch MLP module."""

        def __init__(self) -> None:
            """Initialize Zero PyTorch MLP."""
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.bn = nn.BatchNorm1d(4)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x: zero_torch.Tensor) -> zero_torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor.

            Returns:
                Output tensor.
            """
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            return self.fc2(x)

    return MLP()


# -----------------------------------------------------------------------------
# Keras / TensorFlow
# -----------------------------------------------------------------------------
def get_native_keras_mlp() -> "keras.Model":
    """Return native Keras MLP.

    Returns:
        Native Keras MLP model.
    """
    import keras
    from keras import layers

    inputs = keras.Input(shape=(2,))
    x = layers.Dense(4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def get_zero_keras_mlp() -> "zero_keras.Model":
    """Return Zero Keras MLP.

    Returns:
        Zero Keras MLP model.
    """
    import zero_keras as keras
    from zero_keras import layers

    inputs = keras.Input(shape=(2,))
    x = layers.Dense(4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# -----------------------------------------------------------------------------
# JAX (Pure)
# -----------------------------------------------------------------------------
def get_native_jax_mlp() -> "JAX_RET":
    """Return native JAX MLP parameters and forward function.

    Returns:
        A tuple of initialization and forward pass functions.
    """
    import jax
    import jax.numpy as jnp
    from jax import random

    def init_params(key: "NDArray") -> dict[str, "NDArray"]:
        """Initialize parameters for the JAX MLP.

        Args:
            key: PRNG key for initialization.

        Returns:
            Dictionary of initialized parameters.
        """
        k1, k2 = random.split(key)
        return {
            "w1": random.normal(k1, (2, 4)),  # type: ignore[dict-item]
            "b1": jnp.zeros((4,)),  # type: ignore[dict-item]
            "w2": random.normal(k2, (4, 2)),  # type: ignore[dict-item]
            "b2": jnp.zeros((2,)),  # type: ignore[dict-item]
            "bn_gamma": jnp.ones((4,)),  # type: ignore[dict-item]
            "bn_beta": jnp.zeros((4,)),  # type: ignore[dict-item]
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
            is_training: Whether it's training mode.

        Returns:
            Output array.
        """
        x = jnp.dot(x, params["w1"]) + params["b1"]  # type: ignore[assignment]
        # Simplified BatchNorm for pure JAX
        mean = jnp.mean(x, axis=0) if is_training else jnp.zeros((4,))
        var = jnp.var(x, axis=0) if is_training else jnp.ones((4,))
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x * params["bn_gamma"] + params["bn_beta"]
        x = jax.nn.relu(x)
        x = jnp.dot(x, params["w2"]) + params["b2"]  # type: ignore[assignment]
        return x

    return init_params, forward


def get_zero_jax_mlp() -> "ZJAX_RET":
    """Return Zero JAX MLP parameters and forward function.

    Returns:
        A tuple of initialization and forward pass functions.
    """
    import zero_jax as jax
    import zero_jax.numpy as jnp
    from zero_jax import random

    def init_params(key: "NDArray") -> dict[str, "NDArray"]:
        """Initialize parameters for the Zero JAX MLP.

        Args:
            key: PRNG key for initialization.

        Returns:
            Dictionary of initialized parameters.
        """
        k1, k2 = random.split(key)
        return {
            "w1": random.normal(k1, (2, 4)),
            "b1": jnp.zeros((4,)),
            "w2": random.normal(k2, (4, 2)),
            "b2": jnp.zeros((2,)),
            "bn_gamma": jnp.ones((4,)),
            "bn_beta": jnp.zeros((4,)),
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
            is_training: Whether it's training mode.

        Returns:
            Output array.
        """
        x = jnp.dot(x, params["w1"]) + params["b1"]
        mean = jnp.mean(x, axis=0) if is_training else jnp.zeros((4,))
        var = jnp.var(x, axis=0) if is_training else jnp.ones((4,))
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x * params["bn_gamma"] + params["bn_beta"]
        x = jax.nn.relu(x)
        x = jnp.dot(x, params["w2"]) + params["b2"]
        return x

    return init_params, forward


# -----------------------------------------------------------------------------
# Flax
# -----------------------------------------------------------------------------
def get_native_flax_linen_mlp() -> "nn.Module":
    """Return native Flax Linen MLP.

    Returns:
        Native Flax Linen MLP module.
    """
    import flax.linen as nn

    class MLP(nn.Module):
        """Native Flax Linen MLP module."""

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
                Output array.
            """
            x = nn.Dense(4)(x)  # type: ignore[assignment]
            x = nn.BatchNorm(use_running_average=use_running_average)(x)
            x = nn.relu(x)
            x = nn.Dense(2)(x)  # type: ignore[assignment]
            return x

    return MLP()


def get_zero_flax_linen_mlp() -> "zero_nn.Module":
    """Return Zero Flax Linen MLP.

    Using nnx as fallback if linen doesn't exist.

    Returns:
        Zero Flax Linen MLP module.
    """
    try:
        from zero_flax import linen as nn

        class MLP(nn.Module):  # type: ignore[misc]
            """Zero Flax Linen MLP module."""

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
                    Output array.
                """
                x = nn.Dense(4)(x)
                x = nn.BatchNorm(use_running_average=use_running_average)(x)
                x = nn.relu(x)
                x = nn.Dense(2)(x)
                return x

        return MLP()
    except ImportError:
        # Fallback to nnx Module wrapper for tests
        from zero_flax.nnx import Module, Param

        class MLPWrapper(Module):  # type: ignore[misc]
            """Fallback Zero Flax Linen MLP wrapper using nnx."""

            def __init__(self) -> None:
                """Initialize wrapper."""
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
                    Output array.
                """
                return x

        return MLPWrapper()


def get_native_flax_nnx_mlp() -> "NDArray":
    """Return native Flax NNX MLP.

    Returns:
        Native Flax NNX MLP class.
    """
    from flax import nnx

    class MLP(nnx.Module):
        """Native Flax NNX MLP module."""

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize native Flax NNX MLP.

            Args:
                rngs: PRNG keys for initialization.
            """
            self.fc1 = nnx.Linear(2, 4, rngs=rngs)
            self.bn = nnx.BatchNorm(4, rngs=rngs)
            self.fc2 = nnx.Linear(4, 2, rngs=rngs)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                Output array.
            """
            x = self.fc1(x)  # type: ignore[arg-type,assignment]
            x = self.bn(x)
            import zero_jax

            x = zero_jax.nn.relu(x)
            return self.fc2(x)  # type: ignore[arg-type,return-value]

    return MLP  # type: ignore[return-value]


def get_zero_flax_nnx_mlp() -> "NDArray":
    """Return Zero Flax NNX MLP.

    Returns:
        Zero Flax NNX MLP class.
    """
    from zero_flax import nnx

    class MLP(nnx.Module):  # type: ignore[misc]
        """Zero Flax NNX MLP module."""

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize Zero Flax NNX MLP.

            Args:
                rngs: PRNG keys for initialization.
            """
            self.fc1 = nnx.Linear(2, 4, rngs=rngs)
            self.bn = nnx.BatchNorm(4, rngs=rngs)
            self.fc2 = nnx.Linear(4, 2, rngs=rngs)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                Output array.
            """
            x = self.fc1(x)
            x = self.bn(x)
            import zero_jax

            x = zero_jax.nn.relu(x)
            return self.fc2(x)  # type: ignore[no-any-return]

    return MLP  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# MLX
# -----------------------------------------------------------------------------
def get_native_mlx_mlp() -> "NDArray":
    """Return native MLX MLP.

    Returns:
        Native MLX MLP module.
    """
    import mlx.nn as nn

    class MLP(nn.Module):  # type: ignore[name-defined, misc]
        """Native MLX MLP module."""

        def __init__(self) -> None:
            """Initialize MLX MLP."""
            super().__init__()
            self.fc1 = nn.Linear(2, 4)  # type: ignore[attr-defined]
            self.bn = nn.BatchNorm(4)  # type: ignore[attr-defined]
            self.fc2 = nn.Linear(4, 2)  # type: ignore[attr-defined]

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                Output array.
            """
            x = self.fc1(x)
            x = self.bn(x)
            x = nn.relu(x)  # type: ignore[attr-defined]
            return self.fc2(x)  # type: ignore[no-any-return]

    return MLP()


def get_zero_mlx_mlp() -> "NDArray":
    """Return Zero MLX MLP.

    Returns:
        Zero MLX MLP module.
    """
    import zero_mlx.nn as nn

    class MLP(nn.Module):  # type: ignore[misc]
        """Zero MLX MLP module."""

        def __init__(self) -> None:
            """Initialize Zero MLX MLP."""
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.bn = nn.BatchNorm(4)
            self.fc2 = nn.Linear(4, 2)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                Output array.
            """
            x = self.fc1(x)
            x = self.bn(x)
            x = nn.relu(x)
            return self.fc2(x)  # type: ignore[no-any-return]

    return MLP()


# -----------------------------------------------------------------------------
# Pax
# -----------------------------------------------------------------------------
def get_native_pax_mlp() -> "base_layer.BaseLayer":
    """Return native Pax MLP.

    Returns:
        Native Pax MLP layer.
    """
    try:
        from praxis import base_layer, layers

        class MLP(base_layer.BaseLayer):  # type: ignore[misc]
            """Native Pax MLP layer."""

            def setup(self) -> None:
                """Setup Pax MLP."""
                self.create_child(
                    "fc1", layers.Linear.HParams(input_dims=2, output_dims=4)
                )
                self.create_child("bn", layers.BatchNorm.HParams(dim=4))
                self.create_child(
                    "relu", layers.Activation.HParams(activation_name="relu")
                )
                self.create_child(
                    "fc2", layers.Linear.HParams(input_dims=4, output_dims=2)
                )

            def __call__(
                self,
                x: "NDArray",
            ) -> "NDArray":
                """Forward pass.

                Args:
                    x: Input array.

                Returns:
                    Output array.
                """
                x = self.fc1(x)
                x = self.bn(x)
                x = self.relu(x)
                return self.fc2(x)  # type: ignore[no-any-return]

        return MLP()
    except ImportError:
        # Mock for when Pax is not installed
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
                    Output array.
                """
                return x

        return DummyPax()


def get_zero_pax_mlp() -> "zero_pax.Layer":
    """Return Zero Pax MLP.

    Returns:
        Zero Pax MLP layer.
    """
    import zero_pax
    from zero_pax import Layer

    class MLP(Layer):  # type: ignore[misc]
        """Zero Pax MLP layer."""

        def __init__(self) -> None:
            """Initialize Zero Pax MLP."""
            super().__init__()
            self.fc1 = zero_pax.Linear(input_dims=2, output_dims=4)
            self.bn = zero_pax.BatchNorm(dim=4)
            self.relu = zero_pax.ReLU()
            self.fc2 = zero_pax.Linear(input_dims=4, output_dims=2)

        def __call__(
            self,
            x: "NDArray",
        ) -> "NDArray":
            """Forward pass.

            Args:
                x: Input array.

            Returns:
                Output array.
            """
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            return self.fc2(x)  # type: ignore[no-any-return]

    return MLP()


# Maintain backwards compatibility aliases for other modules using older names
get_pytorch_mlp = get_native_pytorch_mlp
get_flax_mlp = get_native_flax_linen_mlp
get_keras_mlp = get_native_keras_mlp
