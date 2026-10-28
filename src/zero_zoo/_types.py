"""Canonical type definitions and shared protocols for Zero Zoo."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypedDict, TypeVar, Union

import zero_jax.numpy as jnp
import zero_orbax.checkpoint as ocp
from ml_switcheroo_compiler.core.config import config as compiler_config
from ml_switcheroo_compiler.core.tensor import Tensor as SwitcherooTensor
from zero_chex.types import ZeroArray
from zero_jax._types import ArrayLikeSpec
from zero_jax._types import PyTree as JaxPyTree
from zero_jax.numpy._ndarray import (
    _to_tensor,
)
from zero_jax.numpy._ndarray import (
    ndarray as JaxArray,
)
from zero_keras.tensors import KerasTensor
from zero_keras.types import TensorLike as KerasTensorLike
from zero_mlx.array import array as MlxArray
from zero_optax import PyTree as OptaxPyTree
from zero_orbax.types import PyTree as OrbaxPyTree
from zero_pax.types import Tensor as PaxTensor
from zero_tensorflow.core_tensor import Tensor as TfTensor
from zero_torch.tensor import Tensor as TorchTensor

ZooTensor = Union[
    JaxArray,
    TorchTensor,
    TfTensor,
    MlxArray,
    PaxTensor,
    SwitcherooTensor,
    ZeroArray,
]

ParamDict = dict[str, JaxArray]

BatchDict = dict[str, JaxArray]

TTensor = TypeVar("TTensor")


class CallableModelProtocol(Protocol[TTensor]):
    """Generic protocol for callable models mapping input tensors to outputs."""

    def __call__(self, inputs: TTensor) -> TTensor:
        """Execute the model forward pass on `inputs`.

        Args:
            inputs: Input tensor of type `TTensor`.

        Returns:
            Output tensor of type `TTensor`.
        """


class TrainStepResult(TypedDict):
    """Structured result dictionary for a single training step.

    Attributes:
        loss: Scalar loss array for the step.
        params: Updated parameter dictionary after applying gradients.
    """

    loss: JaxArray
    params: ParamDict


class GoldenSeedReport(TypedDict):
    """Structured report returned by `run_golden_seed_test`.

    Attributes:
        loss: 1-D `JaxArray` of step losses.
        weights: Final updated weight `JaxArray`.
    """

    loss: JaxArray
    weights: JaxArray


JaxInitFn = Callable[[JaxArray], ParamDict]


class JaxForwardFn(Protocol):
    """Protocol for pure JAX forward functions.

    Defines the signature for stateless forward passes across pure JAX models.
    """

    def __call__(
        self,
        params: ParamDict,
        x: JaxArray,
        is_training: bool = True,
    ) -> JaxArray:
        """Execute the forward pass of a pure JAX model.

        Args:
            params: Dictionary mapping parameter names to JAX arrays.
            x: Input tensor array.
            is_training: Whether the model is executing in training mode.

        Returns:
            Output JAX array produced by the forward pass.
        """


JaxModelPair = tuple[JaxInitFn, JaxForwardFn]

BackendName = Literal["numpy", "jax", "mlx", "cupy", "dusk", "torch"]

VALID_BACKENDS: tuple[str, ...] = ("numpy", "jax", "mlx", "cupy", "dusk", "torch")


def ensure_eager_mode() -> None:
    """Ensure the `ml-switcheroo-compiler` runtime is configured for eager execution."""
    compiler_config.eager_mode = True


def as_jax(x: ArrayLikeSpec) -> JaxArray:
    """Wrap any `zero_jax` `ArrayLikeSpec` into a concrete `JaxArray`.

    Args:
        x: Input array-like value or `zero_jax` result.

    Returns:
        Wrapped `JaxArray` instance.
    """
    return JaxArray(_to_tensor(x))


def to_compiler_tensor(x: ArrayLikeSpec) -> SwitcherooTensor:
    """Convert any `zero_jax` `ArrayLikeSpec` into a `SwitcherooTensor`.

    Args:
        x: Input `JaxArray` or `ArrayLikeSpec`.

    Returns:
        Underlying `SwitcherooTensor` instance.
    """
    return _to_tensor(x)


def as_restored_jax(
    x: OrbaxPyTree | ocp.Future | OptaxPyTree | ArrayLikeSpec,
) -> JaxArray:
    """Narrow a restored checkpoint or optimizer leaf into a `JaxArray`.

    Args:
        x: Restored checkpoint object, optimizer update leaf, or array-like value.

    Returns:
        Converted `JaxArray` instance.
    """
    if isinstance(x, JaxArray):
        return x
    if isinstance(x, SwitcherooTensor):
        return JaxArray(x)
    return as_jax(0.0)


def scalar_float(x: ArrayLikeSpec) -> float:
    """Extract a Python `float` from a 0-D or 1-element `ArrayLikeSpec`.

    Args:
        x: Scalar `JaxArray` or `ArrayLikeSpec`.

    Returns:
        Extracted Python `float` value.
    """
    return _to_tensor(x).item()


def to_jax_array(
    tensor: ZooTensor | Sequence[float] | Sequence[Sequence[float]],
) -> JaxArray:
    """Convert any supported ecosystem tensor or numeric sequence into a `JaxArray`.

    Args:
        tensor: Ecosystem tensor or nested numeric sequence to convert.

    Returns:
        Converted `JaxArray` instance with preserved shape and values.
    """
    ensure_eager_mode()
    if isinstance(tensor, JaxArray):
        return tensor
    if isinstance(tensor, SwitcherooTensor):
        return as_jax(jnp.asarray(tensor.numpy()))
    if isinstance(tensor, (TorchTensor, TfTensor, MlxArray, PaxTensor, ZeroArray)):
        return as_jax(jnp.asarray(tensor.tolist()))
    return as_jax(jnp.asarray(tensor))


__all__ = [
    "ArrayLikeSpec",
    "BackendName",
    "BatchDict",
    "CallableModelProtocol",
    "GoldenSeedReport",
    "JaxArray",
    "JaxForwardFn",
    "JaxInitFn",
    "JaxModelPair",
    "JaxPyTree",
    "KerasTensor",
    "KerasTensorLike",
    "MlxArray",
    "OptaxPyTree",
    "OrbaxPyTree",
    "ParamDict",
    "PaxTensor",
    "SwitcherooTensor",
    "TfTensor",
    "TorchTensor",
    "TrainStepResult",
    "VALID_BACKENDS",
    "ZeroArray",
    "ZooTensor",
    "as_jax",
    "as_restored_jax",
    "ensure_eager_mode",
    "scalar_float",
    "to_compiler_tensor",
    "to_jax_array",
]
