"""Data generation utilities utilizing the compiler backends."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import typing

    import numpy.typing as npt

    NDArray = npt.NDArray["typing.Any"]
    JAX_INIT_FN = typing.Callable[..., dict[str, NDArray]]
    JAX_FWD_FN = typing.Callable[..., NDArray]
    JAX_RET = tuple[JAX_INIT_FN, JAX_FWD_FN]

    ZJAX_INIT_FN = typing.Callable[..., dict[str, NDArray]]
    ZJAX_FWD_FN = typing.Callable[..., NDArray]
    ZJAX_RET = tuple[ZJAX_INIT_FN, ZJAX_FWD_FN]


def generate_random_tensor(
    shape: tuple[int, ...], seed: int = 42, backend: str = "numpy"
) -> "NDArray":
    """Generate a deterministic random tensor for a specific backend.

    Args:
        shape: The shape of the tensor to generate. Can be empty for scalars.
        seed: The deterministic seed.
        backend: The string name of the backend (e.g., 'numpy', 'torch').

    Returns:
        The generated random tensor.
    """
    if backend not in ["numpy", "jax", "mlx", "cupy", "dusk", "torch"]:
        raise ValueError(f"Unsupported backend: {backend}")

    # Edge Case: Massive Dimensionality Check
    if len(shape) > 10:
        raise ValueError("Dimensionality exceeds safety limit of 10.")

    # Note: In reality, this would call ml-switcheroo-compiler's backend ops.
    # For now, we mock the generation logic using numpy since no other 3rd party
    # dependencies are allowed without proper context,
    # and we are validating the interface.

    rng = np.random.default_rng(seed)

    # Edge Case: Scalar (empty shape)
    if len(shape) == 0:
        val = rng.random()
        return val  # type: ignore[return-value]

    # Edge Case: Empty Tensor (a dimension is 0)
    if any(dim == 0 for dim in shape):
        return np.empty(shape)

    return rng.random(shape)
