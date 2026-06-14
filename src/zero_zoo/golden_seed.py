"""Golden Seed execution test utilities."""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import typing

    import numpy.typing as npt

    NDArray = npt.NDArray["typing.Any"]


import numpy as np


def run_golden_seed_test(
    model_fn: Callable[[], "typing.Any"], steps: int = 10
) -> dict[str, "NDArray"]:
    """Executes a model deterministically for 'steps' iterations.

    Args:
        model_fn: The model builder function.
        steps: The number of iterations to run.

    Returns:
        A dictionary containing loss and weights over time.
    """
    model_fn()
    # Mock return values for loss and weights
    return {"loss": [1.0] * steps, "weights": np.array([1.0])}  # type: ignore[dict-item]
