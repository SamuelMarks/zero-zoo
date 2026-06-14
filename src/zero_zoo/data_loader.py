"""Data loading utilities bridging zero-grain."""

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


def mock_zero_grain_loader(
    dataset_size: int, batch_size: int, variable_seq_len: bool = False
) -> list[dict[str, "NDArray"]]:
    """Mock a zero-grain data ingestion pipeline.

    Args:
        dataset_size: The total number of items in the simulated dataset.
        batch_size: The number of items per batch.
        variable_seq_len: If True, simulates batches with variable sequence lengths.

    Returns:
        A list of batches, where each batch is a dictionary
        containing inputs and targets.
    """
    if batch_size < 0:
        raise ValueError("batch_size cannot be negative.")

    # Edge Case: batch_size = 0
    if batch_size == 0:
        return []

    batches = []
    # Using a simple loop to mock batching since we aren't pulling the real grain dep.
    # Edge Case: out-of-bounds indices are inherently handled by not
    # iterating past dataset_size
    for i in range(0, dataset_size, batch_size):
        current_batch_size = min(batch_size, dataset_size - i)

        if variable_seq_len:
            seq_len = np.random.randint(5, 15)
        else:
            seq_len = 10

        batch_inputs = np.random.randn(current_batch_size, seq_len)
        batch_targets = np.random.randint(0, 2, size=(current_batch_size,))

        batches.append({"inputs": batch_inputs, "targets": batch_targets})

    return batches
