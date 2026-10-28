"""Data loading utilities bridging `zero-grain`."""

from __future__ import annotations

import zero_grain
import zero_jax.numpy as jnp

from ._types import JaxArray, as_jax, ensure_eager_mode, scalar_float
from .data_gen import generate_random_tensor


class SyntheticSequenceDataSource(
    zero_grain.RandomAccessDataSource[dict[str, JaxArray]]
):
    """Deterministic in-memory data source backed by `zero_grain`.

    Attributes:
        dataset_size: Total number of items in the dataset.
        seq_len: Sequence length for each generated input vector.
        seed: Base random seed for deterministic sample generation.
    """

    def __init__(self, dataset_size: int, seq_len: int = 10, seed: int = 42) -> None:
        """Initialize the synthetic sequence data source.

        Args:
            dataset_size: Total number of records in the dataset.
            seq_len: Length of each sequence record.
            seed: Random seed for deterministic generation.
        """
        self.dataset_size: int = dataset_size
        self.seq_len: int = seq_len
        self.seed: int = seed

    def __len__(self) -> int:
        """Return the number of records in the data source.

        Returns:
            Total record count.
        """
        return self.dataset_size

    def __getitem__(self, record_key: int) -> dict[str, JaxArray]:
        """Retrieve a deterministic record at `record_key`.

        Args:
            record_key: Integer index of the record to fetch.

        Returns:
            Dictionary with `"inputs"` and `"targets"` `JaxArray` entries.
        """
        ensure_eager_mode()
        inputs: JaxArray = generate_random_tensor(
            (self.seq_len,),
            seed=self.seed + record_key,
        )
        targets: JaxArray = as_jax(jnp.asarray(record_key % 2))
        return {"inputs": inputs, "targets": targets}


def mock_zero_grain_loader(
    dataset_size: int,
    batch_size: int,
    variable_seq_len: bool = False,
    seed: int = 42,
) -> list[dict[str, JaxArray]]:
    """Load deterministic batches of synthetic data through `zero-grain`.

    Args:
        dataset_size: Total number of items in the dataset.
        batch_size: Number of items per batch.
        variable_seq_len: Whether to vary sequence lengths across batches.
        seed: Random seed for deterministic sequence length and value generation.

    Returns:
        List of batch dictionaries containing `"inputs"` and `"targets"` arrays.

    Raises:
        ValueError: If `batch_size` is negative.
    """
    ensure_eager_mode()
    if batch_size < 0:
        raise ValueError("batch_size cannot be negative.")

    if batch_size == 0 or dataset_size <= 0:
        return []

    source = SyntheticSequenceDataSource(
        dataset_size=dataset_size,
        seq_len=10,
        seed=seed,
    )
    dataset = zero_grain.MapDataset.source(source)
    total_records = len(dataset)

    batches: list[dict[str, JaxArray]] = []
    for batch_idx, start_idx in enumerate(range(0, total_records, batch_size)):
        current_batch_size = min(batch_size, total_records - start_idx)
        seq_len = 5 + ((seed + batch_idx * 3) % 10) if variable_seq_len else 10
        batch_inputs: JaxArray = generate_random_tensor(
            (current_batch_size, seq_len),
            seed=seed + batch_idx,
        )
        target_vals = [
            int(scalar_float(source[idx]["targets"]))
            for idx in range(start_idx, start_idx + current_batch_size)
        ]
        batch_targets: JaxArray = as_jax(jnp.asarray(target_vals))
        batches.append({"inputs": batch_inputs, "targets": batch_targets})

    return batches
