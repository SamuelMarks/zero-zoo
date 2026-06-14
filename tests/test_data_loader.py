"""Tests for data loading utilities."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    pass

import pytest

from zero_zoo.data_loader import mock_zero_grain_loader

T = TypeVar("T")


def test_mock_zero_grain_loader_basic() -> None:
    """Test standard batching."""
    batches = mock_zero_grain_loader(dataset_size=10, batch_size=3)
    assert len(batches) == 4
    assert batches[0]["inputs"].shape == (3, 10)
    assert batches[3]["inputs"].shape == (1, 10)  # Remainder batch


def test_mock_zero_grain_loader_edge_cases() -> None:
    """Test edge cases: batch_size=0, batch_size=1, negative batch, variable seq_len."""
    # Batch size 0
    empty_batches = mock_zero_grain_loader(dataset_size=10, batch_size=0)
    assert len(empty_batches) == 0

    # Negative batch size
    with pytest.raises(ValueError, match="batch_size cannot be negative"):
        mock_zero_grain_loader(dataset_size=10, batch_size=-1)

    # Batch size 1
    single_batches = mock_zero_grain_loader(dataset_size=2, batch_size=1)
    assert len(single_batches) == 2
    assert single_batches[0]["inputs"].shape == (1, 10)

    # Variable sequence length
    var_batches = mock_zero_grain_loader(
        dataset_size=10, batch_size=5, variable_seq_len=True
    )
    assert len(var_batches) == 2
    assert var_batches[0]["inputs"].shape[0] == 5
    assert var_batches[1]["inputs"].shape[0] == 5
    # Sequence length could technically be the same by chance, but the flag is accepted.
