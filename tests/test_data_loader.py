"""Tests for data loading utilities bridging `zero-grain`."""

from __future__ import annotations

import pytest

from zero_zoo.data_loader import SyntheticSequenceDataSource, mock_zero_grain_loader


def test_mock_zero_grain_loader_basic() -> None:
    """Test basic deterministic batching behavior of `mock_zero_grain_loader`."""
    source = SyntheticSequenceDataSource(dataset_size=4, seq_len=6, seed=10)
    assert len(source) == 4
    sample = source[0]
    assert sample["inputs"].shape == (6,)

    batches = mock_zero_grain_loader(dataset_size=10, batch_size=3)
    assert len(batches) == 4
    assert batches[0]["inputs"].shape == (3, 10)
    assert batches[0]["targets"].shape == (3,)
    assert batches[3]["inputs"].shape == (1, 10)
    assert batches[3]["targets"].shape == (1,)


def test_mock_zero_grain_loader_edge_cases() -> None:
    """Test edge cases in `mock_zero_grain_loader`."""
    with pytest.raises(ValueError, match="batch_size cannot be negative."):
        mock_zero_grain_loader(dataset_size=10, batch_size=-1)

    assert mock_zero_grain_loader(dataset_size=10, batch_size=0) == []
    assert mock_zero_grain_loader(dataset_size=0, batch_size=2) == []

    single_batches = mock_zero_grain_loader(dataset_size=3, batch_size=1)
    assert len(single_batches) == 3
    assert single_batches[0]["inputs"].shape == (1, 10)

    var_batches = mock_zero_grain_loader(
        dataset_size=10,
        batch_size=5,
        variable_seq_len=True,
    )
    assert len(var_batches) == 2
    assert var_batches[0]["inputs"].shape[0] == 5
    assert var_batches[1]["inputs"].shape[0] == 5
    assert var_batches[0]["inputs"].shape[1] != var_batches[1]["inputs"].shape[1]
