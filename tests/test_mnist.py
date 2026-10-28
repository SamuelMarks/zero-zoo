"""Tests for MNIST training loops and determinism verification."""

from __future__ import annotations

from zero_zoo.mnist import (
    t_mnist_edge_cases,
    train_mnist_cnn,
    train_mnist_mlp,
    verify_deterministic_training,
)


def test_mnist() -> None:
    """Test MNIST MLP training, CNN training, edge cases, and determinism."""
    assert train_mnist_mlp()
    assert train_mnist_cnn()
    assert t_mnist_edge_cases()
    assert verify_deterministic_training()
