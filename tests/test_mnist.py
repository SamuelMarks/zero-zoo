"""Tests for MNIST."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    pass

from zero_zoo.mnist import (
    t_mnist_edge_cases,
    train_mnist_cnn,
    train_mnist_mlp,
    verify_deterministic_training,
)

T = TypeVar("T")


def test_mnist() -> None:
    """Test MNIST functions."""
    assert train_mnist_mlp()
    assert train_mnist_cnn()
    assert t_mnist_edge_cases()
    assert verify_deterministic_training()
