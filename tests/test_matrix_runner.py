"""Tests for matrix testing harness."""

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    pass


from zero_zoo.matrix_runner import check_equivalence, simulate_stateful_run

T = TypeVar("T")


def test_check_equivalence_basic() -> None:
    """Test basic equivalence."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.00001, 2.0, 3.0])
    c = np.array([1.1, 2.0, 3.0])

    assert check_equivalence(a, b) is True
    assert check_equivalence(a, c) is False


def test_check_equivalence_shapes() -> None:
    """Test shape mismatch."""
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    assert check_equivalence(a, b) is False


def test_check_equivalence_nans() -> None:
    """Test NaN and Inf propagation parity."""
    # Matched NaNs
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    assert check_equivalence(a, b) is True

    # Mismatched NaNs
    c = np.array([1.0, 2.0, np.nan])
    assert check_equivalence(a, c) is False

    # Matched Infs
    d = np.array([np.inf, 2.0])
    e = np.array([np.inf, 2.0])
    assert check_equivalence(d, e) is True

    # Mismatched Infs
    f = np.array([1.0, np.inf])
    assert check_equivalence(d, f) is False

    # All NaNs
    g = np.array([np.nan, np.nan])
    h = np.array([np.nan, np.nan])
    assert check_equivalence(g, h) is True


def test_simulate_stateful_run() -> None:
    """Test stateful simulation guarantees random state isolation."""
    run1 = simulate_stateful_run(42)
    run2 = simulate_stateful_run(42)
    run3 = simulate_stateful_run(99)

    assert np.allclose(run1, run2)
    assert not np.allclose(run1, run3)
