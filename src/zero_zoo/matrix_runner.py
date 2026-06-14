"""Matrix testing harness and equivalence checking utilities."""

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


def check_equivalence(
    output_a: "NDArray", output_b: "NDArray", rtol: float = 1e-4, atol: float = 1e-4
) -> bool:
    """Check mathematical equivalence between two outputs.

    Handles basic precision differences across backends and NaN parity.

    Args:
        output_a: First output tensor (mocked as numpy array).
        output_b: Second output tensor (mocked as numpy array).
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if outputs are equivalent, False otherwise.
    """
    a = np.asarray(output_a)
    b = np.asarray(output_b)

    if a.shape != b.shape:
        return False

    # Edge Case: Test NaN/Inf propagation parity
    # If one has NaNs, the other MUST have NaNs in the exact same spots
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return False

    if np.any(np.isinf(a)) or np.any(np.isinf(b)):
        if not np.array_equal(np.isinf(a), np.isinf(b)):
            return False

    # If both are entirely NaNs/Infs and parity matched above,
    # they are "equivalent" in this context
    if np.all(np.isnan(a)) or np.all(np.isinf(a)):
        return True

    # Filter out NaNs for the allclose check (since NaN != NaN normally)
    mask = ~(np.isnan(a) | np.isinf(a))

    return bool(np.allclose(a[mask], b[mask], rtol=rtol, atol=atol))


def simulate_stateful_run(seed: int = 42) -> "NDArray":
    """Simulate a stateful model run to verify random state isolation.

    Args:
        seed: Random seed.

    Returns:
        A generated state array.
    """
    rng = np.random.default_rng(seed)
    return rng.random((5, 5))
