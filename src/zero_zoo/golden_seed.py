"""Golden Seed deterministic multi-step execution test utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import zero_jax
import zero_jax.numpy as jnp
import zero_optax

from ._types import (
    JaxArray,
    JaxPyTree,
    as_jax,
    as_restored_jax,
    ensure_eager_mode,
    scalar_float,
)

TModel = TypeVar("TModel")


def run_golden_seed_test(
    model_fn: Callable[[], TModel],
    steps: int = 10,
    seed: int = 42,
) -> dict[str, JaxArray]:
    """Execute a model builder and run a deterministic optimization trajectory.

    Args:
        model_fn: Zero-argument callable constructing a model architecture.
        steps: Number of optimization iterations to execute.
        seed: Random seed for parameter initialization.

    Returns:
        Dictionary containing `"loss"` (1-D `JaxArray` of length `steps`) and
        `"weights"` (updated parameter `JaxArray`).
    """
    ensure_eager_mode()
    _ = model_fn()
    optimizer = zero_optax.sgd(learning_rate=0.05)
    base_val = float((seed % 5) + 1)
    weights: JaxArray = as_jax(jnp.asarray([base_val, base_val * 0.5]))
    opt_state = optimizer.init(weights)

    def step_loss(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
        """Compute quadratic objective loss for the first positional parameter array.

        Args:
            *args: Positional PyTree arguments where `args[0]` is the weight vector.
            **kwargs: Unused keyword PyTree arguments.

        Returns:
            Scalar mean squared loss `JaxArray`.
        """
        _ = kwargs
        w = as_jax(args[0])
        return (w * w).mean()

    grad_fn = zero_jax.value_and_grad(step_loss)
    losses: list[float] = []
    for _ in range(steps):
        loss_val_raw, grads_raw = grad_fn(weights)
        losses.append(scalar_float(as_jax(loss_val_raw)))
        grads = as_jax(grads_raw)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = weights + as_restored_jax(updates)

    loss_array: JaxArray = as_jax(jnp.asarray(losses))
    return {"loss": loss_array, "weights": weights}
