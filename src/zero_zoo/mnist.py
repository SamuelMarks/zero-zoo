"""MNIST end-to-end training and determinism verification across `zero-*` libraries."""

from __future__ import annotations

import tempfile
from pathlib import Path

import zero_chex
import zero_jax
import zero_jax.numpy as jnp
import zero_optax
import zero_orbax.checkpoint as ocp
from zero_jax import random

from ._types import (
    JaxArray,
    JaxPyTree,
    ParamDict,
    as_jax,
    as_restored_jax,
    ensure_eager_mode,
    scalar_float,
    to_compiler_tensor,
)
from .cnn import get_zero_jax_cnn
from .data_gen import generate_random_tensor
from .data_loader import mock_zero_grain_loader
from .matrix_runner import check_equivalence
from .mlp import get_zero_jax_mlp


def _run_mlp_training_trajectory(steps: int, seed: int) -> tuple[JaxArray, ParamDict]:
    """Execute a deterministic MLP training trajectory with `zero_optax`.

    Args:
        steps: Number of gradient descent steps to execute.
        seed: Random seed for parameter initialization and synthetic data loading.

    Returns:
        Tuple `(loss_history, final_params)` containing the 1-D loss array and
        updated parameter dictionary.
    """
    ensure_eager_mode()
    init_fn, forward_fn = get_zero_jax_mlp()
    key = as_jax(random.PRNGKey(seed))
    params = init_fn(key)
    batches = mock_zero_grain_loader(dataset_size=8, batch_size=4, seed=seed)
    batch_x = batches[0]["inputs"][:, :2]
    target_y = as_jax(jnp.zeros((4, 2)))
    optimizer = zero_optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(params["w2"])

    def loss_fn(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
        """Compute mean squared error loss wrt output layer weights `w2`.

        Args:
            *args: Positional PyTree arguments where `args[0]` is `w2`.
            **kwargs: Unused keyword PyTree arguments.

        Returns:
            Scalar mean squared error `JaxArray`.
        """
        _ = kwargs
        w2 = as_jax(args[0])
        step_params: ParamDict = {
            "w1": params["w1"],
            "b1": params["b1"],
            "w2": w2,
            "b2": params["b2"],
            "bn_gamma": params["bn_gamma"],
            "bn_beta": params["bn_beta"],
        }
        preds = forward_fn(step_params, batch_x, is_training=True)
        diff = preds - target_y
        return (diff * diff).mean()

    grad_fn = zero_jax.value_and_grad(loss_fn)
    losses: list[float] = []
    w2_cur = params["w2"]
    for _ in range(steps):
        loss_val_raw, grads_raw = grad_fn(w2_cur)
        losses.append(scalar_float(as_jax(loss_val_raw)))
        grads = as_jax(grads_raw)
        updates, opt_state = optimizer.update(grads, opt_state, w2_cur)
        w2_cur = w2_cur + as_restored_jax(updates)

    params["w2"] = w2_cur
    zero_chex.assert_tree_all_finite(params)
    return as_jax(jnp.asarray(losses)), params


def train_mnist_mlp(steps: int = 3, seed: int = 42) -> bool:
    """Run an end-to-end training loop for the Zero JAX MLP on synthetic MNIST batches.

    Args:
        steps: Number of training iterations.
        seed: Random seed for reproducibility.

    Returns:
        `True` if training completes with finite parameters and decreasing loss.
    """
    loss_curve, final_params = _run_mlp_training_trajectory(steps=steps, seed=seed)
    zero_chex.assert_tree_all_finite(loss_curve)
    zero_chex.assert_tree_all_finite(final_params)
    first_loss = scalar_float(loss_curve[0])
    last_loss = scalar_float(loss_curve[steps - 1])
    return loss_curve.shape == (steps,) and last_loss <= first_loss


def train_mnist_cnn(steps: int = 3, seed: int = 42) -> bool:
    """Run an end-to-end training loop for the Zero JAX CNN on 4-D image batches.

    Args:
        steps: Number of training iterations.
        seed: Random seed for reproducibility.

    Returns:
        `True` if CNN training completes with finite updated weights.
    """
    ensure_eager_mode()
    init_fn, forward_fn = get_zero_jax_cnn()
    key = as_jax(random.PRNGKey(seed))
    params = init_fn(key)
    images: JaxArray = generate_random_tensor((4, 4, 4, 1), seed=seed)
    targets: JaxArray = as_jax(jnp.zeros((4, 2)))
    optimizer = zero_optax.sgd(learning_rate=0.02)
    opt_state = optimizer.init(params["w2"])

    def cnn_loss(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
        """Compute CNN mean squared error wrt dense weights `w2`.

        Args:
            *args: Positional PyTree arguments where `args[0]` is `w2`.
            **kwargs: Unused keyword PyTree arguments.

        Returns:
            Scalar loss `JaxArray`.
        """
        _ = kwargs
        w2 = as_jax(args[0])
        step_params: ParamDict = {
            "w1": params["w1"],
            "b1": params["b1"],
            "w2": w2,
            "b2": params["b2"],
            "bn_gamma": params["bn_gamma"],
            "bn_beta": params["bn_beta"],
        }
        preds = forward_fn(step_params, images, is_training=True)
        diff = preds - targets
        return (diff * diff).mean()

    grad_fn = zero_jax.value_and_grad(cnn_loss)
    w2_cur = params["w2"]
    for _ in range(steps):
        loss_val_raw, grads_raw = grad_fn(w2_cur)
        zero_chex.assert_tree_all_finite(as_jax(loss_val_raw))
        grads = as_jax(grads_raw)
        updates, opt_state = optimizer.update(grads, opt_state, w2_cur)
        w2_cur = w2_cur + as_restored_jax(updates)

    params["w2"] = w2_cur
    zero_chex.assert_tree_all_finite(params)
    return True


def t_mnist_edge_cases() -> bool:
    """Verify MNIST training edge cases: empty, single-item, and NaN inputs.

    Returns:
        `True` if all edge-case checks behave as expected.
    """
    ensure_eager_mode()
    empty_batches = mock_zero_grain_loader(dataset_size=4, batch_size=0)
    single_batches = mock_zero_grain_loader(dataset_size=2, batch_size=1)
    init_fn, forward_fn = get_zero_jax_mlp()
    params = init_fn(as_jax(random.PRNGKey(0)))
    single_x = single_batches[0]["inputs"][:, :2]
    single_out = forward_fn(params, single_x, is_training=False)
    nan_input = as_jax(jnp.asarray([[float("nan"), 1.0]]))
    nan_out = forward_fn(params, nan_input, is_training=False)
    return (
        len(empty_batches) == 0
        and single_out.shape == (1, 2)
        and bool(jnp.any(jnp.isnan(nan_out)))
    )


def verify_deterministic_training(seed: int = 42) -> bool:
    """Verify that two identically seeded training runs match identically.

    Also verifies checkpoint persistence and restoration via `zero_orbax.checkpoint`.

    Args:
        seed: Random seed for both training runs.

    Returns:
        `True` if loss curves and checkpointed parameters match identically.
    """
    loss_a, params_a = _run_mlp_training_trajectory(steps=3, seed=seed)
    loss_b, params_b = _run_mlp_training_trajectory(steps=3, seed=seed)
    zero_chex.assert_trees_all_close(params_a, params_b)
    w2_ckpt: JaxArray = params_a["w2"]
    checkpointer = ocp.PyTreeCheckpointer()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "mnist_ckpt"
        checkpointer.save(str(ckpt_dir), to_compiler_tensor(w2_ckpt))
        restored = as_restored_jax(checkpointer.restore(str(ckpt_dir)))
        zero_chex.assert_trees_all_close(w2_ckpt, restored)
    return check_equivalence(loss_a, loss_b)


__all__ = [
    "t_mnist_edge_cases",
    "train_mnist_cnn",
    "train_mnist_mlp",
    "verify_deterministic_training",
]
