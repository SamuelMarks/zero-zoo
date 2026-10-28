"""Tests for top-level `zero_zoo` exports, Golden Seed, and Headless CI utilities."""

from __future__ import annotations

import zero_jax.numpy as jnp

from zero_zoo import (
    get_flax_cnn,
    get_flax_nanogpt,
    get_keras_cnn,
    get_keras_nanogpt,
    get_pytorch_cnn,
    get_pytorch_nanogpt,
    run_golden_seed_test,
    validate_in_browser,
)
from zero_zoo._types import JaxArray, as_jax, scalar_float
from zero_zoo.mlp import get_native_pytorch_mlp


def test_cnn_exports() -> None:
    """Test top-level CNN model getters."""
    assert get_pytorch_cnn() is not None
    assert get_flax_cnn() is not None
    assert get_keras_cnn() is not None


def test_nanogpt_exports() -> None:
    """Test top-level NanoGPT model getters."""
    assert get_pytorch_nanogpt() is not None
    assert get_flax_nanogpt() is not None
    assert get_keras_nanogpt() is not None


def test_golden_seed() -> None:
    """Test deterministic multi-step Golden Seed execution."""
    res = run_golden_seed_test(get_native_pytorch_mlp, steps=3, seed=42)
    assert res["loss"].shape == (3,)
    assert res["weights"].shape == (2,)
    first_loss = scalar_float(res["loss"][0])
    last_loss = scalar_float(res["loss"][2])
    assert last_loss < first_loss


def test_headless_ci() -> None:
    """Test headless CI WASM/WebGPU payload validation for valid and invalid inputs."""
    valid_logs: dict[str, JaxArray] = {"out": as_jax(jnp.asarray([1.0, 2.0]))}
    nan_logs: dict[str, JaxArray] = {"out": as_jax(jnp.asarray([1.0, float("nan")]))}

    assert validate_in_browser(b"wasm_binary", valid_logs) is True
    assert validate_in_browser(b"", valid_logs) is False
    assert validate_in_browser(b"wasm_binary", nan_logs) is False
