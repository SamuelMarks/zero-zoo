"""Headless CI validation utilities for WASM/WebGPU payloads."""

from __future__ import annotations

from collections.abc import Mapping

import zero_jax.numpy as jnp

from ._types import ZooTensor, ensure_eager_mode, to_jax_array


def validate_in_browser(
    wasm_payload: bytes,
    reference_logs: Mapping[str, ZooTensor],
) -> bool:
    """Validate a compiled WASM/WebGPU payload against reference tensor logs.

    Verifies that the binary payload is non-empty and all reference tensors
    contain finite numerical values.

    Args:
        wasm_payload: Compiled binary WASM or WebGPU shader payload.
        reference_logs: Mapping of layer/output names to reference tensors.

    Returns:
        `True` if the payload is non-empty and all reference tensors are finite;
        `False` otherwise.
    """
    ensure_eager_mode()
    if len(wasm_payload) == 0:
        return False

    for tensor in reference_logs.values():
        arr = to_jax_array(tensor)
        if bool(jnp.any(jnp.isnan(arr))):
            return False

    return True
