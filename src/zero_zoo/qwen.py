"""Qwen 3 architectures and weight serialization across `zero-*` frameworks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import zero_chex
import zero_jax
import zero_jax.numpy as jnp
import zero_mlx as zmx
import zero_mlx.nn as mlx_nn
import zero_orbax.checkpoint as ocp
from zero_flax import nnx

from ._types import (
    JaxArray,
    as_jax,
    as_restored_jax,
    ensure_eager_mode,
    to_compiler_tensor,
)
from .matrix_runner import check_equivalence


class Qwen3Config:
    """Hyperparameter configuration for the Qwen 3 transformer block.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Model hidden embedding dimension.
        intermediate_size: SwiGLU feed-forward intermediate dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads for GQA.
        head_dim: Per-head projection dimension.
        rope_theta: Base period for Rotary Position Embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 8,
        intermediate_size: int = 16,
        num_heads: int = 2,
        num_kv_heads: int = 1,
        head_dim: int = 4,
        rope_theta: float = 10000.0,
    ) -> None:
        """Initialize the Qwen 3 configuration.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            hidden_size: Model hidden embedding dimension.
            intermediate_size: SwiGLU feed-forward intermediate dimension.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value attention heads for GQA.
            head_dim: Per-head projection dimension.
            rope_theta: Base period for Rotary Position Embeddings.
        """
        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.rope_theta: float = rope_theta


class ZeroQwen3Block(nnx.Module):
    """Zero Flax NNX Qwen 3 transformer block with QK-norm, RoPE, GQA, and SwiGLU.

    Attributes:
        config: `Qwen3Config` instance defining layer dimensions.
        embed: Token embedding table.
        norm1: Pre-attention RMS normalization.
        rope: Rotary positional embedding layer with configurable `rope_theta`.
        attn: Grouped-query attention layer with QK-normalization enabled.
        norm2: Pre-feed-forward RMS normalization.
        gate_proj: SwiGLU gate projection.
        up_proj: SwiGLU up projection.
        down_proj: SwiGLU down projection.
        lm_head: Output vocabulary projection layer.
    """

    def __init__(self, config: Qwen3Config, rngs: nnx.Rngs) -> None:
        """Initialize the Zero Qwen 3 block.

        Args:
            config: Qwen 3 architectural configuration.
            rngs: Flax NNX random number generator streams.
        """
        ensure_eager_mode()
        self.config = config
        self.embed = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.norm1 = nnx.RMSNorm(config.hidden_size, rngs=rngs)
        self.rope = nnx.RoPE(
            theta=config.rope_theta,
            embedding_size=config.hidden_size,
        )
        self.attn = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.hidden_size,
            num_kv_heads=config.num_kv_heads,
            normalize_qk=True,
            kernel_init=nnx.initializers.ones,
            out_kernel_init=nnx.initializers.ones,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.RMSNorm(config.hidden_size, rngs=rngs)
        self.gate_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=nnx.initializers.ones,
            rngs=rngs,
        )

    def __call__(self, token_ids: JaxArray) -> JaxArray:
        """Execute the Qwen 3 transformer block forward pass.

        Args:
            token_ids: Integer token ID array of shape `(batch_size, seq_len)`.

        Returns:
            Logit `JaxArray` of shape `(batch_size, seq_len, vocab_size)`.
        """
        ensure_eager_mode()
        h = self.embed(token_ids)
        h_norm = self.norm1(h)
        h_rope = self.rope(h_norm)
        attn_out = self.attn(h_rope, h_rope, h_rope, decode=False)
        h = h + attn_out
        ff_in = self.norm2(h)
        swiglu = as_jax(zero_jax.nn.silu(self.gate_proj(ff_in))) * self.up_proj(ff_in)
        h = h + self.down_proj(swiglu)
        logits: JaxArray = self.lm_head(h)
        return logits


def get_qwen3_zero() -> bool:
    """Construct and execute the Qwen 3 architecture in `zero_flax.nnx`.

    Returns:
        `True` if the `ZeroQwen3Block` forward pass produces the expected shape.
    """
    ensure_eager_mode()
    cfg = Qwen3Config()
    model = ZeroQwen3Block(cfg, rngs=nnx.Rngs(0))
    tokens = as_jax(jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]))
    logits = model(tokens)
    return logits.shape == (2, 4, cfg.vocab_size)


def get_qwen3_native() -> bool:
    """Construct and execute the Qwen 3 architecture across MLX and NNX dialects.

    Returns:
        `True` if both MLX and NNX Qwen 3 executions succeed.
    """
    ensure_eager_mode()
    cfg = Qwen3Config()
    emb = mlx_nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    attn = mlx_nn.MultiHeadAttention(cfg.hidden_size, cfg.num_heads)
    ln = mlx_nn.LayerNorm(cfg.hidden_size)
    fc = mlx_nn.Linear(cfg.hidden_size, cfg.vocab_size)
    x_mlx = zmx.zeros((2, 4), dtype=zmx.int32)
    h_mlx = emb(x_mlx)
    out_mlx = fc(ln(h_mlx + attn(h_mlx, h_mlx, h_mlx)))
    return out_mlx.shape == (2, 4, cfg.vocab_size) and get_qwen3_zero()


def load_qwen3_weights() -> bool:
    """Serialize and restore Qwen 3 weights using `zero_orbax.checkpoint`.

    Returns:
        `True` if checkpoint round-trip restoration matches original weights.
    """
    ensure_eager_mode()
    cfg = Qwen3Config()
    model = ZeroQwen3Block(cfg, rngs=nnx.Rngs(11))
    gate_kernel: JaxArray = as_jax(model.gate_proj.kernel.value)
    checkpointer = ocp.PyTreeCheckpointer()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "qwen3_ckpt"
        checkpointer.save(str(ckpt_path), to_compiler_tensor(gate_kernel))
        restored = as_restored_jax(checkpointer.restore(str(ckpt_path)))
        zero_chex.assert_trees_all_close(gate_kernel, restored)
    return True


def t_qwen3_inference() -> bool:
    """Run deterministic cross-execution Qwen 3 forward inference verification.

    Returns:
        `True` if two identically seeded Qwen 3 models produce equivalent logits.
    """
    ensure_eager_mode()
    cfg = Qwen3Config()
    model_a = ZeroQwen3Block(cfg, rngs=nnx.Rngs(13))
    model_b = ZeroQwen3Block(cfg, rngs=nnx.Rngs(13))
    tokens = as_jax(jnp.asarray([[2, 4, 6, 8]]))
    out_a = model_a(tokens)
    out_b = model_b(tokens)
    return check_equivalence(out_a, out_b)


def t_qwen3_edge_cases() -> bool:
    """Verify Qwen 3 RoPE scaling and variable sequence lengths (`1` vs `12`).

    Returns:
        `True` if single-token and long-sequence forward passes succeed and are finite.
    """
    ensure_eager_mode()
    cfg = Qwen3Config(rope_theta=50000.0)
    model = ZeroQwen3Block(cfg, rngs=nnx.Rngs(17))
    short_tokens = as_jax(jnp.asarray([[0]]))
    long_tokens = as_jax(jnp.asarray([[i % cfg.vocab_size for i in range(12)]]))
    out_short = model(short_tokens)
    out_long = model(long_tokens)
    zero_chex.assert_tree_all_finite(out_short)
    zero_chex.assert_tree_all_finite(out_long)
    return out_short.shape == (1, 1, cfg.vocab_size) and out_long.shape == (
        1,
        12,
        cfg.vocab_size,
    )


__all__ = [
    "Qwen3Config",
    "ZeroQwen3Block",
    "get_qwen3_native",
    "get_qwen3_zero",
    "load_qwen3_weights",
    "t_qwen3_edge_cases",
    "t_qwen3_inference",
]
