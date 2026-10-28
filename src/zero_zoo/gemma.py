"""Gemma 4 architectures and weight serialization across `zero-*` frameworks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import zero_chex
import zero_jax
import zero_jax.numpy as jnp
import zero_orbax.checkpoint as ocp
from zero_flax import nnx
from zero_pax.praxis.layers import GroupedQueryAttention as PaxGroupedQueryAttention
from zero_pax.praxis.layers import RmsNorm as PaxRmsNorm
from zero_pax.types import Tensor as PaxTensor

from ._types import (
    JaxArray,
    as_jax,
    as_restored_jax,
    ensure_eager_mode,
    to_compiler_tensor,
)
from .matrix_runner import check_equivalence


class Gemma4Config:
    """Hyperparameter configuration for the Gemma 4 transformer block.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Model hidden embedding dimension.
        intermediate_size: GeGLU feed-forward intermediate dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads for GQA.
        head_dim: Per-head projection dimension.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 8,
        intermediate_size: int = 16,
        num_heads: int = 2,
        num_kv_heads: int = 1,
        head_dim: int = 4,
    ) -> None:
        """Initialize the Gemma 4 configuration.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            hidden_size: Model hidden embedding dimension.
            intermediate_size: GeGLU feed-forward intermediate dimension.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value attention heads for GQA.
            head_dim: Per-head projection dimension.
        """
        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim


class ZeroGemma4Block(nnx.Module):
    """Zero Flax NNX Gemma 4 transformer block with RMSNorm, RoPE, GQA, and GeGLU.

    Attributes:
        config: `Gemma4Config` instance defining layer dimensions.
        embed: Token embedding table.
        norm1: Pre-attention RMS normalization.
        rope: Rotary positional embedding layer.
        attn: Grouped-query multi-head attention layer.
        norm2: Pre-feed-forward RMS normalization.
        gate_proj: GeGLU gate linear projection.
        up_proj: GeGLU up linear projection.
        down_proj: GeGLU down linear projection.
        lm_head: Output vocabulary projection layer.
    """

    def __init__(self, config: Gemma4Config, rngs: nnx.Rngs) -> None:
        """Initialize the Zero Gemma 4 block.

        Args:
            config: Gemma 4 architectural configuration.
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
        self.rope = nnx.RoPE(embedding_size=config.hidden_size)
        self.attn = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.hidden_size,
            num_kv_heads=config.num_kv_heads,
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
        """Execute the Gemma 4 transformer block forward pass.

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
        geglu = as_jax(zero_jax.nn.gelu(self.gate_proj(ff_in))) * self.up_proj(ff_in)
        h = h + self.down_proj(geglu)
        logits: JaxArray = self.lm_head(h)
        return logits


def get_gemma4_zero() -> bool:
    """Construct and execute the Gemma 4 architecture in `zero_flax.nnx`.

    Returns:
        `True` if the `ZeroGemma4Block` forward pass produces the expected shape.
    """
    ensure_eager_mode()
    cfg = Gemma4Config()
    model = ZeroGemma4Block(cfg, rngs=nnx.Rngs(0))
    tokens = as_jax(jnp.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]))
    logits = model(tokens)
    return logits.shape == (2, 4, cfg.vocab_size)


def get_gemma4_native() -> bool:
    """Construct and execute the Gemma 4 architecture across Praxis and JAX dialects.

    Returns:
        `True` if both Praxis RMSNorm + GQA and NNX Gemma 4 blocks succeed.
    """
    ensure_eager_mode()
    cfg = Gemma4Config()
    rms = PaxRmsNorm(dim=cfg.hidden_size)
    gqa = PaxGroupedQueryAttention(
        input_dim=cfg.hidden_size,
        hidden_dim=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        dim_per_head=cfg.head_dim,
    )
    x_pax = PaxTensor([0.5] * (2 * 4 * cfg.hidden_size), (2, 4, cfg.hidden_size))
    h_pax = rms(x_pax)
    attn_pax = gqa(h_pax, h_pax, h_pax)
    return attn_pax.shape == (2, 4, cfg.hidden_size) and get_gemma4_zero()


def load_gemma4_weights() -> bool:
    """Serialize and restore Gemma 4 weights using `zero_orbax.checkpoint`.

    Returns:
        `True` if checkpoint round-trip restoration matches original weights.
    """
    ensure_eager_mode()
    cfg = Gemma4Config()
    model = ZeroGemma4Block(cfg, rngs=nnx.Rngs(42))
    gate_kernel: JaxArray = as_jax(model.gate_proj.kernel.value)
    checkpointer = ocp.PyTreeCheckpointer()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "gemma4_ckpt"
        checkpointer.save(str(ckpt_path), to_compiler_tensor(gate_kernel))
        restored = as_restored_jax(checkpointer.restore(str(ckpt_path)))
        zero_chex.assert_trees_all_close(gate_kernel, restored)
    return True


def t_gemma4_inference() -> bool:
    """Run deterministic cross-execution Gemma 4 forward inference verification.

    Returns:
        `True` if two identically seeded Gemma 4 models produce equivalent logits.
    """
    ensure_eager_mode()
    cfg = Gemma4Config()
    model_a = ZeroGemma4Block(cfg, rngs=nnx.Rngs(7))
    model_b = ZeroGemma4Block(cfg, rngs=nnx.Rngs(7))
    tokens = as_jax(jnp.asarray([[1, 3, 5, 7]]))
    out_a = model_a(tokens)
    out_b = model_b(tokens)
    return check_equivalence(out_a, out_b)


def t_gemma4_edge_cases() -> bool:
    """Verify Gemma 4 boundary token IDs and GeGLU numerical parity.

    Returns:
        `True` if boundary token IDs and GeGLU activation parity pass.
    """
    ensure_eager_mode()
    cfg = Gemma4Config()
    model = ZeroGemma4Block(cfg, rngs=nnx.Rngs(9))
    boundary_tokens = as_jax(jnp.asarray([[0, cfg.vocab_size - 1]]))
    logits = model(boundary_tokens)
    zero_chex.assert_tree_all_finite(logits)
    x_val = as_jax(jnp.asarray([[-1.0, 0.0, 1.0, 2.0]]))
    geglu_a = as_jax(zero_jax.nn.gelu(x_val)) * x_val
    geglu_b = nnx.gelu(x_val) * x_val
    return logits.shape == (1, 2, cfg.vocab_size) and check_equivalence(
        geglu_a,
        geglu_b,
    )


__all__ = [
    "Gemma4Config",
    "ZeroGemma4Block",
    "get_gemma4_native",
    "get_gemma4_zero",
    "load_gemma4_weights",
    "t_gemma4_edge_cases",
    "t_gemma4_inference",
]
