import jax.numpy as jnp

from flax import nnx

from probjax.nn import GaussianFourierEmbedding, Transformer

from jax.typing import ArrayLike
from dataclasses import dataclass


@dataclass
class DMRIEmbeddingConfig:
    num_layers: int = 3
    num_heads: int = 4
    widening_factor: int = 2
    attn_size: int = 16
    dropout_rate: float | None = None
    bvals_embed_dim: int = 3
    signals_embed_dim: int = 3
    bvec_repeats: int = 1
    log_transform_signals: bool = False


class BvalBvecSignalEmbeddingNet(nnx.Module, experimental_pytree=True):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        widening_factor: int = 2,
        attn_size: int = 16,
        dropout_rate: int = None,
        bvals_embed_dim: int = 3,
        signals_embed_dim: int = 3,
        bvec_repeats: int = 1,
        log_transform_signals: bool = False,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.log_transform_signals = log_transform_signals
        self.bvec_repeats = bvec_repeats

        self.initial_layer = nnx.Linear(
            bvals_embed_dim + signals_embed_dim + 3 * bvec_repeats, model_dim, rngs=rngs
        )
        self.embed_bvals = GaussianFourierEmbedding(1, bvals_embed_dim, rngs=rngs)
        self.embed_signals = GaussianFourierEmbedding(1, signals_embed_dim, rngs=rngs)
        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
        )

    def __call__(
        self,
        bvals: ArrayLike,  # B, T
        bvecs: ArrayLike,  # B, T, 3
        signals: ArrayLike,  # B, T
    ):
        if self.log_transform_signals:
            signals = jnp.log(signals + 1e-6)
        bvals = self.embed_bvals(bvals[..., None])
        signals = self.embed_signals(signals[..., None])
        bvecs = jnp.repeat(bvecs, self.bvec_repeats, axis=-1)
        data = jnp.concatenate([bvals, signals, bvecs], axis=-1)
        tokens = self.initial_layer(data)
        out_tokens = self.transformer(tokens)
        return out_tokens
