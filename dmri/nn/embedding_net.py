from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.layers.attention import flex_attention

from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    ssfp_acquisition_scheme,
)


@dataclass
class DMRIEmbeddingConfig:
    num_layers: int = 3
    num_heads: int = 4
    widening_factor: int = 2
    use_flash_attention: bool = False
    attn_size: int = 16
    dropout_rate: float = 0.0
    bvals_embed_dim: int = 3
    signals_embed_dim: int = 3
    bvec_repeats: int = 1
    log_transform_signals: bool = False
    embed_signals: str = "repeat"


class BvalBvecSignalEmbeddingNet(nnx.Module):
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
        dropout_rate: float = 0.0,
        bvals_embed_dim: int = 3,
        signals_embed_dim: int = 3,
        bvec_repeats: int = 1,
        log_transform_signals: bool = False,
        use_flash_attention: bool = False,
        embed_signals="repeat",
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
        if embed_signals == "repeat":
            self.embed_signals = lambda x: jnp.repeat(x, signals_embed_dim, axis=-1)
        elif embed_signals == "fourier":
            self.embed_signals = GaussianFourierEmbedding(
                1, signals_embed_dim, rngs=rngs
            )
        else:
            raise ValueError(f"Invalid embed_signals: {embed_signals}")

        if use_flash_attention:
            attention_fn = flex_attention
        else:
            attention_fn = None

        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            attention_fn=attention_fn,
        )

    def __call__(
        self,
        acq: acquisition_scheme,
        x: ArrayLike,
        deterministic: bool | None = None,
        decode: bool = False,
    ):
        bvals = acq.bvals
        bvecs = acq.bvecs
        signals = x
        if self.log_transform_signals:
            signals = jnp.log(jnp.clip(signals, min=1e-8))
        bvals = self.embed_bvals(bvals[..., None])
        signals = self.embed_signals(signals[..., None])
        bvecs = jnp.repeat(bvecs, self.bvec_repeats, axis=-1)
        data = jnp.concatenate([bvals, signals, bvecs], axis=-1)
        tokens = self.initial_layer(data)
        out_tokens = self.transformer(
            tokens, deterministic=deterministic, decode=decode
        )
        return out_tokens


@dataclass
class SSFPEmbeddingNetConfig:
    num_layers: int = 3
    num_heads: int = 4
    widening_factor: int = 2
    use_flash_attention: bool = False
    attn_size: int = 16
    dropout_rate: float = 0.0
    use_flash_attention: bool = False


class SSFPEmbeddingNet(nnx.Module):
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
        dropout_rate: float = 0.0,
        use_flash_attention: bool = False,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size

        scalar_embed_dim = self.model_dim // 3
        signal_embed_dim = self.model_dim // 3
        bvec_embed_dim = self.model_dim - scalar_embed_dim - signal_embed_dim
        self.embed_scalars = GaussianFourierEmbedding(7, scalar_embed_dim, rngs=rngs)
        self.embed_signals = GaussianFourierEmbedding(1, signal_embed_dim, rngs=rngs)
        # Repeat bvces
        self.embed_bvecs = lambda x: jnp.repeat(
            x[..., None], bvec_embed_dim, axis=-1
        ).reshape(*x.shape[:-1], -1)[..., :bvec_embed_dim]

        if use_flash_attention:
            attention_fn = flex_attention
        else:
            attention_fn = None

        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            attention_fn=attention_fn,
        )

    def __call__(
        self,
        acq: ssfp_acquisition_scheme,
        signals: ArrayLike,
        deterministic: bool | None = None,
        decode: bool = False,
    ):
        # Embed stuff
        T1 = acq.T1
        T2 = acq.T2
        B1 = acq.B1
        diffGradAmps = acq.diffGradAmps
        flipAngles = acq.flipAngles
        TRs = acq.TRs
        diffGradDur = acq.diffGradDur
        scalar = jnp.stack(
            [T1, T2, B1, diffGradAmps, flipAngles, TRs, diffGradDur], axis=-1
        )
        scalar = self.embed_scalars(scalar)
        signal = self.embed_signals(signals[..., None])
        bvecs = self.embed_bvecs(acq.bvecs)
        tokens = jnp.concatenate([scalar, signal, bvecs], axis=-1)
        out_tokens = self.transformer(
            tokens, deterministic=deterministic, decode=decode
        )
        return out_tokens
