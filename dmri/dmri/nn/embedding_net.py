from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.nn.nets.denoising_diffusion_model import EDM
from probjax.nn import MLP
from probjax.nn.utils import AffineFuse

from jax.typing import ArrayLike


class BvalBvecSignalEmbeddingNet(nnx.Module, experimental_pytree=True):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs,
        max_bval: float,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        widening_factor: int = 2,
        attn_size: int = 16,
        dropout_rate: int = None,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.max_bval = max_bval

        self.initial_layer = nnx.Linear(5, model_dim, rngs=rngs)
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
        # Very naive
        bvals = bvals[..., None] / self.max_bval
        signals = signals[..., None]
        data = jnp.concatenate([bvals, bvecs, signals], axis=-1)
        tokens = self.initial_layer(data)
        out_tokens = self.transformer(tokens)
        return out_tokens

