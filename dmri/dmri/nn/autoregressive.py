from functools import partial
from typing import Optional
from dmri.nn.tokenizer import Tokenizer
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.nn.nets.denoising_diffusion_model import EDM
from probjax.nn import MLP
from probjax.nn.utils import AffineFuse

import optax


class BinaryAutoregressiveDecoder(nnx.Module, experimental_pytree=True):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        widening_factor: int = 4,
        attn_size: int = 16,
        dropout_rate: int = None,
        context_dim: Optional[int] = None,
        enable_cross_attention: bool = True,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.context_dim = context_dim

        # Why not use a shared tokenizer with the model mask?
        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            context_dim=context_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            enable_cross_attention=enable_cross_attention,
        )
        self.output = nnx.Linear(model_dim, 1, rngs=rngs)

    def __call__(
        self,
        model_mask,
        tokenizer: Tokenizer,
        context=None,
        y=None,
        mask=None,
        decode=False,
        deterministic=False,
        **kwargs,
    ):
        input_tokens = tokenizer.encode(model_mask=model_mask, **kwargs)
        _, seq_len, model_dim = input_tokens.shape

        assert model_dim == self.model_dim, (
            f"Token dim mismatch, is {model_dim}, expected {self.model_dim}"
        )
        # Autoregressive mask constrained
        base_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        if mask is not None:
            base_mask = base_mask & mask

        if context is not None:
            context = context[..., None, :]

        x = self.transformer(
            input_tokens,
            y,
            y,
            context=context,
            mask=base_mask,
            deterministic=deterministic,
            decode=decode,
        )
        logits = self.output(x)
        return logits[..., :-1, 0]

    def loss_fn(self, params, model_mask, tokenizer, y, **kwargs):
        model_mask_logits = self(model_mask, tokenizer, y=y, **kwargs)
        return jnp.mean(
            optax.sigmoid_binary_cross_entropy(model_mask_logits, model_mask).sum(-1)
        )

    def sample(self, key, tokenizer, y, dim):
        return naive_autoregressive_decoding(self, key, tokenizer, y, dim)


@partial(jax.jit, static_argnums=(2,4,))
def naive_autoregressive_decoding(model, key, tokenizer, y, dim):
    x = jnp.zeros((dim,), dtype=jnp.bool_)

    def scan_fn(carry, k):
        x, i = carry
        logits = model(x.astype(jnp.int32), tokenizer, y=y)
        p_i = jax.nn.sigmoid(logits[i])

        x_i = jax.random.bernoulli(k, p_i)
        x = x.at[i].set(x_i)
        return (x, i + 1), None

    keys = jax.random.split(key, (dim,))
    x, _ = jax.lax.scan(scan_fn, (x, 0), keys)

    return x[0]
