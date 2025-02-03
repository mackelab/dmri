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


class ModelIdentificationDecoder(nnx.Module, experimental_pytree=True):
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
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.context_dim = context_dim

        self.start_token = nnx.Param(jnp.zeros((1, model_dim)))
        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            context_dim=context_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
        )
        self.output = nnx.Linear(model_dim, 1, rngs=rngs)
        self.embed_mask = nnx.Embed(2, model_dim, rngs=rngs)

    def __call__(
        self,
        model_mask,
        context=None,
        mask=None,
        decode=False,
        deterministic=False,
        return_latents=False,
    ):
        # Prepare inputs
        *leading_dims, seq_len = model_mask.shape
        input_embed = self.embed_mask(model_mask.astype(jnp.int32))
        start_token = self.start_token.value
        if input_embed.ndim > 2:
            start_token = jnp.broadcast_to(
                start_token, (*leading_dims, 1, self.model_dim)
            )
        inputs_padded = jnp.concatenate([start_token, input_embed], axis=-2)

        # Autoregressive mask constrained
        base_mask = jnp.tril(jnp.ones((seq_len + 1, seq_len + 1)))
        if mask is not None:
            base_mask = base_mask & mask

        if context is not None:
            context = context[..., None, :]

        x = self.transformer(
            inputs_padded,
            context=context,
            mask=base_mask,
            deterministic=deterministic,
            decode=decode,
        )
        logits = self.output(x)
        if return_latents:
            return logits[..., :-1, 0], inputs_padded[..., :-1, :]
        return logits[..., :-1, 0]

    def sample(self, key, context, dim):
        return naive_autoregressive_decoding(self, key, context, dim)


# OLD
class ModelIdentificationDecoder_old(nnx.Module, experimental_pytree=True):
    model_dim: int = 50
    num_heads: int = 4
    num_layers: int = 2
    widening_factor: int = 2
    attn_size: int = 10

    def __init__(
        self,
        model_dim,
        context_dim,
        rngs,
        num_heads=5,
        num_layers=2,
        widening_factor=2,
        attn_size=10,
        dropout_rate=None,
        context_net=None,
    ):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size

        self.embed = nnx.Embed(2, model_dim, rngs=rngs)
        self.start_token = nnx.Param(jnp.zeros((1, model_dim)))
        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_heads,
            self.attn_size,
            context_dim=model_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
        )
        self.output = nnx.Linear(model_dim, 1, rngs=rngs)
        if context_net is None:
            self.context_embedding = MLP(
                [context_dim, 2 * model_dim, model_dim], rngs=rngs
            )
        else:
            self.context_embedding = context_net

    def __call__(
        self, x, context, decode=False, deterministic=False, return_latents=False
    ):
        # Autoregressive mask
        x = self.embed(x.astype(jnp.int32))
        context = self.context_embedding(context)
        start_token = self.start_token.value
        if x.ndim > 2:
            start_token = jnp.repeat(start_token[None, ...], x.shape[0], axis=0)
        x = jnp.concatenate([start_token, x], axis=-2)
        mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))
        context = context[..., None, :]
        x = self.transformer(
            x, context=context, mask=mask, deterministic=deterministic, decode=decode
        )
        logits = self.output(x)
        if return_latents:
            return logits[..., :-1, 0], x[..., :-1, :]
        return logits[..., :-1, 0]

    def sample(self, key, context, dim):
        return naive_autoregressive_decoding(self, key, context, dim)


@partial(jax.jit, static_argnums=(3,))
def naive_autoregressive_decoding(model, key, context, dim):
    x = jnp.zeros((dim,), dtype=jnp.bool_)

    def scan_fn(carry, k):
        x, i = carry
        logits = model(x.astype(jnp.int32), context)
        p_i = jax.nn.sigmoid(logits[i])

        x_i = jax.random.bernoulli(k, p_i)
        x = x.at[i].set(x_i)
        return (x, i + 1), None

    keys = jax.random.split(key, (dim,))
    x, _ = jax.lax.scan(scan_fn, (x, 0), keys)

    return x[0]
