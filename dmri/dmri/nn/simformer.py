from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.nn.nets.denoising_diffusion_model import EDM
from probjax.nn import MLP
from probjax.nn.utils import AffineFuse


class Simformer(nnx.Module, experimental_pytree=True):
    model_dim: int = 50
    condition_dim: int = 10
    num_heads: int = 4
    num_layers: int = 4
    widening_factor: int = 2
    attn_size: int = 10

    def __init__(
        self,
        tokenizer,
        rngs,
        model_dim=64,
        condition_dim=10,
        num_heads=4,
        num_layers=4,
        attn_size=64,
        widening_factor=2,
        context_embed=None,
    ) -> None:
        self.model_dim = model_dim
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor

        self.tokenizer = tokenizer
        self.time_embedding = GaussianFourierEmbedding(1, self.model_dim, rngs=rngs)
        self.transformer = Transformer(
            self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            rngs=rngs,
            context_dim=self.model_dim * 2,
        )
        self.context_embed = context_embed

    def __call__(
        self,
        t,
        x,
        node_ids,
        condition_mask,
        *args,
        context=None,
        attention_mask=None,
        **kwargs,
    ):
        time_embed = self.time_embedding(t)
        input_embed = self.tokenizer(x, node_ids, condition_mask)
        while time_embed.ndim < input_embed.ndim:
            time_embed = time_embed[..., None, :]

        if context is not None:
            context = (
                self.context_embed(context) if self.context_embed is not None else None
            )
            context = context[..., None, :] if context is not None else None
            # print(time_embed.shape, context.shape)
            context = jnp.concatenate([time_embed, context], axis=-1)
        # print(context.shape)
        output = self.transformer(input_embed, context=context, mask=attention_mask)
        output = self.tokenizer.decode(output, node_ids)
        return output


class Simformer2(nnx.Module, experimental_pytree=True):
    model_dim: int = 50
    condition_dim: int = 10
    num_heads: int = 4
    num_layers: int = 4
    widening_factor: int = 2
    attn_size: int = 10

    def __init__(
        self,
        tokenizer,
        rngs,
        model_dim=64,
        condition_dim=10,
        num_heads=4,
        num_layers=4,
        attn_size=64,
        widening_factor=2,
        context_embed=None,
    ) -> None:
        self.model_dim = model_dim
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor

        self.tokenizer = tokenizer
        self.time_embedding = GaussianFourierEmbedding(1, self.model_dim, rngs=rngs)
        self.transformer = Transformer(
            self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            enable_cross_attention=True,
            rngs=rngs,
            context_dim=self.model_dim,
        )
        self.context_embed = context_embed

    def __call__(
        self,
        t,
        x,
        node_ids,
        *args,
        y=None,
        condition_mask=None,
        context=None,
        attention_mask=None,
        **kwargs,
    ):
        time_embed = self.time_embedding(t)
        input_embed = self.tokenizer(x, node_ids, condition_mask)
        while time_embed.ndim < input_embed.ndim:
            time_embed = time_embed[..., None, :]

        output = self.transformer(
            input_embed, y, y, context=context, mask=attention_mask
        )
        output = self.tokenizer.decode(output, node_ids)
        return output
