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

from probjax.utils.odeint import odeint


class DiffusionTransformer(nnx.Module, experimental_pytree=True):

    def __init__(
        self,
        tokenizer,
        rngs,
        model_dim=64,
        context_dim=64,
        num_heads=4,
        num_layers=4,
        attn_size=16,
        widening_factor=3,
        enable_cross_attention=True,
    ) -> None:
        self.tokenizer = tokenizer
        self.time_embedding = GaussianFourierEmbedding(1, model_dim, rngs=rngs)
        self.transformer = Transformer(
            model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            enable_cross_attention=enable_cross_attention,
            rngs=rngs,
            context_dim=model_dim + context_dim,
        )

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
        _context = time_embed

        if context is not None:
            while context.ndim < input_embed.ndim:
                context = context[..., None, :]
            _context = jnp.concatenate([_context, context], axis=-1)

        output = self.transformer(
            input_embed, y, y, context=_context, mask=attention_mask
        )
        output = self.tokenizer.decode(output, node_ids)
        return output


class EDMSimformer(EDM):
    def __init__(
        self,
        tokenizer,
        rngs,
        model_dim=64,
        context_dim=64,
        num_heads=4,
        num_layers=4,
        attn_size=16,
        widening_factor=3,
        endable_cross_attention=True,
    ):
        transformer = DiffusionTransformer(
            tokenizer,
            rngs,
            model_dim=model_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            enable_cross_attention=endable_cross_attention,
        )
        super().__init__(transformer)

    def sample(self, rng, y, dim, node_ids, context=None, max_noise=None, num_steps=16):
        if max_noise is not None:
            self.max_noise = max_noise
        eps = jax.random.normal(rng, dim) * self.marginal_std(self.max_noise)
        ts = self.solve_schedule(num_steps)


        def drift(t, x):
            t = jnp.atleast_1d(t)
            f = self.drift(t, x)
            g = self.diffusion(t, x)
            score = self.score(t, x, node_ids=node_ids, y=y, context=context)
            return (f - 0.5 * g**2 * score).reshape(x.shape)

        return odeint(drift, eps, ts, method="heun")[-1]
