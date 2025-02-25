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

from dmri.nn.tokenizer import Tokenizer

from probjax.utils.odeint import odeint


class DiffusionTransformer(nnx.Module, experimental_pytree=True):

    def __init__(
        self,
        rngs,
        model_dim=64,
        context_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        enable_cross_attention=True,
    ) -> None:
        self.time_embedding = GaussianFourierEmbedding(1, context_dim, rngs=rngs)
        self.transformer = Transformer(
            model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            enable_cross_attention=enable_cross_attention,
            rngs=rngs,
            context_dim=context_dim,
        )

    def __call__(
        self,
        t,
        x,
        tokenizer: Tokenizer,
        y=None,
        context=None,
        attention_mask=None,
        **kwargs,
    ):
        time_embed = self.time_embedding(t)
        input_embed = tokenizer.encode(x, **kwargs)
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
        output = tokenizer.decode(output, **kwargs)
        return output


class EDMSimformer(EDM):
    def __init__(
        self,
        rngs,
        model_dim=64,
        context_dim=64,
        num_heads=4,
        num_layers=4,
        attn_size=16,
        widening_factor=3,
        enable_cross_attention=True,
    ):
        transformer = DiffusionTransformer(
            rngs,
            model_dim=model_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            enable_cross_attention=enable_cross_attention,
        )
        # Prevent automatic parameter updates
        super().__init__(transformer, loss_kwargs={"update_params": lambda m, p: None})

    def sample(
        self,
        rng,
        tokenizer,
        y,
        dim,
        tokens_cfg=None,
        context=None,
        max_noise=None,
        num_steps=16,
        attention_mask=None,
    ):
        if max_noise is not None:
            self.max_noise = max_noise
        eps = jax.random.normal(rng, dim) * self.marginal_std(self.max_noise)
        ts = self.solve_schedule(num_steps)


        def drift(t, x):
            t = jnp.atleast_1d(t)
            f = self.drift(t, x)
            g = self.diffusion(t, x)
            score = self.score(
                t,
                x,
                tokenizer=tokenizer,
                tokens_cfg=tokens_cfg,
                y=y,
                context=context,
                attention_mask=attention_mask,
            )
            return (f - 0.5 * g**2 * score).reshape(x.shape)

        # Solve the ODE

        state, _ = odeint(
            drift,
            eps,
            ts,
            method="heun",
            filter_state=lambda *args: None,
            return_state=True,
        )


        x = state.y0
        # dt is 0 - ts[0] to get the final sample
        x += -drift(ts[-1], x) * ts[-1]

        return x
