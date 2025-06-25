from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from probjax.nn import GaussianFourierEmbedding, Transformer
from probjax.nn.nets.denoising_diffusion_model import EDM
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint

from dmri.nn.tokenizer import Tokenizer


@dataclass
class DMRIThetaInferenceConfig:
    num_layers: int = 6
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    context_dim: int = 64
    dropout_rate: float | None = None


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
        dropout_rate=None,
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
            dropout_rate=dropout_rate,
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
        # print(x.shape)
        # print(kwargs["tokens_cfg"].shape)
        # print(input_embed.shape)
        while time_embed.ndim < input_embed.ndim:
            time_embed = time_embed[..., None, :]
        _context = time_embed

        # Additional context
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
        dropout_rate=None,
        enable_cross_attention=True,
        loss_type="x0",
    ):
        transformer = DiffusionTransformer(
            rngs,
            model_dim=model_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            dropout_rate=dropout_rate,
            enable_cross_attention=enable_cross_attention,
        )
        # Prevent automatic parameter updates
        super().__init__(transformer, loss_type=loss_type)

    def loss(
        self,
        rng,
        theta,
        tokenizer,
        y,
        model_mask,
        tokens_cfg,
        target_score=None,
        attention_mask=None,
        loss_mask=None,
        weight_by_complexity=False,
        cut_off_tsm=0.1,
    ):
        assert loss_mask is None
        rng0, rng1 = jax.random.split(rng)
        t = self.noise_schedule(rng0, (theta.shape[0],))
        std = self.std_fn(t)
        eps = jax.random.normal(rng1, theta.shape)
        thetas_noisy = theta + std * eps
        if self.loss_type == "x0":
            theta_denoised = self.denoise(
                t,
                thetas_noisy,
                tokenizer,
                y=y,
                model_mask=model_mask,
                tokens_cfg=tokens_cfg,
                attention_mask=attention_mask,
            )
            weight = self.weight_fn(t)
            loss_denoised = weight * jnp.sum(
                (theta_denoised - theta) ** 2, axis=-1, keepdims=True
            )
            loss = loss_denoised
        elif self.loss_type == "v":
            print("using v loss")
            total_std = jnp.sqrt(1.0 + std**2)
            alpha_t = 1.0 / total_std
            sigma_t = std / total_std
            v_target = alpha_t * eps - sigma_t * theta
            theta_denoised = self.denoise(
                t,
                thetas_noisy,
                tokenizer,
                y=y,
                model_mask=model_mask,
                tokens_cfg=tokens_cfg,
                attention_mask=attention_mask,
            )
            eps_pred = (thetas_noisy - theta_denoised) / std
            v = alpha_t * eps_pred - sigma_t * theta_denoised
            weight = self.weight_fn_v(t)
            loss_v = weight * jnp.sum((v - v_target) ** 2, axis=-1, keepdims=True)
            loss = loss_v
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported")

        if target_score is not None and cut_off_tsm > 0.0:
            score_est = (theta_denoised - thetas_noisy) / std**2
            # Target score norm
            target_score_norm = jnp.sqrt(
                jnp.sum(target_score**2, axis=-1, keepdims=True)
            ).mean()
            weight_tsm = (
                1 / target_score_norm * std**2 * jnp.where((std < cut_off_tsm), 1, 0)
            )  # Only use TSM for early times
            loss_score = weight_tsm * jnp.sum(
                (score_est - target_score) ** 2, axis=-1, keepdims=True
            )
            loss += loss_score

            print(
                "Data loss: ", loss_denoised.mean(), "Score loss: ", loss_score.mean()
            )

        if weight_by_complexity:
            loss = loss * (model_mask.sum(axis=-1, keepdims=True) + 0.01)

        return jnp.mean(loss)

    def sample(
        self,
        rng,
        tokenizer,
        y,
        dim,
        tokens_cfg=None,
        model_mask=None,
        context=None,
        attention_mask=None,
        max_noise=None,
        num_steps=16,
        sample_method="ode",
        rho=7,
        min_noise_nugget=0.0,
    ):
        if max_noise is not None:
            self.max_noise = max_noise
        rng, rng_init = jax.random.split(rng)
        eps = jax.random.normal(rng_init, dim) * self.marginal_std(self.max_noise)
        ts = self.solve_schedule(num_steps, rho) + min_noise_nugget

        if sample_method == "ode":

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
                    model_mask=model_mask,
                )
                return (f - 0.5 * g**2 * score).reshape(x.shape)

            state, _ = odeint(
                drift,
                eps,
                ts,
                method="heun",
                filter_state=lambda *args: None,
                return_state=True,
            )

            x = state.y0
            x += -drift(ts[-1], x) * ts[-1]
            return x
        elif sample_method == "sde":

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
                    model_mask=model_mask,
                )
                return f - g**2 * score

            diffusion = self.diffusion

            state, _ = sdeint(
                rng,
                drift,
                diffusion,
                eps,
                ts,
                return_state=True,
                # filter_state=lambda *args: None,
            )
            x = state.y0
            return x

    def log_prob(
        self,
        x,
        tokenizer,
        y,
        tokens_cfg=None,
        context=None,
        attention_mask=None,
        model_mask=None,
        max_noise=None,
        num_steps=16,
        rho=7,
        min_noise_nugget=0.0,
    ):
        if max_noise is not None:
            self.max_noise = max_noise
        ts = self.solve_schedule(num_steps, rho)[::-1] + min_noise_nugget

        def dx_dt_fn(t, z):
            f_ = self.drift(t, z)
            g_ = self.diffusion(t, z)
            s_ = self.score(
                t,
                z,
                tokenizer=tokenizer,
                tokens_cfg=tokens_cfg,
                y=y,
                context=context,
                attention_mask=attention_mask,
                model_mask=model_mask,
            )
            return f_ - 0.5 * g_**2 * s_

        x = x
        logp0 = 0.0

        def drift(t, state):
            data, logp = state
            dx_dt = dx_dt_fn(t, data)
            div = jnp.trace(jax.jacfwd(lambda z: dx_dt_fn(t, z))(data))
            return (dx_dt, div)

        state, _ = odeint(
            drift,
            (x, logp0),
            ts,
            method="heun",
            filter_state=lambda *args: None,
            return_state=True,
        )
        x_final = state.y0
        x_final, logp_final = x_final[:-1], x_final[-1]

        sigma = self.marginal_std(self.max_noise)
        base_logp = -0.5 * jnp.sum(x_final**2) / sigma**2
        base_logp += -0.5 * x_final.shape[-1] * jnp.log(2 * np.pi * sigma**2)
        final = logp_final + base_logp
        return jnp.squeeze(final)

    def sample_and_log_prob(
        self,
        rng,
        tokenizer,
        y,
        dim,
        tokens_cfg=None,
        model_mask=None,
        context=None,
        attention_mask=None,
        max_noise=None,
        num_steps=16,
        rho=7,
        min_noise_nugget=0.0,
    ):
        if max_noise is not None:
            self.max_noise = max_noise
        eps = jax.random.normal(rng, dim) * self.marginal_std(self.max_noise)
        ts = self.solve_schedule(num_steps, rho) + min_noise_nugget

        def dx_dt_fn(t, x):
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
                model_mask=model_mask,
            )
            return f - 0.5 * g**2 * score

        def drift(t, state):
            x, logp = state
            dx_dt = dx_dt_fn(t, x)
            div = jnp.trace(jax.jacfwd(lambda z: dx_dt_fn(t, z))(x))
            return (dx_dt, -div)

        state, _ = odeint(
            drift,
            (eps, 0.0),
            ts,
            method="heun",
            filter_state=lambda *args: None,
            return_state=True,
        )

        y0 = state.y0
        x, logp = y0[:-1], y0[-1]
        # Leave out due to numerical instability
        # dx_dt = dx_dt_fn(ts[-1], x)
        # x += -dx_dt * ts[-1]

        sigma = self.marginal_std(self.max_noise)
        base_logp = -0.5 * jnp.sum(eps**2) / sigma**2
        base_logp += -0.5 * eps.shape[-1] * jnp.log(2 * np.pi * sigma**2)
        logp += base_logp

        return x, jnp.squeeze(logp)
