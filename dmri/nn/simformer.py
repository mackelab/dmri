from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from probjax.nn import (
    AdditiveBinaryFuse,
    GatedFuse,
    GaussianFourierEmbedding,
    Transformer,
)
from probjax.nn.layers.attention import flex_attention
from probjax.nn.nets.denoising_diffusion_model import EDM
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from dmri.nn.tokenizer import Tokenizer


@dataclass
class DMRIThetaInferenceConfig:
    num_layers: int = 6
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    time_embed_dim: int = 64
    dropout_rate: float = 0.0
    use_flash_attention: bool = False
    use_flash_cross_attention: bool = False
    gate_attention: bool = False
    gate_mlp: bool = False
    kv_in_features: Optional[int] = None
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None


class DiffusionTransformer(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        time_embed_dim: int = 64,
        additional_context_dim: int = 0,
        num_heads: int = 4,
        num_layers: int = 6,
        attn_size: int = 16,
        widening_factor: int = 3,
        dropout_rate: float = 0.0,
        enable_cross_attention: bool = True,
        use_flash_attention: bool = False,
        use_flash_cross_attention: bool = False,
        gate_attention: bool = False,
        gate_mlp: bool = False,
        kv_in_features: Optional[int] = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        self.time_embed_dim = time_embed_dim
        self.additional_context_dim = additional_context_dim
        self.total_context_dim = self.time_embed_dim + self.additional_context_dim
        if self.time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be a positive integer.")
        if self.additional_context_dim < 0:
            raise ValueError("additional_context_dim must be non-negative.")
        precision_kwargs = {
            "dtype": dtype,
            "param_dtype": param_dtype,
            "precision": precision,
            "preferred_element_type": preferred_element_type,
        }
        self.time_embedding = GaussianFourierEmbedding(
            1, self.time_embed_dim, rngs=rngs, **precision_kwargs
        )

        attn_fn = flex_attention if use_flash_attention else None
        cross_attn_fn = flex_attention if use_flash_cross_attention else None

        transformer_context_dim = (
            self.total_context_dim if self.total_context_dim > 0 else None
        )
        attn_fuse_cls = AdditiveBinaryFuse if gate_attention else GatedFuse
        mlp_fuse_cls = AdditiveBinaryFuse if gate_mlp else GatedFuse
        self.kv_in_features = kv_in_features if kv_in_features is not None else model_dim
        self.transformer = Transformer(
            model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            kv_in_features=self.kv_in_features,
            widening_factor=widening_factor,
            enable_cross_attention=enable_cross_attention,
            dropout_rate=dropout_rate,
            rngs=rngs,
            context_dim=transformer_context_dim,
            attention_fn=attn_fn,
            cross_attention_fn=cross_attn_fn,
            attn_fuse_cls=attn_fuse_cls,
            mlp_fuse_cls=mlp_fuse_cls,
            **precision_kwargs,
        )

    def __call__(
        self,
        t: ArrayLike,
        x: ArrayLike,
        tokenizer: Tokenizer,
        y: Optional[Array] = None,
        context: Optional[Array] = None,
        attention_mask: Optional[Array] = None,
        **kwargs: Any,
    ) -> Array:
        time_embed = self.time_embedding(t)
        input_embed = tokenizer.encode(x, **kwargs)
        # print(x.shape)
        # print(kwargs["tokens_cfg"].shape)
        # print(input_embed.shape)
        while time_embed.ndim < input_embed.ndim:
            time_embed = time_embed[..., None, :]
        if self.additional_context_dim > 0:
            if context is None:
                extra_context = jnp.zeros(
                    time_embed.shape[:-1] + (self.additional_context_dim,),
                    dtype=time_embed.dtype,
                )
            else:
                extra_context = jnp.asarray(context, dtype=time_embed.dtype)
                while extra_context.ndim < input_embed.ndim:
                    extra_context = extra_context[..., None, :]
                if extra_context.shape[-1] != self.additional_context_dim:
                    raise ValueError(
                        "Context dimensionality mismatch for diffusion transformer."
                    )
            _context = jnp.concatenate([time_embed, extra_context], axis=-1)
        else:
            if context is not None:
                raise ValueError(
                    "Context provided but no additional context dimension configured."
                )
            _context = time_embed

        output = self.transformer(
            input_embed, y, y, context=_context, mask=attention_mask
        )
        output = tokenizer.decode(output, **kwargs)
        return output


class EDMSimformer(EDM):
    def __init__(
        self,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        time_embed_dim: int = 64,
        additional_context_dim: int = 0,
        num_heads: int = 4,
        num_layers: int = 4,
        attn_size: int = 16,
        widening_factor: int = 3,
        dropout_rate: float = 0.0,
        enable_cross_attention: bool = True,
        use_flash_attention: bool = False,
        use_flash_cross_attention: bool = False,
        gate_attention: bool = False,
        gate_mlp: bool = False,
        kv_in_features: Optional[int] = None,
        loss_type: str = "x0",
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ) -> None:
        transformer = DiffusionTransformer(
            rngs,
            model_dim=model_dim,
            time_embed_dim=time_embed_dim,
            additional_context_dim=additional_context_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            dropout_rate=dropout_rate,
            enable_cross_attention=enable_cross_attention,
            use_flash_attention=use_flash_attention,
            use_flash_cross_attention=use_flash_cross_attention,
            gate_attention=gate_attention,
            gate_mlp=gate_mlp,
            kv_in_features=kv_in_features,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        # Prevent automatic parameter updates
        super().__init__(transformer, loss_type=loss_type)

    def loss(
        self,
        rng: RngKey,
        theta: Array,
        tokenizer: Tokenizer,
        y: Array,
        model_mask: ArrayLike,
        tokens_cfg: Array,
        target_score: Optional[Array] = None,
        attention_mask: Optional[ArrayLike] = None,
        loss_mask: Optional[ArrayLike] = None,
        weight_by_complexity: bool = False,
        cut_off_tsm: float = 0.1,
        context: Optional[ArrayLike] = None,
    ) -> Array:
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
                context=context,
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
                context=context,
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

            # print(
            #     "Data loss: ", loss_denoised.mean(), "Score loss: ", loss_score.mean()
            # )

        if weight_by_complexity:
            loss = loss * (model_mask.sum(axis=-1, keepdims=True) + 0.01)

        return jnp.mean(loss)

    def sample(
        self,
        rng: RngKey,
        tokenizer: Tokenizer,
        y: Array,
        dim: Tuple[int, ...] | int,
        tokens_cfg: Optional[Array] = None,
        model_mask: Optional[ArrayLike] = None,
        context: Optional[ArrayLike] = None,
        attention_mask: Optional[ArrayLike] = None,
        max_noise: Optional[float] = None,
        min_noise: Optional[float] = None,
        num_steps: int = 16,
        sample_method: str = "ode",
        rho: float = 7,
    ) -> Array:
        rng, rng_init = jax.random.split(rng)
        eps = jax.random.normal(rng_init, dim) * self.marginal_std(self.max_noise)
        ts = self.solve_schedule(num_steps, rho=rho, min_noise=min_noise, max_noise=max_noise)

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
        x: Array,
        tokenizer: Tokenizer,
        y: Array,
        tokens_cfg: Optional[Array] = None,
        context: Optional[ArrayLike] = None,
        attention_mask: Optional[ArrayLike] = None,
        model_mask: Optional[ArrayLike] = None,
        min_noise: Optional[float] = None,
        max_noise: Optional[float] = None,
        num_steps: int = 16,
        rho: float = 7,
    ) -> Array:
        ts = self.solve_schedule(num_steps, rho=rho, min_noise=min_noise, max_noise=max_noise)[::-1]
        print(x.shape, ts.shape, y.shape, context.shape if context is not None else None)

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
        rng: RngKey,
        tokenizer: Tokenizer,
        y: Array,
        dim: Tuple[int, ...] | int,
        tokens_cfg: Optional[Array] = None,
        model_mask: Optional[ArrayLike] = None,
        context: Optional[ArrayLike] = None,
        attention_mask: Optional[ArrayLike] = None,
        min_noise: Optional[float] = None,
        max_noise: Optional[float] = None,
        num_steps: int = 16,
        rho: float = 7,
    ) -> Tuple[Array, Array]:
        eps = jax.random.normal(rng, dim) * self.marginal_std(self.max_noise)
        ts = self.solve_schedule(num_steps, rho, min_noise=min_noise, max_noise=max_noise)

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
