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
from probjax.core import inverse_and_logabsdet

from functools import partial

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
    normalize_qk_attn: bool = False
    normalize_qk_cross_attn: bool = False
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
        normalize_qk_attn: bool = False,
        normalize_qk_cross_attn: bool = False,
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
        self.kv_in_features = (
            kv_in_features if kv_in_features is not None else model_dim
        )
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
            normalize_qk_attn=normalize_qk_attn,
            normalize_qk_cross_attn=normalize_qk_cross_attn,
            attn_fuse_cls=attn_fuse_cls,
            mlp_fuse_cls=mlp_fuse_cls,
            **precision_kwargs,
        )
        self.out_norm = nnx.LayerNorm(model_dim, rngs=rngs)

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
        output = self.out_norm(output)
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
        normalize_qk_attn: bool = False,
        normalize_qk_cross_attn: bool = False,
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
            normalize_qk_attn=normalize_qk_attn,
            normalize_qk_cross_attn=normalize_qk_cross_attn,
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
        model_mask_arr = jnp.asarray(model_mask)
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
                model_mask=model_mask_arr,
                tokens_cfg=tokens_cfg,
                attention_mask=attention_mask,
                context=context,
            )
            assert isinstance(theta_denoised, jnp.ndarray), (
                "Denoised output is not an ndarray"
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
                model_mask=model_mask_arr,
                tokens_cfg=tokens_cfg,
                attention_mask=attention_mask,
                context=context,
            )
            assert isinstance(theta_denoised, jnp.ndarray), (
                "dennoised output is not an ndarray"
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

        if weight_by_complexity:
            loss = loss * (model_mask_arr.sum(axis=-1, keepdims=True) + 0.01)

        return jnp.mean(loss)

    def sample(
        self,
        rng: RngKey,
        tokenizer: Tokenizer,
        y: Array,
        dim: int,
        tokens_cfg: Optional[Array] = None,
        model_mask: Optional[ArrayLike] = None,
        context: Optional[ArrayLike] = None,
        attention_mask: Optional[ArrayLike] = None,
        sample_method: str = "ode",
        num_steps: int = 64,
        t_min: float | None = None,
        t_max: float | None = None,

    ) -> Array:
        rng, rng_init = jax.random.split(rng)
        eps = jax.random.normal(rng_init, (dim,)) * self.marginal_std(self.train_cfg.t_max)

        if sample_method == "ode":
            return super().sample_ode(eps,t_max=t_max, t_min=t_min, num_steps=num_steps, tokenizer=tokenizer,y=y,model_mask=model_mask,tokens_cfg=tokens_cfg,context=context,attention_mask=attention_mask)
        elif sample_method == "sde":
            return super().sample_sde(rng,eps,t_max=t_max, t_min=t_min, num_steps=num_steps, tokenizer=tokenizer,y=y,model_mask=model_mask,tokens_cfg=tokens_cfg,context=context,attention_mask=attention_mask)
        else:
            raise ValueError(f"Sample method {sample_method} not recognized.")

    def log_prob(
        self,
        x: Array,
        tokenizer: Tokenizer,
        y: Array,
        tokens_cfg: Optional[Array] = None,
        context: Optional[ArrayLike] = None,
        attention_mask: Optional[ArrayLike] = None,
        model_mask: Optional[ArrayLike] = None,
        t_min: float | None = None,
        t_max: float | None     = None,
        num_steps: int = 64,
    ) -> Array:
        if t_min is None:
            t_min = 1e-3
        if t_max is None:
            t_max = self.train_cfg.t_max
        ts = self.solver_cfg.solve_schedule(t_min, t_max, num_steps)[::-1]

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
            data, _ = state
            dx_dt = dx_dt_fn(t, data)
            div = jnp.trace(jax.jacrev(lambda z: dx_dt_fn(t, z))(data))
            return (dx_dt, div)

        x_final = odeint(
            drift,
            (x, logp0),
            ts,
            method="heun",
            collect_trace=False,
        )
        x_final, logp_final = x_final

        sigma = self.marginal_std(t_max)
        base_logp = -0.5 * jnp.sum(x_final**2) / sigma**2
        base_logp += -0.5 * x_final.shape[-1] * jnp.log(2 * np.pi * sigma**2)
        final = logp_final + base_logp

        return jnp.squeeze(final)
