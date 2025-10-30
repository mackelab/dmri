from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from flax import nnx
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from dmri.simulators import MultiCompartment
from dmri.simulators.acquisition_scheme import acquisition_scheme

from .autoregressive import (
    BinaryAutoregressiveDecoder,
    DMRIModelSelectionAmortizedPriorConfig,
    DMRIModelSelectionConfig,
)
from .embedding_net import (
    BvalBvecSignalEmbeddingNet,
    DMRIEmbeddingConfig,
    SSFPEmbeddingNet,
    SSFPEmbeddingNetConfig,
)
from .simformer import DMRIThetaInferenceConfig, EDMSimformer
from .tokenizer import DMRITokenizer, DMRITokenizerPP

EmbeddingModule = BvalBvecSignalEmbeddingNet | SSFPEmbeddingNet
TokenizerType = Type[DMRITokenizer]


@dataclass
class DMRIInferenceModelConfig:
    simulator: type[MultiCompartment]
    model_dim: int = 64
    use_attention_mask: bool = True
    inference_loss_type: str = "v"
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    tokenizer_cls: TokenizerType = DMRITokenizer
    embedding_cls: Type[EmbeddingModule] = BvalBvecSignalEmbeddingNet
    embedding_cfg: Any = field(default_factory=DMRIEmbeddingConfig)
    model_selection_cfg: Any = field(default_factory=DMRIModelSelectionConfig)
    theta_inference_cfg: DMRIThetaInferenceConfig = field(
        default_factory=DMRIThetaInferenceConfig
    )


@dataclass
class DMRIInferenceModelConfigMaskPriorAmortized(DMRIInferenceModelConfig):
    model_selection_cfg: Any = field(
        default_factory=DMRIModelSelectionAmortizedPriorConfig
    )


@dataclass
class DMRIInferenceModelConfigMaskPriorAmortizedPP(
    DMRIInferenceModelConfigMaskPriorAmortized
):
    tokenizer_cls: TokenizerType = DMRITokenizerPP
    embedding_cls: Type[EmbeddingModule] = BvalBvecSignalEmbeddingNet
    embedding_cfg: Any = field(default_factory=DMRIEmbeddingConfig)


@dataclass
class SSFPInferenceModelConfig(DMRIInferenceModelConfig):
    tokenizer_cls: TokenizerType = DMRITokenizerPP
    embedding_cls: Type[EmbeddingModule] = SSFPEmbeddingNet
    embedding_cfg: Any = field(default_factory=SSFPEmbeddingNetConfig)


class DMRIInferenceModel(nnx.Module):
    def __init__(self, cfg: DMRIInferenceModelConfig, rngs: nnx.Rngs) -> None:
        self.cfg: DMRIInferenceModelConfig = cfg
        self.precision_fields: tuple[str, ...] = (
            "dtype",
            "param_dtype",
            "precision",
            "preferred_element_type",
        )

        precision_defaults = {
            key: getattr(cfg, key, None)
            for key in self.precision_fields
            if getattr(cfg, key, None) is not None
        }

        def _config_kwargs(source_cfg: Any) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                key: value
                for key, value in vars(source_cfg).items()
                if value is not None and key not in self.precision_fields
            }
            for key, value in precision_defaults.items():
                kwargs.setdefault(key, value)
            for key in self.precision_fields:
                explicit_value = getattr(source_cfg, key, None)
                if explicit_value is not None:
                    kwargs[key] = explicit_value
            return kwargs

        # Setup embedding net observations
        embedding_kwargs = _config_kwargs(cfg.embedding_cfg)
        self.encoder = cfg.embedding_cls(
            rngs,
            model_dim=cfg.model_dim,
            **embedding_kwargs,
        )
        self.y_seq_dim: int = getattr(self.encoder, "y_seq_dim", cfg.model_dim)
        self.use_y_ctx: bool = bool(
            getattr(self.encoder, "use_global_summary_token", False)
        )
        self.y_glob_dim: int = getattr(self.encoder, "y_glob_dim", cfg.model_dim)
        self.y_ctx_dim: int = self.y_glob_dim if self.use_y_ctx else 0
        # Setup tokenizers
        self.tokenizer: DMRITokenizer = cfg.tokenizer_cls(
            cfg.simulator,
            token_dim=cfg.model_dim,
            rngs=rngs,
            **precision_defaults,
        )

        # Setup model selection network
        selection_cfg = cfg.model_selection_cfg
        selection_kwargs = _config_kwargs(selection_cfg)
        prior_params_embed_dim = selection_kwargs["prior_params_embed_dim"]
        self.requires_mask_prior = prior_params_embed_dim > 0
        selection_kwargs["additional_context_dim"] = self.y_ctx_dim
        selection_kwargs["kv_in_features"] = self.y_seq_dim

        self.model_decoder = BinaryAutoregressiveDecoder(
            rngs,
            model_dim=cfg.model_dim,
            **selection_kwargs,
        )
        self.mask_prior_dim = self.model_decoder.mask_prior_dim

        # Inference decoder
        theta_cfg = cfg.theta_inference_cfg
        theta_kwargs = _config_kwargs(theta_cfg)
        theta_kwargs["additional_context_dim"] = self.y_ctx_dim
        theta_kwargs["kv_in_features"] = self.y_seq_dim
        simformer = EDMSimformer(
            rngs=rngs,
            model_dim=cfg.model_dim,
            loss_type=cfg.inference_loss_type,
            **theta_kwargs,
        )
        self.inference_decoder = simformer

    def __call__(
        self,
        model_mask: Array,
        theta: Array,
        x: Array,
        acq: acquisition_scheme,
        mask_prior: Optional[Array] = None,
        alpha_prior: Optional[Array] = None,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
        t: Optional[ArrayLike] = None,
    ) -> tuple[Array, Array]:
        # Embed model configuration
        tokens_cfg, y_ctx, y, mask_prior = self.embed_inputs(
            model_mask,
            x,
            acq,
            mask_prior=mask_prior,
            alpha_prior=alpha_prior,
            model_idx=model_idx,
            noise_idx=noise_idx,
        )

        # Mode compartment prediction
        model_mask_logits = self.model_decoder(
            model_mask,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            mask_prior=mask_prior,
            additional_context=y_ctx,
        )

        # Get theta predictions
        if t is None:
            t = jnp.ones((theta.shape[0], 1)) * 0.0001
        # Mask out non-selected models
        attention_mask = self.marginalization_mask(
            model_mask, model_idx=model_idx, noise_idx=noise_idx
        )
        theta_pred = self.inference_decoder(
            t,
            theta,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            model_mask=model_mask,
            context=y_ctx,
        )

        return model_mask_logits, theta_pred

    def embed_inputs(
        self,
        model_mask: Array,
        x: Array,
        acq: acquisition_scheme,
        mask_prior: Optional[Array] = None,
        alpha_prior: Optional[Array] = None,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
    ) -> Tuple[Array, Optional[Array], Array, Optional[Array]]:
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask,
            alpha_prior,
            model_idx=model_idx,
            noise_idx=noise_idx,
        )
        # Embed observations and acquisition parameters
        y_ctx, y = self.encoder(acq, x)
        if y.ndim == 2:
            y = y[..., None, :]

        return tokens_cfg, y_ctx, y, mask_prior

    def theta_mask(
        self,
        model_mask: Array,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
    ) -> Array:
        theta_token_mask = self.tokenizer.theta_token_mask(
            model_mask, model_idx=model_idx, noise_idx=noise_idx
        )
        return theta_token_mask

    def marginalization_mask(
        self,
        model_mask: Array,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
    ) -> Array:
        _model_mask_extended = self.tokenizer.theta_token_mask(
            model_mask, model_idx=model_idx, noise_idx=noise_idx
        )
        attention_mask = (
            _model_mask_extended[..., None, :] & _model_mask_extended[..., :, None]
        )
        attention_mask = attention_mask | jnp.eye(
            _model_mask_extended.shape[-1], dtype=bool
        )
        return attention_mask

    def loss_fn(
        self,
        rng: RngKey,
        model_mask: Array,
        theta: Array,
        x: Array,
        acq: acquisition_scheme,
        mask_prior: Array | None = None,
        alpha_prior: Array | None = None,
        target_score: Array | None = None,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
        permute_order: bool = False,
        use_loss_mask: bool = False,
        weight_by_complexity: bool = False,
        cut_off_tsm: float = 0.1,
    ) -> Array:
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask, alpha_prior, model_idx=model_idx, noise_idx=noise_idx
        )
        # Embed observatiosn
        y_ctx, y = self.encoder(acq, x)
        if y.ndim == 2:
            y = y[..., None, :]

        if permute_order:
            rng, permute_rng = jax.random.split(rng)
        else:
            permute_rng = rng

        model_mask_loss = self.model_decoder.loss_fn(
            model_mask,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            permute_order=permute_order,
            rng=permute_rng,
            mask_prior=mask_prior,
            additional_context=y_ctx,
        )
        if self.cfg.use_attention_mask:
            attention_mask = self.marginalization_mask(model_mask)
        else:
            attention_mask = None

        if use_loss_mask:
            loss_mask = ~jax.vmap(
                partial(
                    self.tokenizer.theta_mask,
                    model_idx=model_idx,
                    noise_idx=noise_idx,
                )
            )(model_mask)
        else:
            loss_mask = None

        theta_loss = self.inference_decoder.loss(
            rng,
            theta,
            self.tokenizer,
            y=y,
            model_mask=model_mask,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_score=target_score,
            weight_by_complexity=weight_by_complexity,
            cut_off_tsm=cut_off_tsm,
            context=y_ctx,
        )
        theta_loss /= jnp.sqrt(theta.shape[-1])

        return jnp.concatenate([model_mask_loss[None], theta_loss[None]])

    def sample_mask(
        self,
        rng: RngKey,
        acq: acquisition_scheme,
        x: Array,
        mask_prior: Array | None = None,
    ) -> Array:
        # Update for different model configs
        mask_prior = jnp.asarray(mask_prior) if mask_prior is not None else None
        y_ctx, y = self.encoder(acq, x)
        model_mask = self.model_decoder.sample(
            rng,
            tokenizer=self.tokenizer,
            y=y,
            dim=self.tokenizer.num_models + self.tokenizer.num_noises,
            mask_prior=mask_prior,
            additional_context=y_ctx,
        )
        return model_mask

    def log_prob_mask(
        self,
        model_mask: Array,
        acq: acquisition_scheme,
        x: Array,
        mask_prior: Array | None = None,
    ) -> Array:
        y_ctx, y = self.encoder(acq, x)
        log_prob = self.model_decoder.log_prob(
            model_mask,
            self.tokenizer,
            y=y,
            mask_prior=mask_prior_arr,
            additional_context=y_ctx,
        )
        return log_prob

    def sample_theta(
        self,
        rng: RngKey,
        acq: acquisition_scheme,
        x: Array,
        model_mask: Array,
        num_steps: int = 16,
        max_noise: Optional[float] = None,
        rho: float = 7,
        min_noise_nugget: float = 0.0,
        sample_method: str = "ode",
    ) -> Array:
        y_ctx, y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)
        if y.ndim == 2:
            y = y[..., None, :]

        attention_mask = self.marginalization_mask(model_mask)

        # diffusion sampling
        theta = self.inference_decoder.sample(
            rng,
            y=y,
            tokenizer=self.tokenizer,
            dim=self.cfg.simulator.theta_dim,
            tokens_cfg=tokens_cfg,
            num_steps=num_steps,
            max_noise=max_noise,
            attention_mask=attention_mask,
            model_mask=model_mask,
            sample_method=sample_method,
            rho=rho,
            min_noise_nugget=min_noise_nugget,
            context=y_ctx,
        )

        return theta

    def log_prob_theta(
        self,
        theta: Array,
        acq: acquisition_scheme,
        x: Array,
        model_mask: Array,
        num_steps: int = 16,
        max_noise: Optional[float] = None,
        rho: float = 7,
        min_noise_nugget: float = 0.0,
    ) -> Array:
        y_ctx, y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)
        if y.ndim == 2:
            y = y[..., None, :]

        attention_mask = self.marginalization_mask(model_mask)

        log_prob = self.inference_decoder.log_prob(
            theta,
            y=y,
            tokenizer=self.tokenizer,
            tokens_cfg=tokens_cfg,
            num_steps=num_steps,
            max_noise=max_noise,
            attention_mask=attention_mask,
            model_mask=model_mask,
            rho=rho,
            min_noise_nugget=min_noise_nugget,
            context=y_ctx,
        )

        return log_prob

    def sample_and_log_prob_theta(
        self,
        rng: RngKey,
        acq: acquisition_scheme,
        x: ArrayLike,
        model_mask: Array,
        num_steps: int = 16,
        max_noise: Optional[float] = None,
        rho: float = 7,
        min_noise_nugget: float = 0.0,
    ) -> tuple[Array, Array]:
        y_ctx, y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)
        if y.ndim == 2:
            y = y[..., None, :]

        attention_mask = self.marginalization_mask(model_mask)

        theta, log_prob = self.inference_decoder.sample_and_log_prob(
            rng,
            y=y,
            tokenizer=self.tokenizer,
            dim=self.cfg.simulator.theta_dim,
            tokens_cfg=tokens_cfg,
            num_steps=num_steps,
            max_noise=max_noise,
            attention_mask=attention_mask,
            model_mask=model_mask,
            rho=rho,
            min_noise_nugget=min_noise_nugget,
            context=y_ctx,
        )

        return theta, log_prob

    def score_theta(
        self,
        theta: Array,
        acq: acquisition_scheme,
        x: ArrayLike,
        model_mask: Array,
        t: Optional[ArrayLike] = None,
    ) -> Array:
        if t is None:
            t = jnp.ones((1,)) * 0.01
        else:
            t = jnp.asarray(t)

        y_ctx, y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)
        if y.ndim == 2:
            y = y[..., None, :]

        attention_mask = self.marginalization_mask(model_mask)

        score = self.inference_decoder.score(
            t,
            theta,
            y=y,
            tokenizer=self.tokenizer,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            model_mask=model_mask,
            context=y_ctx,
        )

        return score
