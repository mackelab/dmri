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
from .simformer import DMRIThetaInferenceConfig, EDMSimformer, GaussianFourierEmbedding
from .tokenizer import DMRITokenizer, DMRITokenizerPP

EmbeddingModule = nnx.Module
TokenizerType = Type[DMRITokenizer]


@dataclass
class DMRIInferenceModelConfig:
    simulator: type[MultiCompartment]
    model_dim: int = 64
    use_attention_mask: bool = False
    inference_loss_type: str = "v"
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    tokenizer_cls: TokenizerType = DMRITokenizer
    embedding_cls: Type[EmbeddingModule] = BvalBvecSignalEmbeddingNet
    embedding_cfg: DMRIEmbeddingConfig = field(default_factory=DMRIEmbeddingConfig)
    model_selection_cfg: DMRIModelSelectionConfig = field(
        default_factory=DMRIModelSelectionConfig
    )
    theta_inference_cfg: DMRIThetaInferenceConfig = field(
        default_factory=DMRIThetaInferenceConfig
    )


@dataclass
class DMRIInferenceModelConfigMaskPriorAmortized:
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
    embedding_cfg: DMRIEmbeddingConfig = field(default_factory=DMRIEmbeddingConfig)
    model_selection_cfg: DMRIModelSelectionConfig = field(
        default_factory=DMRIModelSelectionAmortizedPriorConfig
    )
    theta_inference_cfg: DMRIThetaInferenceConfig = field(
        default_factory=DMRIThetaInferenceConfig
    )


@dataclass
class DMRIInferenceModelConfigMaskPriorAmortizedPP:
    simulator: type[MultiCompartment]
    model_dim: int = 64
    use_attention_mask: bool = True
    inference_loss_type: str = "v"
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    tokenizer_cls: TokenizerType = DMRITokenizerPP
    embedding_cls: Type[EmbeddingModule] = BvalBvecSignalEmbeddingNet
    embedding_cfg: DMRIEmbeddingConfig = field(default_factory=DMRIEmbeddingConfig)
    model_selection_cfg: DMRIModelSelectionConfig = field(
        default_factory=DMRIModelSelectionAmortizedPriorConfig
    )
    theta_inference_cfg: DMRIThetaInferenceConfig = field(
        default_factory=DMRIThetaInferenceConfig
    )


@dataclass
class SSFPInferenceModelConfig:
    simulator: type[MultiCompartment]
    model_dim: int = 64
    use_attention_mask: bool = True
    inference_loss_type: str = "v"
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    tokenizer_cls: TokenizerType = DMRITokenizerPP
    embedding_cls: Type[EmbeddingModule] = SSFPEmbeddingNet
    embedding_cfg: SSFPEmbeddingNetConfig = field(
        default_factory=SSFPEmbeddingNetConfig
    )
    model_selection_cfg: DMRIModelSelectionConfig = field(
        default_factory=DMRIModelSelectionConfig
    )
    theta_inference_cfg: DMRIThetaInferenceConfig = field(
        default_factory=DMRIThetaInferenceConfig
    )


class DMRIInferenceModel(nnx.Module):
    def __init__(self, cfg: DMRIInferenceModelConfig, rngs: RngKey) -> None:
        self.cfg: DMRIInferenceModelConfig = cfg
        precision_keys: tuple[str, ...] = (
            "dtype",
            "param_dtype",
            "precision",
            "preferred_element_type",
        )

        def _merge_kwargs(source: dict[str, Any]) -> dict[str, Any]:
            merged: dict[str, Any] = source.copy()
            for key in precision_keys:
                value = merged.get(key)
                if value is None:
                    fallback = getattr(cfg, key, None)
                    if fallback is not None:
                        merged[key] = fallback
            return {k: v for k, v in merged.items() if v is not None}

        # Setup embedding net observations
        embedding_kwargs = _merge_kwargs(vars(cfg.embedding_cfg))
        self.encoder = cfg.embedding_cls(
            rngs,
            model_dim=cfg.model_dim,
            **embedding_kwargs,
        )
        # Setup tokenizers
        tokenizer_kwargs = {
            key: getattr(cfg, key)
            for key in precision_keys
            if getattr(cfg, key, None) is not None
        }
        self.tokenizer: DMRITokenizer = cfg.tokenizer_cls(
            cfg.simulator, token_dim=cfg.model_dim, rngs=rngs, **tokenizer_kwargs
        )

        # Setup model selection network
        selection_cfg = vars(cfg.model_selection_cfg).copy()
        mask_prior_dim = selection_cfg.pop("mask_prior_dim", None)
        selection_kwargs = _merge_kwargs(selection_cfg)

        self.mask_prior_need: bool = False
        context_dim = selection_kwargs.get("context_dim")
        if context_dim is not None:
            # We expect a mask prior input
            self.mask_prior_need = True
            mask_prior_dim = mask_prior_dim or 1
            self.mask_prior_embed: GaussianFourierEmbedding = GaussianFourierEmbedding(
                mask_prior_dim,
                context_dim,
                rngs=rngs,
            )

        self.model_decoder = BinaryAutoregressiveDecoder(
            rngs,
            model_dim=cfg.model_dim,
            **selection_kwargs,
        )

        # Inference decoder
        theta_kwargs = _merge_kwargs(vars(cfg.theta_inference_cfg))
        simformer = EDMSimformer(
            rngs=rngs,
            model_dim=cfg.model_dim,
            **theta_kwargs,
            loss_type=cfg.inference_loss_type,
        )
        self.inference_decoder = simformer

    def __call__(
        self,
        model_mask: Array,
        theta: Array,
        x: Array,
        acq: acquisition_scheme,
        mask_prior: Optional[ArrayLike] = None,
        alpha_prior: Optional[ArrayLike] = None,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
        t: Optional[ArrayLike] = None,
    ) -> tuple[Array, Array]:
        # Embed model configuration
        tokens_cfg, y, mask_prior = self.embed_inputs(
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
            model_mask, self.tokenizer, y=y, tokens_cfg=tokens_cfg, context=mask_prior
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
        )

        return model_mask_logits, theta_pred

    def embed_inputs(
        self,
        model_mask: ArrayLike,
        x: ArrayLike,
        acq: acquisition_scheme,
        mask_prior: Optional[ArrayLike] = None,
        alpha_prior: Optional[ArrayLike] = None,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
    ) -> Tuple[Array, Array, Optional[Array]]:
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask,
            alpha_prior,
            model_idx=model_idx,
            noise_idx=noise_idx,
        )
        # Embed observations and acquisition parameters
        y = self.encoder(acq, x)
        if y.ndim == 2:
            y = y[..., None, :]

        # Embed mask prior
        if mask_prior is not None:
            mask_prior = self.mask_prior_embed(mask_prior)

        return tokens_cfg, y, mask_prior

    def theta_mask(
        self,
        model_mask: ArrayLike,
        model_idx: Optional[List[int]] = None,
        noise_idx: Optional[List[int]] = None,
    ) -> Array:
        theta_token_mask = self.tokenizer.theta_token_mask(
            model_mask, model_idx=model_idx, noise_idx=noise_idx
        )
        return theta_token_mask

    def marginalization_mask(
        self,
        model_mask: ArrayLike,
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
        model_mask: ArrayLike,
        theta: ArrayLike,
        x: ArrayLike,
        acq: acquisition_scheme,
        mask_prior: ArrayLike | None = None,
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
        y = self.encoder(acq, x)
        if y.ndim == 2:
            y = y[..., None, :]

        if self.mask_prior_need:
            assert mask_prior is not None, "Mask prior is required"
            mask_prior = self.mask_prior_embed(mask_prior)

        if permute_order:
            rng, permute_rng = jax.random.split(rng)
        else:
            permute_rng = rng

        model_mask_loss = self.model_decoder.loss_fn(
            model_mask,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            context=mask_prior,
            permute_order=permute_order,
            rng=permute_rng,
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
        )
        theta_loss /= jnp.sqrt(theta.shape[-1])

        return jnp.concatenate([model_mask_loss[None], theta_loss[None]])

    def sample_mask(
        self,
        rng: RngKey,
        acq: acquisition_scheme,
        x: ArrayLike,
        mask_prior: ArrayLike | None = None,
    ) -> Array:
        # Update for different model configs
        y = self.encoder(acq, x)
        if mask_prior is not None:
            mask_prior = self.mask_prior_embed(mask_prior)
        model_mask = self.model_decoder.sample(
            rng,
            tokenizer=self.tokenizer,
            y=y,
            dim=self.tokenizer.num_models + self.tokenizer.num_noises,
            context=mask_prior,
        )
        return model_mask

    def log_prob_mask(
        self,
        model_mask: ArrayLike,
        acq: acquisition_scheme,
        x: ArrayLike,
        mask_prior: ArrayLike | None = None,
    ) -> Array:
        y = self.encoder(acq, x)
        if mask_prior is not None:
            mask_prior = self.mask_prior_embed(mask_prior)
        log_prob = self.model_decoder.log_prob(
            model_mask,
            self.tokenizer,
            y=y,
            context=mask_prior,
        )
        return log_prob

    def sample_theta(
        self,
        rng: RngKey,
        acq: acquisition_scheme,
        x: ArrayLike,
        model_mask: ArrayLike,
        num_steps: int = 16,
        max_noise: Optional[float] = None,
        rho: float = 7,
        min_noise_nugget: float = 0.0,
        sample_method: str = "ode",
    ) -> Array:
        y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

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
        )

        return theta

    def log_prob_theta(
        self,
        theta: Array,
        acq: acquisition_scheme,
        x: ArrayLike,
        model_mask: ArrayLike,
        num_steps: int = 16,
        max_noise: Optional[float] = None,
        rho: float = 7,
        min_noise_nugget: float = 0.0,
    ) -> Array:
        y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

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
        )

        return log_prob

    def sample_and_log_prob_theta(
        self,
        rng: RngKey,
        acq: acquisition_scheme,
        x: ArrayLike,
        model_mask: ArrayLike,
        num_steps: int = 16,
        max_noise: Optional[float] = None,
        rho: float = 7,
        min_noise_nugget: float = 0.0,
    ) -> tuple[Array, Array]:
        y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

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
        )

        return theta, log_prob

    def score_theta(
        self,
        theta: Array,
        acq: acquisition_scheme,
        x: ArrayLike,
        model_mask: ArrayLike,
        t: Optional[ArrayLike] = None,
    ) -> Array:
        if t is None:
            t = jnp.ones((1,)) * 0.01

        y = self.encoder(acq, x)
        tokens_cfg = self.tokenizer.embed_cfgs(model_mask)

        attention_mask = self.marginalization_mask(model_mask)

        score = self.inference_decoder.score(
            t,
            theta,
            y=y,
            tokenizer=self.tokenizer,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
            model_mask=model_mask,
        )

        return score
