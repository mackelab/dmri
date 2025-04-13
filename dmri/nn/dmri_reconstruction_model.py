from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from dmri.simulators import MultiCompartment

from .autoregressive import (
    BinaryAutoregressiveDecoder,
    DMRIModelSelectionAmortizedPriorConfig,
    DMRIModelSelectionConfig,
)
from .embedding_net import BvalBvecSignalEmbeddingNet, DMRIEmbeddingConfig
from .simformer import DMRIThetaInferenceConfig, EDMSimformer, GaussianFourierEmbedding
from .tokenizer import DMRITokenizer, DMRITokenizerPP


@dataclass
class DMRIInferenceModelConfig:
    simulator: type[MultiCompartment]
    model_dim: int = 64
    use_attention_mask: bool = False
    tokenizer = DMRITokenizer
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
    tokenizer = DMRITokenizer
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
    tokenizer = DMRITokenizerPP
    embedding_cfg: DMRIEmbeddingConfig = field(default_factory=DMRIEmbeddingConfig)
    model_selection_cfg: DMRIModelSelectionConfig = field(
        default_factory=DMRIModelSelectionAmortizedPriorConfig
    )
    theta_inference_cfg: DMRIThetaInferenceConfig = field(
        default_factory=DMRIThetaInferenceConfig
    )

class DMRIInferenceModel(nnx.Module, experimental_pytree=True):
    def __init__(self, cfg: DMRIInferenceModelConfig, rngs):
        self.cfg = cfg
        # Setup embedding net observations
        self.encoder = BvalBvecSignalEmbeddingNet(
            rngs,
            model_dim=cfg.model_dim,
            **cfg.embedding_cfg.__dict__,
        )
        # Setup tokenizers
        self.tokenizer = cfg.tokenizer(
            cfg.simulator, token_dim=cfg.model_dim, rngs=rngs
        )

        # Setup model selection network
        params = cfg.model_selection_cfg.__dict__
        if cfg.model_selection_cfg.context_dim is not None:
            # We expect a mask prior input
            self.mask_prior_need = True
            mask_prior_dim = params.pop("mask_prior_dim",1)
            self.mask_prior_embed = GaussianFourierEmbedding(
                mask_prior_dim,
                cfg.model_selection_cfg.context_dim,
                rngs=rngs,
            )

        self.model_decoder = BinaryAutoregressiveDecoder(
            rngs,
            model_dim=cfg.model_dim,
            **cfg.model_selection_cfg.__dict__,
        )

        # Inference decoder
        simformer = EDMSimformer(
            rngs=rngs,
            model_dim=cfg.model_dim,
            **cfg.theta_inference_cfg.__dict__,
        )
        self.inference_decoder = simformer

    def __call__(
        self,
        model_mask: ArrayLike,
        theta: ArrayLike,
        x: ArrayLike,
        bvals: ArrayLike,
        bvecs: ArrayLike,
        mask_prior: Optional[ArrayLike] = None,
        alpha_prior: Optional[ArrayLike] = None,
        model_types: Optional[List[type]] = None,
        noise_types: Optional[List[type]] = None,
        t: Optional[ArrayLike] = None,
    ):
        # Embed model configuration
        tokens_cfg, y, mask_prior = self.embed_inputs(
            model_mask,
            x,
            bvals,
            bvecs,
            mask_prior=mask_prior,
            alpha_prior=alpha_prior,
            model_types=model_types,
            noise_types=noise_types,
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
            model_mask, model_types=model_types, noise_types=noise_types
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
        model_mask,
        x,
        bvals,
        bvecs,
        mask_prior=None,
        alpha_prior=None,
        model_types=None,
        noise_types=None,
    ):
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask, alpha_prior, model_types=model_types, noise_types=noise_types
        )
        # Embed observations and acquisition parameters
        y = self.encoder(bvals, bvecs, x)
        if y.ndim == 2:
            y = y[..., None, :]

        # Embed mask prior
        if mask_prior is not None:
            mask_prior = self.mask_prior_embed(mask_prior)

        return tokens_cfg, y, mask_prior

    def theta_mask(
        self,
        model_mask,
        model_types: Optional[list[type]] = None,
        noise_types: Optional[list[type]] = None,
    ):
        theta_token_mask = self.tokenizer.theta_token_mask(model_mask, model_types=model_types, noise_types=noise_types)
        # Expand this by the


    def marginalization_mask(
        self,
        model_mask,
        model_types: Optional[list[type]] = None,
        noise_types: Optional[list[type]] = None,
    ):
        _model_mask_extended = self.tokenizer.theta_token_mask(
            model_mask, model_types=model_types, noise_types=noise_types
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
        rng,
        model_mask,
        theta,
        x,
        bvals,
        bvecs,
        mask_prior=None,
        alpha_prior=None,
        model_types=None,
        noise_types=None,
        permute_order=False,
        use_loss_mask=False,
    ):
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask, alpha_prior, model_types=model_types, noise_types=noise_types
        )
        # Embed observatiosn
        y = self.encoder(bvals, bvecs, x)
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
            loss_mask = ~jax.vmap(partial(self.tokenizer.theta_mask, model_types=model_types, noise_types=noise_types))(model_mask)
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
        )
        theta_loss /= jnp.sqrt(theta.shape[-1])

        return jnp.concatenate([model_mask_loss[None], theta_loss[None]])

    def sample_mask(self, rng, bvals, bvecs, signals, mask_prior=None):
        # Update for different model configs
        y = self.encoder(bvals, bvecs, signals)
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
        model_mask,
        bvals,
        bvecs,
        signals,
        mask_prior=None,
    ):
        y = self.encoder(bvals, bvecs, signals)
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
        rng,
        bvals,
        bvecs,
        signals,
        model_mask,
        num_steps=16,
        max_noise=None,
    ):
        y = self.encoder(bvals, bvecs, signals)
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
        )

        return theta

    def log_prob_theta(
        self,
        theta,
        bvals,
        bvecs,
        signals,
        model_mask,
        num_steps=16,
        max_noise=None,
    ):
        y = self.encoder(bvals, bvecs, signals)
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
        )

        return log_prob

    def sample_and_log_prob_theta(
        self,
        rng,
        bvals,
        bvecs,
        signals,
        model_mask,
        num_steps=16,
        max_noise=None,
    ):
        y = self.encoder(bvals, bvecs, signals)
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
        )

        return theta, log_prob

    def score_theta(
        self,
        theta,
        bvals,
        bvecs,
        signals,
        model_mask,
        t=None,
    ):
        if t is None:
            t = jnp.ones((1,)) * 0.001

        y = self.encoder(bvals, bvecs, signals)
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
