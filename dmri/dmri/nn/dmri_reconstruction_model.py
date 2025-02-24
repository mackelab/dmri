from .autoregressive import BinaryAutoregressiveDecoder
from .embedding_net import BvalBvecSignalEmbeddingNet
from .simformer import EDMSimformer, GaussianFourierEmbedding
from .tokenizer import StructuredTokenizer, DMRITokenizer

from functools import partial
from typing import List, Optional
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx


from jax.typing import ArrayLike
from dataclasses import dataclass, field

from dmri.simulators import MultiCompartment
import optax

@dataclass
class DMRIEmbeddingConfig:
    num_layers: int = 3
    num_heads: int = 4
    widening_factor: int = 2
    attn_size: int = 16


@dataclass
class DMRIModelSelectionConfig:
    num_layers: int = 4
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    context_dim = None


@dataclass
class DMRIModelSelectionAmortizedPriorConfig:
    num_layers: int = 4
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    context_dim: int = 64
    mask_prior_dim: int = 1


@dataclass
class DMRIThetaInferenceConfig:
    num_layers: int = 6
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16

@dataclass
class DMRIInferenceModelConfig:
    simulator: type[MultiCompartment]
    model_dim: int = 64
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
        self.tokenizer = DMRITokenizer(
            cfg.simulator, token_dim=cfg.model_dim, rngs=rngs
        )

        # Setup model selection network
        params = cfg.model_selection_cfg.__dict__
        if cfg.model_selection_cfg.context_dim is not None:
            # We expect a mask prior input

            self.mask_prior_need = True
            mask_prior_dim = params.pop("mask_prior_dim")
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
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask,
            alpha_prior=alpha_prior,
            model_types=model_types,
            noise_types=noise_types,
        )
        # Embed observatiosn
        y = self.encoder(bvals, bvecs, x)
        if y.ndim == 2:
            y = y[..., None, :]

        # Get model_mask logits
        if self.mask_prior_need:
            assert mask_prior is not None, "Mask prior is required"
            mask_prior = self.mask_prior_embed(mask_prior)
        model_mask_logits = self.model_decoder(
            model_mask, self.tokenizer, y=y, tokens_cfg=tokens_cfg, context=mask_prior
        )

        # Get theta predictions
        if t is None:
            t = jnp.ones((theta.shape[0], 1)) * 0.0001
        # Mask out non-selected models
        # The first token is doing model fractions
        _model_mask_extended = jnp.concatenate(
            [jnp.ones((model_mask.shape[0], 1), dtype=bool), model_mask], axis=-1
        )
        attention_mask = (
            _model_mask_extended[..., None, :] & _model_mask_extended[..., :, None]
        )
        attention_mask = attention_mask | jnp.eye(
            _model_mask_extended.shape[-1], dtype=bool
        )
        theta_pred = self.inference_decoder(
            t,
            theta,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
        )

        return model_mask_logits, theta_pred

    def loss_fn(
        self,
        params,
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
    ):
        nnx.update(self, params)
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

        model_mask_loss = self.model_decoder.loss_fn(
            None,
            model_mask,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            context=mask_prior,
        )

        _model_mask_extended = jnp.concatenate(
            [jnp.ones((model_mask.shape[0], 1), dtype=bool), model_mask], axis=-1
        )
        attention_mask = (
            _model_mask_extended[..., None, :] & _model_mask_extended[..., :, None]
        )
        attention_mask = attention_mask | jnp.eye(
            _model_mask_extended.shape[-1], dtype=bool
        )

        theta_loss = self.inference_decoder.loss(
            None,
            rng,
            theta,
            self.tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            attention_mask=attention_mask,
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

        _model_mask_extended = jnp.concatenate([jnp.array([True]), model_mask])
        attention_mask = _model_mask_extended[None, :] & _model_mask_extended[:, None]
        attention_mask = attention_mask | jnp.eye(
            _model_mask_extended.shape[-1], dtype=bool
        )

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
        )

        return theta
