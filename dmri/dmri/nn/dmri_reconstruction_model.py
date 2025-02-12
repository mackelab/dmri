from .autoregressive import BinaryAutoregressiveDecoder
from .embedding_net import BvalBvecSignalEmbeddingNet
from .simformer import EDMSimformer
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
    max_bval: float = 2000
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
        alpha_prior: Optional[ArrayLike] = None,
        model_types: Optional[List[type]] = None,
        noise_types: Optional[List[type]] = None,
        t: Optional[ArrayLike] = None,
    ):
        # Embed model configuration
        tokens_cfg = self.tokenizer.embed_cfgs(
            model_mask, alpha_prior, model_types=model_types, noise_types=noise_types
        )
        # Embed observatiosn
        y = self.encoder(bvals, bvecs, x)
        if y.ndim == 2:
            y = y[..., None, :]

        print("y", y.shape)
        print("tokens_cfg", tokens_cfg.shape)
        # Get model_mask logits
        model_mask_logits = self.model_decoder(
            model_mask, self.tokenizer, y=y, tokens_cfg=tokens_cfg
        )

        # Get theta predictions
        if t is None:
            t = jnp.ones((theta.shape[0], 1)) * 0.0001
        theta_pred = self.inference_decoder(
            t, theta, self.tokenizer, y=y, tokens_cfg=tokens_cfg
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
        y = y[..., None, :]

        model_mask_loss = self.model_decoder.loss_fn(
            None, model_mask, self.tokenizer, y=y, tokens_cfg=tokens_cfg
        )
        theta_loss = self.inference_decoder.loss(
            None, rng, theta, self.tokenizer, y=y, tokens_cfg=tokens_cfg
        )
        theta_loss /= jnp.sqrt(theta.shape[-1])

        return jnp.concatenate([model_mask_loss[None], theta_loss[None]])

    # def _loss_model_mask(self, params_model_mask, model_mask, y, rng=None):
    #     nnx.update(self.model_decoder, params_model_mask)
    #     logits = self.model_decoder(model_mask, y=y)
    #     loss = optax.sigmoid_binary_cross_entropy(logits, model_mask).sum(-1).mean()
    #     return loss

    # def _loss_inference(
    #     self, params_inference, x, y, node_ids=None, model_mask=None, rng=None
    # ):
    #     if node_ids is None:
    #         node_ids = list(range(self.num_nodes))
    #     model_mask_repeated = jnp.tile(model_mask, (1, 64 // model_mask.shape[-1] + 1))[
    #         :, :64
    #     ]

    #     # Add first "true" to the model
    #     model_mask_first = jnp.ones((model_mask.shape[0], 1), dtype=bool)
    #     _model_mask = jnp.concatenate([model_mask_first, model_mask], axis=-1)
    #     attention_mask = _model_mask[:, None, :] & _model_mask[:, :, None]  # [B, N, N]
    #     attention_mask = (
    #         attention_mask | jnp.eye(_model_mask.shape[-1], dtype=bool)[None, :, :]
    #     )
    #     loss = self.inference_decoder.loss(
    #         params_inference,
    #         rng=rng,
    #         data=x,
    #         node_ids=node_ids,
    #         y=y,
    #         attention_mask=attention_mask,
    #         context=model_mask_repeated,
    #     )
    #     return loss

    # def _loss(
    #     self,
    #     params_encoder,
    #     params_model_mask,
    #     params_inference,
    #     model_mask,
    #     x,
    #     bvals,
    #     bvecs,
    #     signals,
    #     rng=None,
    # ):
    #     nnx.update(self.encoder, params_encoder)
    #     y = self.encoder(bvals, bvecs, signals)
    #     loss_model_mask = self._loss_model_mask(
    #         params_model_mask, model_mask, y, rng=rng
    #     )
    #     loss_inference = self._loss_inference(
    #         params_inference, x, y, model_mask=model_mask, rng=rng
    #     )
    #     loss_inference *= 1 / jnp.sqrt(x.shape[-1])
    #     return loss_model_mask + loss_inference

    # def model_mask_to_theta_mask(self, model_mask):
    #     dims_per_component = list(self.cfg.simulator.split_idx())
    #     theta_mask = []
    #     for i, d in enumerate(dims_per_component):
    #         theta_mask.append(jnp.repeat(model_mask[:, i : i + 1], d, axis=-1))
    #     return jnp.concatenate(theta_mask, axis=-1)

    # def sample_mask(self, rng, bvals, bvecs, signals):
    #     y = self.encoder(bvals, bvecs, signals)
    #     model_mask = self.model_decoder.sample(rng, y, self.num_nodes - 1)
    #     return model_mask

    # def sample_theta(
    #     self,
    #     rng,
    #     bvals,
    #     bvecs,
    #     signals,
    #     model_mask,
    #     node_ids=None,
    #     num_steps=16,
    #     max_noise=None,
    # ):
    #     y = self.encoder(bvals, bvecs, signals)

    #     if node_ids is None:
    #         node_ids = list(range(self.num_nodes))

    #     model_mask_repeated = jnp.tile(model_mask, (1, 64 // model_mask.shape[-1] + 1))[
    #         :, :64
    #     ]
    #     _model_mask_extended = jnp.concatenate([jnp.array([True]), model_mask])
    #     attention_mask = _model_mask_extended[None, :] & _model_mask_extended[:, None]
    #     attention_mask = attention_mask | jnp.eye(
    #         _model_mask_extended.shape[-1], dtype=bool
    #     )

    #     # diffusion sampling
    #     theta = self.inference_decoder.sample(
    #         rng,
    #         y,
    #         self.cfg.simulator.theta_dim,
    #         node_ids,
    #         context=model_mask_repeated,
    #         num_steps=num_steps,
    #         max_noise=max_noise,
    #         attention_mask=attention_mask,
    #     )

    #     return theta
