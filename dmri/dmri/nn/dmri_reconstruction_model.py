from .autoregressive import BinaryAutoregressiveDecoder
from .embedding_net import BvalBvecSignalEmbeddingNet
from .simformer import EDMSimformer
from .tokenizer import StructuredTokenizer

from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx


from jax.typing import ArrayLike
from dataclasses import dataclass

from dmri.simulators import MultiCompartment
import optax


@dataclass
class DMRIInferenceModelConfig:
    simulator: MultiCompartment
    max_bval: float = 2000
    model_dim: int = 64


class DMRIInferenceModel(nnx.Module, experimental_pytree=True):
    def __init__(self, cfg: DMRIInferenceModelConfig, rngs):
        self.cfg = cfg
        self.encoder = BvalBvecSignalEmbeddingNet(
            rngs, cfg.max_bval, model_dim=cfg.model_dim
        )
        self.model_decoder = BinaryAutoregressiveDecoder(rngs, model_dim=cfg.model_dim)

        # Inference
        simulator = cfg.simulator
        dims_per_component = list(simulator.split_idx())
        self.num_nodes = len(dims_per_component)
        simformer = EDMSimformer(
            tokenizer=StructuredTokenizer(dims_per_component, rngs=rngs),
            rngs=rngs,
            model_dim=cfg.model_dim,
        )
        self.inference_decoder = simformer

    def __call__(self, **kwargs):
        raise ValueError("Not implemented")

    def _loss_model_mask(self, params_model_mask, model_mask, y, rng=None):
        nnx.update(self.model_decoder, params_model_mask)
        logits = self.model_decoder(model_mask, y=y)
        loss = optax.sigmoid_binary_cross_entropy(logits, model_mask).sum(-1).mean()
        return loss

    def _loss_inference(
        self, params_inference, x, y, node_ids=None, model_mask=None, rng=None
    ):
        if node_ids is None:
            node_ids = list(range(self.num_nodes))
        model_mask_repeated = jnp.tile(model_mask, (1, 64 // model_mask.shape[-1] + 1))[
            :, :64
        ]
        loss = self.inference_decoder.loss(
            params_inference,
            rng=rng,
            data=x,
            node_ids=node_ids,
            y=y,
            context=model_mask_repeated,
        )
        return loss

    def _loss(
        self,
        params_encoder,
        params_model_mask,
        params_inference,
        model_mask,
        x,
        bvals,
        bvecs,
        signals,
        rng=None,
    ):
        nnx.update(self.encoder, params_encoder)
        y = self.encoder(bvals, bvecs, signals)
        loss_model_mask = self._loss_model_mask(
            params_model_mask, model_mask, y, rng=rng
        )
        loss_inference = self._loss_inference(
            params_inference, x, y, model_mask=model_mask, rng=rng
        )
        loss_inference *= 1 / jnp.sqrt(x.shape[-1])
        return loss_model_mask + loss_inference

    def sample_mask(self, rng, bvals, bvecs, signals):
        y = self.encoder(bvals, bvecs, signals)
        model_mask = self.model_decoder.sample(rng, y, self.num_nodes - 1)
        return model_mask

    def sample_theta(
        self,
        rng,
        bvals,
        bvecs,
        signals,
        model_mask,
        node_ids=None,
        num_steps=16,
        max_noise=None,
    ):
        y = self.encoder(bvals, bvecs, signals)

        if node_ids is None:
            node_ids = list(range(self.num_nodes))

        model_mask_repeated = jnp.tile(model_mask, (1, 64 // model_mask.shape[-1] + 1))[
            :, :64
        ]

        # diffusion sampling
        theta = self.inference_decoder.sample(
            rng,
            y,
            self.cfg.simulator.theta_dim,
            node_ids,
            context=model_mask_repeated,
            num_steps=num_steps,
            max_noise=max_noise,
        )

        return theta
