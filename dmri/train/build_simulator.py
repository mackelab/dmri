from __future__ import annotations

import jax
from omegaconf import DictConfig

from dmri.simulators.config import (
    mask_prior_overrides,
    resolve_acquisition_factories,
    resolve_simulator_model,
    simulator_option,
)


def _simulator_cfg(cfg):
    return cfg.simulator if hasattr(cfg, "simulator") else cfg


def _make_simulation_generator(
    model_class,
    acquisition_factory,
    mask_prior,
    posterior_score,
):
    def simulator(rng, mask_prior_hyperparameter=None, model_mask=None):
        rng0, rng1, rng2, rng3 = jax.random.split(rng, 4)
        acquisition = acquisition_factory(rng0)
        theta = jax.random.normal(rng1, shape=(model_class.theta_dim,))
        rng_prior, rng_mask = jax.random.split(rng2)
        if mask_prior_hyperparameter is None:
            p_mask = mask_prior.sample_hyperparameters(rng_prior)
        else:
            p_mask = mask_prior_hyperparameter
        if model_mask is None:
            model_mask = mask_prior.sample_model_mask(rng_mask, p_mask)

        output = {
            "mask_prior": p_mask,
            "model_mask": model_mask,
            "theta": theta,
            "acq": acquisition,
        }
        if not posterior_score:
            model = model_class.from_theta(theta, model_mask=model_mask)
            output["x"] = model.signal(acquisition, rng=rng3)
            return output

        def posterior_potential(value):
            model = model_class.from_theta(value, model_mask=model_mask)
            observed = jax.lax.stop_gradient(model.signal(acquisition, rng=rng3))
            potential = model.log_likelihood(acquisition, observed).sum()
            potential += jax.scipy.stats.norm.logpdf(value, 0, 1).sum(-1)
            return potential, observed

        score, observed = jax.grad(posterior_potential, has_aux=True)(theta)
        output["x"] = observed
        output["target_score"] = score
        return output

    return simulator


def build_simulator(cfg: DictConfig):
    """Build the signal-model class and training-data generators."""
    simulator_cfg = _simulator_cfg(cfg)
    model_class = resolve_simulator_model(simulator_cfg)
    acquisition_factories = resolve_acquisition_factories(simulator_cfg)
    mask_prior = model_class.create_mask_prior(**mask_prior_overrides(simulator_cfg))
    posterior_score = bool(
        simulator_option(
            simulator_cfg,
            "posterior_score",
            "with_posterior_score",
            False,
        )
    )
    generators = [
        _make_simulation_generator(
            model_class,
            acquisition_factory,
            mask_prior,
            posterior_score,
        )
        for acquisition_factory in acquisition_factories
    ]
    return model_class, generators
