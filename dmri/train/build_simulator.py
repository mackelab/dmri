from omegaconf import DictConfig

from dmri.simulators.acquisition_scheme import (
    random_advanced_reasearch_acquisition_scheme,
    random_hardi_acquisition,
    random_clinical_acquisition,
)

import jax
import jax.numpy as jnp
import importlib

from functools import partial


def build_simulator(cfg: DictConfig):
    sim_type_name = cfg.simulator.sim_type.name
    sim_type_module = importlib.import_module("dmri.simulators.multi_compartment")
    sim_type = getattr(sim_type_module, sim_type_name)

    acq_scheme_name = cfg.simulator.acquisition_scheme.name
    acq_params = cfg.simulator.acquisition_scheme.params
    if acq_scheme_name == "clinical":
        acq_fn = partial(random_clinical_acquisition, **acq_params)
    elif acq_scheme_name == "hardi":
        acq_fn = partial(random_hardi_acquisition, **acq_params)
    elif acq_scheme_name == "advanced_research":
        acq_fn = partial(random_advanced_reasearch_acquisition_scheme, **acq_params)
    else:
        raise ValueError(f"Unknown acquisition scheme {acq_scheme_name}")

    prior_mask_alpha = cfg.simulator.prior_mask_alpha
    prior_mask_beta = cfg.simulator.prior_mask_beta

    def simulator(rng):
        rng0, rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 6)
        acq = acq_fn(rng0)
        theta = jax.random.normal(rng1, shape=(sim_type.theta_dim,))
        # Model mask
        p_mask = jax.random.beta(
            rng2, a=prior_mask_alpha, b=prior_mask_beta, shape=(1,)
        )
        model_mask = jax.random.bernoulli(
            rng3, p=p_mask, shape=(len(sim_type.model_types))
        )
        # Noise mask needs to have only one entry
        model_noise_idx = jax.random.randint(
            rng4, minval=0, maxval=len(sim_type.noise_types), shape=(1,)
        )
        model_noise_mask = jnp.zeros(len(sim_type.noise_types), dtype=jnp.bool)
        model_noise_mask = model_noise_mask.at[model_noise_idx].set(True)
        model_mask = jnp.concatenate([model_mask, model_noise_mask], axis=-1)
        dmri_simulator = sim_type.from_theta(theta, model_mask=model_mask)

        x_o = dmri_simulator.signal(acq, rng=rng5)
        return p_mask, model_mask, theta, x_o, acq

    return sim_type, simulator
