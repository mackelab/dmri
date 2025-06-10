import importlib
from functools import partial

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from dmri.simulators.acquisition_scheme import (
    random_hcp_acquisition,
    random_advanced_reasearch_acquisition_scheme,
    random_clinical_acquisition,
    random_hardi_acquisition,
    random_hcp_large_acquisition,
)


def build_simulator(cfg: DictConfig):
    sim_type_name = cfg.simulator.sim_type.name
    sim_type_module = importlib.import_module("dmri.simulators.multi_compartment")
    sim_type = getattr(sim_type_module, sim_type_name)

    acq_scheme_name = cfg.simulator.acquisition_scheme.name
    acq_params = cfg.simulator.acquisition_scheme.params

    if acq_scheme_name == "multi":
        acq_schemes = acq_params.schemes
    else:
        acq_schemes = [{"name": acq_scheme_name, "params": acq_params}]

    simulators = []

    for acq_scheme in acq_schemes:
        acq_scheme_name = acq_scheme.name
        acq_params = acq_scheme.params
        if acq_scheme_name == "clinical":
            acq_fn = partial(random_clinical_acquisition, **acq_params)
        elif acq_scheme_name == "hardi":
            acq_fn = partial(random_hardi_acquisition, **acq_params)
        elif acq_scheme_name == "advanced_research":
            acq_fn = partial(random_advanced_reasearch_acquisition_scheme, **acq_params)
        elif acq_scheme_name == "hcp":
            acq_fn = partial(random_hcp_acquisition, **acq_params)
        elif acq_scheme_name == "hcp_large":
            acq_fn = partial(random_hcp_large_acquisition, **acq_params)
        else:
            raise ValueError(f"Unknown acquisition scheme {acq_scheme_name}")

        prior_mask_alpha = cfg.simulator.prior_mask_alpha
        prior_mask_beta = cfg.simulator.prior_mask_beta
        with_posterior_score = cfg.simulator.with_posterior_score

        def create_simulator(acq_fn):
            def simulator(rng):
                rng0, rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 6)
                acq = acq_fn(rng0)
                theta = jax.random.normal(rng1, shape=(sim_type.theta_dim,))
                # Model mask
                p_mask = jax.random.beta(
                    rng2, a=prior_mask_alpha, b=prior_mask_beta, shape=(1,)
                )

                # Ensure that at least one compartment is always on
                rng3_1, rng3_2 = jax.random.split(rng3)
                idx_compartment_always_on = jax.random.randint(
                    rng3_1, minval=0, maxval=len(sim_type.model_types), shape=(1,)
                )
                p_mask_extended = jnp.repeat(p_mask, len(sim_type.model_types))
                p_mask_extended = p_mask_extended.at[idx_compartment_always_on].set(0.0)

                model_mask = jax.random.bernoulli(rng3_2, p=p_mask_extended)
                model_mask = model_mask.at[idx_compartment_always_on].set(True)

                # Noise mask needs to have only one entry
                model_noise_idx = jax.random.randint(
                    rng4, minval=0, maxval=len(sim_type.noise_types), shape=(1,)
                )
                model_noise_mask = jnp.zeros(len(sim_type.noise_types), dtype=jnp.bool)
                model_noise_mask = model_noise_mask.at[model_noise_idx].set(True)
                model_mask = jnp.concatenate([model_mask, model_noise_mask], axis=-1)

                if not with_posterior_score:
                    dmri_simulator = sim_type.from_theta(theta, model_mask=model_mask)
                    x_o = dmri_simulator.signal(acq, rng=rng5)
                    return p_mask, model_mask, theta, x_o, acq
                else:
                    def posterior_potential(theta):
                        dmri_simulator = sim_type.from_theta(theta, model_mask=model_mask)
                        x_o = jax.lax.stop_gradient(dmri_simulator.signal(acq, rng=rng5))
                        return dmri_simulator.log_likelihood(acq, x_o).sum() + jax.scipy.stats.norm.logpdf(theta, 0, 1).sum(-1), x_o
                    score, x_o = jax.grad(posterior_potential, has_aux=True)(theta)
                    return p_mask, model_mask, theta, x_o, acq, score

            return simulator

        simulators.append(create_simulator(acq_fn))

    return sim_type, simulators
