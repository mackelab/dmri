import importlib
from functools import partial

import jax
from omegaconf import DictConfig

from dmri.simulators.acquisition_scheme import (
    random_advanced_reasearch_acquisition_scheme,
    random_clinical_acquisition,
    random_hardi_acquisition,
    random_hcp_acquisition,
    random_hcp_large_acquisition,
    random_ssfp_acquisition,
)
from dmri.simulators.mask_prior import BetaBernoulliMaskPrior, TotalParamPenalizedPrior


def build_simulator(cfg: DictConfig):
    sim_type_name = cfg.simulator.sim_type.name
    sim_type_module = importlib.import_module("dmri.simulators.multi_compartment")
    sim_type = getattr(sim_type_module, sim_type_name)

    acq_scheme_name = cfg.simulator.acquisition_scheme.name
    acq_params = cfg.simulator.acquisition_scheme.params
    acq_schemes = acq_params.schemes

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
        elif acq_scheme_name == "ssfp":
            acq_fn = partial(random_ssfp_acquisition, **acq_params)
        else:
            raise ValueError(f"Unknown acquisition scheme {acq_scheme_name}")

        with_posterior_score = cfg.simulator.with_posterior_score

        mask_prior_overrides = {}
        if issubclass(sim_type.mask_prior_cls, BetaBernoulliMaskPrior):
            prior_mask_alpha = getattr(cfg.simulator, "prior_mask_alpha", None)
            prior_mask_beta = getattr(cfg.simulator, "prior_mask_beta", None)
            if prior_mask_alpha is not None:
                mask_prior_overrides["alpha"] = prior_mask_alpha
            if prior_mask_beta is not None:
                mask_prior_overrides["beta"] = prior_mask_beta
        elif issubclass(sim_type.mask_prior_cls, TotalParamPenalizedPrior):
            prior_mask_p0 = getattr(cfg.simulator, "prior_mask_p0", None)
            prior_mask_u_alpha = getattr(cfg.simulator, "prior_mask_u_alpha", None)
            prior_mask_u_beta = getattr(cfg.simulator, "prior_mask_u_beta", None)
            if prior_mask_p0 is not None:
                mask_prior_overrides["p0"] = prior_mask_p0
            if prior_mask_u_alpha is not None:
                mask_prior_overrides["u_alpha"] = prior_mask_u_alpha
            if prior_mask_u_beta is not None:
                mask_prior_overrides["u_beta"] = prior_mask_u_beta
            mask_prior_overrides["num_model_parameters"] = [
                mt.theta_dim for mt in sim_type.model_types
            ]

        def create_simulator(
            acq_fn,
            *,
            mask_overrides=mask_prior_overrides,
            posterior_score=with_posterior_score,
        ):
            mask_prior_dist = sim_type.create_mask_prior(**mask_overrides)

            def simulator(rng, mask_prior_hyperparameter=None, model_mask=None):
                rng0, rng1, rng2, rng3 = jax.random.split(rng, 4)
                acq = acq_fn(rng0)
                theta = jax.random.normal(rng1, shape=(sim_type.theta_dim,))
                rng_prior, rng_mask = jax.random.split(rng2)
                if mask_prior_hyperparameter is None:
                    p_mask = mask_prior_dist.sample_hyperparameters(rng_prior)
                else:
                    p_mask = mask_prior_hyperparameter
                if model_mask is None:
                    model_mask = mask_prior_dist.sample_model_mask(rng_mask, p_mask)

                output = {
                    "mask_prior": p_mask,
                    "model_mask": model_mask,
                    "theta": theta,
                    "acq": acq,
                }

                if not posterior_score:
                    dmri_simulator = sim_type.from_theta(theta, model_mask=model_mask)
                    x = dmri_simulator.signal(acq, rng=rng3)
                    output["x"] = x
                else:

                    def posterior_potential(theta):
                        dmri_simulator = sim_type.from_theta(
                            theta, model_mask=model_mask
                        )
                        x_o = jax.lax.stop_gradient(
                            dmri_simulator.signal(acq, rng=rng3)
                        )
                        return dmri_simulator.log_likelihood(
                            acq, x_o
                        ).sum() + jax.scipy.stats.norm.logpdf(theta, 0, 1).sum(-1), x_o

                    score, x_o = jax.grad(posterior_potential, has_aux=True)(theta)
                    output["x"] = x_o
                    output["target_score"] = score
                return output

            return simulator

        simulators.append(create_simulator(acq_fn))

    return sim_type, simulators
