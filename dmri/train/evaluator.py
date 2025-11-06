from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

def build_pure_eval_fns(graphdef, static, sim_type):
    @jax.jit
    def sample_masks(params, state, rng, data):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()
        p_mask = data['mask_prior']
        thetas = data['theta']
        xs = data['x']
        acq = data['acq']
        sample_fn = jax.vmap(model.sample_mask)
        rngs = jax.random.split(rng, len(thetas))
        masks_sampled = sample_fn(rngs, acq, xs, p_mask)
        return masks_sampled

    @jax.jit
    def log_prob_masks(params, state, data):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

        p_mask = data['mask_prior']
        model_mask = data['model_mask']
        xs = data['x']
        acq = data['acq']
        return jax.vmap(model.log_prob_mask)(model_mask, acq, xs, p_mask)

    @jax.jit
    def sample_thetas(params, state, rng, data, num_steps=64, max_noise=None, min_noise=None, rho=None):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

        model_mask = data['model_mask']
        xs = data['x']
        acq = data['acq']
        thetas = data['theta']
        sample_fn = jax.vmap(model.sample_theta)
        rngs = jax.random.split(rng, len(thetas))
        sample_fn = jax.vmap(
            partial(model.sample_theta, num_steps=num_steps, max_noise=max_noise, min_noise=min_noise, rho=rho)
        )
        return sample_fn(rngs, acq, xs, model_mask)

    @jax.jit
    def log_prob_thetas(params, state, data, num_steps=64, max_noise=None, min_noise=None, rho=None):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

        model_mask = data['model_mask']
        xs = data['x']
        acq = data['acq']
        thetas = data['theta']
        sample_fn = partial(
            model.log_prob_theta, num_steps=num_steps, max_noise=max_noise, min_noise=min_noise, rho=rho
        )
        return jax.vmap(sample_fn)(thetas, acq, xs, model_mask)

    @jax.jit
    def sample_and_log_prob_thetas(
        params, state, rng, data, num_steps=64, max_noise=None, min_noise=None, rho=None
    ):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

        model_mask = data['model_mask']
        xs = data['x']
        acq = data['acq']
        thetas = data['theta']
        rngs = jax.random.split(rng, xs.shape[0])
        thetas, log_probs = jax.vmap(
            partial(
                model.sample_and_log_prob_theta,
                num_steps=num_steps,
                max_noise=max_noise,
                min_noise=min_noise,
                rho=rho,
            )
        )(rngs, acq, xs, model_mask)
        return thetas, log_probs

    @jax.jit
    def true_loglikelihood(data):
        model_mask = data['model_mask']
        thetas = data['theta']
        xs = data['x']
        acq = data['acq']

        def single_ll(theta, model_mask, acq, x):
            simulator = sim_type.from_theta(theta, model_mask=model_mask)
            ll = simulator.log_likelihood(acq, x)
            return ll

        return jax.vmap(single_ll)(thetas, model_mask, acq, xs)

    @jax.jit
    def true_posterior(data):
        ll = true_loglikelihood(data)
        thetas = data['theta']
        prior_logprob = jax.scipy.stats.norm.logpdf(thetas).sum(-1)
        return ll + prior_logprob


    return Evaluator(
        sample_masks=sample_masks,
        log_prob_masks=log_prob_masks,
        sample_thetas=sample_thetas,
        log_prob_thetas=log_prob_thetas,
        sample_and_log_prob_thetas=sample_and_log_prob_thetas,
        true_loglikelihood=true_loglikelihood,
        true_posterior=true_posterior,
    )


class Evaluator(NamedTuple):
    sample_masks: Callable
    log_prob_masks: Callable
    sample_thetas: Callable
    log_prob_thetas: Callable
    sample_and_log_prob_thetas: Callable
    true_loglikelihood: Callable
    true_posterior: Callable
    seed: int = 42

    def eval_nnl_mask(self, params, state, loader, iters=1):
        total_log_prob = 0.0
        i = 0
        if iters == 0:
            return 0.0
        for eval_data in loader:
            total_log_prob += jnp.mean(self.log_prob_masks(params, state, eval_data))
            i += 1
            if i == iters:
                break
        return -float(total_log_prob) / iters

    def eval_nnl_theta(self, params, state, loader, iters=1):
        total_log_prob = 0.0
        i = 0
        if iters == 0:
            return 0.0
        for eval_data in loader:
            total_log_prob += jnp.mean(self.log_prob_thetas(params, state, eval_data))
            i += 1
            if i == iters:
                break
        return -float(total_log_prob) / iters
