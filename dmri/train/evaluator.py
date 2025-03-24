from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx


def build_pure_eval_fns(graphdef, static, sim_type):
    @jax.jit
    def sample_masks(params, state, rng, data):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()
        p_mask, _, thetas, xs, acq = data
        sample_fn = jax.vmap(model.sample_mask)
        rngs = jax.random.split(rng, len(thetas))
        masks_sampled = sample_fn(rngs, acq.bvals, acq.bvecs, xs, p_mask)
        return masks_sampled

    @jax.jit
    def log_prob_masks(params, state, data):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()

        p_mask, model_mask, thetas, xs, acq = data
        return jax.vmap(model.log_prob_mask)(
            model_mask, acq.bvals, acq.bvecs, xs, p_mask
        )

    @jax.jit
    def sample_thetas(params, state, rng, data, num_steps=128, max_noise=40):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()

        _, model_mask, thetas, xs, acq = data
        sample_fn = jax.vmap(model.sample_theta)
        rngs = jax.random.split(rng, len(thetas))
        sample_fn = jax.vmap(
            partial(model.sample_theta, num_steps=num_steps, max_noise=max_noise)
        )
        return sample_fn(rngs, acq.bvals, acq.bvecs, xs, model_mask)

    @jax.jit
    def log_prob_thetas(params, state, data, num_steps=128, max_noise=40):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()

        p_mask, model_mask, thetas, xs, acq = data
        sample_fn = partial(
            model.log_prob_theta, num_steps=num_steps, max_noise=max_noise
        )
        return jax.vmap(sample_fn)(thetas, acq.bvals, acq.bvecs, xs, model_mask)

    @jax.jit
    def sample_and_log_prob_thetas(
        params, state, rng, data, num_steps=128, max_noise=40
    ):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()
        _, model_mask, _, xs, acq = data
        rngs = jax.random.split(rng, xs.shape[0])
        thetas, log_probs = jax.vmap(
            partial(
                model.sample_and_log_prob_theta,
                num_steps=num_steps,
                max_noise=max_noise,
            )
        )(rngs, acq.bvals, acq.bvecs, xs, model_mask)
        return thetas, log_probs

    @jax.jit
    def true_loglikelihood(data):
        p_mask, model_mask, thetas, xs, acq = data

        def single_ll(theta, model_mask, acq, x):
            simulator = sim_type.from_theta(theta, model_mask=model_mask)
            ll = simulator.log_likelihood(acq, x)
            return ll

        return jax.vmap(single_ll)(thetas, model_mask, acq, xs)

    @jax.jit
    def true_posterior(data):
        _, _, thetas, _, _ = data
        ll = true_loglikelihood(data)
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

    def eval_nnl_mask(self, params, state, loader, iters=10):
        total_log_prob = 0.0
        i = 0
        for eval_data in loader:
            total_log_prob += jnp.mean(self.log_prob_masks(params, state, eval_data))
            i += 1
            if i == iters:
                break
        return -float(total_log_prob) / iters

    def eval_nnl_theta(self, params, state, loader, iters=10):
        total_log_prob = 0.0
        i = 0
        for eval_data in loader:
            total_log_prob += jnp.mean(self.log_prob_thetas(params, state, eval_data))
            i += 1
            if i == iters:
                break
        return -float(total_log_prob) / iters

    def eval_effective_sample_size(self, params, state, loader, rng, iters=1, K=10):
        avg_ess = 0.0
        i = 0
        for eval_data in loader:
            log_weights = []
            for _ in range(K):
                rng, rng_eval = jax.random.split(rng)
                thetas_q, log_probs_q = self.sample_and_log_prob_thetas(
                    params, state, rng_eval, eval_data
                )
                eval_data = list(eval_data)
                eval_data[2] = thetas_q
                log_probs_p = self.true_posterior(eval_data)
                log_weight = log_probs_q - log_probs_p
                log_weights.append(log_weight)
            log_weights = jnp.stack(log_weights, axis=0)
            weights = jax.nn.softmax(log_weights, axis=0)
            ess = 1.0 / jnp.sum(weights**2, axis=0)
            normed_ess = ess / K
            avg_ess += jnp.mean(normed_ess)
            i += 1
            if i == iters:
                break
        return float(avg_ess) / iters

    def eval_tarp_mask():
        pass
