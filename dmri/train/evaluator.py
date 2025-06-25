from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from blackjax import hmc, tempered_smc
from blackjax.smc.resampling import systematic


def build_pure_eval_fns(graphdef, static, sim_type):
    @jax.jit
    def sample_masks(params, state, rng, data):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()
        p_mask = data[0]
        thetas = data[2]
        xs = data[3]
        acq = data[4]
        sample_fn = jax.vmap(model.sample_mask)
        rngs = jax.random.split(rng, len(thetas))
        masks_sampled = sample_fn(rngs, acq.bvals, acq.bvecs, xs, p_mask)
        return masks_sampled

    @jax.jit
    def log_prob_masks(params, state, data):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()

        p_mask = data[0]
        model_mask = data[1]
        xs = data[3]
        acq = data[4]
        return jax.vmap(model.log_prob_mask)(
            model_mask, acq.bvals, acq.bvecs, xs, p_mask
        )

    @jax.jit
    def sample_thetas(params, state, rng, data, num_steps=128, max_noise=40):
        model = nnx.merge(graphdef, params, static, state)
        model.eval()

        model_mask = data[1]
        xs = data[3]
        acq = data[4]
        thetas = data[2]
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

        model_mask = data[1]
        xs = data[3]
        acq = data[4]
        thetas = data[2]
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

        model_mask = data[1]
        xs = data[3]
        acq = data[4]
        thetas = data[2]
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
        model_mask = data[1]
        thetas = data[2]
        xs = data[3]
        acq = data[4]

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

    @partial(jax.jit, static_argnames=["K"])
    def smc_ess(params, state, rng, data, K=100):
        model_mask = data[1]
        acq = data[4]
        xs = data[3]

        model = nnx.merge(graphdef, params, static, state)
        model.eval()

        key1, key2 = jax.random.split(rng)
        keys_K = jax.random.split(key1, K)  # noqa: N806

        def sample_thetas(rng):
            keys = jax.random.split(rng, xs.shape[0])
            return jax.vmap(partial(model.sample_theta, num_steps=64, max_noise=80))(
                keys, acq.bvals, acq.bvecs, xs, model_mask
            )

        thetas_post = jax.vmap(sample_thetas)(keys_K)

        print(thetas_post.shape, xs.shape)

        def smc_ess_single(thetas_post, xs, acq, model_mask):
            def log_prior_fn(thetas):
                return jax.scipy.stats.norm.logpdf(thetas).sum(-1)

            def log_likelihood_fn(thetas):
                simulator = sim_type.from_theta(thetas, model_mask=model_mask)
                ll = simulator.log_likelihood(acq, xs)
                return ll

            hmc_kernel = hmc.build_kernel()
            hmc_kernel = partial(
                hmc_kernel,
                step_size=0.001,
                num_integration_steps=20,
                inverse_mass_matrix=jnp.ones(thetas_post.shape[1]),
            )

            resampling_fn = systematic

            smc = tempered_smc(
                log_prior_fn,
                log_likelihood_fn,
                hmc_kernel,
                hmc.init,
                {},
                resampling_fn,
                2,
            )
            state = smc.init(thetas_post)
            state = state._replace(lmbda=0.999)

            def step(state, rng):
                state, i = state
                lmbda = 0.999 + (i + 1) * 0.001 / 1
                new_state, info = smc.step(rng, state, lmbda)
                return (new_state, i + 1), info

            rng_keys = jax.random.split(key2, 1)
            final_state, _ = jax.lax.scan(step, (state, 0), rng_keys)
            final_weights = final_state[0].weights
            ess = 1 / jnp.sum(final_weights**2, axis=0)
            ess /= K
            return ess

        ess = jax.vmap(smc_ess_single, in_axes=(1, 0, 0, 0))(
            thetas_post, xs, acq, model_mask
        )
        return jnp.mean(ess)

    return Evaluator(
        sample_masks=sample_masks,
        log_prob_masks=log_prob_masks,
        sample_thetas=sample_thetas,
        log_prob_thetas=log_prob_thetas,
        sample_and_log_prob_thetas=sample_and_log_prob_thetas,
        true_loglikelihood=true_loglikelihood,
        true_posterior=true_posterior,
        smc_ess=smc_ess,
    )


class Evaluator(NamedTuple):
    sample_masks: Callable
    log_prob_masks: Callable
    sample_thetas: Callable
    log_prob_thetas: Callable
    sample_and_log_prob_thetas: Callable
    true_loglikelihood: Callable
    true_posterior: Callable
    smc_ess: Callable
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

    def eval_effective_sample_size(self, params, state, loader, rng, iters=1, K=100):
        # avg_ess = 0.0
        # i = 0
        # for eval_data in loader:
        #     log_weights = []
        #     for _ in range(K):
        #         rng, rng_eval = jax.random.split(rng)
        #         thetas_q, log_probs_q = self.sample_and_log_prob_thetas(
        #             params, state, rng_eval, eval_data
        #         )
        #         eval_data = list(eval_data)
        #         eval_data[2] = thetas_q
        #         log_probs_p = self.true_posterior(eval_data)
        #         log_weight = log_probs_q - log_probs_p
        #         log_weights.append(log_weight)
        #     log_weights = jnp.stack(log_weights, axis=0)
        #     weights = jax.nn.softmax(log_weights, axis=0)
        #     ess = 1.0 / jnp.sum(weights**2, axis=0)
        #     normed_ess = ess / K
        #     avg_ess += jnp.mean(normed_ess)
        #     i += 1
        #     if i == iters:
        #         break
        # return float(avg_ess) / iters
        ess = 0.0
        i = 0
        for eval_data in loader:
            ess += float(self.smc_ess(params, state, rng, eval_data, K=K))
            i += 1
            if i == iters:
                break
        return ess / iters

    def eval_tarp_mask():
        pass
