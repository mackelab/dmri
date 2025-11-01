from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax import hmc, tempered_smc
from blackjax.smc.resampling import systematic
from flax import nnx


def build_pure_eval_fns(graphdef, static, sim_type):
    @jax.jit
    def sample_masks(params, state, rng, data):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()
        p_mask = data[0]
        thetas = data[2]
        xs = data[3]
        acq = data[4]
        sample_fn = jax.vmap(model.sample_mask)
        rngs = jax.random.split(rng, len(thetas))
        masks_sampled = sample_fn(rngs, acq, xs, p_mask)
        return masks_sampled

    @jax.jit
    def log_prob_masks(params, state, data):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

        p_mask = data[0]
        model_mask = data[1]
        xs = data[3]
        acq = data[4]
        return jax.vmap(model.log_prob_mask)(model_mask, acq, xs, p_mask)

    @jax.jit
    def sample_thetas(params, state, rng, data, num_steps=128, max_noise=40):
        model = nnx.merge(graphdef, params, static, state, copy=True)
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
        return sample_fn(rngs, acq, xs, model_mask)

    @jax.jit
    def log_prob_thetas(params, state, data, num_steps=128, max_noise=40):
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

        model_mask = data[1]
        xs = data[3]
        acq = data[4]
        thetas = data[2]
        sample_fn = partial(
            model.log_prob_theta, num_steps=num_steps, max_noise=max_noise
        )
        return jax.vmap(sample_fn)(thetas, acq, xs, model_mask)

    @jax.jit
    def sample_and_log_prob_thetas(
        params, state, rng, data, num_steps=128, max_noise=40
    ):
        model = nnx.merge(graphdef, params, static, state, copy=True)
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
        )(rngs, acq, xs, model_mask)
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

    def _normalize_weights(weights):
        total = jnp.sum(weights)
        total = jnp.where(total == 0.0, 1.0, total)
        return weights / total

    def _resample_particles(particles, weights, rng):
        weights = _normalize_weights(weights)
        num_particles = particles.shape[0]
        indices = jax.random.choice(
            rng,
            jnp.arange(num_particles),
            shape=(num_particles,),
            replace=True,
            p=weights,
        )
        return particles[indices]

    def _sliced_wasserstein_distance(
        samples, reference, rng, num_projections
    ):
        dim = samples.shape[-1]
        directions = jax.random.normal(rng, (num_projections, dim))
        directions = directions / jnp.maximum(
            jnp.linalg.norm(directions, axis=-1, keepdims=True), 1e-12
        )

        def _distance(direction):
            proj_samples = jnp.sort(samples @ direction)
            proj_reference = jnp.sort(reference @ direction)
            length = min(proj_samples.shape[0], proj_reference.shape[0])
            proj_samples = proj_samples[:length]
            proj_reference = proj_reference[:length]
            return jnp.mean((proj_samples - proj_reference) ** 2)

        distances = jax.vmap(_distance)(directions)
        return jnp.sqrt(jnp.mean(distances))

    @partial(
        jax.jit, static_argnames=["K", "num_projections", "num_smc_steps"]
    )
    def smc_sliced_wasserstein(
        params,
        state,
        rng,
        data,
        K=100,
        num_projections=64,
        num_smc_steps=5,
    ):
        model_mask = data[1]
        acq = data[4]
        xs = data[3]

        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

        key_samples, key_metrics = jax.random.split(rng)
        keys_K = jax.random.split(key_samples, K)  # noqa: N806

        def sample_thetas(rng):
            keys = jax.random.split(rng, xs.shape[0])
            return jax.vmap(
                partial(model.sample_theta, num_steps=64, max_noise=80)
            )(keys, acq, xs, model_mask)

        thetas_post = jax.vmap(sample_thetas)(keys_K)

        def smc_swd_single(
            thetas_post_single, xs_single, acq_single, model_mask_single, rng_single
        ):
            rng_smc, rng_resample, rng_direction = jax.random.split(
                rng_single, 3
            )

            def log_prior_fn(thetas):
                return jax.scipy.stats.norm.logpdf(thetas).sum(-1)

            def log_likelihood_fn(thetas):
                simulator = sim_type.from_theta(
                    thetas, model_mask=model_mask_single
                )
                ll = simulator.log_likelihood(acq_single, xs_single)
                return ll

            hmc_kernel = hmc.build_kernel()
            hmc_kernel = partial(
                hmc_kernel,
                step_size=0.001,
                num_integration_steps=20,
                inverse_mass_matrix=jnp.ones(thetas_post_single.shape[1]),
            )

            smc = tempered_smc(
                log_prior_fn,
                log_likelihood_fn,
                hmc_kernel,
                hmc.init,
                {},
                systematic,
                2,
            )
            tempered_state = smc.init(thetas_post_single)
            tempered_state = tempered_state._replace(lmbda=0.999)

            def step(carry, rng_step):
                current_state, i = carry
                lmbda = jnp.clip(0.999 + (i + 1) * 0.001, a_max=1.0)
                new_state, info = smc.step(rng_step, current_state, lmbda)
                return (new_state, i + 1), info

            keys_scan = jax.random.split(rng_smc, num_smc_steps)
            (final_state, _), _ = jax.lax.scan(
                step, (tempered_state, 0), keys_scan
            )
            final_state = final_state
            corrected_particles = final_state.particles
            corrected_weights = final_state.weights
            resampled = _resample_particles(
                corrected_particles, corrected_weights, rng_resample
            )
            return _sliced_wasserstein_distance(
                thetas_post_single, resampled, rng_direction, num_projections
            )

        keys_data = jax.random.split(key_metrics, xs.shape[0])
        distances = jax.vmap(
            smc_swd_single, in_axes=(1, 0, 0, 0, 0)
        )(thetas_post, xs, acq, model_mask, keys_data)
        return jnp.mean(distances)

    return Evaluator(
        sample_masks=sample_masks,
        log_prob_masks=log_prob_masks,
        sample_thetas=sample_thetas,
        log_prob_thetas=log_prob_thetas,
        sample_and_log_prob_thetas=sample_and_log_prob_thetas,
        true_loglikelihood=true_loglikelihood,
        true_posterior=true_posterior,
        sliced_wasserstein=smc_sliced_wasserstein,
    )


class Evaluator(NamedTuple):
    sample_masks: Callable
    log_prob_masks: Callable
    sample_thetas: Callable
    log_prob_thetas: Callable
    sample_and_log_prob_thetas: Callable
    true_loglikelihood: Callable
    true_posterior: Callable
    sliced_wasserstein: Callable
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

    def eval_sliced_wasserstein_distance(
        self,
        params,
        state,
        loader,
        rng,
        iters=1,
        K=100,
        num_projections=64,
        num_smc_steps=5,
    ):
        distance = 0.0
        i = 0
        for eval_data in loader:
            rng, eval_rng = jax.random.split(rng)
            distance += float(
                self.sliced_wasserstein(
                    params,
                    state,
                    eval_rng,
                    eval_data,
                    K=K,
                    num_projections=num_projections,
                    num_smc_steps=num_smc_steps,
                )
            )
            i += 1
            if i == iters:
                break
        return distance / max(i, 1)

    def eval_tarp_mask():
        pass
