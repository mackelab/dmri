from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from dmri.eval.export_metrics import multiscale_preconditioned_ksd_and_pvalue


def build_pure_eval_fns(graphdef, sim_type):
    @jax.jit
    def sample_masks(params, state, rng, data):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.eval()
        p_mask = data["mask_prior"]
        thetas = data["theta"]
        xs = data["x"]
        acq = data["acq"]
        sample_fn = jax.vmap(model.sample_mask)
        rngs = jax.random.split(rng, len(thetas))
        masks_sampled = sample_fn(rngs, acq, xs, p_mask)
        return masks_sampled

    @jax.jit
    def log_prob_masks(params, state, data):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.eval()

        p_mask = data["mask_prior"]
        model_mask = data["model_mask"]
        xs = data["x"]
        acq = data["acq"]
        return jax.vmap(model.log_prob_mask)(model_mask, acq, xs, p_mask)

    @jax.jit
    def sample_thetas(
        params,
        state,
        rng,
        data,
        num_steps=64,
        t_min=None,
        t_max=None,
    ):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.eval()

        model_mask = data["model_mask"]
        xs = data["x"]
        acq = data["acq"]
        thetas = data["theta"]
        sample_fn = jax.vmap(model.sample_theta)
        rngs = jax.random.split(rng, len(thetas))
        sample_fn = jax.vmap(
            partial(
                model.sample_theta,
                num_steps=num_steps,
                t_max=t_max,
                t_min=t_min,
            )
        )
        return sample_fn(rngs, acq, xs, model_mask)

    def sample_thetas_multi(
        params, state, rng, data, num_samples=16, num_steps=64, t_min=None, t_max=None
    ):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.eval()

        model_mask = data["model_mask"]
        xs = data["x"]
        acq = data["acq"]

        def sample_once(subkey):
            subkeys = jax.random.split(subkey, xs.shape[0])
            sample_fn = jax.vmap(
                partial(
                    model.sample_theta,
                    num_steps=num_steps,
                    t_max=t_max,
                    t_min=t_min,
                )
            )
            return sample_fn(subkeys, acq, xs, model_mask)

        keys = jax.random.split(rng, num_samples)
        samples = jax.vmap(sample_once)(keys)
        return jnp.swapaxes(samples, 0, 1)

    sample_thetas_multi = jax.jit(
        sample_thetas_multi,
        static_argnames=("num_samples", "num_steps", "t_min", "t_max"),
    )

    @jax.jit
    def log_prob_thetas(
        params,
        state,
        data,
        num_steps=64,
        t_min=None,
        t_max=None,
    ):
        model = nnx.merge(graphdef, params, state, copy=True)
        model.eval()

        model_mask = data["model_mask"]
        xs = data["x"]
        acq = data["acq"]
        thetas = data["theta"]
        sample_fn = partial(
            model.log_prob_theta,
            num_steps=num_steps,
            t_max=t_max,
            t_min=t_min,
        )
        return jax.vmap(sample_fn)(thetas, acq, xs, model_mask)

    @jax.jit
    def true_loglikelihood(data):
        model_mask = data["model_mask"]
        thetas = data["theta"]
        xs = data["x"]
        acq = data["acq"]

        def single_ll(theta, model_mask, acq, x):
            simulator = sim_type.from_theta(theta, model_mask=model_mask)
            ll = simulator.log_likelihood(acq, x)
            return ll

        return jax.vmap(single_ll)(thetas, model_mask, acq, xs)

    @jax.jit
    def true_posterior(data):
        ll = true_loglikelihood(data)
        thetas = data["theta"]
        prior_logprob = jax.scipy.stats.norm.logpdf(thetas).sum(-1)
        return ll + prior_logprob

    return Evaluator(
        sample_masks=sample_masks,
        log_prob_masks=log_prob_masks,
        sample_thetas=sample_thetas,
        sample_thetas_multi=sample_thetas_multi,
        log_prob_thetas=log_prob_thetas,
        true_loglikelihood=true_loglikelihood,
        true_posterior=true_posterior,
        sim_type=sim_type,
    )


class Evaluator(NamedTuple):
    sample_masks: Callable
    log_prob_masks: Callable
    sample_thetas: Callable
    sample_thetas_multi: Callable
    log_prob_thetas: Callable
    true_loglikelihood: Callable
    true_posterior: Callable
    sim_type: Any
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

    def eval_ksd(
        self,
        params,
        state,
        loader,
        *,
        iters=1,
        num_samples=16,
        bandwidths=(0.1, 1.0, 10.0),
        n_bootstrap=128,
        ksd_seed=None,
    ):
        if iters == 0:
            return {"ksd": float("nan"), "ksd_pvalue": float("nan")}

        bandwidths_jnp = jnp.asarray(bandwidths)
        key = jax.random.PRNGKey(self.seed if ksd_seed is None else int(ksd_seed))
        total_ksd = 0.0
        total_pvalue = 0.0
        batches = 0

        for eval_data in loader:
            key, sample_key, ksd_key = jax.random.split(key, 3)
            # Shape: (batch, num_samples, theta_dim)
            theta_samples = self.sample_thetas_multi(
                params,
                state,
                sample_key,
                eval_data,
                num_samples=num_samples,
            )
            batch_size = theta_samples.shape[0]
            batch_keys = jax.random.split(ksd_key, batch_size)

            acq = eval_data["acq"]
            xs = eval_data["x"]
            if xs.ndim > 2:
                xs = xs.reshape(xs.shape[0], -1)
            elif xs.ndim == 2:
                xs = xs.reshape(xs.shape[0], -1)
            model_mask = eval_data["model_mask"]

            def _ksd_single(k, th, mask, x, acq):
                ksd2, p_val = multiscale_preconditioned_ksd_and_pvalue(
                    sim_type=self.sim_type,
                    thetas=th,
                    mask=mask,
                    acq=acq,
                    x=x,
                    key=k,
                    bandwidths=bandwidths_jnp,
                    n_bootstrap=n_bootstrap,
                )
                return ksd2, p_val

            ksd_vals, p_vals = jax.vmap(_ksd_single)(
                batch_keys, theta_samples, model_mask, xs, acq
            )
            # Median as this can have heavy tails
            total_ksd += float(jnp.median(ksd_vals))
            total_pvalue += float(jnp.median(p_vals))
            batches += 1
            if batches == iters:
                break

        if batches == 0:
            return {"ksd": float("nan"), "ksd_pvalue": float("nan")}

        return {
            "ksd": total_ksd / batches,
            "ksd_pvalue": total_pvalue / batches,
        }
