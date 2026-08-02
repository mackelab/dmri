from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from blackjax import hmc, tempered_smc
from blackjax.smc.resampling import systematic

from dmri import console
from dmri.eval.autobatch import (
    available_devices,
    batch_sharding_for,
    store_cached_batch_size,
)


def _resolve_devices(
    devices: Sequence[jax.Device] | str | None = None,
    preferred_device_kinds: Sequence[str] | None = ("gpu", "tpu"),
) -> tuple[jax.Device, ...]:
    """Resolve an explicit device list, falling back to preferred kinds when unspecified."""
    if isinstance(devices, str):
        resolved = available_devices(devices)
        if not resolved:
            raise ValueError(f"No JAX devices available for kind '{devices}'.")
        return resolved
    if devices is not None:
        resolved = tuple(devices)
        if not resolved:
            raise ValueError("Device sequence is empty.")
        return resolved
    if preferred_device_kinds:
        for kind in preferred_device_kinds:
            available = available_devices(kind)
            if available:
                return available
    available = tuple(jax.devices())
    if not available:
        raise RuntimeError("No JAX devices are available for evaluation.")
    return available


_OOM_MARKERS = ("out of memory", "resource_exhausted", "failed to allocate")


def _is_oom(error) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _OOM_MARKERS)


def _voxel_progress(desc, total):
    """A progress display over voxels when user-facing output is enabled."""
    return console.voxel_progress(desc, total)


def _pad_to(batch_data, target_size):
    """Pad the leading axis of every array up to ``target_size``."""
    padding = target_size - batch_data[0].shape[0]
    if padding <= 0:
        return batch_data
    return jax.tree_util.tree_map(
        lambda x: jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1)),
        batch_data,
    )


def eval_in_batches(
    fn,
    key,
    *data,
    batch_size=10_000,
    logger=None,
    min_batch_size=100,
    devices: Sequence[jax.Device] | str | None = None,
    preferred_device_kinds: Sequence[str] | None = ("gpu", "tpu"),
    cache_key: str | None = None,
    desc: str | None = None,
):
    """Evaluate a function on data batches, overlapping host prep with device compute.

    Every batch is padded to ``batch_size`` so only a single shape is ever traced;
    the padded rows are trimmed from the results.

    Each row gets its key from a single up-front split over the whole input, so
    the result depends only on ``key`` -- not on the batch size, the device count
    or where the batch boundaries happen to fall.
    """
    eval_results = []
    if logger is not None:
        print_fn = logger.info
    else:
        print_fn = print

    # Capture expected total size for safety check
    expected_total_size = data[0].shape[0]

    resolved_devices = _resolve_devices(devices, preferred_device_kinds)
    num_devices = len(resolved_devices)
    batch_sharding = batch_sharding_for(resolved_devices)

    if num_devices > 1:
        device_desc = ", ".join(
            f"{d.platform}:{d.id}" if hasattr(d, "id") else d.platform
            for d in resolved_devices
        )
        print_fn(f"Sharding batches across {num_devices} devices [{device_desc}]")
    else:
        print_fn(f"Using single device implementation on {resolved_devices[0]}")

    def _finalize(res, original_batch_size):
        """Bring a device result to host as numpy, dropping padded rows."""
        return jax.tree_util.tree_map(
            lambda leaf: np.asarray(jax.device_get(leaf))[:original_batch_size], res
        )

    def _record_batch_size(size):
        if cache_key is not None:
            store_cached_batch_size(cache_key, size)

    # One key per row, decided before batching, so the draws depend on the seed
    # alone and not on where the batch boundaries fall.
    all_keys = np.asarray(jax.random.split(key, expected_total_size))

    current_batch_size = batch_size
    batch_start = 0
    pending_result = None
    progress = _voxel_progress(desc, expected_total_size)

    try:
        while batch_start < data[0].shape[0]:
            try:
                if progress is None:
                    shards = (
                        f" ({current_batch_size // num_devices} per device across "
                        f"{num_devices} devices)"
                        if num_devices > 1
                        else ""
                    )
                    print_fn(
                        f"Evaluating batch {batch_start} with batch size "
                        f"{current_batch_size}{shards}"
                    )
                batch_end = min(batch_start + current_batch_size, data[0].shape[0])
                batch_data = jax.tree_util.tree_map(
                    lambda x, start=batch_start, end=batch_end: x[start:end], data
                )
                # Pad up to the full batch so only one shape is ever traced, keeping
                # the total divisible by the device count so the shards stay even.
                original_batch_size = batch_data[0].shape[0]
                pad_target = min(current_batch_size, expected_total_size)
                pad_target += (-pad_target) % num_devices
                batch_data = _pad_to(batch_data, pad_target)

                batch_keys = _pad_to((all_keys[batch_start:batch_end],), pad_target)[0]
                # Sharding the leading axis lets jit partition the work.
                batch_data_device = jax.device_put(batch_data, batch_sharding)
                batch_keys_device = jax.device_put(batch_keys, batch_sharding)

                # Dispatch computation; fetch the previous result while this runs.
                batch_res = fn(batch_keys_device, *batch_data_device)
                if pending_result is not None:
                    eval_results.append(_finalize(*pending_result))
                pending_result = (batch_res, original_batch_size)
                if progress is not None:
                    progress.update(batch_end - batch_start)
                batch_start = batch_end
            except Exception as error:
                if _is_oom(error) and current_batch_size > min_batch_size * num_devices:
                    reduced = max(current_batch_size // 2, min_batch_size * num_devices)
                    reduced -= reduced % num_devices
                    print_fn(
                        "Out of memory error, reducing batch size from "
                        f"{current_batch_size} to {reduced}"
                    )
                    jax.clear_caches()
                    current_batch_size = reduced
                    _record_batch_size(current_batch_size)
                    if progress is not None:
                        progress.reset()
                        progress.update(batch_start)
                    continue
                if pending_result is not None:
                    eval_results.append(_finalize(*pending_result))
                raise

        if pending_result is not None:
            eval_results.append(_finalize(*pending_result))
    finally:
        if progress is not None:
            progress.close()

    if len(eval_results) == 1:
        result = eval_results[0]
    else:
        result = jax.tree_util.tree_map(
            lambda *parts: np.concatenate(parts, axis=0), *eval_results
        )

    # Safety check: ensure the result has the correct total dimensions
    for leaf in jax.tree_util.tree_leaves(result):
        if leaf.shape[0] != expected_total_size:
            raise ValueError(
                f"Result shape mismatch: expected first dimension to be "
                f"{expected_total_size}, but got {leaf.shape[0]}. This indicates a "
                "bug in the batching logic."
            )

    return result


def build_mask_sample_fn(method, num_samples, model, acq, p_mask, feasible_models=None):
    """Sample model masks, optionally alongside feasible-mask probabilities.

    Both use the same head on the same signal, so when the probabilities are
    wanted they share this pass and its encoding. Returns just the masks when
    ``feasible_models`` is None, else ``(masks, probabilities)``.
    """
    if method != "naive":
        # TODO Add temperature sampling
        raise ValueError(f"Method {method} not supported")

    prior = jnp.array([p_mask])
    feasible = (
        None if feasible_models is None else jnp.asarray(feasible_models, jnp.bool_)
    )

    def per_voxel(key, x):
        y_ctx, y = model._encode_observations(acq, x)
        keys = jax.random.split(key, num_samples)
        masks = jax.vmap(model.sample_mask, in_axes=(0, None, None, None, None, None))(
            keys,
            acq,
            x,
            prior,
            y_ctx,
            y,
        )
        if feasible is None:
            return masks
        log_pmf = jax.vmap(
            model.log_prob_mask, in_axes=(0, None, None, None, None, None)
        )(feasible, acq, x, prior, y_ctx, y)
        return masks, jax.nn.softmax(log_pmf, axis=-1)

    return jax.jit(jax.vmap(per_voxel, in_axes=(0, 0)))


def _batched_over_voxels(fn, model_mask, num_leading_args):
    """vmap ``fn`` over voxels, binding a shared mask when it is not per-voxel."""
    if model_mask is None or model_mask.ndim > 1:
        return jax.jit(jax.vmap(fn, in_axes=(0,) * (num_leading_args + 1)))
    return jax.jit(
        jax.vmap(partial(fn, model_mask=model_mask), in_axes=(0,) * num_leading_args)
    )


def build_base_theta_sample_fn(num_samples, model, acq, model_mask, params):
    """The network sampler alone: signal -> raw theta samples.

    Kept separate from the corrector, which fits a far larger batch.
    """
    num_steps = params.get("num_steps", 25)
    t_max = params.get("t_max", 80)
    t_min = params.get("t_min")
    last_euler_step = params.get("last_euler_step", True)

    def base_sample_fn(key, x, model_mask):
        if num_steps > 0:
            in_axes_model_mask = 0 if model_mask.ndim == 2 else None
            sample_fn = jax.vmap(
                partial(
                    model.sample_theta,
                    num_steps=num_steps,
                    last_euler_step=last_euler_step,
                    t_min=t_min,
                    t_max=t_max,
                ),
                in_axes=(0, None, None, in_axes_model_mask),
            )
            keys = jax.random.split(key, num_samples)
            theta = sample_fn(keys, acq, x, model_mask)
        else:
            theta = jax.random.normal(
                key, (num_samples, model.tokenizer.simulator.theta_dim)
            )
        # The bijection, correctors and likelihood downstream all assume float32,
        # so a half-precision network must not leak into them.
        return theta.astype(jnp.float32)

    return _batched_over_voxels(base_sample_fn, model_mask, 2)


def network_evaluations_per_sample(params):
    """Network evaluations the ODE solver performs for one posterior sample.

    One to initialise, one per interval between the ``num_steps`` grid points,
    and one more if a final Euler correction is taken.
    """
    num_steps = params.get("num_steps", 25)
    if num_steps <= 0:
        return 0
    return 1 + max(num_steps - 1, 0) + int(params.get("last_euler_step", True))


def build_corrector_fn(method, model, acq, model_mask, sim_type, params_corrector):
    """The corrector alone: (raw thetas, signal) -> corrected thetas.

    Operates on theta rather than on network activations, so it is far cheaper
    per voxel and gets its own, much larger batch size.
    """
    corrector = build_corrector(
        method, model, acq, model_mask, sim_type, params_corrector
    )

    def correct_per_x(key, theta, x, model_mask):
        return jnp.asarray(corrector(key, theta, x, model_mask), dtype=jnp.float32)

    return _batched_over_voxels(correct_per_x, model_mask, 3)


def build_model_fn(sim_type):
    def log_likelihood_fn(theta, mask, acq, x):
        simulator = sim_type.from_theta(theta, model_mask=mask)
        ll = simulator.log_likelihood(acq, x)
        ll = jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
        return ll

    def log_prior_fn(theta, mask):
        if mask is not None:
            theta_mask = sim_type.theta_mask(model_mask=mask)
            logpdf = jax.scipy.stats.norm.logpdf(theta, 0, 1)
            logpdf = jnp.where(theta_mask, logpdf, 0.0)
            return jnp.sum(logpdf, axis=-1)
        else:
            return jax.scipy.stats.norm.logpdf(theta, 0, 1).sum()

    def log_posterior_fn(theta, mask, acq, x):
        return log_prior_fn(theta, mask) + log_likelihood_fn(theta, mask, acq, x)

    return log_posterior_fn, log_prior_fn, log_likelihood_fn


def build_corrector(method, model, acq, model_mask, sim_type, params):
    d = model.tokenizer.simulator.theta_dim
    posterior_fn, prior_fn, likelihood_fn = build_model_fn(sim_type)

    num_integration_steps = params.get("num_integration_steps", 5)
    step_size = params.get("step_size", 0.005)
    hmc_kernel = hmc.build_kernel()
    hmc_kernel = partial(
        hmc_kernel,
        num_integration_steps=num_integration_steps,
        step_size=step_size,
        inverse_mass_matrix=jnp.ones(d),
    )

    if method == "auto":
        method = "smc_corrected" if model_mask.ndim < 3 else "mcmc_corrected"

    if method == "uncorrected":

        def corrector(key, theta, x, model_mask):
            return theta

    elif method == "smc_corrected":
        lam_start = params.get("lam_start", 0.99)
        num_steps = params.get("num_steps", 5)
        num_inner_steps = params.get("num_inner_steps", 1)

        def corrector(key, theta, x, model_mask):
            assert model_mask is None or model_mask.ndim <= 1, (
                "Model mask must be None or 1D"
            )
            resampling_fn = systematic
            _log_likelihood_fn = partial(likelihood_fn, mask=model_mask, acq=acq, x=x)
            _prior_fn = partial(prior_fn, mask=model_mask)
            smc = tempered_smc(
                _prior_fn,
                _log_likelihood_fn,
                hmc_kernel,
                hmc.init,
                {},
                resampling_fn,
                num_inner_steps,
            )
            state = smc.init(theta)
            state = state._replace(tempering_param=lam_start)

            def step(state, rng):
                state, i = state
                lmbda = lam_start + (i + 1) * (1 - lam_start) / num_steps
                new_state, _ = smc.step(rng, state, lmbda)
                return (new_state, i + 1), None

            rng_keys = jax.random.split(key, num_steps)
            final_state, _ = jax.lax.scan(step, (state, 0), rng_keys)
            return final_state[0].particles

    elif method == "mcmc_corrected":
        num_steps = params.get("num_steps", 5)

        def _run_single(key, theta_single, x, mask_single):
            alg = hmc(
                partial(posterior_fn, mask=mask_single, acq=acq, x=x),
                step_size,
                jnp.ones(d),
                num_integration_steps,
            )
            state = alg.init(theta_single)

            def step(state, key):
                state, _ = alg.step(key, state)
                return state, None

            keys = jax.random.split(key, num_steps)
            state, _ = jax.lax.scan(step, state, keys)
            return state.position

        def corrector(key, theta, x, model_mask):
            # Support batched theta/model_mask by vmapping over the sample dimension.
            if theta.ndim == 1:
                return _run_single(key, theta, x, model_mask)

            num_samples = theta.shape[0]
            keys = jax.random.split(key, num_samples)

            if model_mask is None or model_mask.ndim == 1:
                vmapped = jax.vmap(
                    lambda k, t: _run_single(k, t, x, model_mask), in_axes=(0, 0)
                )
                return vmapped(keys, theta)

            # model_mask is batched alongside theta; keep x shared across samples.
            vmapped = jax.vmap(
                lambda k, t, m: _run_single(k, t, x, m), in_axes=(0, 0, 0)
            )
            return vmapped(keys, theta, model_mask)

    else:
        raise ValueError(f"Method {method} not supported")

    return corrector
