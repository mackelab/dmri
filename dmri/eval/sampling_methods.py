from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from blackjax import hmc, tempered_smc
from blackjax.smc.resampling import systematic


def _resolve_devices(
    devices: Sequence[jax.Device] | str | None = None,
    preferred_device_kinds: Sequence[str] | None = ("gpu", "tpu"),
) -> tuple[jax.Device, ...]:
    """Resolve an explicit device list, falling back to preferred kinds when unspecified."""
    if isinstance(devices, str):
        resolved = tuple(jax.devices(devices))
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
            available = jax.devices(kind)
            if available:
                return tuple(available)
    available = tuple(jax.devices())
    if not available:
        raise RuntimeError("No JAX devices are available for evaluation.")
    return available


def eval_in_batches(
    fn,
    key,
    *data,
    batch_size=10_000,
    logger=None,
    min_batch_size=100,
    devices: Sequence[jax.Device] | str | None = None,
    preferred_device_kinds: Sequence[str] | None = ("gpu", "tpu"),
):
    """Evaluate a function on data batches, overlapping host prep with device compute."""
    eval_results = []
    if logger is not None:
        print_fn = logger.info
    else:
        print_fn = print

    # Capture expected total size for safety check
    expected_total_size = data[0].shape[0]

    resolved_devices = _resolve_devices(devices, preferred_device_kinds)
    num_devices = len(resolved_devices)

    def _finalize_single_result(res):
        """Bring a device result to host as numpy."""
        return np.asarray(jax.device_get(res))

    def _finalize_pmap_result(res, original_batch_size):
        np_results = np.asarray(res)
        flat_results = np_results.reshape(-1, *np_results.shape[2:])
        return flat_results[:original_batch_size]

    if num_devices > 1:
        # Use pmap when multiple devices are available
        device_desc = ", ".join(
            f"{d.platform}:{d.id}" if hasattr(d, "id") else d.platform
            for d in resolved_devices
        )
        print_fn(f"Using pmap across devices [{device_desc}]")

        # Split batch size across devices
        device_batch_size = max(batch_size // num_devices, min_batch_size)

        # Create pmap function
        @partial(jax.pmap, devices=resolved_devices)
        def pmap_fn(device_key, *device_data):
            # Split the device key for the batch
            batch_keys = jax.random.split(device_key, device_data[0].shape[0])
            return fn(batch_keys, *device_data)

        # Use device_batch_size * num_devices as the effective batch size
        current_batch_size = device_batch_size * num_devices
        batch_start = 0

        pending_result = None
        while batch_start < data[0].shape[0]:
            try:
                key, subkey = jax.random.split(key)
                print_fn(
                    f"Evaluating batch {batch_start} with batch size {current_batch_size} ({device_batch_size} per device across {num_devices} devices)"
                )
                batch_end = min(batch_start + current_batch_size, data[0].shape[0])
                batch_data = jax.tree_util.tree_map(
                    lambda x, start=batch_start, end=batch_end: x[start:end], data
                )

                # Split batch data across devices
                original_batch_size = batch_data[0].shape[0]
                padding_size = (
                    num_devices - (original_batch_size % num_devices)
                ) % num_devices
                if padding_size:
                    batch_data = jax.tree_util.tree_map(
                        lambda x, pad=padding_size: jnp.pad(
                            x, ((0, pad),) + ((0, 0),) * (x.ndim - 1)
                        ),
                        batch_data,
                    )

                device_batch_size_actual = batch_data[0].shape[0] // num_devices

                # Reshape data for pmap (num_devices, device_batch_size, ...)
                pmap_data = jax.tree_util.tree_map(
                    lambda x,
                    devices=num_devices,
                    bs=device_batch_size_actual: x.reshape(devices, bs, *x.shape[1:]),
                    batch_data,
                )

                # Split keys for each device
                device_keys = jax.random.split(subkey, num_devices)

                # Dispatch computation before blocking on previous result
                pmap_results = pmap_fn(device_keys, *pmap_data)
                if pending_result is not None:
                    eval_results.append(_finalize_pmap_result(*pending_result))
                pending_result = (pmap_results, original_batch_size)
                batch_start = batch_end
            except Exception as e:
                if (
                    "out of memory" in str(e).lower()
                    and current_batch_size > min_batch_size * num_devices
                ):
                    print_fn(
                        f"Out of memory error, reducing batch size from {current_batch_size} to {current_batch_size // 2}"
                    )
                    # Clear caches
                    jax.clear_caches()
                    current_batch_size = current_batch_size // 2
                    device_batch_size = max(
                        current_batch_size // num_devices, min_batch_size
                    )
                    continue
                else:
                    if pending_result is not None:
                        eval_results.append(_finalize_pmap_result(*pending_result))
                    raise e
        if pending_result is not None:
            eval_results.append(_finalize_pmap_result(*pending_result))
    else:
        # Fallback to original single-device implementation
        single_device = resolved_devices[0]
        print_fn(f"Using single device implementation on {single_device}")
        current_batch_size = batch_size
        batch_start = 0

        pending_result = None
        while batch_start < data[0].shape[0]:
            try:
                key, subkey = jax.random.split(key)
                print_fn(
                    f"Evaluating batch {batch_start} with batch size {current_batch_size}"
                )
                batch_end = min(batch_start + current_batch_size, data[0].shape[0])
                batch_data = jax.tree_util.tree_map(
                    lambda x, start=batch_start, end=batch_end: x[start:end], data
                )
                batch_keys = jax.random.split(subkey, batch_data[0].shape[0])
                batch_data_device = jax.device_put(batch_data, single_device)
                batch_keys_device = jax.device_put(batch_keys, single_device)

                # Dispatch computation; fetch previous result while this runs to overlap host/device work.
                batch_res = fn(batch_keys_device, *batch_data_device)
                if pending_result is not None:
                    eval_results.append(_finalize_single_result(pending_result))
                pending_result = batch_res
                batch_start = batch_end
            except Exception as e:
                if (
                    "out of memory" in str(e).lower()
                    and current_batch_size > min_batch_size
                ):
                    print_fn(
                        f"Out of memory error, reducing batch size from {current_batch_size} to {current_batch_size // 2}"
                    )
                    # Clear caches
                    jax.clear_caches()
                    current_batch_size = current_batch_size // 2
                    continue
                else:
                    if pending_result is not None:
                        eval_results.append(_finalize_single_result(pending_result))
                    raise e
        if pending_result is not None:
            eval_results.append(_finalize_single_result(pending_result))

    result = np.concatenate(eval_results, axis=0)

    # Safety check: ensure the result has the correct total dimensions
    if result.shape[0] != expected_total_size:
        raise ValueError(
            f"Result shape mismatch: expected first dimension to be {expected_total_size}, "
            f"but got {result.shape[0]}. This indicates a bug in the batching logic."
        )

    return result


def build_mask_sample_fn(method, num_samples, model, acq, p_mask):
    if method == "naive":

        def sample_mask_per_x(key, x):
            keys = jax.random.split(key, num_samples)
            return jax.vmap(model.sample_mask, in_axes=(0, None, None, None))(
                keys, acq, x, jnp.array([p_mask])
            )

        sample_mask_per_x = jax.jit(jax.vmap(sample_mask_per_x, in_axes=(0, 0)))

        return sample_mask_per_x
    else:
        # TODO Add temperature sampling
        raise ValueError(f"Method {method} not supported")


def build_theta_sample_fn(
    method, num_samples, model, acq, model_mask, sim_type, params, params_corrector
):
    num_steps = params.get("num_steps", 25)
    t_max = params.get("t_max", 80)
    t_min = params.get("t_min", 5e-3)

    def base_sample_fn(key, x, model_mask):
        if num_steps > 0:
            K = num_samples
            in_axes_model_mask = 0 if model_mask.ndim == 2 else None
            sample_fn = jax.vmap(
                partial(model.sample_theta, num_steps=num_steps, t_min=t_min, t_max=t_max),
                in_axes=(0, None, None, in_axes_model_mask),
            )
            keys = jax.random.split(key, K)
            theta = sample_fn(keys, acq, x, model_mask)
        else:
            theta = jax.random.normal(
                key, (num_samples, model.tokenizer.simulator.theta_dim)
            )
        return theta

    corrector = build_corrector(
        method, model, acq, model_mask, sim_type, params_corrector
    )

    # Combine sampling and correction
    def sample_theta_per_x(key, x, model_mask):
        key1, key2 = jax.random.split(key)
        theta = base_sample_fn(key1, x, model_mask)
        theta_corrected = corrector(key2, theta, x, model_mask)
        return theta_corrected

    if model_mask is None or model_mask.ndim > 1:
        sample_theta_per_x = jax.jit(jax.vmap(sample_theta_per_x, in_axes=(0, 0, 0)))
    else:
        sample_theta_per_x = jax.jit(
            jax.vmap(partial(sample_theta_per_x, model_mask=model_mask), in_axes=(0, 0))
        )

    return sample_theta_per_x


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
    print(model_mask.shape)

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
