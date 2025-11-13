from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from dmri.eval.export_theta import embed_in_full_brain_array
from dmri.eval.sampling_methods import eval_in_batches
from dmri.utils.dmriutils import export_nifti

def compute_posterior_nll_metric(
    cfg,
    sim_type,
    acq,
    full_data_flat_in_brain,
    model_parameters_brain,
    model_mask,
    brain_mask_flat,
    data_norm,
    orig_data,
    out_path,
    *,
    devices=None,
):
    """Compute and export ground truth metrics.

    Args:
        cfg: Configuration object
        sim_type: Simulator type
        acq: Acquisition scheme
        full_data_flat_in_brain: Flattened data within brain mask
        model_parameters_brain: Model parameters for each voxel
        brain_mask_flat: Flattened brain mask
        data_norm: Normalized data
        orig_data: Reference image (affine + shape info)
        out_path: Output path for the error maps
    """

    def gt_metrics(_, x, theta, model_mask):
        simulator = sim_type.from_theta(theta, model_mask=model_mask)
        theta_mask = sim_type.theta_mask(model_mask)
        prior = jax.scipy.stats.norm.logpdf(theta, loc=0.0, scale=1.0)
        prior = jnp.sum(prior * theta_mask, axis=-1)
        ll = simulator.log_likelihood(x, acq)
        log_posterior = ll + prior
        return -log_posterior  # Negative log-likelihood (MSE for Gaussian noise)

    if model_mask is None or model_mask.ndim <= 1:
        metrics_fn = partial(gt_metrics, model_mask=model_mask)
        in_axes1 = (0, 0, 0)  # Only theta and x are batched
        in_axes2 = (None, None, 0)  # Only theta is batched in inner vmap
        data_eval = (full_data_flat_in_brain, model_parameters_brain)
    else:
        metrics_fn = gt_metrics
        in_axes1 = (0, 0, 0, 0)
        in_axes2 = (None, None, 0, None if model_mask.ndim < 3 else 0)
        data_eval = (full_data_flat_in_brain, model_parameters_brain, model_mask)

    _gt_metrics = jax.vmap(
        jax.vmap(metrics_fn, in_axes=in_axes2), in_axes=in_axes1
    )
    metrics_maps = eval_in_batches(
        _gt_metrics,
        jax.random.PRNGKey(0),
        *data_eval,
        batch_size=20_000,
        devices=devices,
    )
    metrics_maps = jnp.mean(metrics_maps, axis=1)  # Average over samples
    full_metrics_maps = embed_in_full_brain_array(
        metrics_maps, brain_mask_flat.astype(np.bool), data_norm.shape[:-1]
    )
    export_nifti(full_metrics_maps, orig_data, out_path, "metric_posterior_nll.nii.gz")




def compute_reconstruction_error(
    cfg,
    sim_type,
    acq,
    full_data_flat_in_brain,
    model_parameters_brain,
    model_mask,
    brain_mask_flat,
    data_norm,
    orig_data,
    out_path,
    *,
    devices=None,
):
    """Compute and export reconstruction error maps.

    Args:
        cfg: Configuration object
        sim_type: Simulator type
        acq: Acquisition scheme
        full_data_flat_in_brain: Flattened data within brain mask
        model_parameters_brain: Model parameters for each voxel
        models_selected_brain: Selected models for each voxel (can be None)
        brain_mask_flat: Flattened brain mask
        data_norm: Normalized data
        orig_data: Reference image (affine + shape info)
        out_path: Output path for the error maps
    """

    def reconstruction_error(_, x, theta, mask):
        simulator = sim_type.from_theta(theta, model_mask=mask)
        signal = simulator.signal(acq)
        return jnp.mean(jnp.abs(signal - x), axis=-1)

    if model_mask is None or model_mask.ndim <= 1:
        error_fn = partial(reconstruction_error, mask=model_mask)
        in_axes1 = (0, 0, 0)  # Only theta and x are batched
        in_axes2 = (None, None, 0)  # Only theta is batched in inner vmap
        data_eval = (full_data_flat_in_brain, model_parameters_brain)
    else:
        error_fn = reconstruction_error
        in_axes1 = (0, 0, 0, 0)
        in_axes2 = (None, None, 0, None if model_mask.ndim < 3 else 0)
        data_eval = (full_data_flat_in_brain, model_parameters_brain, model_mask)

    _reconstruction_error = jax.vmap(
        jax.vmap(error_fn, in_axes=in_axes2), in_axes=in_axes1
    )

    error_maps = eval_in_batches(
        _reconstruction_error,
        jax.random.PRNGKey(0),
        *data_eval,
        batch_size=20_000,
        devices=devices,
    )

    full_error_maps = embed_in_full_brain_array(
        error_maps, brain_mask_flat.astype(np.bool), data_norm.shape[:-1]
    )
    export_nifti(full_error_maps, orig_data, out_path, "error_reconstruction.nii.gz")
