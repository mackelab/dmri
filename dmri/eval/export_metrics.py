"""Configurable evaluation metrics exported as NIfTI volumes.

This module replaces the previous ad-hoc metric helpers with a small registry
that can be configured from Hydra.  Each metric definition controls how the
metric is evaluated, how the per-voxel samples are reduced, optional summary
statistics, and where the resulting files are stored.

The public entry point `run_configured_metrics` is meant to be invoked by the
Hydra evaluation script once sampling has finished.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

try:  # Hydra is optional for callers outside the CLI.
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError:  # pragma: no cover - Hydra is available in normal runs.
    DictConfig = ListConfig = OmegaConf = None  # type: ignore[assignment]

from dmri.eval.export_theta import embed_in_full_brain_array
from dmri.eval.sampling_methods import eval_in_batches
from dmri.utils.dmriutils import export_nifti

MetricFunction = Callable[[jax.Array, jax.Array, jax.Array, Any], jax.Array]


@dataclass(slots=True)
class MetricAggregation:
    """Specification for a summary statistic."""

    type: str = "mean"
    name: str | None = None
    q: float | Sequence[float] | None = None  # For percentile aggregator


@dataclass(slots=True)
class MetricSpec:
    """Structured configuration for a metric."""

    key: str
    type: str
    output_filename: str
    batch_size: int = 20_000
    enabled: bool = True
    sample_reduction: str = "mean"  # mean/median/none
    sample_axis: int | None = 1
    requires_theta_samples: bool = True
    write_summary: bool = True
    summary_filename: str | None = None
    aggregations: tuple[MetricAggregation, ...] = field(default_factory=tuple)
    preferred_device_kinds: tuple[str, ...] = ("gpu", "tpu")
    options: Mapping[str, Any] = field(default_factory=dict)

    def normalized_summary_filename(self) -> str | None:
        if not self.write_summary:
            return None
        return self.summary_filename or f"{self.key}_summary.json"


@dataclass(slots=True)
class MetricResult:
    key: str
    nifti_path: str | None
    summary_path: str | None
    aggregations: Mapping[str, float]


@dataclass(slots=True)
class MetricContext:
    cfg: Any
    sim_type: Any
    acq: Any
    full_data_flat_in_brain: np.ndarray
    model_parameters_brain: np.ndarray | None
    model_mask: np.ndarray | None
    model_mask_samples: np.ndarray | None
    brain_mask_flat: np.ndarray
    data_norm: np.ndarray
    orig_data: Any
    out_path: str | None = None
    true_model_parameters: np.ndarray | None = None
    true_model_mask: np.ndarray | None = None


DEFAULT_OUTPUT_FILENAMES: Mapping[str, str] = {
    "reconstruction_mse": "error_reconstruction.nii.gz",
    "posterior_nll": "metric_posterior_nll.nii.gz",
    "ksd": "metric_ksd.nii.gz",
    "sbc_marginal_coverage": "metric_sbc_marginal_coverage.nii.gz",
    "sbc_model_mask": "metric_sbc_model_mask.nii.gz",
    "model_selection_calibration": "metric_model_selection_calibration_error.nii.gz",
    "model_selection_classification": "metric_model_selection_f1.nii.gz",
}


def run_configured_metrics(
    metrics_cfg: Mapping[str, Any] | Sequence[Any] | None,
    context: MetricContext,
    out_path: str,
    *,
    devices: Sequence[jax.Device] | str | None = None,
    logger: Any | None = None,
) -> MutableMapping[str, MetricResult]:
    """Evaluate and export all enabled metrics.

    Args:
        metrics_cfg: Mapping or list describing metric specifications.  When
            using Hydra, this typically comes from `cfg.export.metrics`.
        context: Pre-computed tensors and metadata shared by all metrics.
        out_path: Destination directory.
        devices: Optional device specification forwarded to batching helper.
        logger: Optional logger for informational messages.

    Returns:
        A mapping from metric key to :class:`MetricResult` for bookkeeping.
    """

    specs = _normalize_metric_specs(metrics_cfg)
    if not specs:
        if logger is not None:
            logger.info("No metrics configured; skipping metric export.")
        return {}

    os.makedirs(out_path, exist_ok=True)
    results: MutableMapping[str, MetricResult] = {}
    for spec in specs:
        if not spec.enabled:
            if logger is not None:
                logger.info("Skipping metric '%s' (disabled).", spec.key)
            continue
        if spec.requires_theta_samples and context.model_parameters_brain is None:
            raise ValueError(
                f"Metric '{spec.key}' requires theta samples but none are available."
            )
        metric_fn = _METRIC_REGISTRY.get(spec.type)
        if metric_fn is None:
            raise ValueError(
                f"Metric '{spec.key}' references unknown type '{spec.type}'."
            )
        if logger is not None:
            logger.info("Computing metric '%s' (%s)", spec.key, spec.type)
        metric_values = metric_fn(spec, context, devices)
        if metric_values is None:
            if logger is not None:
                logger.info("Metric '%s' returned no values (skipped).", spec.key)
            continue
        nifti_path = _export_metric_map(spec, metric_values, context, out_path)
        aggregations = _summarize_metric(spec, metric_values)
        summary_path = _write_metric_summary(spec, aggregations, out_path)
        results[spec.key] = MetricResult(
            key=spec.key,
            nifti_path=nifti_path,
            summary_path=summary_path,
            aggregations=aggregations,
        )
        if logger is not None and aggregations:
            agg_text = ", ".join(
                f"{name}={value:.6g}" for name, value in aggregations.items()
            )
            logger.info("Metric '%s' summary: %s", spec.key, agg_text)
    return results


# ---------------------------------------------------------------------------
# Metric implementations


def _compute_posterior_nll(
    spec: MetricSpec,
    context: MetricContext,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray:
    """Compute voxel-wise negative log-likelihood under the posterior samples."""

    sim_type = context.sim_type
    acq = context.acq

    def gt_metrics(_, x, theta, model_mask):
        simulator = sim_type.from_theta(theta, model_mask=model_mask)
        theta_mask = sim_type.theta_mask(model_mask)
        prior = jax.scipy.stats.norm.logpdf(theta, loc=0.0, scale=1.0)
        prior = jnp.sum(prior * theta_mask, axis=-1)
        ll = simulator.log_likelihood(acq, x)
        log_posterior = ll + prior
        return -log_posterior

    values = _evaluate_metric_function(spec, context, gt_metrics, devices)
    return _reduce_samples(values, spec)


def _compute_reconstruction_mse(
    spec: MetricSpec,
    context: MetricContext,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray:
    """Compute absolute reconstruction error for each voxel."""

    sim_type = context.sim_type
    acq = context.acq

    def reconstruction_error(_, x, theta, model_mask):
        simulator = sim_type.from_theta(theta, model_mask=model_mask)
        signal = simulator.signal(acq)
        return jnp.mean(jnp.abs(signal - x), axis=-1)

    values = _evaluate_metric_function(spec, context, reconstruction_error, devices)
    return _reduce_samples(values, spec)


def _compute_swd_metric(
    spec: MetricSpec,
    context: MetricContext,
) -> np.ndarray | None:
    """Compute sliced Wasserstein distance against a reference sample file."""

    options = dict(spec.options or {})
    ref_path = options.get("reference_samples_path")
    if not ref_path:
        return None

    if context.model_parameters_brain is None:
        raise ValueError(
            f"Metric '{spec.key}' requires theta samples but none are available."
        )

    ref_path = os.path.expanduser(str(ref_path))
    if not os.path.isabs(ref_path):
        ref_path = os.path.abspath(ref_path)

    reference = _load_reference_theta_samples(ref_path)
    theta = np.asarray(context.model_parameters_brain)

    sample_axis = spec.sample_axis if spec.sample_axis is not None else 1
    theta = np.moveaxis(theta, sample_axis, 1)
    reference = np.moveaxis(reference, sample_axis, 1)

    if theta.shape != reference.shape:
        raise ValueError(
            f"Metric '{spec.key}' reference shape {reference.shape} does not match"
            f" theta samples {theta.shape}."
        )

    num_voxels_full = theta.shape[0]
    subset_indices = _select_voxel_subset(spec, num_voxels_full)
    if subset_indices is not None:
        theta = theta[subset_indices]
        reference = reference[subset_indices]

    num_voxels = theta.shape[0]
    num_samples = theta.shape[1]
    feature_dim = int(np.prod(theta.shape[2:]))
    theta = theta.reshape(num_voxels, num_samples, feature_dim)
    reference = reference.reshape(num_voxels, num_samples, feature_dim)

    theta = np.nan_to_num(theta, copy=False)
    reference = np.nan_to_num(reference, copy=False)

    num_directions = int(options.get("num_directions", 128))
    if num_directions <= 0:
        raise ValueError(
            f"Metric '{spec.key}' expects num_directions > 0, got {num_directions}."
        )
    seed = options.get("random_seed", 0)
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(num_directions, feature_dim))
    norms = np.linalg.norm(projections, axis=1, keepdims=True)
    projections = projections / np.clip(norms, 1e-12, None)

    distances = np.zeros(num_voxels, dtype=np.float64)
    for direction in projections:
        proj_theta = np.tensordot(theta, direction, axes=([2], [0]))
        proj_reference = np.tensordot(reference, direction, axes=([2], [0]))
        proj_theta.sort(axis=1)
        proj_reference.sort(axis=1)
        diff = proj_theta - proj_reference
        w2 = np.mean(diff * diff, axis=1)
        distances += np.sqrt(np.maximum(w2, 0.0))

    distances /= num_directions
    distances = distances.astype(np.float32)
    if subset_indices is not None:
        distances = _scatter_metric_values(distances, subset_indices, num_voxels_full)
    return distances


def _compute_ksd_metric(
    spec: MetricSpec,
    context: MetricContext,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray:
    """Compute the (preconditioned, multi-scale) KSD per voxel."""

    if context.model_parameters_brain is None:
        raise ValueError(
            f"Metric '{spec.key}' requires theta samples but none are available."
        )

    options = dict(spec.options or {})
    bandwidths = np.asarray(
        options.get("bandwidths", (0.05, 0.1, 0.5)), dtype=np.float32
    ).reshape(-1)
    if bandwidths.size == 0:
        raise ValueError(f"Metric '{spec.key}' requires at least one bandwidth.")
    n_bootstrap = int(options.get("n_bootstrap", 256))
    if n_bootstrap <= 0:
        raise ValueError(
            f"Metric '{spec.key}' expects n_bootstrap > 0, got {n_bootstrap}."
        )
    max_samples = options.get("max_samples", options.get("num_samples"))
    seed = int(options.get("random_seed", getattr(context.cfg, "seed", 0)))
    export_pvalue = bool(options.get("export_pvalue", True))
    pvalue_filename = options.get("pvalue_output_filename", "metric_ksd_pvalue.nii.gz")

    theta = np.asarray(context.model_parameters_brain)
    sample_axis = 1 if spec.sample_axis is None else spec.sample_axis
    theta = np.moveaxis(theta, sample_axis, 1)
    if max_samples is not None:
        theta = theta[:, : int(max_samples)]
    num_voxels_full = theta.shape[0]
    subset_indices = _select_voxel_subset(spec, num_voxels_full)
    if subset_indices is not None:
        theta = theta[subset_indices]
    if theta.shape[1] < 2:
        values = np.full(theta.shape[0], np.nan, dtype=np.float32)
        if subset_indices is not None:
            values = _scatter_metric_values(values, subset_indices, num_voxels_full)
        return values
    theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    theta = theta.reshape(theta.shape[0], theta.shape[1], -1)

    data = np.asarray(context.full_data_flat_in_brain)
    if subset_indices is not None:
        data = data[subset_indices]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    sim_type = context.sim_type
    num_voxels = theta.shape[0]
    model_mask = context.model_mask
    if model_mask is not None:
        mask_arr = np.asarray(model_mask)
        if mask_arr.ndim == 1:
            mask_arr = np.broadcast_to(mask_arr, (num_voxels, mask_arr.shape[-1]))
        elif mask_arr.shape[0] != num_voxels_full:
            mask_arr = np.broadcast_to(
                mask_arr, (num_voxels_full,) + mask_arr.shape[1:]
            )
        if subset_indices is not None and mask_arr.shape[0] == num_voxels_full:
            mask_arr = mask_arr[subset_indices]
        model_mask = mask_arr.astype(np.bool_)

    bandwidths_jnp = jnp.asarray(bandwidths)

    def _ksd_single(key, theta_voxel, x_voxel, mask_voxel):
        ksd2, p_value = multiscale_preconditioned_ksd_and_pvalue(
            sim_type=sim_type,
            thetas=theta_voxel,
            mask=mask_voxel,
            acq=context.acq,
            x=x_voxel,
            key=key,
            bandwidths=bandwidths_jnp,
            n_bootstrap=n_bootstrap,
        )
        return jnp.stack((ksd2, p_value))

    if model_mask is None:

        def _ksd_single_no_mask(key, theta_voxel, x_voxel):
            return _ksd_single(key, theta_voxel, x_voxel, None)

        ksd_fn = jax.vmap(_ksd_single_no_mask, in_axes=(0, 0, 0))
        values = eval_in_batches(
            ksd_fn,
            jax.random.PRNGKey(seed),
            theta,
            data,
            batch_size=spec.batch_size,
            devices=devices,
            preferred_device_kinds=spec.preferred_device_kinds,
        )
    else:
        ksd_fn = jax.vmap(_ksd_single, in_axes=(0, 0, 0, 0))
        values = eval_in_batches(
            ksd_fn,
            jax.random.PRNGKey(seed),
            theta,
            data,
            model_mask,
            batch_size=spec.batch_size,
            devices=devices,
            preferred_device_kinds=spec.preferred_device_kinds,
        )
    values = np.asarray(values, dtype=np.float32)
    ksd_values = values[..., 0]
    p_values = values[..., 1]

    if subset_indices is not None:
        ksd_values = _scatter_metric_values(ksd_values, subset_indices, num_voxels_full)
        p_values = _scatter_metric_values(p_values, subset_indices, num_voxels_full)

    if export_pvalue and context.out_path:
        full_map = embed_in_full_brain_array(
            p_values,
            context.brain_mask_flat.astype(np.bool_),
            context.data_norm.shape[:-1],
        )
        export_nifti(full_map, context.orig_data, context.out_path, pvalue_filename)

    return ksd_values


def _compute_sbc_marginal_coverage(
    spec: MetricSpec,
    context: MetricContext,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray | None:
    """Compute marginal rank coverage for SBC on synthetic data."""

    del devices  # Unused; present for signature compatibility.

    if context.true_model_parameters is None:
        # SBC is only applicable when we have ground truth parameters (synthetic data).
        return None
    if context.model_parameters_brain is None:
        raise ValueError(
            f"Metric '{spec.key}' requires theta samples but none are available."
        )

    posterior_samples = np.asarray(context.model_parameters_brain)
    true_thetas = np.asarray(context.true_model_parameters)
    if posterior_samples.shape[0] != true_thetas.shape[0]:
        raise ValueError(
            "Mismatch between posterior samples and ground truth theta shapes: "
            f"{posterior_samples.shape} vs {true_thetas.shape}."
        )

    sample_axis = 1 if spec.sample_axis is None else spec.sample_axis
    if sample_axis < 0:
        sample_axis += posterior_samples.ndim
    if sample_axis < 0 or sample_axis >= posterior_samples.ndim:
        raise ValueError(
            f"Metric '{spec.key}' sample_axis={spec.sample_axis} is out of bounds for "
            f"posterior sample array with shape {posterior_samples.shape}."
        )
    posterior_samples = np.moveaxis(posterior_samples, sample_axis, 1)

    num_voxels = posterior_samples.shape[0]
    num_samples = posterior_samples.shape[1]
    if num_samples == 0:
        return np.full(true_thetas.shape, np.nan, dtype=np.float32)

    posterior_samples = posterior_samples.reshape(num_voxels, num_samples, -1)
    true_thetas = true_thetas.reshape(num_voxels, -1)
    if posterior_samples.shape[2] != true_thetas.shape[1]:
        raise ValueError(
            "Posterior samples and ground truth thetas have incompatible shapes after "
            f"flattening: {posterior_samples.shape} vs {true_thetas.shape}."
        )

    mask_source = (
        context.true_model_mask
        if context.true_model_mask is not None
        else context.model_mask
    )
    model_mask = _broadcast_model_mask(mask_source, num_voxels)

    coverage = np.full_like(true_thetas, np.nan, dtype=np.float32)
    for voxel_idx in range(num_voxels):
        theta_mask = None
        if model_mask is not None:
            theta_mask = np.asarray(context.sim_type.theta_mask(model_mask[voxel_idx]))
        coverage[voxel_idx] = _marginal_rank_fraction(
            posterior_samples[voxel_idx],
            true_thetas[voxel_idx],
            theta_mask,
        )
    pvalues, sample_counts = _compute_uniformity_pvalues(coverage)
    if context.out_path:
        pvalue_filename = spec.options.get(
            "pvalue_output_filename", "sbc_uniformity_pvalues.json"
        )
        _write_uniformity_pvalues(
            pvalues,
            sample_counts,
            spec,
            context,
            pvalue_filename,
        )
    return coverage


def _compute_sbc_model_mask(
    spec: MetricSpec,
    context: MetricContext,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray | None:
    """Compute SBC rank coverage for model-mask components on synthetic data."""

    del devices  # Unused

    if context.true_model_mask is None:
        # SBC only applicable with ground-truth masks (synthetic data).
        return None

    posterior_mask_samples = context.model_mask_samples
    if posterior_mask_samples is None:
        raise ValueError(
            f"Metric '{spec.key}' requires model mask samples but none are available."
        )

    posterior_mask_samples = np.asarray(posterior_mask_samples)
    true_mask = np.asarray(context.true_model_mask, dtype=np.bool_)

    num_voxels = true_mask.shape[0]
    if true_mask.ndim == 1:
        true_mask = np.broadcast_to(true_mask, (num_voxels, true_mask.shape[-1]))

    if posterior_mask_samples.shape[0] != num_voxels:
        raise ValueError(
            "Posterior mask samples and ground truth mask must share the first dimension: "
            f"{posterior_mask_samples.shape} vs {true_mask.shape}."
        )
    if posterior_mask_samples.ndim < 2:
        raise ValueError(
            f"Metric '{spec.key}' expects model mask samples with a sample axis; "
            f"got shape {posterior_mask_samples.shape}."
        )

    sample_axis = 1 if spec.sample_axis is None else spec.sample_axis
    if sample_axis < 0:
        sample_axis += posterior_mask_samples.ndim
    if sample_axis < 0 or sample_axis >= posterior_mask_samples.ndim:
        raise ValueError(
            f"Metric '{spec.key}' sample_axis={spec.sample_axis} is out of bounds for "
            f"posterior mask array with shape {posterior_mask_samples.shape}."
        )
    posterior_mask_samples = np.moveaxis(posterior_mask_samples, sample_axis, 1)

    posterior_mask_samples = posterior_mask_samples.reshape(
        num_voxels, posterior_mask_samples.shape[1], -1
    )
    true_mask = true_mask.reshape(num_voxels, -1)
    if posterior_mask_samples.shape[2] != true_mask.shape[1]:
        raise ValueError(
            "Posterior mask samples and ground truth mask have incompatible shapes after "
            f"flattening: {posterior_mask_samples.shape} vs {true_mask.shape}."
        )

    coverage = np.full_like(true_mask, np.nan, dtype=np.float32)
    for voxel_idx in range(num_voxels):
        coverage[voxel_idx] = _marginal_rank_fraction(
            posterior_mask_samples[voxel_idx],
            true_mask[voxel_idx],
            theta_mask=None,
        )

    pvalues, sample_counts = _compute_uniformity_pvalues(coverage)
    if context.out_path:
        pvalue_filename = spec.options.get(
            "pvalue_output_filename", "sbc_model_mask_uniformity_pvalues.json"
        )
        _write_uniformity_pvalues(
            pvalues,
            sample_counts,
            spec,
            context,
            pvalue_filename,
        )
    return coverage


def _compute_model_selection_calibration(
    spec: MetricSpec,
    context: MetricContext,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray:
    """Calibration error per voxel/component for model-selection probabilities."""

    del devices  # Unused

    probabilities, true_mask = _prepare_model_selection_inputs(spec, context)
    num_bins = int(spec.options.get("num_bins", 10))
    if num_bins <= 0:
        raise ValueError(f"Metric '{spec.key}' expects num_bins > 0, got {num_bins}.")
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    flat_probs = probabilities.reshape(-1).astype(np.float64)
    flat_labels = true_mask.reshape(-1).astype(np.float64)
    valid = np.isfinite(flat_probs) & np.isfinite(flat_labels)
    if not np.any(valid):
        return np.full_like(probabilities, np.nan, dtype=np.float32)
    flat_probs = np.clip(flat_probs[valid], 0.0, 1.0)
    flat_labels = flat_labels[valid]

    bin_indices = np.digitize(flat_probs, bin_edges[1:-1], right=False)
    counts = np.bincount(bin_indices, minlength=num_bins).astype(np.int64)
    acc = np.zeros(num_bins, dtype=np.float64)
    conf = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        mask = bin_indices == b
        if counts[b] == 0:
            continue
        acc[b] = float(np.mean(flat_labels[mask]))
        conf[b] = float(np.mean(flat_probs[mask]))
    bin_error = np.abs(acc - conf)
    total = flat_probs.size
    weights = counts / total
    ece = float(np.sum(weights * bin_error)) if total > 0 else np.nan
    non_empty = counts > 0
    mce = float(np.max(bin_error[non_empty])) if np.any(non_empty) else np.nan

    per_sample_error = np.full(probabilities.size, np.nan, dtype=np.float32)
    per_sample_error[valid] = bin_error[bin_indices].astype(np.float32)
    per_voxel_error = per_sample_error.reshape(probabilities.shape)

    reliability_filename = spec.options.get(
        "reliability_output_filename", "model_selection_reliability.json"
    )
    if context.out_path:
        _write_reliability_diagram(
            context.out_path,
            reliability_filename,
            spec.key,
            bin_edges,
            bin_centers,
            acc,
            conf,
            counts,
            ece,
            mce,
        )
    return per_voxel_error


def _compute_model_selection_classification(
    spec: MetricSpec,
    context: MetricContext,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray:
    """Per-voxel F1 for model-selection predictions along with summary metrics."""

    del devices  # Unused

    probabilities, true_mask = _prepare_model_selection_inputs(spec, context)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    threshold = float(spec.options.get("threshold", 0.5))
    predicted_mask = probabilities >= threshold
    labels = true_mask.astype(bool)

    tp = np.sum(predicted_mask & labels, axis=-1).astype(np.float64)
    fp = np.sum(predicted_mask & ~labels, axis=-1).astype(np.float64)
    fn = np.sum(~predicted_mask & labels, axis=-1).astype(np.float64)
    tn = np.sum(~predicted_mask & ~labels, axis=-1).astype(np.float64)
    num_components = labels.shape[-1]

    denom_f1 = 2.0 * tp + fp + fn
    f1_per_voxel = np.divide(
        2.0 * tp,
        denom_f1,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denom_f1 > 0,
    )

    sample_accuracy = np.mean(predicted_mask == labels, axis=-1).astype(np.float64)
    hamming_loss = np.mean(predicted_mask != labels, axis=-1).astype(np.float64)
    exact_match = np.all(predicted_mask == labels, axis=-1).astype(np.float64)

    tp_c = np.sum(predicted_mask & labels, axis=0).astype(np.float64)
    fp_c = np.sum(predicted_mask & ~labels, axis=0).astype(np.float64)
    fn_c = np.sum(~predicted_mask & labels, axis=0).astype(np.float64)
    denom_precision = tp_c + fp_c
    denom_recall = tp_c + fn_c
    denom_f1_class = 2.0 * tp_c + fp_c + fn_c
    per_class_precision = np.divide(
        tp_c,
        denom_precision,
        out=np.full_like(tp_c, np.nan, dtype=np.float64),
        where=denom_precision > 0,
    )
    per_class_recall = np.divide(
        tp_c,
        denom_recall,
        out=np.full_like(tp_c, np.nan, dtype=np.float64),
        where=denom_recall > 0,
    )
    per_class_f1 = np.divide(
        2.0 * tp_c,
        denom_f1_class,
        out=np.full_like(tp_c, np.nan, dtype=np.float64),
        where=denom_f1_class > 0,
    )

    micro_precision = _safe_divide(tp.sum(), tp.sum() + fp.sum())
    micro_recall = _safe_divide(tp.sum(), tp.sum() + fn.sum())
    micro_f1 = _safe_divide(2.0 * tp.sum(), 2.0 * tp.sum() + fp.sum() + fn.sum())
    micro_accuracy = _safe_divide(
        tp.sum() + tn.sum(),
        tp.sum() + tn.sum() + fp.sum() + fn.sum(),
    )

    macro_precision = (
        np.nanmean(per_class_precision) if per_class_precision.size else np.nan
    )
    macro_recall = np.nanmean(per_class_recall) if per_class_recall.size else np.nan
    macro_f1 = np.nanmean(per_class_f1) if per_class_f1.size else np.nan

    summary_filename = spec.options.get(
        "classification_output_filename",
        "model_selection_classification.json",
    )
    if context.out_path:
        _write_classification_summary(
            context.out_path,
            summary_filename,
            spec.key,
            threshold,
            micro_precision,
            micro_recall,
            micro_f1,
            micro_accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            float(np.nanmean(sample_accuracy)) if sample_accuracy.size else np.nan,
            float(np.nanmean(hamming_loss)) if hamming_loss.size else np.nan,
            float(np.nanmean(exact_match)) if exact_match.size else np.nan,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            num_components,
        )

    return f1_per_voxel.astype(np.float32)


_METRIC_REGISTRY: Mapping[
    str,
    Callable[
        [MetricSpec, MetricContext, Sequence[jax.Device] | str | None],
        np.ndarray | None,
    ],
] = {
    "posterior_nll": _compute_posterior_nll,
    "reconstruction_mse": _compute_reconstruction_mse,
    "sliced_wasserstein": lambda spec, context, devices: _compute_swd_metric(
        spec, context
    ),
    "ksd": _compute_ksd_metric,
    "sbc_marginal_coverage": _compute_sbc_marginal_coverage,
    "sbc_model_mask": _compute_sbc_model_mask,
    "model_selection_calibration": _compute_model_selection_calibration,
    "model_selection_classification": _compute_model_selection_classification,
}


# ---------------------------------------------------------------------------
# Helpers


def _select_voxel_subset(spec: MetricSpec, num_voxels: int) -> np.ndarray | None:
    options = dict(spec.options or {})
    subset_size = options.get("voxel_subset_size")
    subset_fraction = options.get("voxel_subset_fraction")
    if subset_size is None and subset_fraction is None:
        return None

    target_size = num_voxels
    if subset_fraction is not None:
        fraction = float(subset_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"Metric '{spec.key}' expects voxel_subset_fraction in (0, 1], "
                f"got {fraction}."
            )
        target_size = max(1, round(num_voxels * fraction))
    if subset_size is not None:
        size = int(subset_size)
        if size <= 0:
            raise ValueError(
                f"Metric '{spec.key}' expects voxel_subset_size > 0, got {size}."
            )
        target_size = min(target_size, size)
    if target_size >= num_voxels:
        return None

    seed = int(options.get("voxel_subset_seed", options.get("random_seed", 0)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_voxels, size=target_size, replace=False))


def _scatter_metric_values(
    values: np.ndarray, indices: np.ndarray, num_voxels: int
) -> np.ndarray:
    if values.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Subset values and indices differ: {values.shape[0]} vs {indices.shape[0]}."
        )
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float32
    full = np.full((num_voxels,) + values.shape[1:], np.nan, dtype=dtype)
    full[indices] = values
    return full


def _evaluate_metric_function(
    spec: MetricSpec,
    context: MetricContext,
    metric_fn: MetricFunction,
    devices: Sequence[jax.Device] | str | None,
) -> np.ndarray:
    """Apply `metric_fn` over all voxels in the brain mask."""

    model_mask = context.model_mask
    full_data_flat_in_brain = context.full_data_flat_in_brain
    model_parameters_brain = context.model_parameters_brain
    if model_parameters_brain is None:
        raise ValueError("Metric evaluation requires theta samples; none provided.")

    if model_mask is None or getattr(model_mask, "ndim", 0) <= 1:
        metrics_fn = partial(metric_fn, model_mask=model_mask)
        in_axes1 = (0, 0, 0)
        in_axes2 = (None, None, 0)
        data_eval = (full_data_flat_in_brain, model_parameters_brain)
    else:
        metrics_fn = metric_fn
        in_axes1 = (0, 0, 0, 0)
        mask_axes = None if model_mask.ndim < 3 else 0
        in_axes2 = (None, None, 0, mask_axes)
        data_eval = (full_data_flat_in_brain, model_parameters_brain, model_mask)

    _batched_fn = jax.vmap(jax.vmap(metrics_fn, in_axes=in_axes2), in_axes=in_axes1)
    values = eval_in_batches(
        _batched_fn,
        jax.random.PRNGKey(0),
        *data_eval,
        batch_size=spec.batch_size,
        devices=devices,
        preferred_device_kinds=spec.preferred_device_kinds,
    )
    return np.asarray(values)


def _reduce_samples(values: np.ndarray, spec: MetricSpec) -> np.ndarray:
    reduction = (spec.sample_reduction or "none").lower()
    if reduction in ("none", ""):
        return values
    if spec.sample_axis is None:
        raise ValueError(
            f"Metric '{spec.key}' requested sample reduction but has no sample axis."
        )
    axis = spec.sample_axis
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ValueError(
            f"Metric '{spec.key}' sample_axis={spec.sample_axis} is out of bounds for"
            f" array with shape {values.shape}."
        )
    if reduction == "mean":
        return np.mean(values, axis=axis)
    if reduction == "median":
        return np.median(values, axis=axis)
    raise ValueError(
        f"Metric '{spec.key}' uses unsupported sample_reduction '{spec.sample_reduction}'."
    )


def _broadcast_model_mask(
    model_mask: np.ndarray | None, num_voxels: int
) -> np.ndarray | None:
    """Broadcast the model mask to (num_voxels, ...) if present."""
    if model_mask is None:
        return None
    mask_arr = np.asarray(model_mask)
    if mask_arr.ndim == 1:
        return np.broadcast_to(mask_arr, (num_voxels, mask_arr.shape[-1])).astype(
            np.bool_
        )
    if mask_arr.shape[0] != num_voxels:
        mask_arr = np.broadcast_to(mask_arr, (num_voxels,) + mask_arr.shape[1:])
    return mask_arr.astype(np.bool_)


def _marginal_rank_fraction(
    posterior_samples: np.ndarray,
    true_theta: np.ndarray,
    theta_mask: np.ndarray | None,
) -> np.ndarray:
    """Return per-dimension rank fraction of the true theta within posterior samples."""

    flat_samples = np.asarray(posterior_samples, dtype=np.float64).reshape(
        posterior_samples.shape[0], -1
    )
    flat_theta = np.asarray(true_theta, dtype=np.float64).reshape(-1)
    if flat_samples.shape[1] != flat_theta.shape[0]:
        raise ValueError(
            f"Incompatible shapes for SBC coverage: posterior {flat_samples.shape} "
            f"vs theta {flat_theta.shape}."
        )

    if theta_mask is not None:
        mask = np.asarray(theta_mask, dtype=bool).reshape(-1)
        if mask.size not in (1, flat_theta.size):
            raise ValueError(
                f"SBC theta mask with size {mask.size} is incompatible with theta "
                f"size {flat_theta.size}."
            )
        mask = np.broadcast_to(mask, flat_theta.shape)
        flat_theta = np.where(mask, flat_theta, np.nan)
        flat_samples = np.where(mask, flat_samples, np.nan)

    finite_samples = np.isfinite(flat_samples)
    sample_counts = np.sum(finite_samples, axis=0)
    ranks = np.sum((flat_samples < flat_theta) & finite_samples, axis=0)

    coverage = np.full(flat_theta.shape, np.nan, dtype=np.float64)
    np.divide(
        ranks,
        sample_counts,
        out=coverage,
        where=sample_counts > 0,
    )
    return coverage.astype(np.float32)


def _compute_uniformity_pvalues(
    coverage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension KS p-values for uniformity of coverage ranks."""

    flat = np.asarray(coverage)
    if flat.ndim == 1:
        flat = flat[:, None]
    flat = flat.reshape(-1, flat.shape[-1])

    pvalues = np.full(flat.shape[1], np.nan, dtype=np.float32)
    counts = np.zeros(flat.shape[1], dtype=np.int64)
    for dim in range(flat.shape[1]):
        values = flat[:, dim]
        values = values[np.isfinite(values)]
        counts[dim] = values.size
        if values.size == 0:
            continue
        pvalues[dim] = _ks_uniform_pvalue(values)
    return pvalues, counts


def _prepare_model_selection_inputs(
    spec: MetricSpec, context: MetricContext
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, true_mask) aligned to (num_voxels, num_components)."""

    if context.true_model_mask is None:
        raise ValueError(
            f"Metric '{spec.key}' requires true_model_mask; only available for synthetic data."
        )
    if context.model_mask is None:
        raise ValueError(
            f"Metric '{spec.key}' requires model selection predictions but model_mask is None."
        )

    true_mask = np.asarray(context.true_model_mask, dtype=np.bool_)
    num_voxels = true_mask.shape[0]
    if true_mask.ndim == 1:
        true_mask = np.broadcast_to(true_mask, (num_voxels, true_mask.shape[0]))

    pred_mask = np.asarray(context.model_mask)
    if pred_mask.ndim == 1:
        pred_mask = np.broadcast_to(pred_mask, (num_voxels, pred_mask.shape[0]))
    elif pred_mask.shape[0] != num_voxels:
        pred_mask = np.broadcast_to(pred_mask, (num_voxels,) + pred_mask.shape[1:])

    sample_axis = spec.sample_axis if spec.sample_axis is not None else 1
    if pred_mask.ndim >= 3:
        axis = sample_axis
        if axis < 0:
            axis += pred_mask.ndim
        if axis < 0 or axis >= pred_mask.ndim:
            raise ValueError(
                f"Metric '{spec.key}' sample_axis={spec.sample_axis} is out of bounds for "
                f"model_mask with shape {pred_mask.shape}."
            )
        pred_mask = np.moveaxis(pred_mask, axis, 1)
        probabilities = np.mean(pred_mask, axis=1)
    else:
        probabilities = pred_mask.astype(np.float32)

    if probabilities.shape[0] != num_voxels:
        raise ValueError(
            f"Metric '{spec.key}' predictions have shape {probabilities.shape}, "
            f"expected first dimension {num_voxels}."
        )
    if probabilities.shape[-1] != true_mask.shape[-1]:
        raise ValueError(
            f"Metric '{spec.key}' predictions last dimension {probabilities.shape[-1]} "
            f"does not match true_model_mask {true_mask.shape[-1]}."
        )
    return probabilities.astype(np.float32), true_mask


def _ks_uniform_pvalue(values: np.ndarray) -> float:
    """Kolmogorov–Smirnov p-value for uniform[0,1] without SciPy."""

    sorted_vals = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_vals.size
    if n == 0:
        return np.nan

    cdf = np.arange(1, n + 1, dtype=np.float64) / n
    d_plus = np.max(cdf - sorted_vals)
    d_minus = np.max(sorted_vals - (np.arange(n, dtype=np.float64) / n))
    d_stat = max(d_plus, d_minus)

    en = np.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d_stat

    # Asymptotic Kolmogorov distribution (Smirnov series)
    p_value = 0.0
    for k in range(1, 101):
        term = (-1) ** (k - 1) * np.exp(-2 * (k**2) * (lam**2))
        p_value += term
        if abs(term) < 1e-10:
            break
    p_value = max(0.0, min(1.0, 2.0 * p_value))
    return float(p_value)


def _write_uniformity_pvalues(
    pvalues: np.ndarray,
    sample_counts: np.ndarray,
    spec: MetricSpec,
    context: MetricContext,
    filename: str,
) -> str:
    """Persist per-dimension KS p-values for SBC coverage."""

    output_path = Path(context.out_path) / filename
    payload = {
        "metric": spec.key,
        "type": spec.type,
        "pvalues": {f"dim_{i}": float(val) for i, val in enumerate(pvalues)},
        "num_samples": {f"dim_{i}": int(n) for i, n in enumerate(sample_counts)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    return str(output_path)


def _write_reliability_diagram(
    out_path: str,
    filename: str,
    metric_key: str,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    accuracy: np.ndarray,
    confidence: np.ndarray,
    counts: np.ndarray,
    ece: float,
    mce: float,
) -> str:
    """Persist reliability diagram statistics."""

    output_path = Path(out_path) / filename
    payload = {
        "metric": metric_key,
        "bins": {
            "edges": [float(x) for x in bin_edges],
            "centers": [float(x) for x in bin_centers],
            "count": [int(c) for c in counts],
            "accuracy": [_finite_or_none(x) for x in accuracy],
            "confidence": [_finite_or_none(x) for x in confidence],
        },
        "ece": _finite_or_none(ece),
        "mce": _finite_or_none(mce),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    return str(output_path)


def _write_classification_summary(
    out_path: str,
    filename: str,
    metric_key: str,
    threshold: float,
    micro_precision: float,
    micro_recall: float,
    micro_f1: float,
    micro_accuracy: float,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    mean_sample_accuracy: float,
    mean_hamming_loss: float,
    exact_match_rate: float,
    per_class_precision: np.ndarray,
    per_class_recall: np.ndarray,
    per_class_f1: np.ndarray,
    num_components: int,
) -> str:
    """Persist multi-label classification summary statistics."""

    output_path = Path(out_path) / filename
    payload: dict[str, Any] = {
        "metric": metric_key,
        "threshold": threshold,
        "micro": {
            "precision": _finite_or_none(micro_precision),
            "recall": _finite_or_none(micro_recall),
            "f1": _finite_or_none(micro_f1),
            "accuracy": _finite_or_none(micro_accuracy),
        },
        "macro": {
            "precision": _finite_or_none(macro_precision),
            "recall": _finite_or_none(macro_recall),
            "f1": _finite_or_none(macro_f1),
        },
        "sample": {
            "mean_accuracy": _finite_or_none(mean_sample_accuracy),
            "mean_hamming_loss": _finite_or_none(mean_hamming_loss),
            "exact_match_rate": _finite_or_none(exact_match_rate),
        },
    }
    per_class: dict[str, dict[str, float | None]] = {}
    for i in range(num_components):
        per_class[f"class_{i}"] = {
            "precision": _finite_or_none(per_class_precision[i]),
            "recall": _finite_or_none(per_class_recall[i]),
            "f1": _finite_or_none(per_class_f1[i]),
        }
    payload["per_class"] = per_class
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    return str(output_path)


def _safe_divide(num: float, denom: float) -> float:
    if denom == 0:
        return np.nan
    return float(num) / float(denom)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _export_metric_map(
    spec: MetricSpec,
    metric_values: np.ndarray,
    context: MetricContext,
    out_path: str,
) -> str:
    brain_mask = context.brain_mask_flat.astype(np.bool_)
    brain_shape = context.data_norm.shape[:-1]
    full_metric_map = embed_in_full_brain_array(metric_values, brain_mask, brain_shape)
    filename = spec.output_filename or DEFAULT_OUTPUT_FILENAMES.get(spec.type)
    if filename is None:
        raise ValueError(
            f"Metric '{spec.key}' did not provide an output filename and has no default."
        )
    export_nifti(full_metric_map, context.orig_data, out_path, filename)
    return os.path.join(out_path, filename)


def _summarize_metric(spec: MetricSpec, metric_values: np.ndarray) -> dict[str, float]:
    if not spec.aggregations:
        return {}
    flat = np.asarray(metric_values).astype(np.float64)
    flat = flat.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {}
    summaries: dict[str, float] = {}
    for agg in spec.aggregations:
        agg_type = agg.type.lower()
        if agg_type == "mean":
            name = agg.name or "mean"
            summaries[name] = float(np.mean(flat))
        elif agg_type == "median":
            name = agg.name or "median"
            summaries[name] = float(np.median(flat))
        elif agg_type == "std":
            name = agg.name or "std"
            summaries[name] = float(np.std(flat))
        elif agg_type in {"min", "max"}:
            name = agg.name or agg_type
            reducer = np.min if agg_type == "min" else np.max
            summaries[name] = float(reducer(flat))
        elif agg_type == "percentile":
            q_values = _ensure_sequence(agg.q)
            if q_values is None:
                raise ValueError(
                    f"Percentile aggregation for metric '{spec.key}' misses 'q'."
                )
            percentiles = np.percentile(flat, q_values)
            if np.isscalar(percentiles):
                percentiles = [float(percentiles)]
            for q, value in zip(q_values, percentiles):
                summaries[_format_percentile_name(agg, q)] = float(value)
        else:
            raise ValueError(
                f"Metric '{spec.key}' uses unsupported aggregation '{agg.type}'."
            )
    return summaries


def _write_metric_summary(
    spec: MetricSpec,
    aggregations: Mapping[str, float],
    out_path: str,
) -> str | None:
    summary_filename = spec.normalized_summary_filename()
    if summary_filename is None or not aggregations:
        return None
    summary_path = Path(out_path) / summary_filename
    payload = {
        "metric": spec.key,
        "type": spec.type,
        "aggregations": aggregations,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    return str(summary_path)


def _normalize_metric_specs(
    metrics_cfg: Mapping[str, Any] | Sequence[Any] | None,
) -> list[MetricSpec]:
    if metrics_cfg is None:
        return []

    def to_container(cfg_entry: Any) -> Any:
        if OmegaConf is not None and isinstance(cfg_entry, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg_entry, resolve=True)
        return cfg_entry

    specs: list[MetricSpec] = []
    if isinstance(metrics_cfg, Mapping):
        iterable: Iterable[tuple[str, Any]] = metrics_cfg.items()
    elif isinstance(metrics_cfg, Sequence):
        iterable = enumerate(metrics_cfg)
    else:
        raise TypeError(
            f"metrics_cfg must be a mapping or sequence, got {type(metrics_cfg)}"
        )

    for key, raw_spec in iterable:
        data = to_container(raw_spec) or {}
        metric_key = str(data.get("key", key))
        metric_type = data.get("type")
        if metric_type is None:
            raise ValueError(f"Metric '{metric_key}' is missing a 'type'.")
        if metric_type in {
            "model_selection_calibration",
            "model_selection_classification",
            "sbc_model_mask",
        }:
            data = dict(data)
            data.setdefault("requires_theta_samples", False)
            data.setdefault("sample_reduction", "none")
            if not data.get("aggregations"):
                if metric_type == "model_selection_calibration":
                    data["aggregations"] = [
                        {"type": "mean", "name": "ece"},
                        {"type": "max", "name": "mce"},
                    ]
                elif metric_type == "model_selection_classification":
                    data["aggregations"] = [
                        {"type": "mean", "name": "mean_f1"},
                        {"type": "median", "name": "median_f1"},
                    ]
                else:  # sbc_model_mask
                    data["aggregations"] = [
                        {"type": "mean", "name": "mean"},
                        {"type": "median", "name": "median"},
                        {"type": "percentile", "q": [5, 95], "name": "p{q}"},
                    ]
        output_filename = data.get(
            "output_filename", DEFAULT_OUTPUT_FILENAMES.get(metric_type)
        )
        aggregations = tuple(
            MetricAggregation(
                type=agg.get("type", "mean"),
                name=agg.get("name"),
                q=agg.get("q"),
            )
            for agg in data.get("aggregations", [])
        )
        spec = MetricSpec(
            key=metric_key,
            type=metric_type,
            output_filename=output_filename,
            batch_size=int(data.get("batch_size", 20_000)),
            enabled=bool(data.get("enabled", True)),
            sample_reduction=data.get("sample_reduction", "mean"),
            sample_axis=data.get("sample_axis", 1),
            requires_theta_samples=bool(data.get("requires_theta_samples", True)),
            write_summary=bool(data.get("write_summary", True)),
            summary_filename=data.get("summary_filename"),
            aggregations=aggregations,
            preferred_device_kinds=_normalize_device_kinds(
                data.get("preferred_device_kinds", ("gpu", "tpu"))
            ),
            options=dict(data.get("options", {})),
        )
        specs.append(spec)
    return specs


def _ensure_sequence(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _format_percentile_name(agg: MetricAggregation, q: float) -> str:
    if agg.name:
        return agg.name.format(q=q)
    q_int = int(q)
    if abs(q - q_int) < 1e-6:
        return f"p{q_int}"
    return f"p{str(q).replace('.', '_')}"


def _normalize_device_kinds(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    raise TypeError(
        f"preferred_device_kinds must be a string or sequence, got {value!r}"
    )


def _load_reference_theta_samples(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference sample file '{path}' does not exist.")

    if path.endswith(".npz"):
        with np.load(path) as data:
            if "thetas" in data:
                return np.asarray(data["thetas"])
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            raise ValueError(
                f"Reference npz '{path}' must contain an array named 'thetas' or a single array."
            )
    array = np.load(path)
    return np.asarray(array)


def _cfg_lookup(cfg_section: Any, key: str, default: Any = None) -> Any:
    if cfg_section is None:
        return default
    if isinstance(cfg_section, Mapping):
        return cfg_section.get(key, default)
    return getattr(cfg_section, key, default)


def compute_swd_to_reference(
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
):
    """Backward-compatible wrapper to export SWD maps from reference samples."""

    export_cfg = getattr(cfg, "export", None)
    reference_path = _cfg_lookup(export_cfg, "reference_samples_path")
    if not reference_path:
        return None

    num_directions = int(_cfg_lookup(export_cfg, "swd_num_directions", 128))
    random_seed = _cfg_lookup(export_cfg, "swd_random_seed", getattr(cfg, "seed", 0))
    output_filename = _cfg_lookup(
        export_cfg, "swd_output_filename", "metric_swd.nii.gz"
    )

    context = MetricContext(
        cfg=cfg,
        sim_type=sim_type,
        acq=acq,
        full_data_flat_in_brain=full_data_flat_in_brain,
        model_parameters_brain=model_parameters_brain,
        model_mask=model_mask,
        model_mask_samples=None,
        brain_mask_flat=brain_mask_flat,
        data_norm=data_norm,
        orig_data=orig_data,
        out_path=out_path,
    )
    spec = MetricSpec(
        key="sliced_wasserstein",
        type="sliced_wasserstein",
        output_filename=output_filename,
        sample_reduction="mean",
        sample_axis=1,
        options={
            "reference_samples_path": reference_path,
            "num_directions": num_directions,
            "random_seed": random_seed,
        },
    )

    distances = _compute_swd_metric(spec, context)
    if distances is None:
        return None

    full_map = embed_in_full_brain_array(
        distances,
        brain_mask_flat.astype(np.bool_),
        data_norm.shape[:-1],
    )
    export_nifti(full_map, orig_data, out_path, output_filename)
    return os.path.join(out_path, output_filename)


def multiscale_preconditioned_ksd_and_pvalue(
    sim_type,
    thetas,
    mask,
    acq,
    x,
    key,
    bandwidths,
    sigma=0.1,
    precond_matrix=None,
    n_bootstrap: int = 512,
    eps: float = 1e-8,
):
    """
    Multi-scale, preconditioned Kernel Stein Discrepancy (KSD^2) + wild-bootstrap p-value,
    restricted to the dimensions indicated by `theta_mask`.

    This version is written to reduce peak memory usage:
      - avoids (N, N, D) tensors (no explicit pairwise differences),
      - avoids (M, N, N) tensor across bandwidths,
      - avoids (N_bootstrap, N, N) intermediates in the bootstrap.
    """

    thetas = jnp.asarray(thetas)
    n, d = thetas.shape
    dtype = thetas.dtype

    # theta_mask indicates which dimensions are relevant (shape: (D,))
    theta_mask = sim_type.theta_mask(mask).astype(dtype)  # (D,)
    d_eff = jnp.sum(theta_mask)  # effective dimension (scalar)

    # ---------- log posterior ----------
    def log_posterior(theta, mask, acq, x):
        simulator = sim_type.from_theta(theta, model_mask=mask)
        prior = jax.scipy.stats.norm.logpdf(theta, loc=0.0, scale=1.0)
        # Only apply prior on masked dimensions
        prior = jnp.sum(prior * theta_mask, axis=-1)
        ll = simulator.log_likelihood(acq, x)
        return ll + prior

    # ---------- 0. Build / use preconditioner P ----------
    if precond_matrix is None:
        # Empirical covariance over *masked* dimensions
        theta_mean = jnp.mean(thetas, axis=0)
        theta_centered = thetas - theta_mean
        theta_centered_masked = theta_centered * theta_mask  # (N, D)

        cov = (theta_centered_masked.T @ theta_centered_masked) / jnp.maximum(
            n - 1, 1
        )  # (D, D)

        jitter = 1e-3 * jnp.eye(d, dtype=dtype)
        L = jnp.linalg.cholesky(cov + jitter)

        # Whitening: P = L^{-1}
        P = jnp.linalg.inv(L)
        # For score transform we need P^{-T} = (P^{-1})^T = L^T
        P_inv_T = L.T
    else:
        P = jnp.asarray(precond_matrix, dtype=dtype)
        P_inv_T = jnp.linalg.inv(P).T

    # Normalize bandwidths
    bandwidths = jnp.atleast_1d(bandwidths).astype(dtype)
    h2s = jnp.maximum(bandwidths**2, eps)  # (M,)
    m = h2s.shape[0]

    # ---------- 1. Score in theta, then transform to z coordinates ----------
    def score_theta(theta):
        g = jax.grad(log_posterior, argnums=0)(theta, mask, acq, x)  # (D,)
        return g * theta_mask

    scores_theta = jax.vmap(score_theta)(thetas)  # (N, D)
    scores_z = scores_theta @ P_inv_T  # (N, D)
    scores_z = scores_z * theta_mask  # (N, D)

    # ---------- 2. Transform samples to z-space for distances ----------
    z = thetas @ P.T  # (N, D)
    z_masked = z * theta_mask  # (N, D)

    # Pairwise squared distances using Gram trick (no (N, N, D))
    z_sq_norm = jnp.sum(z_masked**2, axis=1)  # (N,)
    # sq_dist_z[i,j] = ||z_i||^2 + ||z_j||^2 - 2 z_i·z_j
    gram_z = z_masked @ z_masked.T  # (N, N)
    sq_dist_z = z_sq_norm[:, None] + z_sq_norm[None, :] - 2.0 * gram_z  # (N, N)
    sq_dist_z = jnp.maximum(sq_dist_z, 0.0)  # numerical safety

    # Precompute Stein-related Gram matrices (all (N, N))
    S = scores_z  # (N, D)
    gram_s = S @ S.T  # (N, N)  term1: s_i^T s_j
    SZT = S @ z_masked.T  # (N, N)  A_ij = s_i^T z_j
    b = jnp.diag(SZT)  # (N,)   b_i = s_i^T z_i

    # ---------- 3. Multi-scale Stein kernel (memory-friendly) ----------
    def body_h2(i, H):
        h2 = h2s[i]

        # RBF kernel in z
        k = jnp.exp(-sq_dist_z / (2.0 * h2))  # (N, N)

        # Term 1: s_i^T s_j * k_ij
        term1 = gram_s * k  # (N, N)

        # Term 2 and 3 via analytic forms (no (N, N, D) tensors):
        # term2(i,j) = k/h2 * (s_i^T z_i - s_i^T z_j) = k/h2 * (b_i - SZT_ij)
        # term3(i,j) = -k/h2 * (s_j^T z_i - s_j^T z_j) = -k/h2 * (SZT_ji - b_j)
        k_over_h2 = k / h2
        term2 = k_over_h2 * (b[:, None] - SZT)  # (N, N)
        term3 = -k_over_h2 * (SZT.T - b[None, :])  # (N, N)

        # Term 4: trace Hessian
        # trace = k * (d_eff / h2 - ||z_i - z_j||^2 / h2^2)
        trace_hess = k * (d_eff / h2 - sq_dist_z / (h2**2))

        # Weight for multi-scale combination
        weight = sigma * 1 / len(h2s)
        H_update = weight * (term1 + term2 + term3 + trace_hess)

        return H + H_update

    H0 = jnp.zeros_like(sq_dist_z)
    H = jax.lax.fori_loop(0, m, body_h2, H0)  # (N, N)

    # Remove diagonal for U-statistic
    diag_H = jnp.diag(jnp.diag(H))
    H_no_diag = H - diag_H
    denom = n * (n - 1)
    ksd2_obs = jnp.sum(H_no_diag) / denom
    ksd = jnp.sqrt(jnp.maximum(ksd2_obs, 0.0))

    # ---------- 4. Wild bootstrap (memory-friendly) ----------
    def bootstrap_body(i, state):
        key_i, count = state
        key_i, subkey = jax.random.split(key_i)

        # Rademacher ±1
        xi = 2.0 * jax.random.bernoulli(subkey, 0.5, shape=(n,)) - 1.0  # (N,)

        # T_star = xi^T H_no_diag xi / (n(n-1))
        Hx = H_no_diag @ xi  # (N,)
        T_star = jnp.dot(xi, Hx) / denom

        count = count + (T_star >= ksd2_obs)
        return (key_i, count)

    init_state = (key, jnp.array(0.0, dtype=dtype))
    _, ge_count = jax.lax.fori_loop(0, n_bootstrap, bootstrap_body, init_state)
    p_value = (1.0 + ge_count) / (n_bootstrap + 1.0)

    return ksd, p_value
