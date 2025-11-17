"""Configurable evaluation metrics exported as NIfTI volumes.

This module replaces the previous ad-hoc metric helpers with a small registry
that can be configured from Hydra.  Each metric definition controls how the
metric is evaluated, how the per-voxel samples are reduced, optional summary
statistics, and where the resulting files are stored.

The public entry point `run_configured_metrics` is meant to be invoked by the
Hydra evaluation script once sampling has finished.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from functools import partial
import json
import os
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
    brain_mask_flat: np.ndarray
    data_norm: np.ndarray
    orig_data: Any
    out_path: str | None = None


DEFAULT_OUTPUT_FILENAMES: Mapping[str, str] = {
    "reconstruction_mse": "error_reconstruction.nii.gz",
    "posterior_nll": "metric_posterior_nll.nii.gz",
    "ksd": "metric_ksd.nii.gz",
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
                logger.info(
                    "Metric '%s' returned no values (skipped).", spec.key
                )
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
    return distances.astype(np.float32)


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
    mask_threshold = float(options.get("mask_threshold", 0.5))
    max_samples = options.get("max_samples", options.get("num_samples"))
    seed = int(options.get("random_seed", getattr(context.cfg, "seed", 0)))
    export_pvalue = bool(options.get("export_pvalue", True))
    pvalue_filename = options.get(
        "pvalue_output_filename", "metric_ksd_pvalue.nii.gz"
    )

    theta = np.asarray(context.model_parameters_brain)
    sample_axis = 1 if spec.sample_axis is None else spec.sample_axis
    theta = np.moveaxis(theta, sample_axis, 1)
    if max_samples is not None:
        theta = theta[:, : int(max_samples)]
    if theta.shape[1] < 2:
        return np.full(theta.shape[0], np.nan, dtype=np.float32)
    theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    theta = theta.reshape(theta.shape[0], theta.shape[1], -1)

    data = np.asarray(context.full_data_flat_in_brain)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    sim_type = context.sim_type
    num_voxels = theta.shape[0]
    num_components = len(sim_type.model_types) + len(sim_type.noise_types)
    model_mask = _normalize_model_mask_for_ksd(
        context.model_mask, num_voxels, num_components, mask_threshold
    )

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

    if export_pvalue and context.out_path:
        full_map = embed_in_full_brain_array(
            p_values,
            context.brain_mask_flat.astype(np.bool_),
            context.data_norm.shape[:-1],
        )
        export_nifti(full_map, context.orig_data, context.out_path, pvalue_filename)

    return ksd_values


_METRIC_REGISTRY: Mapping[
    str,
    Callable[[MetricSpec, MetricContext, Sequence[jax.Device] | str | None], np.ndarray | None],
] = {
    "posterior_nll": _compute_posterior_nll,
    "reconstruction_mse": _compute_reconstruction_mse,
    "sliced_wasserstein": lambda spec, context, devices: _compute_swd_metric(spec, context),
    "ksd": _compute_ksd_metric,
}


# ---------------------------------------------------------------------------
# Helpers


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
            "metrics_cfg must be a mapping or sequence, got " f"{type(metrics_cfg)}"
        )

    for key, raw_spec in iterable:
        data = to_container(raw_spec) or {}
        metric_key = str(data.get("key", key))
        metric_type = data.get("type")
        if metric_type is None:
            raise ValueError(f"Metric '{metric_key}' is missing a 'type'.")
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
    raise TypeError(f"preferred_device_kinds must be a string or sequence, got {value!r}")


def _normalize_model_mask_for_ksd(
    model_mask: Any,
    num_voxels: int,
    num_components: int,
    threshold: float,
) -> np.ndarray:
    """Broadcast and clean the model mask for KSD evaluation."""

    if model_mask is None:
        return np.ones((num_voxels, num_components), dtype=np.bool_)

    mask = np.asarray(model_mask)
    if mask.ndim == 1:
        if mask.shape[0] != num_components:
            raise ValueError(
                f"Model mask has length {mask.shape[0]}, expected {num_components}."
            )
        return np.broadcast_to(mask.astype(np.bool_), (num_voxels, mask.shape[-1]))

    if mask.ndim == 2:
        if mask.shape[-1] != num_components:
            raise ValueError(
                f"Model mask shape {mask.shape} is incompatible with expected components {num_components}."
            )
        if mask.shape[0] == num_voxels:
            return mask.astype(np.bool_)
        return np.broadcast_to(mask.astype(np.bool_), (num_voxels, mask.shape[-1]))

    # If we have a sample dimension, average and threshold to get a deterministic mask.
    if mask.ndim >= 3:
        if mask.shape[-1] != num_components:
            raise ValueError(
                f"Model mask shape {mask.shape} is incompatible with expected components {num_components}."
            )
        mask = np.mean(mask, axis=1) >= float(threshold)
        mask = mask.astype(np.bool_)
        return np.broadcast_to(mask, (num_voxels, mask.shape[-1]))

    raise ValueError(f"Unsupported model mask shape {mask.shape} for KSD metric.")


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

    If `precond_matrix` is None, we estimate a whitening preconditioner from the
    empirical covariance of `thetas` (only over masked dimensions):
        cov ≈ Cov(theta_masked),  L L^T = cov + jitter
        P = L^{-1}
    and work in z = P @ theta coordinates.

    Args
    ----
    sim_type :
        Object that provides theta_mask(mask) and from_theta(theta, model_mask=mask).
    thetas : (N, D) array
        Samples in parameter space theta.
    mask, acq, x :
        Extra arguments forwarded to log_posterior(theta, mask, acq, x).
    key : PRNGKey
        Random key for wild bootstrap.
    bandwidths : array-like, shape (M,) or scalar
        Fixed RBF bandwidths h in z-space.
    precond_matrix : (D, D) array or None
        Constant preconditioner P. If None, build P as a whitening transform
        from the sample covariance of `thetas` (masked).
    n_bootstrap : int
        Number of wild-bootstrap replicates.
    eps : float
        Small constant to avoid numerical issues.

    Returns
    -------
    ksd2_obs : scalar
        Observed (preconditioned, multi-scale) KSD^2 (U-statistic).
    p_value : scalar
        Wild-bootstrap p-value (right-tailed).
    """

    thetas = jnp.asarray(thetas)
    n, d = thetas.shape

    # theta_mask indicates which dimensions are relevant (shape: (D,))
    # assume 0/1 or bool; convert to float for arithmetic
    theta_mask = sim_type.theta_mask(mask)
    theta_mask = theta_mask.astype(thetas.dtype)        # (D,)
    d_eff = jnp.sum(theta_mask)                        # effective dimension

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

        # Zero out irrelevant dims before covariance
        theta_centered_masked = theta_centered * theta_mask  # (N, D)
        cov = (theta_centered_masked.T @ theta_centered_masked) / jnp.maximum(
            n - 1, 1
        )                                                   # (D, D)

        # Cholesky + jitter for stability
        jitter = 1e-6 * jnp.eye(d, dtype=thetas.dtype)
        L = jnp.linalg.cholesky(cov + jitter)

        # Whitening: P = L^{-1}
        P = jnp.linalg.inv(L)
        # For score transform we need P^{-T} = (P^{-1})^T = L^T
        P_inv_T = L.T
    else:
        P = jnp.asarray(precond_matrix)
        P_inv_T = jnp.linalg.inv(P).T

    # Normalize bandwidths
    bandwidths = jnp.atleast_1d(bandwidths).astype(thetas.dtype)
    h2s = jnp.maximum(bandwidths**2, eps)  # (M,)

    # ---------- 1. Score in theta, then transform to z coordinates ----------
    def score_theta(theta):
        # theta: (D,)
        g = jax.grad(log_posterior, argnums=0)(theta, mask, acq, x)  # (D,)
        # Ensure irrelevant dimensions are ignored (should already be 0 from prior mask,
        # but this makes it explicit)
        return g * theta_mask

    scores_theta = jax.vmap(score_theta)(thetas)        # (N, D)

    # s_z = P^{-T} s_theta   (chain rule for z = P theta)
    scores_z = scores_theta @ P_inv_T                   # (N, D)
    # Mask again in z-space to be completely safe
    scores_z = scores_z * theta_mask                    # (N, D)

    # ---------- 2. Transform samples to z-space for distances ----------
    # z_i = P theta_i
    z = thetas @ P.T                                    # (N, D)

    # Only use masked dimensions in distances
    z_masked = z * theta_mask                           # (N, D)
    diff_z = z_masked[:, None, :] - z_masked[None, :, :]  # (N, N, D)
    sq_dist_z = jnp.sum(diff_z**2, axis=-1)             # (N, N)

    # ---------- 3. Stein kernel for one bandwidth in z-space ----------
    def stein_kernel_single_h2(h2):
        k = jnp.exp(-sq_dist_z / (2.0 * h2))            # (N, N)

        s = scores_z                                   # (N, D)
        # Term 1: s_i^T s_j * k_ij
        s_dot = s @ s.T                                # (N, N)
        term1 = s_dot * k

        # RBF derivatives in z (masked):
        # ∇_z k = -k * (z_i - z_j) / h2  on relevant dims only
        grad_zprime = k[..., None] * diff_z / h2       # (N, N, D)
        grad_z      = -grad_zprime                     # (N, N, D)

        # Term 2: s_i^T ∇_{z'} k(z_i, z_j)
        term2 = jnp.einsum("id,ijd->ij", s, grad_zprime)

        # Term 3: s_j^T ∇_z k(z_i, z_j)
        term3 = jnp.einsum("jd,ijd->ij", s, grad_z)

        # Term 4: trace ∇_{z,z'} k
        # For masked dimensions, the effective dimension is d_eff.
        # trace = k * (d_eff / h2 - ||z_i - z_j||^2 / h2^2)
        trace_hess = k * (d_eff / h2 - sq_dist_z / (h2**2))

        H = term1 + term2 + term3 + trace_hess         # (N, N)
        return H

    # ---------- 4. Multi-scale Stein kernel (average over bandwidths) ----------
    Hs = jax.vmap(stein_kernel_single_h2)(h2s)         # (M, N, N)
    weights = sigma / h2s[:, None, None]**2                # (M, 1, 1)
    weights = weights            # Normalize weights
    Hs = Hs * weights                                   # Weighted kernels
    H = jnp.sum(Hs, axis=0)                           # (N, N)

    # Remove diagonal for U-statistic
    H_no_diag = H - jnp.diag(jnp.diag(H))
    ksd2_obs = jnp.sum(H_no_diag) / (n * (n - 1))

    # ---------- 5. Wild bootstrap for p-value ----------
    keys = jax.random.split(key, n_bootstrap)

    def one_bootstrap(k_boot):
        # Rademacher weights ξ_i ∈ {-1, +1}
        xi = jax.random.choice(k_boot, jnp.array([-1.0, 1.0]), shape=(n,))
        xi_outer = xi[:, None] * xi[None, :]           # (N, N)
        H_star = H_no_diag * xi_outer                  # diag stays zero
        T_star = jnp.sum(H_star) / (n * (n - 1))
        return T_star

    boot_stats = jax.vmap(one_bootstrap)(keys)         # (n_bootstrap,)
    p_value = (1.0 + jnp.sum(boot_stats >= ksd2_obs)) / (n_bootstrap + 1.0)

    return ksd2_obs, p_value
