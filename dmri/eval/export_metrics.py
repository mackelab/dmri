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


DEFAULT_OUTPUT_FILENAMES: Mapping[str, str] = {
    "reconstruction_mse": "error_reconstruction.nii.gz",
    "posterior_nll": "metric_posterior_nll.nii.gz",
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

    def reconstruction_error(_, x, theta, mask):
        simulator = sim_type.from_theta(theta, model_mask=mask)
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


_METRIC_REGISTRY: Mapping[
    str,
    Callable[[MetricSpec, MetricContext, Sequence[jax.Device] | str | None], np.ndarray | None],
] = {
    "posterior_nll": _compute_posterior_nll,
    "reconstruction_mse": _compute_reconstruction_mse,
    "sliced_wasserstein": lambda spec, context, devices: _compute_swd_metric(spec, context),
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
        metrics_fn = jax.tree_util.Partial(metric_fn, mask=model_mask)
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
