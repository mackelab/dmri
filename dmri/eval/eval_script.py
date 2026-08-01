import logging
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import hydra
import jax
import jax.numpy as jnp
import nibabel as nb
import numpy as np
from flax import nnx
from omegaconf import DictConfig, ListConfig, OmegaConf

from dmri.config import runtime_config
from dmri.eval import console
from dmri.eval.autobatch import (
    available_devices,
    cache_dir,
    load_cached_batch_size,
    make_cache_key,
    replicated_sharding_for,
    resolve_batch_size,
)
from dmri.eval.data_sources import generate_synthetic_data
from dmri.eval.export_metrics import MetricContext, run_configured_metrics
from dmri.eval.export_models import get_model_selection_exporter
from dmri.eval.export_theta import (
    export_thetas_raw,
    export_thetas_to_files_ball3stick,
)
from dmri.eval.load_data import load_and_process_data
from dmri.eval.precision import is_half, normalize_precision
from dmri.eval.sampling_methods import (
    build_base_theta_sample_fn,
    build_corrector_fn,
    build_mask_sample_fn,
    eval_in_batches,
    network_evaluations_per_sample,
)
from dmri.eval.selection import select_models
from dmri.hub import load_pretrained
from dmri.simulators import acquisition_scheme
from dmri.train.utils import load_checkpoint

# memory_fraction = 0.98  # Use 98% of available memory
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

# Compilation cache!
jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

logo = r"""

 /$$$$$$$  /$$      /$$ /$$$$$$$  /$$$$$$
| $$__  $$| $$$    /$$$| $$__  $$|_  $$_/
| $$  \ $$| $$$$  /$$$$| $$  \ $$  | $$
| $$  | $$| $$ $$/$$ $$| $$$$$$$/  | $$
| $$  | $$| $$  $$$| $$| $$__  $$  | $$
| $$  | $$| $$\  $ | $$| $$  \ $$  | $$
| $$$$$$$/| $$ \/  | $$| $$  | $$ /$$$$$$
|_______/ |__/     |__/|__/  |__/|______/
"""


def _first_device(kind: str):
    devices = available_devices(kind)
    return devices[0] if devices else None


def _device_scope(device):
    if device is None:
        return nullcontext()
    # Avoid forcing a single device when multiple accelerators are visible.
    if len(jax.local_devices()) > 1:
        return nullcontext()
    return jax.default_device(device)


def _to_cpu_array(value):
    if value is None or isinstance(value, np.ndarray):
        return value
    if isinstance(value, jax.Array):
        return np.asarray(value, dtype=value.dtype)
    return value


def _replicate_model(model, devices, logger=None):
    """Put a full copy of every model array on each evaluation device."""
    if not devices:
        return model
    sharding = replicated_sharding_for(devices)
    state = nnx.state(model)
    replicated = jax.tree_util.tree_map(
        lambda x: (
            jax.device_put(x, sharding) if isinstance(x, (jax.Array, np.ndarray)) else x
        ),
        state,
    )
    nnx.update(model, replicated)
    if logger is not None and len(devices) > 1:
        logger.info("Replicated model parameters across %d devices", len(devices))
    return model


def _to_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text in ("", "None", "null"):
        return None
    return int(text)


def _parse_slice_value(value, axis_length=None):
    if isinstance(value, slice):
        return value
    if value is None:
        return slice(None, None, None)
    if isinstance(value, int):
        idx = value
        if axis_length is not None and idx < 0:
            idx = axis_length + idx
        if axis_length is not None and (idx < 0 or idx >= axis_length):
            raise ValueError(
                f"Slice index {value} is out of bounds for axis length {axis_length}."
            )
        return slice(idx, idx + 1, None)
    if isinstance(value, str):
        text = value.strip()
        if text in ("", ":"):
            return slice(None, None, None)
        if ":" not in text:
            return _parse_slice_value(_to_int(text), axis_length)
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid slice specification '{value}'.")
        parsed = [_to_int(part) for part in parts]
        while len(parsed) < 3:
            parsed.append(None)
        return slice(parsed[0], parsed[1], parsed[2])
    if isinstance(value, (list, tuple)):
        parts = list(value)
        if len(parts) == 0:
            return slice(None, None, None)
        if len(parts) == 1:
            return _parse_slice_value(parts[0], axis_length)
        if len(parts) > 3:
            raise ValueError(f"Invalid slice specification '{value}'.")
        parsed = [_to_int(part) for part in parts]
        while len(parsed) < 3:
            parsed.append(None)
        return slice(parsed[0], parsed[1], parsed[2])
    raise TypeError(f"Unsupported slice specification: {value!r}")


def _slice_has_effect(spec):
    return not (spec.start is None and spec.stop is None and spec.step is None)


def _slice_to_string(spec):
    start = "" if spec.start is None else spec.start
    stop = "" if spec.stop is None else spec.stop
    step = "" if spec.step is None else spec.step
    if step != "":
        return f"{start}:{stop}:{step}"
    if stop != "":
        return f"{start}:{stop}"
    if start != "":
        return str(start)
    return ":"


def _build_slice_tuple(slice_cfg, volume_shape):
    if slice_cfg is None:
        return None
    if isinstance(slice_cfg, DictConfig):
        slice_cfg = OmegaConf.to_container(slice_cfg, resolve=True)
    if not isinstance(slice_cfg, dict):
        raise TypeError(
            f"Slice configuration must be a mapping, got {type(slice_cfg)}."
        )
    axes = ("x", "y", "z")
    slices = []
    any_slice_applied = False
    for axis_index in range(min(len(volume_shape), len(axes))):
        axis_name = axes[axis_index]
        raw_value = slice_cfg.get(axis_name)
        spec = _parse_slice_value(raw_value, axis_length=volume_shape[axis_index])
        if _slice_has_effect(spec):
            any_slice_applied = True
        slices.append(spec)
    for _ in range(len(slices), len(volume_shape)):
        slices.append(slice(None, None, None))
    if not any_slice_applied:
        return None
    return tuple(slices)


def _restrict_brain_mask_to_slice(slice_cfg, brain_mask, logger):
    """Narrow a brain mask to a sub-volume, returning the mask and slice tuple.

    Restricting the mask before the signal is read means excluded voxels are
    never loaded at all.
    """
    slice_tuple = _build_slice_tuple(slice_cfg, brain_mask.shape)
    if slice_tuple is None:
        return brain_mask, None

    selection_mask = np.zeros_like(brain_mask, dtype=bool)
    selection_mask[slice_tuple] = True

    voxels_before = int(np.count_nonzero(brain_mask))
    brain_mask = np.logical_and(brain_mask, selection_mask)
    voxels_after = int(np.count_nonzero(brain_mask))

    if voxels_after == 0:
        raise ValueError(
            "Slice selection resulted in zero voxels inside the brain mask."
        )

    slice_strings = [_slice_to_string(spec) for spec in slice_tuple[:3]]
    cfg_for_logging = (
        OmegaConf.to_container(slice_cfg, resolve=True)
        if isinstance(slice_cfg, DictConfig)
        else slice_cfg
    )
    logger.info(
        "Applying slice %s -> axes (%s), keeping %d/%d voxels",
        cfg_for_logging,
        ", ".join(slice_strings),
        voxels_after,
        voxels_before,
    )

    return brain_mask, slice_tuple


def _cfg_get(cfg, key, default=None):
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _find_theta_samples_file(export_dir, filename="raw_thetas.nii.gz"):
    if not os.path.isdir(export_dir):
        return None
    matches = []
    for root, _, files in os.walk(export_dir):
        if filename in files:
            matches.append(os.path.join(root, filename))
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple {filename} files under export directory {export_dir}: {matches}"
        )
    return matches[0] if matches else None


def _has_model_selection_to_export(models_selected, models_sampled):
    return models_selected is not None or models_sampled is not None


def _should_compute_feasible_model_probabilities(cfg, export_cfg, export_type):
    return bool(
        cfg.sample_mask
        and _cfg_get(export_cfg, "export_feasible_model_probabilities", False)
        and export_type == "ball3stick"
    )


def _load_theta_samples(
    path,
    brain_mask_flat,
    brain_shape,
    expected_num_samples,
    expected_theta_dim,
):
    theta = np.asarray(nb.load(path).dataobj, dtype=np.float32)
    spatial_dims = len(brain_shape)
    num_voxels = int(np.count_nonzero(brain_mask_flat))
    if theta.shape[:spatial_dims] == tuple(brain_shape):
        theta = theta.reshape(-1, *theta.shape[spatial_dims:])[brain_mask_flat]
    elif theta.shape[0] != num_voxels:
        raise ValueError(
            f"Theta samples shape {theta.shape} does not match brain shape {brain_shape} "
            f"or its {num_voxels} in-mask voxels."
        )
    if theta.ndim == 2:
        theta = theta[:, None, :]
    if theta.ndim != 3:
        raise ValueError(
            "Theta samples must have shape (voxels, samples, theta_dim) or "
            f"(voxels, theta_dim), got {theta.shape}."
        )
    if theta.shape[-1] != expected_theta_dim:
        raise ValueError(
            f"Expected theta dimension {expected_theta_dim}, got {theta.shape[-1]}."
        )
    if expected_num_samples is not None and theta.shape[1] != expected_num_samples:
        raise ValueError(
            f"Expected {expected_num_samples} theta samples, got {theta.shape[1]}."
        )
    return theta


def _default_ksd_metric(seed):
    return {
        "key": "ksd",
        "type": "ksd",
        "output_filename": "metric_ksd.nii.gz",
        "batch_size": 512,
        "sample_reduction": "none",
        "sample_axis": 1,
        "requires_theta_samples": True,
        "write_summary": True,
        "summary_filename": "ksd_summary.json",
        "options": {
            "bandwidths": (0.05, 0.1, 0.5),
            "n_bootstrap": 256,
            "mask_threshold": 0.5,
            "max_samples": None,
            "random_seed": seed,
        },
        "aggregations": [
            {"type": "mean", "name": "mean"},
            {"type": "median", "name": "median"},
            {"type": "percentile", "q": [5, 95], "name": "p{q}"},
        ],
    }


def _ensure_default_ksd_metric(metrics_cfg, seed):
    """Ensure the KSD metric is present without mutating the Hydra config."""

    if metrics_cfg is None:
        return None, False
    if isinstance(metrics_cfg, (DictConfig, ListConfig)) and len(metrics_cfg) == 0:
        return metrics_cfg, False

    metrics_py = metrics_cfg
    if isinstance(metrics_cfg, (DictConfig, ListConfig)):
        metrics_py = OmegaConf.to_container(metrics_cfg, resolve=True)
        if metrics_py in (None, {}, []):
            return metrics_cfg, False

    def _is_ksd(spec):
        if spec is None:
            return False
        if isinstance(spec, Mapping):
            spec_type = spec.get("type")
            spec_key = spec.get("key")
            return spec_type == "ksd" or spec_key == "ksd"
        return False

    added = False
    if isinstance(metrics_py, Mapping):
        if not any(_is_ksd(spec) for spec in metrics_py.values()):
            metrics_py = dict(metrics_py)
            metrics_py["ksd"] = _default_ksd_metric(seed)
            added = True
    elif isinstance(metrics_py, list):
        if not any(_is_ksd(spec) for spec in metrics_py):
            metrics_py = list(metrics_py) + [_default_ksd_metric(seed)]
            added = True

    return metrics_py, added


def _iter_metric_specs(metrics_cfg):
    """Yield (key, spec_dict) pairs from a metrics configuration."""
    if metrics_cfg is None:
        return []
    if isinstance(metrics_cfg, Mapping):
        iterable = metrics_cfg.items()
    elif isinstance(metrics_cfg, Sequence) and not isinstance(
        metrics_cfg, (str, bytes)
    ):
        iterable = enumerate(metrics_cfg)
    else:
        raise TypeError(
            f"metrics_cfg must be a mapping or sequence, got {type(metrics_cfg)}."
        )

    specs = []
    for key, raw_spec in iterable:
        spec = raw_spec
        if isinstance(raw_spec, (DictConfig, ListConfig)):
            spec = OmegaConf.to_container(raw_spec, resolve=True)
        if spec is None:
            spec = {}
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"Metric specification for '{key}' must be a mapping, got {type(spec)}."
            )
        specs.append((key, spec))
    return specs


def _metric_requires_theta(spec, key):
    """Return True if the metric spec requires theta samples."""
    requires_theta = spec.get("requires_theta_samples")
    if requires_theta is not None:
        return bool(requires_theta)
    metric_type = spec.get("type")
    if metric_type in (
        "model_selection_calibration",
        "model_selection_classification",
        "sbc_model_mask",
    ):
        return False
    if metric_type is None:
        raise ValueError(f"Metric '{key}' is missing a 'type'.")
    return True


def _filter_metrics_without_theta(metrics_cfg):
    """Drop metrics that require theta samples and report which were removed."""
    filtered_specs = []
    skipped = []
    for key, spec in _iter_metric_specs(metrics_cfg):
        metric_key = str(spec.get("key", key))
        if _metric_requires_theta(spec, metric_key):
            skipped.append(metric_key)
            continue
        if "key" not in spec:
            spec = dict(spec)
            spec["key"] = metric_key
        filtered_specs.append(spec)
    return filtered_specs, skipped


def _metrics_are_model_selection_only(metrics_cfg) -> bool:
    """Return True if all configured metrics are model-selection metrics."""
    if not metrics_cfg:
        return False
    model_sel_types = {"model_selection_calibration", "model_selection_classification"}
    has_any = False
    for _, spec in _iter_metric_specs(metrics_cfg):
        has_any = True
        metric_type = spec.get("type")
        if metric_type not in model_sel_types:
            return False
    return has_any


def _maybe_run_metrics(
    metrics_cfg_raw,
    cfg,
    log,
    checkpoint_root,
    export_name,
    sim_type,
    acq,
    full_data_flat_in_brain,
    model_parameters_brain,
    model_mask,
    model_mask_samples,
    brain_mask_flat,
    brain_shape,
    export_template,
    thetas_synth,
    true_model_mask,
    heavy_device,
    eval_devices,
    *,
    add_default_ksd,
    output_dir=None,
):
    """Normalize and run metrics configuration."""

    metrics_cfg = metrics_cfg_raw
    added_ksd_metric = False
    add_default_ksd = (
        add_default_ksd
        and model_parameters_brain is not None
        and not _metrics_are_model_selection_only(metrics_cfg_raw)
    )
    if add_default_ksd:
        metrics_cfg, added_ksd_metric = _ensure_default_ksd_metric(
            metrics_cfg_raw, seed=cfg.seed
        )

    metrics_cfg_to_run = metrics_cfg
    if model_parameters_brain is None and metrics_cfg:
        metrics_cfg_to_run, skipped_metrics = _filter_metrics_without_theta(metrics_cfg)
        if skipped_metrics:
            log.warning(
                "Skipping metrics requiring theta samples (%s) because none are available.",
                ", ".join(skipped_metrics),
            )

    if metrics_cfg_to_run:
        out_path = os.path.join(
            output_dir or os.path.join(checkpoint_root, cfg.model_name), export_name
        )
        metric_context = MetricContext(
            cfg=cfg,
            sim_type=sim_type,
            acq=acq,
            full_data_flat_in_brain=full_data_flat_in_brain,
            model_parameters_brain=model_parameters_brain,
            model_mask=model_mask,
            model_mask_samples=model_mask_samples,
            brain_mask_flat=brain_mask_flat,
            brain_shape=brain_shape,
            orig_data=export_template,
            out_path=out_path,
            true_model_parameters=thetas_synth,
            true_model_mask=true_model_mask,
        )
        with _device_scope(heavy_device):
            run_configured_metrics(
                metrics_cfg_to_run,
                metric_context,
                out_path,
                devices=eval_devices,
                logger=log,
            )
        if added_ksd_metric:
            log.info("Added default KSD metric to export metrics configuration.")


def _resolve_data_path(data_cfg):
    explicit_path = _cfg_get(data_cfg, "path")
    if explicit_path not in (None, "", "null", "None"):
        return os.path.expanduser(str(explicit_path))
    data_folder = _cfg_get(data_cfg, "data_folder")
    if data_folder in (None, "", "null", "None"):
        raise ValueError(
            "Please set either data.path or data.data_folder to select the input data."
        )
    data_root = _cfg_get(data_cfg, "data_root")
    if data_root in (None, "", "null", "None"):
        data_root = os.path.join(os.getcwd(), "data")
    return os.path.join(os.path.expanduser(str(data_root)), str(data_folder))


def _resolve_checkpoint_root(cfg):
    explicit_path = _cfg_get(cfg, "path_checkpoint")
    if explicit_path not in (None, "", "null", "None"):
        return os.path.expanduser(str(explicit_path))
    results_root = _cfg_get(cfg, "results_root")
    if results_root in (None, "", "null", "None"):
        results_root = os.path.join(os.getcwd(), "results")
    results_root = os.path.expanduser(str(results_root))
    results_folder = _cfg_get(cfg, "results_folder")
    if results_folder in (None, "", "null", "None"):
        return results_root
    return os.path.join(results_root, str(results_folder))


def main():
    """Main script function"""
    print(logo)
    _main()


@hydra.main(config_path="../../conf", config_name="eval.yaml", version_base=None)
def _main(cfg: DictConfig):
    """Evaluate score based inference"""
    cfg = runtime_config(cfg, "eval")
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    cpu_device = _first_device("cpu")
    try:
        gpu_device = _first_device("gpu")
    except Exception:
        gpu_device = None
    heavy_device = gpu_device or cpu_device
    eval_devices = _resolve_eval_devices()

    log.info(f"CPU device: {cpu_device}")
    log.info(f"GPU device: {gpu_device}")
    log.info(f"Heavy compute device: {heavy_device}")
    log.info(f"Eval devices: {eval_devices}")

    with _device_scope(cpu_device):
        _run_eval_pipeline(cfg, log, heavy_device, eval_devices)


def _resolve_eval_devices():
    """Return the preferred device set for heavy eval (GPUs > TPUs > default)."""
    for kind in ("gpu", "tpu"):
        available = available_devices(kind)
        if available:
            return available
    return tuple(jax.devices())


def _run_eval_pipeline(
    cfg: DictConfig, log: logging.Logger, heavy_device, eval_devices
):
    """Main evaluation pipeline executed under the CPU default device."""
    log.info(f"Model name: {cfg.model_name}")
    data_cfg = getattr(cfg, "data", cfg)
    checkpoint_root = _resolve_checkpoint_root(cfg)
    configured_output_dir = _cfg_get(cfg, "output_dir")
    output_dir = (
        os.path.abspath(os.path.expanduser(str(configured_output_dir)))
        if configured_output_dir not in (None, "", "null", "None")
        else os.path.join(checkpoint_root, cfg.model_name)
    )
    log.info(f"Output directory: {output_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")

    # Set seed
    log.info(f"Setting seed: {cfg.seed}")
    key = jax.random.PRNGKey(cfg.seed)

    precision = normalize_precision(_cfg_get(cfg, "precision"))
    if is_half(precision):
        log.info(
            "Using %s for the network forward pass; parameters, the sampler and "
            "the export stay float32.",
            precision,
        )

    # Load model and simulator first (needed for synthetic data)
    pretrained_repo_id = _cfg_get(cfg, "pretrained_repo_id")
    if pretrained_repo_id not in (None, "", "null", "None"):
        log.info(
            "Loading pretrained model %s from Hugging Face repo %s",
            cfg.model_name,
            pretrained_repo_id,
        )
        pretrained = load_pretrained(
            model_name=cfg.model_name,
            repo_id=str(pretrained_repo_id),
            which=_cfg_get(cfg, "checkpoint", "best"),
            revision=_cfg_get(cfg, "pretrained_revision"),
            cache_dir=_cfg_get(cfg, "pretrained_cache_dir"),
            local_files_only=bool(_cfg_get(cfg, "local_files_only", False)),
            precision=precision,
        )
        model = pretrained.model
    else:
        log.info(f"Loading model from {checkpoint_root}/{cfg.model_name}")
        path_checkpoint = os.path.join(checkpoint_root, cfg.model_name)
        checkpoint, model, _ = load_checkpoint(
            path_checkpoint,
            which=_cfg_get(cfg, "checkpoint", "latest"),
            precision=precision,
        )
        graphdef, params, static, state = nnx.split(
            model, nnx.Param, nnx.Intermediate, ...
        )
        params = checkpoint.get(cfg.params_name)
        if params is None:
            params = checkpoint["params"]
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.eval()

    _replicate_model(model, eval_devices, log)
    sim_type = model.tokenizer.simulator

    source = _cfg_get(data_cfg, "source", "real")

    slice_cfg = getattr(data_cfg, "slice", None)
    clip_quantile = 0.999 if cfg.data.clip_outliers else None

    if source == "synthetic":
        log.info("Generating synthetic evaluation data using simulator.")
        (
            data,
            data_norm,
            brain_mask,
            bvals,
            bvecs,
            true_model_mask,
            thetas_synth,
            acq_synth,
            key,
        ) = generate_synthetic_data(data_cfg, sim_type, key)
        use_true_model_mask = cfg.data.use_true_model_mask_for_synthetic

        brain_mask, volume_slice = _restrict_brain_mask_to_slice(
            slice_cfg, brain_mask, log
        )
        full_data_flat_in_brain = data_norm.reshape(-1, data_norm.shape[-1])[
            brain_mask.reshape(-1), :
        ].astype(np.float32)
        np.nan_to_num(
            full_data_flat_in_brain, nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )
        if clip_quantile is not None and full_data_flat_in_brain.size:
            upper = np.quantile(full_data_flat_in_brain, clip_quantile)
            np.clip(full_data_flat_in_brain, 0, upper, out=full_data_flat_in_brain)
    else:
        data_path = _resolve_data_path(data_cfg)
        log.info(f"Loading data from {data_path}")
        volume_slice = None

        def _mask_filter(mask, _log=log):
            nonlocal volume_slice
            mask, volume_slice = _restrict_brain_mask_to_slice(slice_cfg, mask, _log)
            return mask

        # The signal is read straight into the flat in-brain layout, normalised
        # and clipped there; out-of-mask voxels are never loaded.
        data, full_data_flat_in_brain, brain_mask, bvals, bvecs = load_and_process_data(
            data_path,
            data_cfg.brain_mask,
            data_cfg.mri_data,
            data_cfg.bvals_data,
            data_cfg.bvecs_data,
            data_cfg.round_bvals,
            mask_filter=_mask_filter,
            clip_quantile=clip_quantile,
        )
        true_model_mask = None
        thetas_synth = None
        acq_synth = None
        use_true_model_mask = False

    export_template = SimpleNamespace(
        affine=data.affine,
        shape=data.shape,
        volume_slice=volume_slice,
    )

    brain_shape = brain_mask.shape
    brain_mask_flat = brain_mask.reshape(-1)
    acq = acq_synth if acq_synth is not None else acquisition_scheme(bvals, bvecs)
    log.info(
        f"Full data flat in brain quantiles 1%, 10%, 50%, 90%, 99%: {np.quantile(full_data_flat_in_brain, [0.01, 0.1, 0.5, 0.9, 0.99])}"
    )

    with _device_scope(heavy_device):
        stage_batch_sizes = _autotune_batch_sizes(
            cfg,
            key,
            model,
            acq,
            sim_type,
            full_data_flat_in_brain,
            log,
            eval_devices,
        )

    key, key_masks = jax.random.split(key)
    # Sample masks if needed
    if cfg.sample_mask:
        log.info("Sampling masks")
        with _device_scope(heavy_device):
            models_sampled_brain, feasible_model_probabilities = sample_mask(
                cfg,
                key_masks,
                model,
                acq,
                full_data_flat_in_brain,
                log,
                devices=eval_devices,
                batch_size=stage_batch_sizes.get(STAGE_MASK),
            )
        models_sampled_brain = _to_cpu_array(models_sampled_brain)
        feasible_model_probabilities = _to_cpu_array(feasible_model_probabilities)
        avg_freq = np.mean(models_sampled_brain, axis=(0, 1))
        log.info(f"Average frequency of all models: {avg_freq}")
    else:
        models_sampled_brain = None
        feasible_model_probabilities = None
    log.info(
        f"Models sampled brain shape: {models_sampled_brain.shape if models_sampled_brain is not None else 'None'}"
    )

    # Run selection of models if needed
    if cfg.select_models:
        log.info("Selecting models")
        with _device_scope(heavy_device):
            models_selected_brain = select_models(
                cfg,
                key_masks,
                full_data_flat_in_brain,
                log,
                model.tokenizer.simulator,
                model,
                acq,
                model_mask_samples=models_sampled_brain,
                devices=eval_devices,
            )
        models_selected_brain = _to_cpu_array(models_selected_brain)
    else:
        models_selected_brain = None
    log.info(
        f"Models selected brain shape: {models_selected_brain.shape if models_selected_brain is not None else 'None'}"
    )
    # Set default mask without model selection
    default_mask = _cfg_get(cfg, "default_mask", None)
    if use_true_model_mask and true_model_mask is not None:
        log.info("Using true model mask from synthetic data generation.")
        default_mask = true_model_mask
    elif default_mask is not None:
        log.info("Using default model mask from config.")
        default_mask = jnp.array(default_mask, dtype=jnp.bool)

    # Sample theta
    key, key_theta = jax.random.split(key)
    if cfg.sample_theta:
        export_dir = os.path.join(output_dir, cfg.export.name)
        theta_path = (
            _find_theta_samples_file(export_dir)
            if bool(_cfg_get(cfg, "reuse_theta_samples", False))
            else None
        )
        if theta_path is not None:
            log.info("Loading existing theta samples from %s", theta_path)
            model_parameters_brain = _load_theta_samples(
                theta_path,
                brain_mask_flat,
                brain_mask.shape,
                _cfg_get(cfg, "theta_sample.num_samples", None),
                sim_type.theta_dim,
            )
        else:
            log.info("Sampling thetas")
            with _device_scope(heavy_device):
                model_parameters_brain = sample_theta(
                    cfg,
                    key_theta,
                    model,
                    acq,
                    full_data_flat_in_brain,
                    log,
                    model_mask=models_selected_brain,
                    devices=eval_devices,
                    default_mask=default_mask,
                    batch_sizes=stage_batch_sizes,
                )
        model_parameters_brain = _to_cpu_array(model_parameters_brain)
        _warn_on_non_finite_thetas(model_parameters_brain, precision, log)
    else:
        model_parameters_brain = None
    log.info(
        f"Model parameters brain shape: {model_parameters_brain.shape if model_parameters_brain is not None else 'None'}"
    )
    # Only use ground-truth thetas when we actually sampled posterior thetas.
    # When cfg.sample_theta is False we should leave model_parameters_brain as None
    # so theta-dependent metrics are skipped rather than run on incompatible data.
    if cfg.sample_theta and model_parameters_brain is None and thetas_synth is not None:
        model_parameters_brain = thetas_synth
    model_parameter_nans = (
        int(np.isnan(model_parameters_brain).sum())
        if model_parameters_brain is not None
        else 0
    )
    log.info(f"Model parameters nans: {model_parameter_nans}")

    # Export model selection and (optionally) mask samples
    export_cfg = getattr(cfg, "export_model_selection", None)
    if export_cfg is None:
        raise ValueError("Missing export_model_selection configuration.")
    export_name = _cfg_get(
        cfg, "export_model_selection.name", "model_selection_results"
    )
    export_type = _cfg_get(cfg, "export_model_selection.type", "ball3stick")
    if _has_model_selection_to_export(models_selected_brain, models_sampled_brain):
        log.info("Exporting model selection of type '%s'", export_type)
        exporter = get_model_selection_exporter(export_type)
        if (
            feasible_model_probabilities is None
            and _should_compute_feasible_model_probabilities(
                cfg, export_cfg, export_type
            )
        ):
            if getattr(export_cfg, "feasible_models", None) is None:
                log.warning(
                    "export_model_selection.feasible_models is not set; skipping feasible model probabilities."
                )
            else:
                log.info("Evaluating feasible model probabilities for export.")
                with _device_scope(heavy_device):
                    feasible_model_probabilities = (
                        _compute_feasible_model_probabilities(
                            cfg,
                            export_cfg,
                            model,
                            acq,
                            full_data_flat_in_brain,
                            logger=log,
                            devices=eval_devices,
                            batch_size=stage_batch_sizes.get(STAGE_FEASIBLE),
                        )
                    )
                feasible_model_probabilities = _to_cpu_array(
                    feasible_model_probabilities
                )

        def _export_masks(masks, target_name, label):
            if masks is None:
                return False
            out_path = os.path.join(output_dir, target_name)
            log.info("Exporting model selection %s to %s", label, out_path)
            exporter(
                cfg,
                export_cfg,
                masks,
                out_path,
                export_template,
                brain_mask_flat,
                brain_shape,
                feasible_model_probabilities,
            )
            return True

        exported_selection = _export_masks(
            models_selected_brain, export_name, label="results"
        )

        export_samples_enabled = bool(
            _cfg_get(cfg, "export_model_selection.export_samples", True)
        )
        samples_name_cfg = _cfg_get(cfg, "export_model_selection.samples_name", None)
        samples_name = (
            samples_name_cfg
            if samples_name_cfg not in (None, "")
            else f"{export_name}_mask_samples"
        )
        if models_sampled_brain is not None:
            if export_samples_enabled:
                _export_masks(models_sampled_brain, samples_name, label="mask samples")
            elif not exported_selection:
                _export_masks(models_sampled_brain, export_name, label="mask samples")

        # Evaluate metrics for sampled masks if configured
        mask_metrics_cfg_raw = _cfg_get(cfg, "export_model_selection.metrics", None)
        if cfg.sample_mask and mask_metrics_cfg_raw:
            _maybe_run_metrics(
                mask_metrics_cfg_raw,
                cfg,
                log,
                checkpoint_root,
                export_name,
                sim_type,
                acq,
                full_data_flat_in_brain,
                model_parameters_brain,
                models_sampled_brain,
                models_sampled_brain,
                brain_mask_flat,
                brain_shape,
                export_template,
                thetas_synth,
                true_model_mask,
                heavy_device,
                eval_devices,
                add_default_ksd=False,
                output_dir=output_dir,
            )
    log.info("Exporting inferred parameters.")
    # Export samples
    if model_parameters_brain is not None:
        out_path = os.path.join(output_dir, cfg.export.name)
        export_cfg = cfg.export
        export_type = (
            export_cfg.get("type", "ball3stick")
            if isinstance(export_cfg, DictConfig)
            else getattr(export_cfg, "type", "ball3stick")
        )
        if export_type == "none":
            log.info("Skipping inferred parameter export (export.type=none).")
        elif export_type == "raw":
            export_thetas_raw(
                cfg,
                model_parameters_brain,
                true_model_mask,
                thetas_synth,
                brain_mask_flat,
                brain_shape,
                out_path,
                export_template,
            )
        else:
            with _device_scope(heavy_device):
                export_thetas_to_files_ball3stick(
                    cfg,
                    model_parameters_brain,
                    sim_type,
                    models_selected_brain
                    if models_selected_brain is not None
                    else default_mask,
                    brain_mask_flat,
                    brain_shape,
                    out_path,
                    export_template,
                )

    if models_selected_brain is None:
        models_selected_brain = default_mask

    # Evaluate metrics (including model selection metrics that do not need theta samples)
    metrics_cfg_raw = getattr(cfg.export, "metrics", None)
    _maybe_run_metrics(
        metrics_cfg_raw,
        cfg,
        log,
        checkpoint_root,
        cfg.export.name,
        sim_type,
        acq,
        full_data_flat_in_brain,
        model_parameters_brain,
        models_selected_brain,
        models_sampled_brain,
        brain_mask_flat,
        brain_shape,
        export_template,
        thetas_synth,
        true_model_mask,
        heavy_device,
        eval_devices,
        add_default_ksd=True,
        output_dir=output_dir,
    )


def _warn_on_non_finite_thetas(thetas, precision, logger):
    """Report non-finite samples, naming half precision when it is in play."""
    if thetas is None:
        return 0.0
    non_finite = int(np.count_nonzero(~np.isfinite(thetas)))
    if not non_finite:
        return 0.0
    fraction = non_finite / thetas.size
    message = (
        f"{non_finite} non-finite theta values ({fraction:.2%} of samples). "
        "The affected voxels will export as zeros."
    )
    if is_half(precision):
        logger.warning(
            "%s This run used %s; rerun with precision=fp32 if the maps look wrong.",
            message,
            precision,
        )
    else:
        logger.warning("%s", message)
    return fraction


def _resolved(node):
    """Config subtree as a plain, comparable value."""
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _stage_shape_inputs(cfg, stage):
    """Config that determines this stage's graph, and so its memory footprint.

    Per stage, so changing the ODE step count does not re-tune mask sampling.
    """
    if stage.startswith("theta"):
        return (
            cfg.theta_sample.num_samples,
            _resolved(_cfg_get(cfg, "theta_sample.params")),
            _resolved(_cfg_get(cfg, "theta_sample.corrector")),
        )
    return (
        cfg.mask_sample.n_samples,
        _cfg_get(cfg, "mask_sample.method"),
        _cfg_get(cfg, "mask_sample.p_mask"),
        len(_feasible_models_for_mask_pass(cfg) or ()),
    )


def _batch_cache_key(cfg, stage, data, devices, *extra):
    """Identify a batching workload so a probed size can be reused across runs.

    Everything that changes the stage's memory footprint must be in here; the
    source fingerprint is added by `make_cache_key`.
    """
    device = devices[0] if devices else None
    stats = None
    try:
        stats = device.memory_stats() if device is not None else None
    except Exception:
        stats = None
    return make_cache_key(
        stage,
        cfg.model_name,
        data.shape[-1],
        *_stage_shape_inputs(cfg, stage),
        _cfg_get(cfg, "model_selection.name"),
        # Half precision admits a larger batch, so it needs its own entry.
        normalize_precision(_cfg_get(cfg, "precision")),
        getattr(device, "device_kind", "cpu"),
        (stats or {}).get("bytes_limit"),
        len(devices) if devices else 1,
        *extra,
    )


def _resolve_stage_batch_size(cfg, stage, fn, key, data, *extra_args, logger, devices):
    """Pick a batch size for one batched stage, honouring the config override."""
    return resolve_batch_size(
        fn,
        data,
        *extra_args,
        key_example=key,
        configured=cfg[stage].eval_batch_size,
        devices=devices,
        num_voxels=data.shape[0],
        cache_key=_batch_cache_key(cfg, stage, data, devices),
        logger=logger,
        override=cfg.get("batch_size"),
    )


def _feasible_models(cfg):
    """Feasible masks the export wants scored, if any."""
    export_cfg = _cfg_get(cfg, "export_model_selection")
    if export_cfg is None:
        return None
    export_type = _cfg_get(
        cfg, "export_model_selection.type", _cfg_get(cfg, "export.type")
    )
    if not _should_compute_feasible_model_probabilities(cfg, export_cfg, export_type):
        return None
    return _cfg_get(cfg, "export_model_selection.feasible_models")


def _feasible_models_for_mask_pass(cfg):
    """Feasible masks folded into the mask-sampling pass, which shares its head."""
    return _feasible_models(cfg) if cfg.sample_mask else None


def _standalone_feasible_models(cfg):
    """Feasible masks needing their own pass, because no mask pass will run."""
    return None if cfg.sample_mask else _feasible_models(cfg)


def _theta_mask_stand_in(cfg, sim_type):
    """Shape-accurate model mask for probing, before the real one is sampled.

    Only rank and trailing shape matter, so one row is enough.
    """
    num_comp = len(sim_type.model_types) + len(sim_type.noise_types)
    selection = _cfg_get(cfg, "model_selection.name", "none")
    if selection == "none" and _cfg_get(cfg, "default_mask") is None:
        return jnp.ones(num_comp, dtype=jnp.bool_)
    return jnp.ones((1, cfg.theta_sample.num_samples, num_comp), dtype=jnp.bool_)


#: Stage names. Shared by the autotuner and the runner so a size measured for
#: one stage is always looked up under the same key.
STAGE_MASK = "mask_sample"
STAGE_THETA_BASE = "theta_base"
STAGE_THETA_CORRECTOR = "theta_corrector"
STAGE_FEASIBLE = "feasible_models"


@dataclass(frozen=True)
class BatchStage:
    """One batched pass over the voxel axis, and how to size it."""

    name: str
    label: str
    build: Callable[[], Callable]
    #: Config section holding this stage's explicit `eval_batch_size`, if any.
    config_section: str = "mask_sample"
    #: Arguments the stage takes before the signal (the corrector's thetas).
    leading_args: tuple = ()
    #: Arguments it takes after it (a per-voxel model mask).
    trailing_args: tuple = ()

    def probe_args(self, data):
        return (*self.leading_args, data, *self.trailing_args)

    def configured(self, cfg):
        return _cfg_get(cfg, f"{self.config_section}.eval_batch_size")

    def cache_key(self, cfg, data, devices):
        return _batch_cache_key(cfg, self.name, data, devices)


def _batch_stages(cfg, model, acq, sim_type):
    """The batched passes this configuration will run, in order."""
    stages = []

    if cfg.sample_mask:
        feasible_models = _feasible_models_for_mask_pass(cfg)
        stages.append(
            BatchStage(
                name=STAGE_MASK,
                label="mask sampling"
                + (" + model selection" if feasible_models else ""),
                build=lambda: build_mask_sample_fn(
                    cfg.mask_sample.method,
                    cfg.mask_sample.n_samples,
                    model,
                    acq,
                    cfg.mask_sample.p_mask,
                    feasible_models=feasible_models,
                ),
            )
        )

    if cfg.sample_theta:
        mask = _theta_mask_stand_in(cfg, sim_type)
        per_voxel_mask = (mask,) if mask.ndim > 1 else ()
        stages.append(
            BatchStage(
                name=STAGE_THETA_BASE,
                label="theta sampling",
                config_section="theta_sample",
                build=lambda: build_base_theta_sample_fn(
                    cfg.theta_sample.num_samples,
                    model,
                    acq,
                    mask,
                    cfg.theta_sample.params,
                ),
                trailing_args=per_voxel_mask,
            )
        )

        corrector_name = _cfg_get(cfg, "theta_sample.corrector.name", "none")
        if corrector_name not in (None, "none", "uncorrected"):
            thetas = jnp.zeros(
                (1, cfg.theta_sample.num_samples, sim_type.theta_dim), jnp.float32
            )
            stages.append(
                BatchStage(
                    name=STAGE_THETA_CORRECTOR,
                    label="corrector",
                    config_section="theta_sample",
                    build=lambda: build_corrector_fn(
                        corrector_name,
                        model,
                        acq,
                        mask,
                        sim_type,
                        cfg.theta_sample.corrector,
                    ),
                    leading_args=(thetas,),
                    trailing_args=per_voxel_mask,
                )
            )

    feasible = _standalone_feasible_models(cfg)
    if feasible is not None:
        stages.append(
            BatchStage(
                name=STAGE_FEASIBLE,
                label="model selection",
                build=lambda: _build_feasible_probability_fn(cfg, feasible, model, acq),
            )
        )

    return stages


def _autotune_batch_sizes(cfg, key, model, acq, sim_type, data, log, devices):
    """Resolve every stage's batch size before any sampling starts.

    Probing compiles the stage graphs: slow, but once per machine. Doing it here
    keeps that cost in one labelled place rather than stalling the run repeatedly.
    """
    stages = _batch_stages(cfg, model, acq, sim_type)
    if not stages:
        return {}

    override = cfg.get("batch_size")
    resolved = {}
    pending = []
    for stage in stages:
        cached = (
            None
            if override
            else load_cached_batch_size(stage.cache_key(cfg, data, devices))
        )
        if cached:
            resolved[stage.name] = min(cached, data.shape[0])
        else:
            pending.append(stage)

    if not pending:
        log.info(
            "Using cached batch sizes: %s",
            ", ".join(f"{name}={size}" for name, size in resolved.items()),
        )
        return resolved

    message = (
        f"Autotuning batch sizes for this device (one-time; cached in {cache_dir()})"
    )
    log.info(message)
    console.say(f"\n{message}")
    started = time.perf_counter()
    for index, stage in enumerate(pending, start=1):
        stage_started = time.perf_counter()
        resolved[stage.name] = resolve_batch_size(
            stage.build(),
            *stage.probe_args(data),
            key_example=key,
            configured=stage.configured(cfg),
            devices=devices,
            num_voxels=data.shape[0],
            cache_key=stage.cache_key(cfg, data, devices),
            logger=log,
            override=override,
        )
        summary = (
            f"  [{index}/{len(pending)}] {stage.label:<36} "
            f"{resolved[stage.name]:>7,} vox/batch  "
            f"{time.perf_counter() - stage_started:5.1f}s"
        )
        log.info(summary)
        console.say(summary)

    done = (
        f"Autotuning done in {time.perf_counter() - started:.1f}s "
        "- later runs reuse it."
    )
    log.info(done)
    console.say(done + "\n")
    return resolved


def sample_mask(cfg, key, model, acq, data, logger, devices=None, batch_size=None):
    """Sample model masks, and the feasible-mask probabilities if wanted.

    Returns ``(masks, feasible_probabilities)``, the latter None unless the
    export asks for them.
    """
    feasible_models = _feasible_models_for_mask_pass(cfg)
    sample_mask_fn = build_mask_sample_fn(
        cfg.mask_sample.method,
        cfg.mask_sample.n_samples,
        model,
        acq,
        cfg.mask_sample.p_mask,
        feasible_models=feasible_models,
    )
    batch_size = batch_size or _resolve_stage_batch_size(
        cfg, STAGE_MASK, sample_mask_fn, key, data, logger=logger, devices=devices
    )
    models_sampled_brain = eval_in_batches(
        sample_mask_fn,
        key,
        data,
        batch_size=batch_size,
        logger=logger,
        devices=devices,
        cache_key=_batch_cache_key(cfg, STAGE_MASK, data, devices),
        desc="Sampling model masks",
    )
    if feasible_models is None:
        return models_sampled_brain, None
    return models_sampled_brain


def sample_theta(
    cfg,
    key,
    model,
    acq,
    data,
    logger,
    model_mask=None,
    devices=None,
    default_mask=None,
    batch_sizes=None,
):
    """Sample theta parameters"""
    sim_type = model.tokenizer.simulator
    num_comp = len(sim_type.model_types) + len(sim_type.noise_types)

    logger.info(f"Running inference with {sim_type}")
    logger.info(
        f"Model mask shape: {model_mask.shape if model_mask is not None else 'None'}"
    )

    if model_mask is None:
        if default_mask is not None:
            model_mask = default_mask
            logger.info(f"Using default mask from config {default_mask}.")
        else:
            model_mask = jnp.ones(num_comp, dtype=jnp.bool)
    else:
        model_mask = jnp.array(model_mask, dtype=jnp.bool)

        nans_in_samples = jnp.isnan(model_mask).sum()
        logger.info(f"Number of NaNs in model_mask: {nans_in_samples}")
        model_mask = jnp.where(jnp.isnan(model_mask), True, model_mask)

        assert model_mask.shape[0] == data.shape[0], (
            "model_mask must have the same number of voxels as the data"
        )
        assert model_mask.shape[-1] == num_comp, (
            "model_mask must have the same number of components as the model"
        )

    name = cfg.theta_sample.corrector.name
    logger.info(f"Using theta sampling corrector: {name}")
    nfe = network_evaluations_per_sample(cfg.theta_sample.params)
    logger.info(
        "%d ODE steps -> %d network evaluations per sample, %d samples per voxel",
        cfg.theta_sample.params.get("num_steps", 0),
        nfe,
        cfg.theta_sample.num_samples,
    )

    batched_mask = model_mask is not None and model_mask.ndim > 1
    extra_args = (model_mask,) if batched_mask else ()
    key_base, key_corrector = jax.random.split(key)

    labels = {
        STAGE_THETA_BASE: "Sampling parameters",
        STAGE_THETA_CORRECTOR: "Refining parameters",
    }

    def _run_stage(stage, fn, stage_key, *stage_data):
        cache_key = _batch_cache_key(cfg, stage, data, devices)
        batch_size = (batch_sizes or {}).get(stage) or resolve_batch_size(
            fn,
            *stage_data,
            key_example=stage_key,
            configured=cfg.theta_sample.eval_batch_size,
            devices=devices,
            num_voxels=data.shape[0],
            cache_key=cache_key,
            logger=logger,
            override=cfg.get("batch_size"),
        )
        return eval_in_batches(
            fn,
            stage_key,
            *stage_data,
            batch_size=batch_size,
            logger=logger,
            devices=devices,
            cache_key=cache_key,
            desc=labels.get(stage),
        )

    # Separate passes: the corrector fits a ~7x larger batch than the sampler.
    logger.info("Sampling thetas (network sampler)")
    base_fn = build_base_theta_sample_fn(
        cfg.theta_sample.num_samples, model, acq, model_mask, cfg.theta_sample.params
    )
    thetas_full = _run_stage(STAGE_THETA_BASE, base_fn, key_base, data, *extra_args)

    if name in (None, "none", "uncorrected"):
        return thetas_full

    logger.info("Correcting thetas (%s)", name)
    corrector_fn = build_corrector_fn(
        name, model, acq, model_mask, sim_type, cfg.theta_sample.corrector
    )
    return _run_stage(
        STAGE_THETA_CORRECTOR,
        corrector_fn,
        key_corrector,
        thetas_full,
        data,
        *extra_args,
    )


def _build_feasible_probability_fn(cfg, feasible_models, model, acq):
    """Probability over the feasible model masks, per voxel."""
    feasible_models = jnp.asarray(feasible_models, dtype=jnp.bool_)
    p_mask_arr = jnp.array([cfg.mask_sample.p_mask])

    def eval_feasible_log_probs(x):
        return jax.vmap(model.log_prob_mask, in_axes=(0, None, None, None))(
            feasible_models, acq, x, p_mask_arr
        )

    batched = jax.jit(jax.vmap(eval_feasible_log_probs, in_axes=(0,)))

    def _eval_fn(keys, batch_data):
        del keys  # Deterministic computation; keys are unused.
        return jax.nn.softmax(batched(batch_data), axis=-1)

    return _eval_fn


def _compute_feasible_model_probabilities(
    cfg, export_cfg, model, acq, data, logger=None, devices=None, batch_size=None
):
    """Compute probabilities over feasible model masks for each voxel."""
    if data.shape[0] == 0:
        return None

    _eval_fn = _build_feasible_probability_fn(
        cfg, export_cfg.feasible_models, model, acq
    )

    key = jax.random.PRNGKey(0)
    cache_key = _batch_cache_key(cfg, STAGE_FEASIBLE, data, devices)
    batch_size = batch_size or resolve_batch_size(
        _eval_fn,
        data,
        key_example=key,
        configured=cfg.mask_sample.eval_batch_size,
        devices=devices,
        num_voxels=data.shape[0],
        cache_key=cache_key,
        logger=logger,
        override=cfg.get("batch_size"),
    )
    return eval_in_batches(
        _eval_fn,
        key,
        data,
        batch_size=batch_size,
        logger=logger,
        devices=devices,
        cache_key=cache_key,
        desc="Scoring feasible models",
    )


def embed_in_full_brain_array(to_embed, brain_mask_flat, brain_shape):
    """Embed a tensor in the full brain array"""
    event_shape = to_embed.shape[1:]
    full_brain = np.zeros(
        (brain_mask_flat.shape[0],) + event_shape, dtype=to_embed.dtype
    )
    full_brain[brain_mask_flat, ...] = to_embed
    full_brain = full_brain.reshape(brain_shape + event_shape)
    return full_brain
