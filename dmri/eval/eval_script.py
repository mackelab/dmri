import logging
import os
import socket
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from types import SimpleNamespace

import jax

# memory_fraction = 0.98  # Use 98% of available memory
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

# Compilation cache!
jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


import hydra
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig, ListConfig, OmegaConf

from dmri.eval.data_sources import generate_synthetic_data
from dmri.eval.export_metrics import MetricContext, run_configured_metrics
from dmri.eval.export_models import get_model_selection_exporter
from dmri.eval.export_theta import (
    export_thetas_raw,
    export_thetas_to_files_ball3stick,
)
from dmri.eval.load_data import load_and_process_data
from dmri.eval.sampling_methods import (
    build_mask_sample_fn,
    build_theta_sample_fn,
    eval_in_batches,
)
from dmri.eval.selection import select_models
from dmri.simulators import acquisition_scheme
from dmri.train.utils import load_checkpoint

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
    devices = jax.devices(kind)
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
        return np.asarray(value)
    return value


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


def _apply_slice_to_brain(slice_cfg, brain_mask, data_norm, logger):
    slice_tuple = _build_slice_tuple(slice_cfg, brain_mask.shape)
    if slice_tuple is None:
        return brain_mask, data_norm, None

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

    data_norm = np.where(brain_mask[..., None], data_norm, 0.0)
    return brain_mask, data_norm, slice_tuple


def _cfg_get(cfg, key, default=None):
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


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
    elif isinstance(metrics_cfg, Sequence) and not isinstance(metrics_cfg, (str, bytes)):
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
    data_norm,
    export_template,
    thetas_synth,
    true_model_mask,
    heavy_device,
    eval_devices,
    *,
    add_default_ksd,
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
        metrics_cfg_to_run, skipped_metrics = _filter_metrics_without_theta(
            metrics_cfg
        )
        if skipped_metrics:
            log.warning(
                "Skipping metrics requiring theta samples (%s) because none are available.",
                ", ".join(skipped_metrics),
            )

    if metrics_cfg_to_run:
        out_path = os.path.join(checkpoint_root, cfg.model_name, export_name)
        metric_context = MetricContext(
            cfg=cfg,
            sim_type=sim_type,
            acq=acq,
            full_data_flat_in_brain=full_data_flat_in_brain,
            model_parameters_brain=model_parameters_brain,
            model_mask=model_mask,
            model_mask_samples=model_mask_samples,
            brain_mask_flat=brain_mask_flat,
            data_norm=data_norm,
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


@hydra.main(config_path="../../conf_eval", config_name="config.yaml", version_base=None)
def _main(cfg: DictConfig):
    """Evaluate score based inference"""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    cpu_device = _first_device("cpu")
    try:
        gpu_device = _first_device("gpu")
    except:
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
        try:
            available = tuple(jax.devices(kind))
        except:
            available = False
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
    output_dir = os.path.join(checkpoint_root, cfg.model_name)
    log.info(f"Output directory: {output_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")

    # Set seed
    log.info(f"Setting seed: {cfg.seed}")
    key = jax.random.PRNGKey(cfg.seed)

    # Load model and simulator first (needed for synthetic data)
    log.info(f"Loading model from {checkpoint_root}/{cfg.model_name}")
    path_checkpoint = os.path.join(checkpoint_root, cfg.model_name)
    checkpoint, model, _ = load_checkpoint(path_checkpoint)
    graphdef, params, static, state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)
    params = checkpoint[cfg.params_name]
    model = nnx.merge(graphdef, params, static, state, copy=True)
    model.eval()
    sim_type = model.tokenizer.simulator

    source = _cfg_get(data_cfg, "source", "real")

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
    else:
        data_path = _resolve_data_path(data_cfg)
        log.info(f"Loading data from {data_path}")
        data, data_norm, brain_mask, bvals, bvecs = load_and_process_data(
            data_path,
            data_cfg.brain_mask,
            data_cfg.mri_data,
            data_cfg.bvals_data,
            data_cfg.bvecs_data,
            data_cfg.round_bvals,
        )
        true_model_mask = None
        thetas_synth = None
        acq_synth = None
        use_true_model_mask = False

    # Optionally restrict processing to a sub-volume
    brain_mask, data_norm, volume_slice = _apply_slice_to_brain(
        getattr(data_cfg, "slice", None), brain_mask, data_norm, log
    )
    export_template = SimpleNamespace(
        affine=data.affine,
        shape=data.shape,
        volume_slice=volume_slice,
    )

    # Only infer within the brain mask
    full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
    brain_mask_flat = brain_mask.reshape(-1)
    acq = acq_synth if acq_synth is not None else acquisition_scheme(bvals, bvecs)
    full_data_flat_in_brain = full_data_flat[brain_mask_flat, :]
    full_data_flat_in_brain = np.nan_to_num(
        full_data_flat_in_brain, nan=0.0, posinf=0.0, neginf=0.0
    )
    log.info(
        f"Full data flat in brain quantiles 1%, 10%, 50%, 90%, 99%: {np.quantile(full_data_flat_in_brain, [0.01, 0.1, 0.5, 0.9, 0.99])}"
    )

    # Clip outliers
    if cfg.data.clip_outliers:
        exclude_outliers = np.quantile(full_data_flat_in_brain, 0.999)
        full_data_flat_in_brain = np.clip(full_data_flat_in_brain, 0, exclude_outliers)

    key, key_masks = jax.random.split(key)
    # Sample masks if needed
    if cfg.sample_mask:
        log.info("Sampling masks")
        with _device_scope(heavy_device):
            models_sampled_brain = sample_mask(
                cfg,
                key_masks,
                model,
                acq,
                full_data_flat_in_brain,
                log,
                devices=eval_devices,
            )
        models_sampled_brain = _to_cpu_array(models_sampled_brain)
        avg_freq = np.mean(models_sampled_brain, axis=(0, 1))
        log.info(f"Average frequency of all models: {avg_freq}")
    else:
        models_sampled_brain = None
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
            )
        model_parameters_brain = _to_cpu_array(model_parameters_brain)
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
    exporter = get_model_selection_exporter(export_type)
    feasible_model_probabilities = None
    export_feasible_probs = bool(
        _cfg_get(export_cfg, "export_feasible_model_probabilities", False)
    )
    if export_feasible_probs and export_type == "ball3stick":
        if getattr(export_cfg, "feasible_models", None) is None:
            log.warning(
                "export_model_selection.feasible_models is not set; skipping feasible model probabilities."
            )
        else:
            log.info("Evaluating feasible model probabilities for export.")
            with _device_scope(heavy_device):
                feasible_model_probabilities = _compute_feasible_model_probabilities(
                    cfg, export_cfg, model, acq, full_data_flat_in_brain
                )
            feasible_model_probabilities = _to_cpu_array(feasible_model_probabilities)

    def _export_masks(masks, target_name, label):
        if masks is None:
            return False
        out_path = os.path.join(checkpoint_root, cfg.model_name, target_name)
        log.info("Exporting model selection %s to %s", label, out_path)
        exporter(
            cfg,
            export_cfg,
            masks,
            out_path,
            export_template,
            brain_mask_flat,
            data_norm.shape[:-1],
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
            data_norm,
            export_template,
            thetas_synth,
            true_model_mask,
            heavy_device,
            eval_devices,
            add_default_ksd=False,
        )

    # Export samples
    if model_parameters_brain is not None:
        out_path = os.path.join(checkpoint_root, cfg.model_name, cfg.export.name)
        export_cfg = cfg.export
        export_type = (
            export_cfg.get("type", "ball3stick")
            if isinstance(export_cfg, DictConfig)
            else getattr(export_cfg, "type", "ball3stick")
        )
        if export_type == "raw":
            export_thetas_raw(
                cfg,
                model_parameters_brain,
                true_model_mask,
                thetas_synth,
                brain_mask_flat,
                data_norm.shape[:-1],
                out_path,
                export_template,
            )
        else:
            export_thetas_to_files_ball3stick(
                cfg,
                model_parameters_brain,
                sim_type,
                None,
                brain_mask_flat,
                data_norm.shape[:-1],
                out_path,
                export_template,
            )

    if models_selected_brain is None:
        models_selected_brain = default_mask
    if models_selected_brain is None and model_masks_synth is not None:
        models_selected_brain = model_masks_synth

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
        data_norm,
        export_template,
        thetas_synth,
        true_model_mask,
        heavy_device,
        eval_devices,
        add_default_ksd=True,
    )


def sample_mask(cfg, key, model, acq, data, logger, devices=None):
    """Sample a mask from the model"""
    sample_mask_fn = build_mask_sample_fn(
        cfg.mask_sample.method,
        cfg.mask_sample.n_samples,
        model,
        acq,
        cfg.mask_sample.p_mask,
    )
    models_sampled_brain = eval_in_batches(
        sample_mask_fn,
        key,
        data,
        batch_size=cfg.mask_sample.eval_batch_size,
        logger=logger,
        devices=devices,
    )
    return models_sampled_brain


def sample_theta(
    cfg, key, model, acq, data, logger, model_mask=None, devices=None, default_mask=None
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
    sample_theta_fn = build_theta_sample_fn(
        name,
        cfg.theta_sample.num_samples,
        model,
        acq,
        model_mask,
        sim_type,
        cfg.theta_sample.params,
        cfg.theta_sample.corrector,
    )

    if model_mask is not None and model_mask.ndim > 1:
        thetas_full = eval_in_batches(
            sample_theta_fn,
            key,
            data,
            model_mask,
            batch_size=cfg.theta_sample.eval_batch_size,
            logger=logger,
            devices=devices,
        )
    else:
        thetas_full = eval_in_batches(
            sample_theta_fn,
            key,
            data,
            batch_size=cfg.theta_sample.eval_batch_size,
            logger=logger,
            devices=devices,
        )
    return thetas_full


def _compute_feasible_model_probabilities(cfg, export_cfg, model, acq, data):
    """Compute probabilities over feasible model masks for each voxel."""
    feasible_models = jnp.array(export_cfg.feasible_models, dtype=jnp.bool)
    p_mask = cfg.mask_sample.p_mask

    def eval_feasible_log_probs(x):
        model_logpmf = jax.vmap(model.log_prob_mask, in_axes=(0, None, None, None))(
            feasible_models, acq, x, jnp.array([p_mask])
        )
        return model_logpmf

    batch_size = 10_000
    probabilities = []
    for i in range(0, data.shape[0], batch_size):
        batch_data = data[i : i + batch_size]
        batch_logpmf = jax.vmap(eval_feasible_log_probs, in_axes=(0,))(batch_data)
        batch_probs = jax.nn.softmax(batch_logpmf, axis=-1)
        probabilities.append(batch_probs)

    if not probabilities:
        return None
    return np.concatenate(probabilities, axis=0)


def embed_in_full_brain_array(to_embed, brain_mask_flat, brain_shape):
    """Embed a tensor in the full brain array"""
    event_shape = to_embed.shape[1:]
    full_brain = np.zeros(
        (brain_mask_flat.shape[0],) + event_shape, dtype=to_embed.dtype
    )
    full_brain[brain_mask_flat, ...] = to_embed
    full_brain = full_brain.reshape(brain_shape + event_shape)
    return full_brain
