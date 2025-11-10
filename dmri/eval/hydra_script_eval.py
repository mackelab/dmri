import logging
import os
import socket
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
from omegaconf import DictConfig, OmegaConf

from dmri.eval.export_metrics import compute_reconstruction_error
from dmri.eval.export_models import export_model_selection_to_files
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


def _resolve_data_path(cfg):
    explicit_path = _cfg_get(cfg, "path")
    if explicit_path not in (None, "", "null", "None"):
        return os.path.expanduser(str(explicit_path))
    data_folder = _cfg_get(cfg, "data_folder")
    if data_folder in (None, "", "null", "None"):
        raise ValueError(
            "Please set either cfg.path or cfg.data_folder to select the input data."
        )
    data_root = _cfg_get(cfg, "data_root")
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

    log.info(f"Model name: {cfg.model_name}")
    data_path = _resolve_data_path(cfg)
    checkpoint_root = _resolve_checkpoint_root(cfg)
    output_dir = os.path.join(checkpoint_root, cfg.model_name)
    log.info(f"Output directory: {output_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")

    # Set seed
    log.info(f"Setting seed: {cfg.seed}")
    key = jax.random.PRNGKey(cfg.seed)

    log.info(f"Loading data from {data_path}")

    # Load data
    data, data_norm, brain_mask, bvals, bvecs = load_and_process_data(
        data_path,
        cfg.brain_mask,
        cfg.mri_data,
        cfg.bvals_data,
        cfg.bvecs_data,
        cfg.round_bvals,
    )

    # Optionally restrict processing to a sub-volume
    brain_mask, data_norm, volume_slice = _apply_slice_to_brain(
        cfg.get("slice", None), brain_mask, data_norm, log
    )
    export_template = SimpleNamespace(
        affine=data.affine,
        shape=data.shape,
        volume_slice=volume_slice,
    )

    # Only infer within the brain mask
    full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
    brain_mask_flat = brain_mask.reshape(-1)
    acq = acquisition_scheme(bvals, bvecs)
    full_data_flat_in_brain = full_data_flat[brain_mask_flat, :]
    full_data_flat_in_brain = np.nan_to_num(
        full_data_flat_in_brain, nan=0.0, posinf=0.0, neginf=0.0
    )
    log.info(
        f"Full data flat in brain quantiles 1%, 10%, 50%, 90%, 99%: {np.quantile(full_data_flat_in_brain, [0.01, 0.1, 0.5, 0.9, 0.99])}"
    )

    # Clip outliers
    if cfg.clip_outliers:
        exclude_outliers = np.quantile(full_data_flat_in_brain, 0.999)
        full_data_flat_in_brain = np.clip(full_data_flat_in_brain, 0, exclude_outliers)

    # Build model and simulator
    log.info(f"Loading model from {checkpoint_root}/{cfg.model_name}")
    path_checkpoint = os.path.join(checkpoint_root, cfg.model_name)
    checkpoint, model, _ = load_checkpoint(path_checkpoint)
    graphdef, params, static, state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)
    params = checkpoint[cfg.params_name]
    model = nnx.merge(graphdef, params, static, state, copy=True)
    model.eval()
    sim_type = model.tokenizer.simulator

    key, key_masks = jax.random.split(key)
    # Sample masks if needed
    if cfg.sample_mask:
        log.info("Sampling masks")
        models_sampled_brain = sample_mask(
            cfg, key_masks, model, acq, full_data_flat_in_brain, log
        )
        avg_freq = jnp.mean(models_sampled_brain, axis=1).mean(0)
        log.info(f"Average frequency of all models: {avg_freq}")
    else:
        models_sampled_brain = None
    log.info(
        f"Models sampled brain shape: {models_sampled_brain.shape if models_sampled_brain is not None else 'None'}"
    )

    # Run selection of models if needed
    if cfg.select_models:
        log.info("Selecting models")
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
    else:
        models_selected_brain = None
    log.info(
        f"Models selected brain shape: {models_selected_brain.shape if models_selected_brain is not None else 'None'}"
    )

    # Sample theta
    key, key_theta = jax.random.split(key)
    if cfg.sample_theta:
        log.info("Sampling thetas")
        model_parameters_brain = sample_theta(
            cfg,
            key_theta,
            model,
            acq,
            full_data_flat_in_brain,
            log,
            model_mask=models_selected_brain,
        )
    else:
        model_parameters_brain = None
    log.info(
        f"Model parameters brain shape: {model_parameters_brain.shape if model_parameters_brain is not None else 'None'}"
    )
    log.info(f"Model parameters nans: {np.isnan(model_parameters_brain).sum()}")

    # Export model selection
    if models_selected_brain is not None:
        out_path = os.path.join(
            checkpoint_root, cfg.model_name, cfg.export_model_selection.name
        )
        export_model_selection_to_files(
            cfg,
            models_selected_brain,
            out_path,
            export_template,
            brain_mask_flat,
            data_norm.shape[:-1],
            model,
            acq,
            full_data_flat_in_brain,
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

    # Compute reconstruction error
    if cfg.export.export_reconstruction_error:
        out_path = os.path.join(checkpoint_root, cfg.model_name, cfg.export.name)
        compute_reconstruction_error(
            cfg,
            sim_type,
            acq,
            full_data_flat_in_brain,
            model_parameters_brain,
            models_selected_brain,
            brain_mask_flat,
            data_norm,
            export_template,
            out_path,
        )


def sample_mask(cfg, key, model, acq, data, logger):
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
    )
    return models_sampled_brain


def sample_theta(cfg, key, model, acq, data, logger, model_mask=None):
    """Sample theta parameters"""
    sim_type = model.tokenizer.simulator
    num_comp = len(sim_type.model_types) + len(sim_type.noise_types)

    logger.info(f"Running inference with {sim_type}")
    logger.info(
        f"Model mask shape: {model_mask.shape if model_mask is not None else 'None'}"
    )

    if model_mask is None:
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
        )
    else:
        thetas_full = eval_in_batches(
            sample_theta_fn,
            key,
            data,
            batch_size=cfg.theta_sample.eval_batch_size,
            logger=logger,
        )
    return thetas_full


def embed_in_full_brain_array(to_embed, brain_mask_flat, brain_shape):
    """Embed a tensor in the full brain array"""
    event_shape = to_embed.shape[1:]
    full_brain = np.zeros(
        (brain_mask_flat.shape[0],) + event_shape, dtype=to_embed.dtype
    )
    full_brain[brain_mask_flat, ...] = to_embed
    full_brain = full_brain.reshape(brain_shape + event_shape)
    return full_brain
