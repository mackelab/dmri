import logging
import os
import socket

import jax

memory_fraction = 0.98  # Use 98% of available memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

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
from dmri.eval.export_theta import export_thetas_to_files_ball3stick
from dmri.eval.load_data import load_and_process_data
from dmri.eval.sampling_methods import (
    build_mask_sample_fn,
    build_theta_sample_fn,
    eval_in_batches,
)
from dmri.eval.selection import select_models
from dmri.simulators.acquisition_scheme import ssfp_acquisition_scheme
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
    output_dir = os.path.join(cfg.path_checkpoint, cfg.model_name)
    log.info(f"Output directory: {output_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")

    # Set seed
    log.info(f"Setting seed: {cfg.seed}")
    key = jax.random.PRNGKey(cfg.seed)

    log.info(f"Loading data from {cfg.path}")

    # Load data
    data_type_params = dict(cfg.data_type)
    data, data_norm, brain_mask, acq = load_and_process_data(
        cfg.path, **data_type_params
    )

    # Only infer within the brain mask
    full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
    brain_mask_flat = brain_mask.reshape(-1)
    full_data_flat_in_brain = full_data_flat[brain_mask_flat, :]
    full_data_flat_in_brain = np.nan_to_num(
        full_data_flat_in_brain, nan=0.0, posinf=0.0, neginf=0.0
    )
    # Flatten acq if necessary
    if isinstance(acq, ssfp_acquisition_scheme):
        acq.T1_raw = jnp.array(acq.T1_raw[brain_mask])
        acq.T2_raw = jnp.array(acq.T2_raw[brain_mask])
        acq.B1 = jnp.array(acq.B1[brain_mask])
    log.info(
        f"Full data flat in brain quantiles 1%, 10%, 50%, 90%, 99%: {np.quantile(full_data_flat_in_brain, [0.01, 0.1, 0.5, 0.9, 0.99])}"
    )
    log.info(f"Acquisition scheme: {jax.tree_util.tree_map(lambda x: x.shape, acq)}")

    # Clip outliers
    if cfg.data_type.clip_outliers:
        exclude_outliers = np.quantile(full_data_flat_in_brain, 0.999)
        full_data_flat_in_brain = np.clip(full_data_flat_in_brain, 0, exclude_outliers)

    # Build model and simulator
    log.info(f"Loading model from {cfg.path_checkpoint}/{cfg.model_name}")
    path_checkpoint = os.path.join(cfg.path_checkpoint, cfg.model_name)
    checkpoint, model, _ = load_checkpoint(path_checkpoint)
    graphdef, params, static, state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)
    params = checkpoint[cfg.params_name]
    model = nnx.merge(graphdef, params, static, state)
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
            cfg.path_checkpoint, cfg.model_name, cfg.export_model_selection.name
        )
        export_model_selection_to_files(
            cfg,
            models_selected_brain,
            out_path,
            data,
            brain_mask_flat,
            data_norm.shape[:-1],
            model,
            acq,
            full_data_flat_in_brain,
        )

    # Export samples
    if model_parameters_brain is not None:
        out_path = os.path.join(cfg.path_checkpoint, cfg.model_name, cfg.export.name)
        export_thetas_to_files_ball3stick(
            cfg,
            model_parameters_brain,
            sim_type,
            None,
            brain_mask_flat,
            data_norm.shape[:-1],
            out_path,
            data,
        )

    # Compute reconstruction error
    if cfg.export.export_reconstruction_error:
        out_path = os.path.join(cfg.path_checkpoint, cfg.model_name, cfg.export.name)
        compute_reconstruction_error(
            cfg,
            sim_type,
            acq,
            full_data_flat_in_brain,
            model_parameters_brain,
            models_selected_brain,
            brain_mask_flat,
            data_norm,
            data,
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
        print("this case")
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
