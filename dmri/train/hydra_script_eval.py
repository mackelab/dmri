from functools import partial
import importlib
import logging
import os
import random
import socket
import time

from dmri.simulators import acquisition_scheme
from dmri.simulators.local_signal_models.ball import MultiShellStaticBall
import hydra
import jax
import numpy as np
import optax
import wandb
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from dmri.train.build_model import build_model
from dmri.train.build_simulator import build_simulator
from dmri.train.checkpointing import CheckpointManager
from dmri.train.evaluator import build_pure_eval_fns

import jax.numpy as jnp
from dmri.utils.dmriutils import export_nifti
# Backends


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


from dipy.io.gradients import read_bvals_bvecs
import nibabel as nb
from dmri.train.utils import load_cfg, load_checkpoint
from blackjax import hmc, tempered_smc
from blackjax.smc.resampling import systematic
from dmri.utils.dmriutils import reorder_angles_3fib, sph2cart, make_dyads

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
    data = nb.load(os.path.join(cfg.path, cfg.mri_data))
    data_norm = data.get_fdata()
    data_norm = data_norm.astype(np.float32)

    # Load brain mask
    brain_mask = nb.load(os.path.join(cfg.path, cfg.brain_mask))
    brain_mask = brain_mask.get_fdata().astype(np.bool)

    # Load bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(os.path.join(cfg.path, cfg.bvals), os.path.join(cfg.path, cfg.bvecs))
    _bvals = bvals.astype(np.float32)
    bvecs = bvecs.astype(np.float32)


    # Round bvals to nearest (0, 1000, 2000)
    if cfg.round_bvals:
        bvals_rounded = np.round(_bvals / 1000) * 1000
        bvals = np.clip(bvals_rounded, 0, 2000)
    else:
        bvals = _bvals

    idx = np.argsort(bvals)
    bvals = bvals[idx]
    bvecs = bvecs[idx]
    data_norm = data_norm[...,idx]

    log.info(f"Bvals: {bvals}")


    # Normalized data
    b0_mask = bvals == 0
    S0 = np.mean(data_norm[...,b0_mask], axis=-1, keepdims=True)
    data_norm = data_norm / S0
    data_norm = np.where(brain_mask[...,None], data_norm, 0.)
    data_norm = np.where(np.isnan(data_norm), 0., data_norm)
    S0 = np.where(brain_mask, S0[...,0], 0.)

    full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
    brain_mask_flat = brain_mask.reshape(-1)

    acq = acquisition_scheme(bvals, bvecs)
    full_data_flat_in_brain = full_data_flat[brain_mask_flat,:]

    full_data_flat_in_brain = np.nan_to_num(full_data_flat_in_brain, nan=0.0, posinf=0.0, neginf=0.0)

    log.info(f"Full data flat in brain quantiles 1%, 10%, 50%, 90%, 99%: {np.quantile(full_data_flat_in_brain, [0.01, 0.1, 0.5, 0.9, 0.99])}")

    if cfg.clip_outliers:
        exclude_outliers = np.quantile(full_data_flat_in_brain, 0.999)
        full_data_flat_in_brain = np.clip(full_data_flat_in_brain, 0, exclude_outliers)

    # Build model and simulator
    log.info(f"Loading model from {cfg.path_checkpoint}/{cfg.model_name}")
    path_checkpoint = os.path.join(cfg.path_checkpoint, cfg.model_name)
    checkpoint, model, simulator = load_checkpoint(path_checkpoint)
    graphdef, params, static, state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)
    params = checkpoint[cfg.params_name]

    model = nnx.merge(graphdef, params, static, state)
    model.eval()
    sim_type = model.tokenizer.simulator

    key, key_masks = jax.random.split(key)
    # Sample masks
    if cfg.sample_mask:
        log.info("Sampling masks")
        models_sampled_brain = sample_mask(cfg,key_masks, model, acq,full_data_flat_in_brain, log)
        avg_freq = jnp.mean(models_sampled_brain, axis=1).mean(0)
        log.info(f"Average frequency of all models: {avg_freq}")
    else:
        models_sampled_brain = None
    log.info(f"Models sampled brain shape: {models_sampled_brain.shape if models_sampled_brain is not None else 'None'}")

    # Run selection of models
    if cfg.select_models:
        log.info("Selecting models")
        models_selected_brain = select_models(cfg, key_masks, full_data_flat_in_brain, log, model.tokenizer.simulator, model, acq, model_mask_samples=models_sampled_brain)
    else:
        models_selected_brain = None
    log.info(f"Models selected brain shape: {models_selected_brain.shape if models_selected_brain is not None else 'None'}")

    # Sample theta
    key, key_theta = jax.random.split(key)
    if cfg.sample_theta:
        log.info("Sampling thetas")
        model_parameters_brain = sample_theta(cfg,key_theta, model, acq,full_data_flat_in_brain, log, model_mask=models_selected_brain)
    else:
        model_parameters_brain = None
    log.info(f"Model parameters brain shape: {model_parameters_brain.shape if model_parameters_brain is not None else 'None'}")
    log.info(f"Model parameters nans: {np.isnan(model_parameters_brain).sum()}")

    # Export model selection
    if models_selected_brain is not None:
        out_path = os.path.join(cfg.path_checkpoint, cfg.model_name, cfg.export_model_selection.name)
        export_model_selection_to_files(cfg, models_selected_brain, out_path, data, brain_mask_flat, data_norm.shape[:-1], model, acq, full_data_flat_in_brain)


    # Export samples
    if model_parameters_brain is not None:
        out_path = os.path.join(cfg.path_checkpoint, cfg.model_name, cfg.export.name)
        export_thetas_to_files_ball3stick(cfg,model_parameters_brain, sim_type, None, brain_mask_flat, data_norm.shape[:-1], out_path, data)


    # Compute reconstruction error
    if cfg.export.export_reconstruction_error:
        out_path = os.path.join(cfg.path_checkpoint, cfg.model_name, cfg.export.name)

        def reconstruction_error(theta, mask, x):
            simulator = sim_type.from_theta(theta, model_mask=mask)
            signal = simulator.signal(acq)
            return jnp.mean(jnp.abs(signal - x), axis=-1)

        # Compute error maps
        in_axes1 = (0, None if models_selected_brain is None or models_selected_brain.ndim == 1 else 0, 0)
        in_axes2 = (0, None if models_selected_brain is None or models_selected_brain.ndim < 3 else 0, None)
        _reconstruction_error = jax.vmap(jax.vmap(reconstruction_error, in_axes=in_axes2), in_axes=in_axes1)
        batch_size = 20_000

        error_maps = []
        for i in range(0, full_data_flat_in_brain.shape[0], batch_size):
            batch_end = min(i + batch_size, full_data_flat_in_brain.shape[0])
            batch_data = full_data_flat_in_brain[i:batch_end]
            batch_thetas = model_parameters_brain[i:batch_end]
            batch_model_mask = models_selected_brain[i:batch_end]
            errors = _reconstruction_error(batch_thetas, batch_model_mask, batch_data)
            mean_error = errors if errors.ndim == 1 else errors.mean(-1)
            error_maps.append(mean_error)
        error_maps = np.concatenate(error_maps, axis=0)
        full_error_maps = embed_in_full_brain_array(error_maps, brain_mask_flat.astype(np.bool), data_norm.shape[:-1])
        export_nifti(full_error_maps, data, out_path, "error_reconstruction.nii.gz")




def sample_mask(cfg,key, model, acq,data, logger):
    """Sample a mask from the model"""
    print(cfg)
    def sample_mask_per_x(key, x):
        keys = jax.random.split(key, cfg.mask_sample.n_samples)
        return jax.vmap(model.sample_mask, in_axes=(0,None,None, None, None))(keys, acq.bvals, acq.bvecs, x, jnp.array([cfg.mask_sample.p_mask]))

    sample_mask_per_x = jax.jit(jax.vmap(sample_mask_per_x, in_axes=(0,0)))

    batch_size = cfg.mask_sample.eval_batch_size
    models_selected_brain = []
    for batch_start in range(0, data.shape[0], batch_size):
        logger.info(f"Sampling model cfgs: Batch {batch_start}")
        key, subkey = jax.random.split(key)
        batch_end = min(batch_start + batch_size, data.shape[0])
        batch_data = data[batch_start:batch_end]
        batch_keys = jax.random.split(subkey, batch_data.shape[0])
        batch_masks = sample_mask_per_x(batch_keys, batch_data)
        batch_masks = np.array(batch_masks, dtype=np.bool)
        models_selected_brain.append(batch_masks)
    models_selected_brain = np.concatenate(models_selected_brain, axis=0)

    return models_selected_brain


def select_models(cfg, key, data, logger, sim_type, model, acq, model_mask_samples=None):
    """Select models based on the model mask"""
    num_comp = len(sim_type.model_types) + len(sim_type.noise_types)

    if  cfg.model_selection.name == "none":
        return None
    else:
        model_mask = jnp.array(model_mask_samples, dtype=jnp.bool)

        nans_in_samples = jnp.isnan(model_mask).sum()
        logger.info(f"Number of NaNs in model_mask: {nans_in_samples}")
        model_mask = jnp.where(jnp.isnan(model_mask), True, model_mask)

        incorporate_models = cfg.model_selection.name

        if incorporate_models == "average":
            logger.info("Sampling parameters for each model")
            assert model_mask.ndim == 3, "model_mask must be 3D if incorporate_models is ball3stick_average"
            num_mask_samples = model_mask.shape[1]
            if num_mask_samples >= cfg.theta_sample.num_samples:
                # Select random num_mask_samples from model_mask
                if num_mask_samples > cfg.theta_sample.num_samples:
                    selected_mask_samples = jax.random.choice(key, model_mask, (cfg.theta_sample.num_samples,), axis=1)
                    model_mask = selected_mask_samples
            else:
                raise ValueError(f"num_mask_samples ({num_mask_samples}) must be greater than or equal to num_samples ({cfg.theta_sample.num_samples})")
            average_freq = jnp.mean(model_mask, axis=1).mean(0)
            logger.info(f"Average frequency of all models: {average_freq}")

        elif incorporate_models == "best":
            logger.info("Sampling best model only")
            # Select the most frequent mask
            feasible_models = jnp.array(cfg.model_selection.feasible_models, dtype=jnp.bool)
            p_mask = cfg.mask_sample.p_mask
            def eval_feasible_log_probs(x):
                model_logpmf = jax.vmap(model.log_prob_mask, in_axes=(0,None,None,None,None))(feasible_models, acq.bvals, acq.bvecs, x, jnp.array([p_mask]))
                return model_logpmf

            batch_size = 10_000
            models_selected = []
            for i in range(0, data.shape[0], batch_size):
                batch_data = data[i:i+batch_size]
                batch_logpmf = jax.vmap(eval_feasible_log_probs, in_axes=(0,))(batch_data)
                batch_mask = batch_logpmf.argmax(axis=-1)
                batch_mask = np.array(feasible_models[batch_mask], dtype=np.bool)
                models_selected.append(batch_mask)
            model_mask = np.concatenate(models_selected, axis=0)

        assert model_mask.shape[0] == data.shape[0], "model_mask must have the same number of voxels as the data"
        assert model_mask.shape[-1] == num_comp, "model_mask must have the same number of components as the model"

    return model_mask


def sample_theta(cfg, key, model, acq, data, logger, model_mask=None):
    sim_type = model.tokenizer.simulator
    num_comp = len(sim_type.model_types) + len(sim_type.noise_types)

    logger.info(f"Running inference with {sim_type}")
    logger.info(f"Model mask shape: {model_mask.shape if model_mask is not None else 'None'}")

    if model_mask is None:
        model_mask = jnp.ones(num_comp, dtype=jnp.bool)
    else:
        model_mask = jnp.array(model_mask, dtype=jnp.bool)

        nans_in_samples = jnp.isnan(model_mask).sum()
        logger.info(f"Number of NaNs in model_mask: {nans_in_samples}")
        model_mask = jnp.where(jnp.isnan(model_mask), True, model_mask)

        assert model_mask.shape[0] == data.shape[0], "model_mask must have the same number of voxels as the data"
        assert model_mask.shape[-1] == num_comp, "model_mask must have the same number of components as the model"

    sample_theta_per_x = build_theta_sample_fn(cfg.theta_sample.method, cfg.theta_sample.num_samples, model, acq, data, model_mask, sim_type)

    batch_size = 20_000
    thetas_full = []
    key, subkey = jax.random.split(key)
    for batch_start in range(0, data.shape[0], batch_size):
        logger.info(f"Sampling theta: Batch {batch_start}")
        subkey, subkey_ = jax.random.split(subkey)
        batch_end = min(batch_start + batch_size, data.shape[0])
        batch_data = data[batch_start:batch_end]
        batch_keys = jax.random.split(subkey_, batch_data.shape[0])
        if model_mask is None or model_mask.ndim == 1:
            batch_thetas = sample_theta_per_x(batch_keys, batch_data, model_mask)
        else:
            batch_model_mask = model_mask[batch_start:batch_end]
            batch_thetas = sample_theta_per_x(batch_keys, batch_data, batch_model_mask)
        batch_thetas = np.array(batch_thetas)
        thetas_full.append(batch_thetas)
    thetas_full = np.concatenate(thetas_full, axis=0)

    return thetas_full


def build_theta_sample_fn(method, num_samples,model, acq, data, model_mask, sim_type):

    d = model.tokenizer.simulator.theta_dim
    if method == "smc_corrected":
        def log_likelihood_fn(theta, mask, acq, x):
            simulator = sim_type.from_theta(theta, model_mask=mask)
            ll = simulator.log_likelihood(acq, x)
            ll = jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
            return ll

        def log_prior_fn(theta):
            return jax.scipy.stats.norm.logpdf(theta, 0, 1).sum()

        def smc(rng,thetas,model_mask, acq, x_o):
            hmc_kernel = hmc.build_kernel()
            hmc_kernel = partial(hmc_kernel, step_size=0.005, num_integration_steps=10, inverse_mass_matrix=jnp.ones(d))

            resampling_fn = systematic
            _log_likelihood_fn = partial(log_likelihood_fn, mask=model_mask, acq=acq, x=x_o)
            smc = tempered_smc(log_prior_fn, _log_likelihood_fn, hmc_kernel, hmc.init, {}, resampling_fn, 2)
            state = smc.init(thetas)
            state =state._replace(lmbda=0.99)

            def step(state, rng):
                state, i = state
                lmbda = 0.99 + (i+1)*0.01/5
                new_state, info = smc.step(rng, state, lmbda)
                return (new_state, i+1), info

            rng_keys = jax.random.split(rng, 5)
            final_state, _ = jax.lax.scan(step, (state, 0), rng_keys)
            return final_state[0].particles


        def sample_mcmc_correct(key, x, model_mask):
            key, subkey = jax.random.split(key)
            K = num_samples
            propose_fn = partial(model.sample_theta, num_steps=64 , max_noise=80)
            key_k = jax.random.split(key, K)
            in_axes_model_mask = 0 if model_mask.ndim == 2 else None
            theta = jax.vmap(propose_fn, in_axes=(0,None,None,None, in_axes_model_mask))(key_k, acq.bvals, acq.bvecs, x,model_mask)
            theta_corr = smc(subkey, theta, model_mask, acq, x)
            return theta_corr


        if model_mask is None or model_mask.ndim <= 1:
            in_axes = (0, 0, None)
            sample_theta_per_x = jax.jit(jax.vmap(sample_mcmc_correct, in_axes=in_axes))
        else:
            sample_theta_per_x = jax.jit(jax.vmap(sample_mcmc_correct))
    elif method == "uncorrected":
        def sample_theta_per_x(key, x, model_mask):
            K = num_samples
            in_axes_model_mask = 0 if model_mask.ndim == 2 else None
            sample_fn = jax.vmap(partial(model.sample_theta, num_steps=64, max_noise=80), in_axes=(0,None,None,None, in_axes_model_mask))
            keys = jax.random.split(key, K)
            theta = sample_fn(keys, acq.bvals, acq.bvecs, x, model_mask)
            return theta
        if model_mask is None or model_mask.ndim <= 1:
            in_axes = (0, 0, None)
            sample_theta_per_x = jax.jit(jax.vmap(sample_theta_per_x, in_axes=in_axes))
        else:
            sample_theta_per_x = jax.jit(jax.vmap(sample_theta_per_x))
    elif method == "mcmc_corrected":

        def log_likelihood_fn(theta, mask, acq, x):
            simulator = sim_type.from_theta(theta, model_mask=mask)
            ll = simulator.log_likelihood(acq, x)
            ll = jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
            return ll

        def log_prior_fn(theta):
            return jax.scipy.stats.norm.logpdf(theta, 0, 1).sum()

        def log_posterior_fn(theta, mask, acq, x):
            return log_prior_fn(theta) + log_likelihood_fn(theta, mask, acq, x)

        def mcmc(key, theta, x, model_mask):
            alg = hmc(partial(log_posterior_fn, mask=model_mask, acq=acq, x=x), 0.001, jnp.ones(d), 10)

            state = alg.init(theta)
            def step(state, key):
                state, info = alg.step(key, state)
                return state, info

            keys = jax.random.split(key, 20)
            state, _ = jax.lax.scan(step, state, keys)
            return state.position



        def sample_theta_per_x(key, x, model_mask):
            key, subkey = jax.random.split(key)
            K = num_samples
            propose_fn = partial(model.sample_theta, num_steps=64 , max_noise=80)
            key_k = jax.random.split(key, K)
            in_axes_model_mask = 0 if model_mask.ndim == 2 else None
            theta = jax.vmap(propose_fn, in_axes=(0,None,None,None, in_axes_model_mask))(key_k, acq.bvals, acq.bvecs, x,model_mask)
            key_k2 = jax.random.split(subkey, K)
            theta_corr = jax.vmap(mcmc, in_axes=(0,0,None,in_axes_model_mask))(key_k2, theta, x, model_mask)
            return theta_corr

        if model_mask is None or model_mask.ndim <= 1:
            in_axes = (0, 0, None)
            sample_theta_per_x = jax.jit(jax.vmap(sample_theta_per_x, in_axes=in_axes))
        else:
            sample_theta_per_x = jax.jit(jax.vmap(sample_theta_per_x))

    else:
        raise ValueError(f"Method {method} not supported")

    return sample_theta_per_x



def export_thetas_to_files_ball3stick(cfg,thetas, sim_type, model_mask, brain_mask_flat, brain_shape, out_path, orig_data):
    """Export thetas to files for ball3stick"""

    if not os.path.exists(out_path):
        os.makedirs(out_path)


    def to_fractions(theta):
        return sim_type.from_theta(theta, model_mask=model_mask).model_fractions

    def to_diffusivities(theta):
        return sim_type.from_theta(theta, model_mask=model_mask).model_compartments[0].lam



    def direction_s1(theta):
        return sim_type.from_theta(theta, model_mask=model_mask).model_compartments[1].mu

    def direction_s2(theta):
        return sim_type.from_theta(theta, model_mask=model_mask).model_compartments[2].mu

    def direction_s3(theta):
        return sim_type.from_theta(theta, model_mask=model_mask).model_compartments[3].mu

    def snr(theta):
        return sim_type.from_theta(theta, model_mask=model_mask).noise_compartments[0].snr



    fractions = np.array(jax.vmap(jax.vmap(to_fractions))(thetas))
    diffusitivity = np.array(jax.vmap(jax.vmap(to_diffusivities))(thetas))
    mu1 = np.array(jax.vmap(jax.vmap(direction_s1))(thetas))
    mu2 = np.array(jax.vmap(jax.vmap(direction_s2))(thetas))
    mu3 = np.array(jax.vmap(jax.vmap(direction_s3))(thetas))
    snr = np.array(jax.vmap(jax.vmap(snr))(thetas))

    # To save multishell stds
    is_multi_shell = sim_type.model_types[0] is MultiShellStaticBall
    if is_multi_shell:
        def to_diffusivities_std(theta):
            return sim_type.from_theta(theta, model_mask=model_mask).model_compartments[0].lam_std
        diffusitivity_std = np.array(jax.vmap(jax.vmap(to_diffusivities_std))(thetas))
        diffusitivity_std_mean = np.mean(diffusitivity_std, axis=1)

        full_diffusitivity_std_mean = embed_in_full_brain_array(diffusitivity_std_mean, brain_mask_flat, brain_shape)
        export_nifti(full_diffusitivity_std_mean, orig_data, out_path, "mean_dstdsamples_std.nii.gz")


    if model_mask is not None:
        # Mask out voxels that are not in the model
        if model_mask.ndim == 2:
            model_mask = model_mask[...,None,:]
            model_mask = np.repeat(model_mask, fractions.shape[1], axis=-1)
        fractions = np.where(model_mask, fractions, 0)
    else:
        fractions = fractions

    if cfg.export.sort_by_fractions:
        # Keep ball fraction unchanged
        fractions_new = np.zeros_like(fractions)
        fractions_new[...,0] = fractions[...,0]

        # Initialize arrays for sorted parameters
        mu1_new = np.zeros_like(mu1)
        mu2_new = np.zeros_like(mu2)
        mu3_new = np.zeros_like(mu3)

        for i in range(fractions.shape[0]):
            for j in range(fractions.shape[1]):
                # Get indices that would sort stick fractions in descending order
                idx = np.argsort(-fractions[i,j,1:]) # Negative to sort descending

                # Sort stick fractions
                fractions_new[i,j,1:] = fractions[i,j,1:][idx]
                # Stack and sort corresponding mu parameters
                mus = np.stack([mu1[i,j], mu2[i,j], mu3[i,j]], axis=0)
                mus_sorted = mus[idx]

                # Unpack sorted mus
                mu1_new[i,j] = mus_sorted[0]
                mu2_new[i,j] = mus_sorted[1]
                mu3_new[i,j] = mus_sorted[2]

        # Replace original arrays with sorted versions
        fractions = fractions_new
        mu1 = mu1_new
        mu2 = mu2_new
        mu3 = mu3_new

        # Chekc that fraction still sums to 1
        assert np.allclose(np.sum(fractions, axis=-1), 1.0), "Fraction does not sum to 1"

        # Check that sorting worked
        assert np.all(fractions[...,1] >= fractions[...,2]), "f1 should be greater than f2"
        assert np.all(fractions[...,2] >= fractions[...,3]), "f2 should be greater than f3"


    # Moments
    fractions_mean = np.mean(fractions, axis=1)
    diffusitivity_mean = np.mean(diffusitivity, axis=1)



    f0_mean = fractions_mean[...,0]
    f1_mean = fractions_mean[...,1]
    f2_mean = fractions_mean[...,2]
    f3_mean = fractions_mean[...,3]
    fsum_mean = f1_mean + f2_mean + f3_mean

    if cfg.export.export_stds:
        fractions_std = np.std(fractions, axis=1)
        diffusitivity_std = np.std(diffusitivity, axis=1)
        f0_std = fractions_std[...,0]
        f1_std = fractions_std[...,1]
        f2_std = fractions_std[...,2]
        f3_std = fractions_std[...,3]
        fsum_std = f1_std + f2_std + f3_std

    diffusitivity_mean = diffusitivity_mean

    # Export diffusitivity
    full_diffusitivity_mean = embed_in_full_brain_array(diffusitivity_mean, brain_mask_flat, brain_shape)
    full_diffusitivity_samples = embed_in_full_brain_array(diffusitivity, brain_mask_flat, brain_shape)

    export_nifti(full_diffusitivity_mean, orig_data, out_path, "mean_dsamples.nii.gz")
    export_nifti(full_diffusitivity_samples, orig_data, out_path, "merged_dsamples.nii.gz")


    if cfg.export.export_stds:
        full_diffusitivity_std = embed_in_full_brain_array(diffusitivity_std, brain_mask_flat, brain_shape)
        export_nifti(full_diffusitivity_std, orig_data, out_path, "std_dsamples.nii.gz")

    # Export SNR
    snr_mean = np.mean(snr, axis=1)


    full_snr_mean = embed_in_full_brain_array(snr_mean, brain_mask_flat, brain_shape)
    full_snr_samples = embed_in_full_brain_array(snr, brain_mask_flat, brain_shape)

    if cfg.export.export_stds:
        snr_std = np.std(snr, axis=1)
        full_snr_std = embed_in_full_brain_array(snr_std, brain_mask_flat, brain_shape)
        export_nifti(full_snr_std, orig_data, out_path, "std_snrsamples.nii.gz")

    export_nifti(full_snr_mean, orig_data, out_path, "mean_snrsamples.nii.gz")
    export_nifti(full_snr_samples, orig_data, out_path, "merged_snrsamples.nii.gz")

    # Export mean fractions
    full_f0_mean = embed_in_full_brain_array(f0_mean, brain_mask_flat, brain_shape)
    full_f1_mean = embed_in_full_brain_array(f1_mean, brain_mask_flat, brain_shape)
    full_f2_mean = embed_in_full_brain_array(f2_mean, brain_mask_flat, brain_shape)
    full_f3_mean = embed_in_full_brain_array(f3_mean, brain_mask_flat, brain_shape)
    full_fsum_mean = embed_in_full_brain_array(fsum_mean, brain_mask_flat, brain_shape)

    nfib1_pred = np.sum(f1_mean > 0.05, axis=-1).astype(np.float32)
    nfib2_pred = np.sum(f2_mean > 0.05, axis=-1).astype(np.float32)
    nfib3_pred = np.sum(f3_mean > 0.05, axis=-1).astype(np.float32)
    num_fib_pred = nfib1_pred + nfib2_pred + nfib3_pred

    # Some usefull auxiliary
    f2_f1_ratio = f2_mean / f1_mean
    f3_f1_ratio = f3_mean / f1_mean
    full_f2_f1_ratio = embed_in_full_brain_array(f2_f1_ratio, brain_mask_flat, brain_shape)
    full_f3_f1_ratio = embed_in_full_brain_array(f3_f1_ratio, brain_mask_flat, brain_shape)
    full_num_fib_pred = embed_in_full_brain_array(num_fib_pred, brain_mask_flat, brain_shape)

    # Export as nifti
    export_nifti(full_f0_mean, orig_data, out_path, "mean_f0samples.nii.gz")
    export_nifti(full_f1_mean, orig_data, out_path, "mean_f1samples.nii.gz")
    export_nifti(full_f2_mean, orig_data, out_path, "mean_f2samples.nii.gz")
    export_nifti(full_f3_mean, orig_data, out_path, "mean_f3samples.nii.gz")
    export_nifti(full_fsum_mean, orig_data, out_path, "mean_fsumsamples.nii.gz")
    export_nifti(full_f2_f1_ratio, orig_data, out_path, "mean_f2_f1_ratiosamples.nii.gz")
    export_nifti(full_f3_f1_ratio, orig_data, out_path, "mean_f3_f1_ratiosamples.nii.gz")
    export_nifti(full_num_fib_pred, orig_data, out_path, "mean_num_fib_predsamples.nii.gz")
    # Export std fractions
    if cfg.export.export_stds:
        full_f0_std = embed_in_full_brain_array(f0_std, brain_mask_flat, brain_shape)
        full_f1_std = embed_in_full_brain_array(f1_std, brain_mask_flat, brain_shape)
        full_f2_std = embed_in_full_brain_array(f2_std, brain_mask_flat, brain_shape)
        full_f3_std = embed_in_full_brain_array(f3_std, brain_mask_flat, brain_shape)
        full_fsum_std = embed_in_full_brain_array(fsum_std, brain_mask_flat, brain_shape)

        export_nifti(full_f0_std, orig_data, out_path, "std_f0samples.nii.gz")
        export_nifti(full_f1_std, orig_data, out_path, "std_f1samples.nii.gz")
        export_nifti(full_f2_std, orig_data, out_path, "std_f2samples.nii.gz")
        export_nifti(full_f3_std, orig_data, out_path, "std_f3samples.nii.gz")
        export_nifti(full_fsum_std, orig_data, out_path, "std_fsumsamples.nii.gz")

    # Export samples fractions
    full_fractions_samples = embed_in_full_brain_array(fractions, brain_mask_flat, brain_shape)
    export_nifti(full_fractions_samples[...,0], orig_data, out_path, "merged_f0samples.nii.gz")
    export_nifti(full_fractions_samples[...,1], orig_data, out_path, "merged_f1samples.nii.gz")
    export_nifti(full_fractions_samples[...,2], orig_data, out_path, "merged_f2samples.nii.gz")
    export_nifti(full_fractions_samples[...,3], orig_data, out_path, "merged_f3samples.nii.gz")


    # Angles
    mu1_theta = mu1[...,0]
    mu2_theta = mu2[...,0]
    mu3_theta = mu3[...,0]

    mu1_phi = mu1[...,1]
    mu2_phi = mu2[...,1]
    mu3_phi = mu3[...,1]


    # Export theta angles samples
    full_mu1_theta = embed_in_full_brain_array(mu1_theta, brain_mask_flat, brain_shape)
    full_mu2_theta = embed_in_full_brain_array(mu2_theta, brain_mask_flat, brain_shape)
    full_mu3_theta = embed_in_full_brain_array(mu3_theta, brain_mask_flat, brain_shape)
    export_nifti(full_mu1_theta, orig_data, out_path, "merged_th1samples.nii.gz")
    export_nifti(full_mu2_theta, orig_data, out_path, "merged_th2samples.nii.gz")
    export_nifti(full_mu3_theta, orig_data, out_path, "merged_th3samples.nii.gz")

    # Export phi angles samples
    full_mu1_phi = embed_in_full_brain_array(mu1_phi, brain_mask_flat, brain_shape)
    full_mu2_phi = embed_in_full_brain_array(mu2_phi, brain_mask_flat, brain_shape)
    full_mu3_phi = embed_in_full_brain_array(mu3_phi, brain_mask_flat, brain_shape)
    export_nifti(full_mu1_phi, orig_data, out_path, "merged_ph1samples.nii.gz")
    export_nifti(full_mu2_phi, orig_data, out_path, "merged_ph2samples.nii.gz")
    export_nifti(full_mu3_phi, orig_data, out_path, "merged_ph3samples.nii.gz")

    # Cartesian samples
    mu1_cart = jax.vmap(jax.vmap(sph2cart, in_axes=(0,0)), in_axes=(0,0))(mu1_theta, mu1_phi)
    mu2_cart = jax.vmap(jax.vmap(sph2cart, in_axes=(0,0)), in_axes=(0,0))(mu2_theta, mu2_phi)
    mu3_cart = jax.vmap(jax.vmap(sph2cart, in_axes=(0,0)), in_axes=(0,0))(mu3_theta, mu3_phi)

    # Export cartesian samples
    full_mu1_cart = embed_in_full_brain_array(mu1_cart, brain_mask_flat, brain_shape)
    full_mu2_cart = embed_in_full_brain_array(mu2_cart, brain_mask_flat, brain_shape)
    full_mu3_cart = embed_in_full_brain_array(mu3_cart, brain_mask_flat, brain_shape)
    export_nifti(full_mu1_cart, orig_data, out_path, "merged_cart1samples.nii.gz")
    export_nifti(full_mu2_cart, orig_data, out_path, "merged_cart2samples.nii.gz")
    export_nifti(full_mu3_cart, orig_data, out_path, "merged_cart3samples.nii.gz")

    # Dyads unarranged
    dyads1_in_brain_unordered, dyads1_disp_unordered = jax.vmap(make_dyads)(mu1_theta, mu1_phi)
    dyads2_in_brain_unordered, dyads2_disp_unordered = jax.vmap(make_dyads)(mu2_theta, mu2_phi)
    dyads3_in_brain_unordered, dyads3_disp_unordered = jax.vmap(make_dyads)(mu3_theta, mu3_phi)

    # Export dyads unordered
    full_dyads1_in_brain_unordered = embed_in_full_brain_array(dyads1_in_brain_unordered, brain_mask_flat, brain_shape)
    full_dyads2_in_brain_unordered = embed_in_full_brain_array(dyads2_in_brain_unordered, brain_mask_flat, brain_shape)
    full_dyads3_in_brain_unordered = embed_in_full_brain_array(dyads3_in_brain_unordered, brain_mask_flat, brain_shape)
    export_nifti(full_dyads1_in_brain_unordered, orig_data, out_path, "dyads1.nii.gz")
    export_nifti(full_dyads2_in_brain_unordered, orig_data, out_path, "dyads2.nii.gz")
    export_nifti(full_dyads3_in_brain_unordered, orig_data, out_path, "dyads3.nii.gz")

    # Export dyads dispersion
    full_dyads1_disp_unordered = embed_in_full_brain_array(dyads1_disp_unordered, brain_mask_flat, brain_shape)
    full_dyads2_disp_unordered = embed_in_full_brain_array(dyads2_disp_unordered, brain_mask_flat, brain_shape)
    full_dyads3_disp_unordered = embed_in_full_brain_array(dyads3_disp_unordered, brain_mask_flat, brain_shape)
    export_nifti(full_dyads1_disp_unordered, orig_data, out_path, "dyads1_dispersion.nii.gz")
    export_nifti(full_dyads2_disp_unordered, orig_data, out_path, "dyads2_dispersion.nii.gz")
    export_nifti(full_dyads3_disp_unordered, orig_data, out_path, "dyads3_dispersion.nii.gz")



    # Reorder
    # Dyads reordered
    if cfg.export.reorder_dyads:

        reordered_path = os.path.join(out_path, "reordered")
        if not os.path.exists(reordered_path):
            os.makedirs(reordered_path)

        f1 = fractions[...,1]
        f2 = fractions[...,2]
        f3 = fractions[...,3]
        mu1_reordered, mu2_reordered, mu3_reordered, f1_reordered, f2_reordered, f3_reordered = jax.vmap(reorder_angles_3fib)(mu1, mu2, mu3, f1, f2, f3)


        dyads1_in_brain_reordered, dyads1_disp_reordered = jax.vmap(make_dyads)(mu1_reordered[...,0], mu1_reordered[...,1])
        dyads2_in_brain_reordered, dyads2_disp_reordered = jax.vmap(make_dyads)(mu2_reordered[...,0], mu2_reordered[...,1])
        dyads3_in_brain_reordered, dyads3_disp_reordered = jax.vmap(make_dyads)(mu3_reordered[...,0], mu3_reordered[...,1])

        fractions_reordered = np.concatenate([fractions[...,0][...,None], f1_reordered[...,None], f2_reordered[...,None], f3_reordered[...,None]], axis=-1)
        assert np.allclose(np.sum(fractions_reordered, axis=-1), 1.0), "Fraction does not sum to 1"
        # Export reordered
        full_fractions_reordered = embed_in_full_brain_array(fractions_reordered, brain_mask_flat, brain_shape)


        export_nifti(full_fractions_reordered, orig_data, reordered_path, "merged_fsamples.nii.gz")

        f0_reordered = fractions_reordered[...,0]
        f1_reordered = fractions_reordered[...,1]
        f2_reordered = fractions_reordered[...,2]
        f3_reordered = fractions_reordered[...,3]
        fsum_reordered = f1_reordered + f2_reordered + f3_reordered

        full_f0_reordered = embed_in_full_brain_array(f0_reordered, brain_mask_flat, brain_shape)
        full_f1_reordered = embed_in_full_brain_array(f1_reordered, brain_mask_flat, brain_shape)
        full_f2_reordered = embed_in_full_brain_array(f2_reordered, brain_mask_flat, brain_shape)
        full_f3_reordered = embed_in_full_brain_array(f3_reordered, brain_mask_flat, brain_shape)
        full_fsum_reordered = embed_in_full_brain_array(fsum_reordered, brain_mask_flat, brain_shape)
        export_nifti(full_f0_reordered, orig_data, reordered_path, "merged_f0samples.nii.gz")
        export_nifti(full_f1_reordered, orig_data, reordered_path, "merged_f1samples.nii.gz")
        export_nifti(full_f2_reordered, orig_data, reordered_path, "merged_f2samples.nii.gz")
        export_nifti(full_f3_reordered, orig_data, reordered_path, "merged_f3samples.nii.gz")
        export_nifti(full_fsum_reordered, orig_data, reordered_path, "merged_fsumsamples.nii.gz")

        f0_std_reordered = fractions_reordered[...,0]
        f1_std_reordered = fractions_reordered[...,1]
        f2_std_reordered = fractions_reordered[...,2]
        f3_std_reordered = fractions_reordered[...,3]
        fsum_std_reordered = f1_std_reordered + f2_std_reordered + f3_std_reordered

        if cfg.export.export_stds:
            full_f0_std_reordered = embed_in_full_brain_array(f0_std_reordered, brain_mask_flat, brain_shape)
            full_f1_std_reordered = embed_in_full_brain_array(f1_std_reordered, brain_mask_flat, brain_shape)
            full_f2_std_reordered = embed_in_full_brain_array(f2_std_reordered, brain_mask_flat, brain_shape)
            full_f3_std_reordered = embed_in_full_brain_array(f3_std_reordered, brain_mask_flat, brain_shape)
            full_fsum_std_reordered = embed_in_full_brain_array(fsum_std_reordered, brain_mask_flat, brain_shape)
            export_nifti(full_f0_std_reordered, orig_data, reordered_path, "std_f0samples.nii.gz")
            export_nifti(full_f1_std_reordered, orig_data, reordered_path, "std_f1samples.nii.gz")
            export_nifti(full_f2_std_reordered, orig_data, reordered_path, "std_f2samples.nii.gz")
            export_nifti(full_f3_std_reordered, orig_data, reordered_path, "std_f3samples.nii.gz")
            export_nifti(full_fsum_std_reordered, orig_data, reordered_path, "std_fsumsamples.nii.gz")


        full_dyads1_in_brain_reordered = embed_in_full_brain_array(dyads1_in_brain_reordered, brain_mask_flat, brain_shape)
        full_dyads2_in_brain_reordered = embed_in_full_brain_array(dyads2_in_brain_reordered, brain_mask_flat, brain_shape)
        full_dyads3_in_brain_reordered = embed_in_full_brain_array(dyads3_in_brain_reordered, brain_mask_flat, brain_shape)
        export_nifti(full_dyads1_in_brain_reordered, orig_data, reordered_path, "dyads1.nii.gz")
        export_nifti(full_dyads2_in_brain_reordered, orig_data, reordered_path, "dyads2.nii.gz")
        export_nifti(full_dyads3_in_brain_reordered, orig_data, reordered_path, "dyads3.nii.gz")

        full_dyads1_disp_reordered = embed_in_full_brain_array(dyads1_disp_reordered, brain_mask_flat, brain_shape)
        full_dyads2_disp_reordered = embed_in_full_brain_array(dyads2_disp_reordered, brain_mask_flat, brain_shape)
        full_dyads3_disp_reordered = embed_in_full_brain_array(dyads3_disp_reordered, brain_mask_flat, brain_shape)
        export_nifti(full_dyads1_disp_reordered, orig_data, reordered_path, "dyads1_dispersion.nii.gz")
        export_nifti(full_dyads2_disp_reordered, orig_data, reordered_path, "dyads2_dispersion.nii.gz")
        export_nifti(full_dyads3_disp_reordered, orig_data, reordered_path, "dyads3_dispersion.nii.gz")




def export_model_selection_to_files(cfg, model_mask, out_path, orig_data, brain_mask_flat, brain_shape, model, acq, data):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Export the samples
    full_model_mask = embed_in_full_brain_array(model_mask, brain_mask_flat, brain_shape).astype(np.float32)
    export_nifti(full_model_mask, orig_data, out_path, "merged_model_mask.nii.gz")

    marginal_probabilities = jnp.mean(model_mask, axis=1)
    full_marginal_probabilities = embed_in_full_brain_array(marginal_probabilities, brain_mask_flat, brain_shape)
    export_nifti(full_marginal_probabilities, orig_data, out_path, "mean_marginal_probabilities.nii.gz")

    # Get feasible models from config
    feasible_models = jnp.array(cfg.export_model_selection.feasible_models, dtype=jnp.bool)
    p_mask = cfg.mask_sample.p_mask

    # Calculate log probabilities for each feasible model
    def eval_feasible_log_probs(x):
        model_logpmf = jax.vmap(model.log_prob_mask, in_axes=(0,None,None,None,None))(feasible_models, acq.bvals, acq.bvecs, x, jnp.array([p_mask]))
        return model_logpmf

    # Process in batches to avoid memory issues
    batch_size = 10_000
    probabilities = []
    for i in range(0, data.shape[0], batch_size):
        batch_data = data[i:i+batch_size]
        batch_logpmf = jax.vmap(eval_feasible_log_probs, in_axes=(0,))(batch_data)
        # Convert log probabilities to probabilities
        batch_probs = jax.nn.softmax(batch_logpmf, axis=-1)
        probabilities.append(batch_probs)
    probabilities = np.concatenate(probabilities, axis=0)

    # Export probabilities for each feasible model
    for i in range(len(feasible_models)):
        p_model_i = probabilities[...,i]
        full_p_model_i = embed_in_full_brain_array(p_model_i, brain_mask_flat, brain_shape)
        export_nifti(full_p_model_i, orig_data, out_path, f"p_feasible_model_{i}.nii.gz")


def embed_in_full_brain_array(to_embed, brain_mask_flat, brain_shape):
    """Embed a tensor in the full brain array"""
    event_shape = to_embed.shape[1:]
    full_brain = np.zeros((brain_mask_flat.shape[0],) + event_shape, dtype=to_embed.dtype)
    full_brain[brain_mask_flat,...] = to_embed
    full_brain = full_brain.reshape(brain_shape + event_shape)

    return full_brain
