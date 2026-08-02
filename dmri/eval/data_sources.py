from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    random_advanced_reasearch_acquisition_scheme,
    random_clinical_acquisition,
    random_hardi_acquisition,
    random_hcp_acquisition,
    random_hcp_large_acquisition,
    random_ssfp_acquisition,
)

ACQ_SCHEME_BUILDERS = {
    "clinical": random_clinical_acquisition,
    "hardi": random_hardi_acquisition,
    "advanced_research": random_advanced_reasearch_acquisition_scheme,
    "hcp": random_hcp_acquisition,
    "hcp_large": random_hcp_large_acquisition,
    "ssfp": random_ssfp_acquisition,
}


def _generate_random_bvecs(num_gradients: int, rng):
    vecs = jax.random.normal(rng, (num_gradients, 3))
    vecs = vecs / jnp.clip(jnp.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)
    return np.asarray(vecs, dtype=np.float32)


def _to_container(cfg):
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _build_acq_fn_from_name(name, params):
    if name not in ACQ_SCHEME_BUILDERS:
        raise ValueError(f"Unknown acquisition scheme {name}")
    builder = ACQ_SCHEME_BUILDERS[name]
    param_container = _to_container(params) or {}
    return partial(builder, **param_container)


def _get_acquisition_fns(acq_cfg):
    if isinstance(acq_cfg, (str, bytes)):
        return [_build_acq_fn_from_name(acq_cfg, {})]

    cfg = _to_container(acq_cfg) or {}
    if isinstance(cfg, dict) and "acquisitions" in cfg:
        return [instantiate(item) for item in cfg["acquisitions"]]
    if isinstance(cfg, dict) and "_target_" in cfg:
        return [instantiate(cfg)]
    acq_scheme_name = cfg.get("name") if isinstance(cfg, dict) else None
    acq_params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    acq_schemes = acq_params.get("schemes") if isinstance(acq_params, dict) else None
    if acq_schemes:
        schemes = _to_container(acq_schemes)
        return [
            _build_acq_fn_from_name(scheme["name"], scheme.get("params", {}))
            for scheme in schemes
        ]
    if acq_scheme_name is None:
        raise ValueError(
            "acquisition_scheme must be a name string or mapping with a name field"
        )
    return [_build_acq_fn_from_name(acq_scheme_name, acq_params)]


def _sample_acquisition_from_cfg(acq_cfg, key):
    acq_fns = _get_acquisition_fns(acq_cfg)
    key, scheme_key, acq_key = jax.random.split(key, 3)
    if len(acq_fns) == 1:
        acq_fn = acq_fns[0]
    else:
        idx = int(jax.random.randint(scheme_key, (), 0, len(acq_fns)))
        acq_fn = acq_fns[idx]
    acq = acq_fn(acq_key)
    bvals = np.asarray(acq.bvals, dtype=np.float32)
    bvecs = np.asarray(acq.bvecs, dtype=np.float32)
    return key, acq, bvals, bvecs


def generate_synthetic_data(data_cfg, sim_type, key):
    """Generate synthetic diffusion-MRI data from a simulator configuration.

    Parameters
    ----------
    data_cfg:
        Hydra/OmegaConf dict with ``num_voxels``, ``acquisition_scheme`` or
        ``bvals``/``bvecs``, and optionally ``model_mask_hyperparameter``.
    sim_type:
        A :class:`MultiCompartment` subclass used to sample masks and signals.
    key:
        JAX PRNG key.

    Returns
    -------
    tuple
        ``(orig_data, data_norm, brain_mask, bvals, bvecs, model_masks, thetas, acq, key)``
        arranged as a pseudo-volume suitable for the evaluation pipeline.
    """
    num_voxels = int(getattr(data_cfg, "num_voxels", 256))
    acq_cfg = (
        data_cfg.get("acquisition_scheme", None)
        if hasattr(data_cfg, "get")
        else getattr(data_cfg, "acquisition_scheme", None)
    )
    if acq_cfg is not None:
        key, acq, bvals, bvecs = _sample_acquisition_from_cfg(acq_cfg, key)
    else:
        bvals = np.asarray(
            getattr(data_cfg, "bvals", [0, 1000, 2000]), dtype=np.float32
        )
        bvecs_cfg = getattr(data_cfg, "bvecs", None)
        if bvecs_cfg is None:
            key, bvec_key = jax.random.split(key)
            bvecs = _generate_random_bvecs(len(bvals), bvec_key)
        else:
            bvecs = np.asarray(bvecs_cfg, dtype=np.float32)
        acq = acquisition_scheme(bvals, bvecs)

    # Optionally fix the mask-prior hyperparameters instead of sampling them
    fixed_mask_hyperparameters = (
        data_cfg.get("model_mask_hyperparameter", None)
        if hasattr(data_cfg, "get")
        else getattr(data_cfg, "model_mask_hyperparameter", None)
    )
    fixed_mask_hyperparameters = _to_container(fixed_mask_hyperparameters)

    mask_prior = sim_type.create_mask_prior()

    if fixed_mask_hyperparameters is not None:
        fixed_mask_hyperparameters = jnp.asarray(
            fixed_mask_hyperparameters, dtype=jnp.float32
        )
        if fixed_mask_hyperparameters.ndim == 0:
            fixed_mask_hyperparameters = fixed_mask_hyperparameters[None]

    def _sample_single(rng):
        rng_theta, rng_mask, rng_sig = jax.random.split(rng, 3)
        theta = jax.random.normal(rng_theta, (sim_type.theta_dim,))
        if fixed_mask_hyperparameters is not None:
            model_mask = mask_prior.sample_model_mask(
                rng_mask, fixed_mask_hyperparameters
            )
        else:
            mask_sample = mask_prior.sample(rng_mask)
            model_mask = mask_sample.model_mask

        simulator = sim_type.from_theta(theta, model_mask=model_mask)
        signal = simulator.signal(acq, rng=rng_sig)
        return signal, model_mask, theta

    keys = jax.random.split(key, num_voxels)
    signals, model_masks, thetas = jax.vmap(_sample_single)(keys)
    signals = np.asarray(signals, dtype=np.float32)
    model_masks = np.asarray(model_masks)
    thetas = np.asarray(thetas, dtype=np.float32)

    # Arrange into pseudo-volume of shape (num_voxels, 1, 1, gradients)
    data_norm = signals.reshape((num_voxels, 1, 1, -1))
    brain_mask = np.ones((num_voxels, 1, 1), dtype=bool)
    orig_data = SimpleNamespace(
        affine=np.eye(4, dtype=np.float32),
        shape=data_norm.shape,
        volume_slice=None,
    )
    return orig_data, data_norm, brain_mask, bvals, bvecs, model_masks, thetas, acq, key
