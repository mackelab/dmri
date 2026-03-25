#!/usr/bin/env python3
"""Evaluate FODs for the full brain and save them in batches."""

from __future__ import annotations

import argparse
import os
from functools import partial
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.rumba import RumbaSDModel
from flax import nnx

from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.train.utils import load_checkpoint
from sh_lib import choose_sigma_deg, spherical_kde_vmf_from_samples


LOGGER = logging.getLogger(__name__)


def parse_methods(text: str) -> list[str]:
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def iter_batches(total: int, batch_size: int, start_idx: int = 0):
    for start in range(start_idx, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


def _computed_voxels(existing: np.ndarray) -> np.ndarray:
    finite = np.isfinite(existing).all(axis=-1)
    nonzero = np.any(existing != 0.0, axis=-1)
    return finite & nonzero


def find_resume_start(
    flat_out: np.ndarray, brain_indices: np.ndarray, batch_size: int
) -> int:
    total = brain_indices.shape[0]
    for start, end in iter_batches(total, batch_size):
        existing = flat_out[brain_indices[start:end]]
        computed = _computed_voxels(existing)
        if not np.all(computed):
            return start
    return total


def open_output_map(
    out_path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
    resume: bool,
) -> tuple[np.memmap, bool]:
    if resume and out_path.exists():
        out_map = np.lib.format.open_memmap(out_path, mode="r+")
        if out_map.shape != shape:
            raise ValueError(
                f"Existing output {out_path} has shape {out_map.shape}, expected {shape}."
            )
        return out_map, True
    out_map = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    out_map[:] = 0.0
    return out_map, False


def open_output_array(
    out_path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
    resume: bool,
) -> tuple[np.ndarray, bool]:
    if resume and out_path.exists():
        data = np.load(out_path)
        if data.shape != shape:
            raise ValueError(
                f"Existing output {out_path} has shape {data.shape}, expected {shape}."
            )
        return data, True
    return np.zeros(shape, dtype=dtype), False


def _fsync_path(path: Path) -> None:
    try:
        fd = os.open(path.as_posix(), os.O_RDONLY)
    except OSError as exc:
        LOGGER.warning("fsync open failed for %s (%s).", path, exc)
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        LOGGER.warning("fsync failed for %s (%s).", path, exc)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path.as_posix(), os.O_RDONLY)
    except OSError as exc:
        LOGGER.warning("dir fsync open failed for %s (%s).", path, exc)
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        LOGGER.warning("dir fsync failed for %s (%s).", path, exc)
    finally:
        os.close(fd)


def _save_array(out_path: Path, data: np.ndarray, do_fsync: bool) -> None:
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as handle:
        np.save(handle, data)
        handle.flush()
        if do_fsync:
            os.fsync(handle.fileno())
    os.replace(tmp_path, out_path)
    if do_fsync:
        _fsync_path(out_path)
        _fsync_dir(out_path.parent)


def flush_and_fsync(memmap_obj: np.memmap, do_fsync: bool) -> None:
    memmap_obj.flush()
    if not do_fsync:
        return
    if hasattr(memmap_obj, "_mmap") and memmap_obj._mmap is not None:
        memmap_obj._mmap.flush()
    try:
        fd = os.open(memmap_obj.filename, os.O_RDWR)
    except OSError as exc:
        LOGGER.warning("fsync open failed for %s (%s).", memmap_obj.filename, exc)
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        LOGGER.warning("fsync failed for %s (%s).", memmap_obj.filename, exc)
    finally:
        os.close(fd)


def _model_type_name(model_type) -> str:
    if hasattr(model_type, "__name__"):
        return model_type.__name__
    name = str(model_type).split(".")[-1].replace("'", "")
    return name.rstrip(">")


def build_default_masks(model) -> dict[str, np.ndarray]:
    model_types = model.tokenizer.simulator.model_types
    num_models = model.tokenizer.num_models
    num_noises = model.tokenizer.num_noises
    mask_len = num_models + num_noises

    indices: dict[str, list[int]] = {}
    for i, m_type in enumerate(model_types):
        name = _model_type_name(m_type)
        indices.setdefault(name, []).append(i)

    if "Ball" not in indices:
        raise ValueError("Ball component not found in model types.")
    if "Stick" not in indices or len(indices["Stick"]) < 3:
        raise ValueError("Need at least three Stick components for B3S mask.")
    dti_key = "Dti" if "Dti" in indices else "Tensor"
    if dti_key not in indices or len(indices[dti_key]) < 3:
        raise ValueError("Need at least three Dti/Tensor components for B3T mask.")

    mask_b3s = np.zeros(mask_len, dtype=bool)
    mask_b3t = np.zeros(mask_len, dtype=bool)
    mask_b3s[indices["Ball"][0]] = True
    mask_b3s[indices["Stick"][:3]] = True
    mask_b3t[indices["Ball"][0]] = True
    mask_b3t[indices[dti_key][:3]] = True
    if num_noises > 0:
        mask_b3s[num_models] = True
        mask_b3t[num_models] = True
    return {"b3s": mask_b3s, "b3t": mask_b3t}


def parse_custom_masks(items: list[str], mask_len: int) -> dict[str, np.ndarray]:
    masks = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Custom mask '{item}' must be name:bits.")
        name, bits_str = item.split(":", 1)
        bits_str = bits_str.strip()
        if "," not in bits_str and " " not in bits_str and len(bits_str) == mask_len:
            bits = list(bits_str)
        else:
            bits = [b for b in bits_str.replace(",", " ").split() if b]
        if len(bits) != mask_len:
            raise ValueError(
                f"Mask '{name}' expected length {mask_len} but got {len(bits)}."
            )
        mask = np.array(
            [b.lower() in ("1", "true", "t", "yes", "y") for b in bits],
            dtype=bool,
        )
        masks[name.strip().lower()] = mask
    return masks


def parse_mask_prior(text: str, mask_prior_dim: int) -> jnp.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Mask prior cannot be empty.")
    if len(values) == 1 and mask_prior_dim > 1:
        return jnp.full((mask_prior_dim,), values[0])
    if len(values) != mask_prior_dim:
        raise ValueError(
            f"Mask prior dim {mask_prior_dim} does not match values {len(values)}."
        )
    return jnp.asarray(values)


def apply_slice_mask(
    brain_mask: np.ndarray, axis: str | int, index: int
) -> np.ndarray:
    if isinstance(axis, str):
        axis = axis.lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid slice axis '{axis}'. Use x, y, or z.")
        axis = axis_map[axis]
    if axis not in (0, 1, 2):
        raise ValueError("Slice axis must be 0, 1, or 2.")

    mask = np.zeros_like(brain_mask, dtype=bool)
    if axis == 0:
        mask[index, :, :] = True
    elif axis == 1:
        mask[:, index, :] = True
    else:
        mask[:, :, index] = True
    return brain_mask & mask


def apply_roi_mask(
    brain_mask: np.ndarray, roi: tuple[int, int, int, int, int, int]
) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = roi
    if not (0 <= x0 < x1 <= brain_mask.shape[0]):
        raise ValueError("ROI x range is out of bounds.")
    if not (0 <= y0 < y1 <= brain_mask.shape[1]):
        raise ValueError("ROI y range is out of bounds.")
    if not (0 <= z0 < z1 <= brain_mask.shape[2]):
        raise ValueError("ROI z range is out of bounds.")
    roi_mask = np.zeros_like(brain_mask, dtype=bool)
    roi_mask[x0:x1, y0:y1, z0:z1] = True
    return brain_mask & roi_mask


def load_model(checkpoint_path: Path):
    checkpoint, model, simulator = load_checkpoint(str(checkpoint_path))
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema", checkpoint.get("params", params))
    state = checkpoint.get("model_state", state)
    model = nnx.merge(graphdef, params, state)
    model.eval()
    return model, simulator


def load_data_from_arrays(
    data: np.ndarray,
    brain_mask: np.ndarray | None,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    outlier_quantile: float | None,
):
    if brain_mask is None:
        brain_mask = np.ones(data.shape[:-1], dtype=bool)
    brain_mask = brain_mask.astype(bool)

    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)

    bvals = np.round(bvals / 1000.0) * 1000.0
    bvals = np.clip(bvals, 0.0, None)
    b0_mask = bvals == 0

    if np.any(b0_mask):
        s0 = np.mean(data[..., b0_mask], axis=-1)
    else:
        LOGGER.warning("No b0 volumes found; skipping normalization.")
        s0 = np.ones(data.shape[:-1], dtype=np.float32)
    s0 = np.where(np.isfinite(s0) & (s0 > 0), s0, 1.0)

    data_norm = data / s0[..., None]
    data_norm = np.where(brain_mask[..., None], data_norm, 0.0)
    data_norm = np.where(np.isnan(data_norm), 0.0, data_norm)

    idx = np.argsort(bvals)
    bvals = bvals[idx]
    bvecs = bvecs[idx]
    data_norm = data_norm[..., idx]

    full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
    brain_mask_flat = brain_mask.reshape(-1).astype(bool)
    data_brain_raw = full_data_flat[brain_mask_flat]
    data_brain_raw = np.nan_to_num(
        data_brain_raw, nan=0.0, posinf=0.0, neginf=0.0
    )
    data_brain = data_brain_raw
    if outlier_quantile is not None:
        cutoff = np.quantile(data_brain_raw, outlier_quantile)
        data_brain = np.clip(data_brain_raw, 0.0, cutoff)

    return (
        data_norm,
        brain_mask,
        data_brain,
        data_brain_raw,
        brain_mask_flat,
        bvals,
        bvecs,
    )


def load_data_from_folder(data_dir: Path, outlier_quantile: float | None):
    data_path = data_dir / "data.nii.gz"
    mask_path = data_dir / "nodif_brain_mask.nii.gz"
    bvals_path = data_dir / "bvals"
    bvecs_path = data_dir / "bvecs"

    data, _ = load_nifti(data_path.as_posix())
    brain_mask, _ = load_nifti(mask_path.as_posix())
    raw_bvals, bvecs = read_bvals_bvecs(bvals_path.as_posix(), bvecs_path.as_posix())
    return load_data_from_arrays(data, brain_mask, raw_bvals, bvecs, outlier_quantile)


def load_data_from_files(
    data_path: Path,
    bvals_path: Path,
    bvecs_path: Path,
    mask_path: Path | None,
    outlier_quantile: float | None,
):
    data, _ = load_nifti(data_path.as_posix())
    brain_mask = None
    if mask_path is not None:
        brain_mask, _ = load_nifti(mask_path.as_posix())
    raw_bvals, bvecs = read_bvals_bvecs(bvals_path.as_posix(), bvecs_path.as_posix())
    return load_data_from_arrays(data, brain_mask, raw_bvals, bvecs, outlier_quantile)


def load_data_from_dipy(dataset: str, outlier_quantile: float | None):
    fimg, fbvals, fbvecs = get_fnames(dataset)
    data, _ = load_nifti(fimg)
    raw_bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    brain_mask = np.ones(data.shape[:-1], dtype=bool)
    return load_data_from_arrays(data, brain_mask, raw_bvals, bvecs, outlier_quantile)


def _clean_samples_jax(
    samples_xyz: jax.Array, eps: float = 1e-12
) -> tuple[jax.Array, jax.Array]:
    finite = jnp.all(jnp.isfinite(samples_xyz), axis=-1)
    samples_clean = jnp.where(finite[:, None], samples_xyz, 0.0)
    norms = jnp.linalg.norm(samples_clean, axis=-1)
    valid = finite & (norms > eps)
    norm_safe = jnp.where(valid, norms, 1.0)
    X = samples_clean / norm_safe[:, None]
    X = jnp.where(valid[:, None], X, 0.0)
    return X, valid


def _kappa_from_sigma_deg_jax(sigma_deg: jax.Array) -> jax.Array:
    sigma_rad = jnp.deg2rad(sigma_deg)
    return 1.0 / (sigma_rad * sigma_rad + 1e-12)


def _choose_sigma_deg_jax(
    samples_xyz: jax.Array,
    sigma_min: float = 3.0,
    sigma_max: float = 18.0,
    antipodal: bool = True,
) -> jax.Array:
    default_sigma = jnp.clip(
        jnp.asarray(12.0, dtype=samples_xyz.dtype), sigma_min, sigma_max
    )
    if samples_xyz.shape[0] < 5:
        return default_sigma

    X, valid = _clean_samples_jax(samples_xyz)
    weights = valid.astype(X.dtype)
    neff = jnp.sum(weights)
    neff_safe = jnp.maximum(neff, 1.0)
    spacing_rad = jnp.sqrt(4.0 * jnp.pi / neff_safe)
    spacing_deg = jnp.rad2deg(spacing_rad)

    T = (X.T * weights) @ X / (neff_safe + 1e-12)
    T = 0.5 * (T + T.T)
    tr = jnp.trace(T)
    T = T / (tr + 1e-12)
    T = 0.5 * (T + T.T) + jnp.eye(3, dtype=X.dtype) * 1e-8
    evals = jnp.linalg.eigvalsh(T)
    evals = jnp.sort(evals)
    l1 = evals[-1]
    l2 = evals[-2]
    c = (l1 - l2) / (l1 + 1e-12)
    c = jnp.clip(c, 0.0, 1.0)

    factor = 1.1 - 0.6 * c
    sigma = factor * spacing_deg
    if not antipodal:
        sigma = sigma * 0.9
    sigma = jnp.clip(sigma, sigma_min, sigma_max)
    return jnp.where(neff >= 5.0, sigma, default_sigma)


def _pmf_kde_from_samples_jax(
    samples_xyz: jax.Array,
    sphere_vertices: jax.Array,
    sigma_deg: jax.Array,
    antipodal: bool = True,
) -> jax.Array:
    X, valid = _clean_samples_jax(samples_xyz)
    weights = valid.astype(X.dtype)
    dots = sphere_vertices @ X.T
    kappa = _kappa_from_sigma_deg_jax(jnp.asarray(sigma_deg, dtype=X.dtype))
    dots = jnp.clip(kappa * dots, -50.0, 50.0)
    if antipodal:
        K = jnp.cosh(dots)
    else:
        K = jnp.exp(dots)
    sf = jnp.sum(K * weights[None, :], axis=1)
    s = jnp.sum(sf)
    pmf = jnp.where(s > 0, sf / s, jnp.zeros_like(sf))
    return jnp.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0)


def _pmf_histogram_from_samples_jax(
    samples_xyz: jax.Array,
    sphere_vertices: jax.Array,
    antipodal: bool = True,
) -> jax.Array:
    X, valid = _clean_samples_jax(samples_xyz)
    weights = valid.astype(X.dtype)
    dots = sphere_vertices @ X.T
    if antipodal:
        dots = jnp.abs(dots)
    idx = jnp.argmax(dots, axis=0).astype(jnp.int32)
    counts = jnp.zeros((sphere_vertices.shape[0],), dtype=weights.dtype)
    counts = counts.at[idx].add(weights)
    s = jnp.sum(counts)
    pmf = jnp.where(s > 0, counts / s, jnp.zeros_like(counts))
    return jnp.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0)


def _pmf_kde_from_samples_np(
    samples_xyz: np.ndarray,
    sphere,
    sigma_deg: float | None,
) -> np.ndarray:
    if sigma_deg is None or sigma_deg <= 0:
        sigma_deg = choose_sigma_deg(samples_xyz, antipodal=True)
    sf = spherical_kde_vmf_from_samples(
        samples_xyz,
        sphere,
        sigma_deg=float(sigma_deg),
        antipodal=True,
        normalize="sum",
    )
    return np.nan_to_num(sf, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_sphere_vertices(sphere) -> jax.Array:
    vertices = jnp.asarray(sphere.vertices, dtype=jnp.float32)
    return vertices / (jnp.linalg.norm(vertices, axis=1, keepdims=True) + 1e-12)


@partial(jax.jit, static_argnames=("map_batch_size",))
def _kde_batch_fixed_sigma(
    samples: jax.Array,
    sphere_vertices: jax.Array,
    sigma_deg: jax.Array,
    map_batch_size: int,
) -> jax.Array:
    def sample_one(sample_xyz):
        return _pmf_kde_from_samples_jax(
            sample_xyz, sphere_vertices, sigma_deg, antipodal=True
        )

    return jax.lax.map(sample_one, samples, batch_size=map_batch_size)


@partial(jax.jit, static_argnames=("map_batch_size",))
def _kde_batch_auto_sigma(
    samples: jax.Array,
    sphere_vertices: jax.Array,
    map_batch_size: int,
) -> jax.Array:
    def sample_one(sample_xyz):
        sigma = _choose_sigma_deg_jax(sample_xyz, antipodal=True)
        return _pmf_kde_from_samples_jax(
            sample_xyz, sphere_vertices, sigma, antipodal=True
        )

    return jax.lax.map(sample_one, samples, batch_size=map_batch_size)


@partial(jax.jit, static_argnames=("map_batch_size",))
def _pmf_batch_histogram(
    samples: jax.Array,
    sphere_vertices: jax.Array,
    map_batch_size: int,
) -> jax.Array:
    def sample_one(sample_xyz):
        return _pmf_histogram_from_samples_jax(
            sample_xyz, sphere_vertices, antipodal=True
        )

    return jax.lax.map(sample_one, samples, batch_size=map_batch_size)


@partial(jax.jit, static_argnames=("num_keys",))
def _make_voxel_keys(
    base_key: jax.Array, voxel_ids: jax.Array, num_keys: int
) -> jax.Array:
    voxel_keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(voxel_ids)
    return jax.vmap(lambda k: jax.random.split(k, num_keys))(voxel_keys)


def _maybe_device_put(data: np.ndarray | jax.Array, name: str):
    if isinstance(data, jax.Array):
        return data
    try:
        return jax.device_put(data)
    except Exception as exc:
        LOGGER.warning("Keeping %s on host (%s).", name, exc)
        return data


def _get_batch_x(
    data_brain: np.ndarray | jax.Array,
    start: int,
    end: int,
    todo_mask: np.ndarray | None,
) -> jax.Array:
    batch_x = data_brain[start:end]
    if todo_mask is not None:
        if isinstance(batch_x, jax.Array):
            batch_x = batch_x[jnp.asarray(todo_mask)]
        else:
            batch_x = batch_x[todo_mask]
    return jnp.asarray(batch_x)


def make_theta_sampler_fixed(model, acq, num_steps: int, t_min: float, model_mask):
    def sample_thetas(keys, x):
        def sample_one(key):
            return model.sample_theta(
                key,
                acq,
                x,
                model_mask=model_mask,
                num_steps=num_steps,
                t_min=t_min,
            )

        return jax.lax.map(sample_one, keys)

    return jax.jit(sample_thetas)


def make_theta_sampler_varying(model, acq, num_steps: int, t_min: float):
    def sample_thetas(keys, x, model_masks):
        def sample_one(key, mask):
            return model.sample_theta(
                key,
                acq,
                x,
                model_mask=mask,
                num_steps=num_steps,
                t_min=t_min,
            )

        return jax.vmap(sample_one)(keys, model_masks)

    return jax.jit(sample_thetas)


def make_mask_sampler(model, acq, mask_prior):
    def sample_masks(keys, x):
        def sample_one(key):
            return model.sample_mask(key, acq=acq, x=x, mask_prior=mask_prior)

        return jax.vmap(sample_one)(keys)

    return jax.jit(sample_masks)


def make_fod_sampler_fixed(sim_type, fod_samples: int, model_mask):
    def sample_one(theta, rng):
        simulator = sim_type.from_theta(theta, model_mask=model_mask)
        return simulator.to_fod().sample(rng, (fod_samples,))

    return jax.jit(jax.vmap(sample_one, in_axes=(0, 0)))


def make_fod_sampler_varying(sim_type, fod_samples: int):
    def sample_one(theta, mask, rng):
        simulator = sim_type.from_theta(theta, model_mask=mask)
        return simulator.to_fod().sample(rng, (fod_samples,))

    return jax.jit(jax.vmap(sample_one, in_axes=(0, 0, 0)))


@partial(jax.jit, static_argnames=("theta_sampler", "theta_samples", "map_batch_size"))
def sample_thetas_fixed_batch(
    theta_sampler,
    keys_theta: jax.Array,
    batch_x: jax.Array,
    theta_samples: int,
    map_batch_size: int,
) -> jax.Array:
    def sample_one(args):
        key, x = args
        keys_theta = jax.random.split(key, theta_samples)
        return theta_sampler(keys_theta, x)

    return jax.lax.map(
        sample_one,
        (keys_theta, batch_x),
        batch_size=map_batch_size,
    )


@partial(
    jax.jit,
    static_argnames=("mask_sampler", "theta_sampler", "theta_samples", "map_batch_size"),
)
def sample_thetas_varying_batch(
    mask_sampler,
    theta_sampler,
    keys_mask: jax.Array,
    keys_theta: jax.Array,
    batch_x: jax.Array,
    theta_samples: int,
    map_batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    def sample_one(args):
        key_mask, key_theta, x = args
        keys_mask = jax.random.split(key_mask, theta_samples)
        model_masks = mask_sampler(keys_mask, x)
        keys_theta = jax.random.split(key_theta, theta_samples)
        thetas = theta_sampler(keys_theta, x, model_masks)
        return model_masks, thetas

    return jax.lax.map(
        sample_one,
        (keys_mask, keys_theta, batch_x),
        batch_size=map_batch_size,
    )


@partial(jax.jit, static_argnames=("fod_sampler", "theta_samples", "map_batch_size"))
def sample_fods_from_thetas_fixed(
    fod_sampler,
    keys_fod: jax.Array,
    thetas: jax.Array,
    theta_samples: int,
    map_batch_size: int,
) -> jax.Array:
    def sample_one(args):
        key, theta = args
        keys_fod = jax.random.split(key, theta_samples)
        samples = fod_sampler(theta, keys_fod)
        return samples.reshape((-1, 3))

    return jax.lax.map(
        sample_one,
        (keys_fod, thetas),
        batch_size=map_batch_size,
    )


@partial(jax.jit, static_argnames=("fod_sampler", "theta_samples", "map_batch_size"))
def sample_fods_from_thetas_varying(
    fod_sampler,
    keys_fod: jax.Array,
    model_masks: jax.Array,
    thetas: jax.Array,
    theta_samples: int,
    map_batch_size: int,
) -> jax.Array:
    def sample_one(args):
        key, mask, theta = args
        keys_fod = jax.random.split(key, theta_samples)
        samples = fod_sampler(theta, mask, keys_fod)
        return samples.reshape((-1, 3))

    return jax.lax.map(
        sample_one,
        (keys_fod, model_masks, thetas),
        batch_size=map_batch_size,
    )


def evaluate_rumba(
    data_norm: np.ndarray,
    data_brain: np.ndarray,
    brain_indices: np.ndarray,
    gtab,
    sphere,
    out_map: np.ndarray,
    out_path: Path,
    write_mode: str,
    batch_size: int,
    auto_response: bool,
    roi_radii: int,
    fa_thr: float,
    resume: bool,
    fsync_every: int,
) -> None:
    if auto_response:
        LOGGER.info("Estimating RUMBA response (roi_radii=%d, fa_thr=%.2f).", roi_radii, fa_thr)
        response, _ = auto_response_ssst(gtab, data_norm, roi_radii=roi_radii, fa_thr=fa_thr)
        rumba = RumbaSDModel(gtab, wm_response=response[0], gm_response=None, sphere=sphere)
    else:
        rumba = RumbaSDModel(gtab, sphere=sphere)

    if write_mode not in ("memmap", "array"):
        raise ValueError(f"Unknown write mode '{write_mode}'.")
    flat_out = out_map.reshape(-1, out_map.shape[-1])
    resume_start = 0
    if resume:
        resume_start = find_resume_start(flat_out, brain_indices, batch_size)
        if resume_start >= data_brain.shape[0]:
            LOGGER.info("RUMBA: all voxels already computed; skipping.")
            return
        if resume_start > 0:
            LOGGER.info("RUMBA resume: starting at index %d.", resume_start)
    wrote_any = False
    batch_count = 0
    for start, end in iter_batches(data_brain.shape[0], batch_size, start_idx=resume_start):
        LOGGER.info("RUMBA batch %d:%d", start, end)
        batch_indices = brain_indices[start:end]
        batch = data_brain[start:end]
        if resume:
            existing = flat_out[batch_indices]
            computed = _computed_voxels(existing)
            if np.all(computed):
                continue
            if np.any(computed):
                batch = batch[~computed]
                batch_indices = batch_indices[~computed]
        rumba_fit = rumba.fit(batch)
        combined = np.asarray(rumba_fit.combined_odf_iso, dtype=np.float32)
        flat_out[batch_indices] = combined
        batch_count += 1
        do_fsync = fsync_every > 0 and (batch_count % fsync_every == 0)
        if write_mode == "memmap":
            flush_and_fsync(out_map, do_fsync)
        elif do_fsync:
            _save_array(out_path, out_map, True)
        wrote_any = True
    if wrote_any and fsync_every > 0:
        if write_mode == "memmap":
            flush_and_fsync(out_map, True)
        else:
            _save_array(out_path, out_map, True)
    elif wrote_any and write_mode == "array":
        _save_array(out_path, out_map, False)


def evaluate_fixed_mask(
    name: str,
    model,
    acq,
    sim_type,
    data_brain: np.ndarray | jax.Array,
    brain_indices: np.ndarray,
    out_map: np.ndarray,
    out_path: Path,
    write_mode: str,
    model_mask: np.ndarray,
    rng: jax.Array,
    theta_samples: int,
    fod_samples: int,
    num_steps: int,
    t_min: float,
    batch_size: int,
    sphere,
    sigma_deg: float | None,
    pmf_mode: str,
    resume: bool,
    fsync_every: int,
) -> jax.Array:
    mask = jnp.asarray(model_mask)
    theta_sampler = make_theta_sampler_fixed(model, acq, num_steps, t_min, mask)
    fod_sampler = make_fod_sampler_fixed(sim_type, fod_samples, mask)
    sphere_vertices = _prepare_sphere_vertices(sphere)
    sigma_fixed = None
    if sigma_deg is not None and sigma_deg > 0:
        sigma_fixed = jnp.asarray(sigma_deg, dtype=jnp.float32)
    if pmf_mode not in ("kde", "hist"):
        raise ValueError(f"Unknown pmf mode '{pmf_mode}'.")

    if write_mode not in ("memmap", "array"):
        raise ValueError(f"Unknown write mode '{write_mode}'.")
    flat_out = out_map.reshape(-1, out_map.shape[-1])
    resume_start = 0
    if resume:
        resume_start = find_resume_start(flat_out, brain_indices, batch_size)
        if resume_start >= data_brain.shape[0]:
            LOGGER.info("%s: all voxels already computed; skipping.", name)
            return rng
        if resume_start > 0:
            LOGGER.info("%s resume: starting at index %d.", name, resume_start)
    wrote_any = False
    batch_count = 0
    for start, end in iter_batches(data_brain.shape[0], batch_size, start_idx=resume_start):
        LOGGER.info("%s batch %d:%d", name, start, end)
        batch_indices = brain_indices[start:end]
        computed = None
        if resume:
            existing = flat_out[batch_indices]
            computed = _computed_voxels(existing)
        todo_mask = None
        if computed is not None:
            todo_mask = ~computed
            if not np.any(todo_mask):
                continue
            batch_indices = batch_indices[todo_mask]
        batch_x_jnp = _get_batch_x(data_brain, start, end, todo_mask)
        map_batch_size = min(batch_size, batch_x_jnp.shape[0])
        voxel_ids = jnp.asarray(batch_indices, dtype=jnp.uint32)
        key_pairs = _make_voxel_keys(rng, voxel_ids, 2)
        keys_theta = key_pairs[:, 0]
        keys_fod = key_pairs[:, 1]
        thetas = sample_thetas_fixed_batch(
            theta_sampler,
            keys_theta,
            batch_x_jnp,
            theta_samples,
            map_batch_size,
        )
        samples = sample_fods_from_thetas_fixed(
            fod_sampler,
            keys_fod,
            thetas,
            theta_samples,
            map_batch_size,
        )
        kde_map_batch = min(4, batch_x_jnp.shape[0])
        if pmf_mode == "hist":
            sf_kde = _pmf_batch_histogram(samples, sphere_vertices, kde_map_batch)
            values = np.asarray(sf_kde, dtype=np.float32)
        else:
            samples_np = np.asarray(samples)
            values = np.zeros((samples_np.shape[0], sphere_vertices.shape[0]), dtype=np.float32)
            for offset, sample_set in enumerate(samples_np):
                values[offset] = _pmf_kde_from_samples_np(sample_set, sphere, sigma_deg)
        flat_out[batch_indices] = values
        batch_count += 1
        do_fsync = fsync_every > 0 and (batch_count % fsync_every == 0)
        if write_mode == "memmap":
            flush_and_fsync(out_map, do_fsync)
        elif do_fsync:
            _save_array(out_path, out_map, True)
        wrote_any = True
    if wrote_any and fsync_every > 0:
        if write_mode == "memmap":
            flush_and_fsync(out_map, True)
        else:
            _save_array(out_path, out_map, True)
    elif wrote_any and write_mode == "array":
        _save_array(out_path, out_map, False)
    return rng


def evaluate_sampled_masks(
    model,
    acq,
    sim_type,
    data_brain: np.ndarray | jax.Array,
    brain_indices: np.ndarray,
    out_map: np.ndarray,
    out_path: Path,
    write_mode: str,
    rng: jax.Array,
    mask_prior: jnp.ndarray,
    theta_samples: int,
    fod_samples: int,
    num_steps: int,
    t_min: float,
    batch_size: int,
    sphere,
    sigma_deg: float | None,
    pmf_mode: str,
    resume: bool,
    fsync_every: int,
) -> jax.Array:
    mask_sampler = make_mask_sampler(model, acq, mask_prior)
    theta_sampler = make_theta_sampler_varying(model, acq, num_steps, t_min)
    fod_sampler = make_fod_sampler_varying(sim_type, fod_samples)
    sphere_vertices = _prepare_sphere_vertices(sphere)
    sigma_fixed = None
    if sigma_deg is not None and sigma_deg > 0:
        sigma_fixed = jnp.asarray(sigma_deg, dtype=jnp.float32)
    if pmf_mode not in ("kde", "hist"):
        raise ValueError(f"Unknown pmf mode '{pmf_mode}'.")

    if write_mode not in ("memmap", "array"):
        raise ValueError(f"Unknown write mode '{write_mode}'.")
    flat_out = out_map.reshape(-1, out_map.shape[-1])
    resume_start = 0
    if resume:
        resume_start = find_resume_start(flat_out, brain_indices, batch_size)
        if resume_start >= data_brain.shape[0]:
            LOGGER.info("all: all voxels already computed; skipping.")
            return rng
        if resume_start > 0:
            LOGGER.info("all resume: starting at index %d.", resume_start)
    wrote_any = False
    batch_count = 0
    for start, end in iter_batches(data_brain.shape[0], batch_size, start_idx=resume_start):
        LOGGER.info("all batch %d:%d", start, end)
        batch_indices = brain_indices[start:end]
        computed = None
        if resume:
            existing = flat_out[batch_indices]
            computed = _computed_voxels(existing)
        todo_mask = None
        if computed is not None:
            todo_mask = ~computed
            if not np.any(todo_mask):
                continue
            batch_indices = batch_indices[todo_mask]
        batch_x_jnp = _get_batch_x(data_brain, start, end, todo_mask)
        map_batch_size = min(batch_size, batch_x_jnp.shape[0])
        voxel_ids = jnp.asarray(batch_indices, dtype=jnp.uint32)
        key_triplets = _make_voxel_keys(rng, voxel_ids, 3)
        keys_mask = key_triplets[:, 0]
        keys_theta = key_triplets[:, 1]
        keys_fod = key_triplets[:, 2]
        model_masks, thetas = sample_thetas_varying_batch(
            mask_sampler,
            theta_sampler,
            keys_mask,
            keys_theta,
            batch_x_jnp,
            theta_samples,
            map_batch_size,
        )
        samples = sample_fods_from_thetas_varying(
            fod_sampler,
            keys_fod,
            model_masks,
            thetas,
            theta_samples,
            map_batch_size,
        )
        kde_map_batch = min(4, batch_x_jnp.shape[0])
        if pmf_mode == "hist":
            sf_kde = _pmf_batch_histogram(samples, sphere_vertices, kde_map_batch)
            values = np.asarray(sf_kde, dtype=np.float32)
        else:
            samples_np = np.asarray(samples)
            values = np.zeros((samples_np.shape[0], sphere_vertices.shape[0]), dtype=np.float32)
            for offset, sample_set in enumerate(samples_np):
                values[offset] = _pmf_kde_from_samples_np(sample_set, sphere, sigma_deg)
        flat_out[batch_indices] = values
        batch_count += 1
        do_fsync = fsync_every > 0 and (batch_count % fsync_every == 0)
        if write_mode == "memmap":
            flush_and_fsync(out_map, do_fsync)
        elif do_fsync:
            _save_array(out_path, out_map, True)
        wrote_any = True
    if wrote_any and fsync_every > 0:
        if write_mode == "memmap":
            flush_and_fsync(out_map, True)
        else:
            _save_array(out_path, out_map, True)
    elif wrote_any and write_mode == "array":
        _save_array(out_path, out_map, False)
    return rng


def save_metadata(
    out_dir: Path,
    brain_mask: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    sphere,
    methods: list[str],
    slice_axis: str | None,
    slice_index: int | None,
    roi: tuple[int, int, int, int, int, int] | None,
) -> None:
    np.savez_compressed(
        out_dir / "fod_metadata.npz",
        brain_mask=brain_mask.astype(bool),
        bvals=bvals.astype(np.float32),
        bvecs=bvecs.astype(np.float32),
        sphere_vertices=np.asarray(sphere.vertices, dtype=np.float32),
        sphere_faces=np.asarray(sphere.faces, dtype=np.int32),
        methods=np.asarray(methods),
        slice_axis=np.asarray(slice_axis) if slice_axis is not None else None,
        slice_index=np.asarray(slice_index) if slice_index is not None else None,
        roi=np.asarray(roi) if roi is not None else None,
    )


def configure_logging() -> None:
    root = logging.getLogger()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    has_stream = False
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            has_stream = True
            break
    if not has_stream:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save full-brain FODs for model-based and baseline methods."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/macke/mgloeckler90/dmri/results/updated_all_3_6_8_128_model_prior"
        ),
        help="Path to the training run directory with checkpoints.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
        help="Root directory containing the data folder.",
    )
    parser.add_argument(
        "--data-folder",
        default="data",
        help="Subfolder under data-root with NIfTI + bvals/bvecs.",
    )
    parser.add_argument(
        "--data-source",
        choices=("folder", "files", "dipy"),
        default="folder",
        help="Load data from a folder, explicit files, or a dipy dataset.",
    )
    parser.add_argument(
        "--data-nifti",
        type=Path,
        default=None,
        help="Path to diffusion NIfTI (used with --data-source=files).",
    )
    parser.add_argument(
        "--data-bvals",
        type=Path,
        default=None,
        help="Path to bvals file (used with --data-source=files).",
    )
    parser.add_argument(
        "--data-bvecs",
        type=Path,
        default=None,
        help="Path to bvecs file (used with --data-source=files).",
    )
    parser.add_argument(
        "--data-mask",
        type=Path,
        default=None,
        help="Optional brain mask NIfTI (used with --data-source=files).",
    )
    parser.add_argument(
        "--dipy-dataset",
        default="small_25",
        help="Dataset name for dipy.data.get_fnames (used with --data-source=dipy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write FOD volumes.",
    )
    parser.add_argument(
        "--write-mode",
        choices=("memmap", "array"),
        default="array",
        help="Write mode: memmap (default) or array.",
    )
    parser.add_argument(
        "--methods",
        default="rumba,b3s,b3t,all",
        help="Comma-separated: rumba,b3s,b3t,all or custom masks.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--theta-samples", type=int, default=500)
    parser.add_argument("--fod-samples", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--t-min", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outlier-quantile", type=float, default=0.999)
    parser.add_argument(
        "--mask-prior",
        type=str,
        default="1.0",
        help="Mask prior hyperparameters (comma-separated).",
    )
    parser.add_argument("--sphere", default="symmetric724")
    parser.add_argument(
        "--sigma-deg",
        type=float,
        default=None,
        help="Fixed bandwidth in degrees for spherical KDE (default: auto).",
    )
    parser.add_argument(
        "--pmf-mode",
        choices=("kde", "hist"),
        default="kde",
        help="PMF estimation: KDE (default) or histogram on sphere.",
    )
    parser.add_argument(
        "--custom-mask",
        action="append",
        default=[],
        help="Add fixed mask as name:bits (comma/space-separated 0/1).",
    )
    parser.add_argument(
        "--slice-axis",
        type=str,
        default=None,
        help="Axis for single-slice evaluation (x, y, or z).",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Index along slice-axis for single-slice evaluation.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Axis-aligned ROI as x0,x1,y0,y1,z0,z1 (end indices exclusive).",
    )
    parser.add_argument(
        "--rumba-auto-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate response function for RUMBA (default: true).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing outputs and skip computed voxels.",
    )
    parser.add_argument("--rumba-roi-radii", type=int, default=10)
    parser.add_argument("--rumba-fa-thr", type=float, default=0.7)
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=5,
        help="Force fsync every N batches (0 disables; array mode saves full file).",
    )
    args = parser.parse_args()

    configure_logging()

    methods = parse_methods(args.methods)
    if not methods:
        raise ValueError("No methods selected.")

    if args.data_source == "folder":
        data_dir = args.data_root / args.data_folder
        (
            data_norm,
            brain_mask,
            data_brain,
            data_brain_raw,
            brain_mask_flat,
            bvals,
            bvecs,
        ) = load_data_from_folder(data_dir, args.outlier_quantile)
    elif args.data_source == "files":
        if args.data_nifti is None or args.data_bvals is None or args.data_bvecs is None:
            raise ValueError("--data-nifti, --data-bvals, and --data-bvecs are required.")
        (
            data_norm,
            brain_mask,
            data_brain,
            data_brain_raw,
            brain_mask_flat,
            bvals,
            bvecs,
        ) = load_data_from_files(
            args.data_nifti,
            args.data_bvals,
            args.data_bvecs,
            args.data_mask,
            args.outlier_quantile,
        )
    else:
        (
            data_norm,
            brain_mask,
            data_brain,
            data_brain_raw,
            brain_mask_flat,
            bvals,
            bvecs,
        ) = load_data_from_dipy(args.dipy_dataset, args.outlier_quantile)

    slice_axis = args.slice_axis.lower() if args.slice_axis is not None else None
    slice_index = args.slice_index
    if (slice_axis is None) != (slice_index is None):
        raise ValueError("Both --slice-axis and --slice-index must be provided together.")
    roi = None
    if args.roi is not None:
        parts = [p.strip() for p in args.roi.split(",") if p.strip()]
        if len(parts) != 6:
            raise ValueError("ROI must be six comma-separated integers.")
        roi_vals = tuple(int(p) for p in parts)
        roi = roi_vals

    if slice_axis is not None and roi is not None:
        raise ValueError("Use either --slice-axis or --roi, not both.")

    if slice_axis is not None:
        axis_map = {"x": 0, "y": 1, "z": 2}
        if slice_axis not in axis_map:
            raise ValueError("Slice axis must be one of: x, y, z.")
        axis_dim = data_norm.shape[axis_map[slice_axis]]
        if slice_index < 0 or slice_index >= axis_dim:
            raise ValueError("Slice index out of range for the selected axis.")
        brain_mask = apply_slice_mask(brain_mask, slice_axis, slice_index)
        brain_mask_flat = brain_mask.reshape(-1).astype(bool)
        full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
        data_brain_raw = np.nan_to_num(
            full_data_flat[brain_mask_flat], nan=0.0, posinf=0.0, neginf=0.0
        )
        data_brain = data_brain_raw
        if args.outlier_quantile is not None:
            cutoff = np.quantile(data_brain_raw, args.outlier_quantile)
            data_brain = np.clip(data_brain_raw, 0.0, cutoff)
    elif roi is not None:
        brain_mask = apply_roi_mask(brain_mask, roi)
        brain_mask_flat = brain_mask.reshape(-1).astype(bool)
        full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
        data_brain_raw = np.nan_to_num(
            full_data_flat[brain_mask_flat], nan=0.0, posinf=0.0, neginf=0.0
        )
        data_brain = data_brain_raw
        if args.outlier_quantile is not None:
            cutoff = np.quantile(data_brain_raw, args.outlier_quantile)
            data_brain = np.clip(data_brain_raw, 0.0, cutoff)

    acq = acquisition_scheme(np.asarray(bvals), np.asarray(bvecs))
    gtab = gradient_table(bvals, bvecs)
    sphere = get_sphere(name=args.sphere)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    volume_shape = data_norm.shape[:3]
    num_dirs = len(sphere.vertices)

    save_metadata(
        out_dir,
        brain_mask,
        bvals,
        bvecs,
        sphere,
        methods,
        slice_axis,
        slice_index,
        roi,
    )

    brain_indices = np.flatnonzero(brain_mask_flat)

    fixed_masks = {}
    model = None
    sim_type = None
    mask_prior = None

    needs_model = any(m in ("b3s", "b3t", "all") for m in methods) or bool(args.custom_mask)
    if needs_model:
        model, _ = load_model(args.checkpoint)
        sim_type = model.tokenizer.simulator
        fixed_masks.update(build_default_masks(model))
        if args.custom_mask:
            mask_len = model.tokenizer.num_models + model.tokenizer.num_noises
            fixed_masks.update(parse_custom_masks(args.custom_mask, mask_len))
        params_per_comp = [m.theta_dim for m in sim_type.model_types]
        mask_prior = sim_type.create_mask_prior(num_model_parameters=params_per_comp)
        mask_prior = parse_mask_prior(args.mask_prior,mask_prior.mask_prior_dim)
    data_brain_device = data_brain
    if needs_model:
        data_brain_device = _maybe_device_put(data_brain, "data_brain")

    outputs: dict[str, Path] = {}
    rng = jax.random.PRNGKey(args.seed)

    for method in methods:
        if method == "rumba":
            out_path = out_dir / "fod_rumba.npy"
            if args.write_mode == "memmap":
                out_map, resumed = open_output_map(
                    out_path,
                    volume_shape + (num_dirs,),
                    np.float32,
                    args.resume,
                )
            else:
                out_map, resumed = open_output_array(
                    out_path,
                    volume_shape + (num_dirs,),
                    np.float32,
                    args.resume,
                )
            if resumed:
                flat_out = out_map.reshape(-1, out_map.shape[-1])
                existing = _computed_voxels(flat_out[brain_indices])
                LOGGER.info(
                    "Resuming rumba: %d/%d voxels already computed.",
                    int(existing.sum()),
                    int(existing.size),
                )
            evaluate_rumba(
                data_norm,
                data_brain_raw,
                brain_indices,
                gtab,
                sphere,
                out_map,
                out_path,
                args.write_mode,
                args.batch_size,
                args.rumba_auto_response,
                args.rumba_roi_radii,
                args.rumba_fa_thr,
                args.resume,
                args.fsync_every,
            )
            if args.write_mode == "memmap":
                out_map.flush()
            outputs[method] = out_path
            continue

        if model is None or sim_type is None:
            raise ValueError(f"Method '{method}' requires a model checkpoint.")

        out_path = out_dir / f"fod_{method}.npy"
        if args.write_mode == "memmap":
            out_map, resumed = open_output_map(
                out_path,
                volume_shape + (num_dirs,),
                np.float32,
                args.resume,
            )
        else:
            out_map, resumed = open_output_array(
                out_path,
                volume_shape + (num_dirs,),
                np.float32,
                args.resume,
            )
        if resumed:
            flat_out = out_map.reshape(-1, out_map.shape[-1])
            existing = _computed_voxels(flat_out[brain_indices])
            LOGGER.info(
                "Resuming %s: %d/%d voxels already computed.",
                method,
                int(existing.sum()),
                int(existing.size),
            )

        rng, method_key = jax.random.split(rng)
        if method == "all":
            if mask_prior is None:
                raise ValueError("Mask prior must be set for sampled masks.")
            method_key = evaluate_sampled_masks(
                model,
                acq,
                sim_type,
                data_brain_device,
                brain_indices,
                out_map,
                out_path,
                args.write_mode,
                method_key,
                mask_prior,
                args.theta_samples,
                args.fod_samples,
                args.num_steps,
                args.t_min,
                args.batch_size,
                sphere,
                args.sigma_deg,
                args.pmf_mode,
                args.resume,
                args.fsync_every,
            )
        else:
            if method not in fixed_masks:
                raise ValueError(f"Unknown method '{method}'.")
            method_key = evaluate_fixed_mask(
                method,
                model,
                acq,
                sim_type,
                data_brain_device,
                brain_indices,
                out_map,
                out_path,
                args.write_mode,
                fixed_masks[method],
                method_key,
                args.theta_samples,
                args.fod_samples,
                args.num_steps,
                args.t_min,
                args.batch_size,
                sphere,
                args.sigma_deg,
                args.pmf_mode,
                args.resume,
                args.fsync_every,
            )
        if args.write_mode == "memmap":
            out_map.flush()
        outputs[method] = out_path

    LOGGER.info("Finished. Outputs: %s", outputs)


if __name__ == "__main__":
    main()
