#!/usr/bin/env python3
"""Compute training RMSE and leave-one-out generalization error."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dmri.eval.load_data import load_and_process_data
from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.train.utils import load_checkpoint

B3S_MASK = jnp.array([True, True, True, True] + [False] * 6 + [True, False])
B3T_MASK = jnp.array([True] + [False] * 6 + [True] * 3 + [True, False])

MASK_LOOKUP = {
    "b3s": B3S_MASK,
    "b3t": B3T_MASK,
    "all": None,
    "auto": None,
}

LOGGER = logging.getLogger(__name__)


def _parse_lam_suffix(raw: str) -> float:
    if not raw:
        raise ValueError("Empty allXX suffix.")
    raw = raw.strip().lower()
    if raw.isdigit():
        scale = 10 ** max(len(raw) - 1, 0)
        lam = int(raw) / scale
        if not np.isfinite(lam):
            raise ValueError(f"Invalid allXX suffix: {raw!r}")
        return lam
    raw = raw.replace("p", ".", 1)
    try:
        lam = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid allXX suffix: {raw!r}") from exc
    if not np.isfinite(lam):
        raise ValueError(f"Invalid allXX suffix: {raw!r}")
    return lam


def _resolve_mask_choice(
    mask_name: str, default_lam: float, conv_checkpoint: Path
) -> tuple[np.ndarray | None, float, Path | None]:
    if mask_name in MASK_LOOKUP:
        return MASK_LOOKUP[mask_name], default_lam, None
    if mask_name.startswith("all_conv"):
        suffix = mask_name[len("all_conv") :]
        if not suffix:
            return None, default_lam, conv_checkpoint
        return None, _parse_lam_suffix(suffix), conv_checkpoint
    if mask_name.startswith("all"):
        suffix = mask_name[3:]
        if not suffix:
            return None, default_lam, None
        return None, _parse_lam_suffix(suffix), None
    raise ValueError(f"Unknown mask choice: {mask_name}")


def parse_args(repo_root: Path) -> argparse.Namespace:
    default_checkpoint = repo_root / "results/updated_all_3_6_8_128_model_prior"
    default_conv_checkpoint = (
        repo_root / "results/updated_all_3_6_8_128_model_prior_with_convs_2gpus"
    )
    default_data_path = repo_root / "data/data"

    parser = argparse.ArgumentParser(
        description=(
            "Compute training RMSE and leave-one-out generalization error for a "
            "Simformer checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Path to the training run directory to load a checkpoint from.",
    )
    parser.add_argument(
        "--conv-checkpoint",
        type=Path,
        default=default_conv_checkpoint,
        help="Path to the convolutional training run directory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path,
        help="Path to the diffusion dataset directory.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=-1,
        help="Slice index to evaluate; set to -1 to use the full volume.",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.999,
        help="Upper quantile used to clip data outliers.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=0,
        help="Maximum number of voxels to evaluate (0 means all).",
    )
    parser.add_argument(
        "--subsample-voxels",
        type=int,
        default=0,
        help=(
            "Randomly subsample this many voxels after masking "
            "(applied after --max-voxels; 0 means all)."
        ),
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=0,
        help="Random seed used for voxel subsampling.",
    )
    parser.add_argument(
        "--mask",
        action="append",
        help=(
            "Mask choice to evaluate; repeat for multiple. "
            "Use b3s/b3t/all/auto, all_conv, or allXX (e.g., all0.5, all0p5, all05)."
        ),
    )
    parser.add_argument(
        "--eval-simformer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate Simformer mask-based errors.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of RNG seeds to average over.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of MC samples per seed when estimating MSE.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of voxels per evaluation batch.",
    )
    parser.add_argument(
        "--loo-samples",
        type=int,
        default=0,
        help="Number of gradient directions to evaluate for LOO (0 means all).",
    )
    parser.add_argument(
        "--loo-offset",
        type=int,
        default=0,
        help="Offset into gradient directions for LOO evaluation.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Mask prior hyperparameter value.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for mask/theta.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional path to write per-voxel MSE arrays (npz).",
    )
    parser.add_argument(
        "--mask-erosion-iters",
        type=int,
        default=2,
        help="Binary erosion iterations applied to brain mask (0 to disable).",
    )
    parser.add_argument(
        "--eval-rumba",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate RUMBA training and LOO errors.",
    )
    parser.add_argument(
        "--eval-csd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate CSD training and LOO errors.",
    )
    parser.add_argument(
        "--eval-dti",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate DTI training and LOO errors.",
    )
    parser.add_argument(
        "--rumba-roi-radii",
        type=int,
        default=10,
        help="ROI radius for auto_response_ssst when computing RUMBA/CSD response.",
    )
    parser.add_argument(
        "--rumba-fa-thr",
        type=float,
        default=0.7,
        help="FA threshold for auto_response_ssst when computing RUMBA/CSD response.",
    )
    parser.add_argument(
        "--rumba-loo-samples",
        type=int,
        default=0,
        help="Number of gradient directions to evaluate for RUMBA LOO (0 means all).",
    )
    parser.add_argument(
        "--rumba-loo-offset",
        type=int,
        default=0,
        help="Offset into gradient directions for RUMBA LOO evaluation.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path):
    checkpoint, model, simulator = load_checkpoint(str(checkpoint_path))
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema", checkpoint.get("params", params))
    model = nnx.merge(graphdef, params, state)
    model.eval()
    return model, simulator


def prepare_brain_data(
    data_norm,
    brain_mask,
    outlier_quantile,
    slice_index,
    max_voxels,
    subsample_voxels,
    subsample_seed,
    mask_erosion_iters: int,
):
    if mask_erosion_iters is not None and mask_erosion_iters > 0:
        from scipy.ndimage import binary_erosion

        brain_mask = binary_erosion(
            brain_mask.astype(bool), iterations=mask_erosion_iters
        )

    if slice_index is not None and slice_index >= 0:
        data_norm_slice = data_norm[:, :, slice_index : slice_index + 1, :]
        brain_mask_slice = brain_mask[:, :, slice_index : slice_index + 1]
    else:
        data_norm_slice = data_norm
        brain_mask_slice = brain_mask

    full_data_flat = data_norm_slice.reshape(-1, data_norm_slice.shape[-1])
    brain_mask_flat = brain_mask_slice.reshape(-1).astype(bool)
    full_data_flat_in_brain = full_data_flat[brain_mask_flat, :]
    full_data_flat_in_brain = np.nan_to_num(
        full_data_flat_in_brain, nan=0.0, posinf=0.0, neginf=0.0
    )
    if outlier_quantile is not None:
        exclude_outliers = np.quantile(full_data_flat_in_brain, outlier_quantile)
        full_data_flat_in_brain = np.clip(full_data_flat_in_brain, 0.0, exclude_outliers)

    if max_voxels and max_voxels > 0:
        full_data_flat_in_brain = full_data_flat_in_brain[:max_voxels]
    if subsample_voxels and subsample_voxels > 0:
        rng = np.random.default_rng(subsample_seed)
        n_voxels = full_data_flat_in_brain.shape[0]
        if subsample_voxels < n_voxels:
            indices = rng.choice(n_voxels, size=subsample_voxels, replace=False)
            full_data_flat_in_brain = full_data_flat_in_brain[indices]
            LOGGER.info(
                "Subsampled voxels: keeping %d of %d (seed=%d).",
                subsample_voxels,
                n_voxels,
                subsample_seed,
            )
        else:
            LOGGER.info(
                "Subsample voxels (%d) >= available voxels (%d); using all voxels.",
                subsample_voxels,
                n_voxels,
            )
    LOGGER.info(
        "Prepared data with %d voxels and %d directions.",
        full_data_flat_in_brain.shape[0],
        full_data_flat_in_brain.shape[1],
    )
    return data_norm_slice, brain_mask_slice, full_data_flat_in_brain


def eval_rumba_errors(
    data_norm_slice: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    full_data_flat_in_brain: np.ndarray,
    eval_indices: np.ndarray,
    roi_radii: int,
    fa_thr: float,
):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import auto_response_ssst
    from dipy.reconst.rumba import RumbaSDModel

    gtab_full = gradient_table(bvals, bvecs)
    response, _ = auto_response_ssst(
        gtab_full, data_norm_slice, roi_radii=roi_radii, fa_thr=fa_thr
    )
    rumba = RumbaSDModel(gtab_full, voxelwise=True, wm_response=response[0])
    fit = rumba.fit(full_data_flat_in_brain)
    pred = fit.predict(gtab=gtab_full)
    resid = full_data_flat_in_brain - pred
    train_mse = np.mean(resid**2, axis=-1)

    eval_indices = np.asarray(eval_indices, dtype=int)
    if eval_indices.size == 0:
        raise ValueError("No evaluation indices provided for RUMBA LOO.")

    loo_mse_sum = np.zeros(full_data_flat_in_brain.shape[0], dtype=np.float64)
    n_dirs = full_data_flat_in_brain.shape[-1]
    all_indices = np.arange(n_dirs)
    for eval_index in eval_indices:
        idx_train = np.delete(all_indices, eval_index)
        gtab_train = gradient_table(bvals[idx_train], bvecs[idx_train])
        gtab_eval = gradient_table(bvals[[eval_index]], bvecs[[eval_index]])
        rumba = RumbaSDModel(gtab_train, voxelwise=True, wm_response=response[0])
        fit = rumba.fit(full_data_flat_in_brain[:, idx_train])
        pred = fit.predict(gtab=gtab_eval)
        resid = full_data_flat_in_brain[:, [eval_index]] - pred
        loo_mse_sum += np.mean(resid**2, axis=-1)

    loo_mse = loo_mse_sum / float(eval_indices.size)
    return train_mse, loo_mse


def eval_csd_errors(
    data_norm_slice: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    full_data_flat_in_brain: np.ndarray,
    eval_indices: np.ndarray,
    roi_radii: int,
    fa_thr: float,
):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import (
        ConstrainedSphericalDeconvModel,
        mask_for_response_ssst,
        response_from_mask_ssst,
    )

    gtab_full = gradient_table(bvals, bvecs=bvecs)
    mask = mask_for_response_ssst(
        gtab_full, data_norm_slice, roi_radii=roi_radii, fa_thr=fa_thr
    )
    response, _ = response_from_mask_ssst(gtab_full, data_norm_slice, mask)

    csd = ConstrainedSphericalDeconvModel(gtab_full, response)
    fit = csd.fit(full_data_flat_in_brain)
    pred = fit.predict(gtab=gtab_full)
    resid = full_data_flat_in_brain - pred
    train_mse = np.mean(resid**2, axis=-1)

    eval_indices = np.asarray(eval_indices, dtype=int)
    if eval_indices.size == 0:
        raise ValueError("No evaluation indices provided for CSD LOO.")

    loo_mse_sum = np.zeros(full_data_flat_in_brain.shape[0], dtype=np.float64)
    n_dirs = full_data_flat_in_brain.shape[-1]
    all_indices = np.arange(n_dirs)
    for eval_index in eval_indices:
        idx_train = np.delete(all_indices, eval_index)
        gtab_train = gradient_table(bvals[idx_train], bvecs=bvecs[idx_train])
        gtab_eval = gradient_table(bvals[[eval_index]], bvecs=bvecs[[eval_index]])
        csd = ConstrainedSphericalDeconvModel(gtab_train, response)
        fit = csd.fit(full_data_flat_in_brain[:, idx_train])
        pred = fit.predict(gtab=gtab_eval)
        resid = full_data_flat_in_brain[:, [eval_index]] - pred
        loo_mse_sum += np.mean(resid**2, axis=-1)

    loo_mse = loo_mse_sum / float(eval_indices.size)
    return train_mse, loo_mse


def eval_dti_errors(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    full_data_flat_in_brain: np.ndarray,
    eval_indices: np.ndarray,
):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    gtab_full = gradient_table(bvals, bvecs=bvecs)
    fit = TensorModel(gtab_full).fit(full_data_flat_in_brain)
    pred = fit.predict(gtab=gtab_full)
    resid = full_data_flat_in_brain - pred
    train_mse = np.mean(resid**2, axis=-1)

    eval_indices = np.asarray(eval_indices, dtype=int)
    if eval_indices.size == 0:
        raise ValueError("No evaluation indices provided for DTI LOO.")

    loo_mse_sum = np.zeros(full_data_flat_in_brain.shape[0], dtype=np.float64)
    n_dirs = full_data_flat_in_brain.shape[-1]
    all_indices = np.arange(n_dirs)
    for eval_index in eval_indices:
        idx_train = np.delete(all_indices, eval_index)
        gtab_train = gradient_table(bvals[idx_train], bvecs=bvecs[idx_train])
        gtab_eval = gradient_table(bvals[[eval_index]], bvecs=bvecs[[eval_index]])
        fit = TensorModel(gtab_train).fit(full_data_flat_in_brain[:, idx_train])
        pred = fit.predict(gtab=gtab_eval)
        resid = full_data_flat_in_brain[:, [eval_index]] - pred
        loo_mse_sum += np.mean(resid**2, axis=-1)

    loo_mse = loo_mse_sum / float(eval_indices.size)
    return train_mse, loo_mse


def _loo_indices(n_total: int, eval_index: jnp.ndarray) -> jnp.ndarray:
    base = jnp.arange(n_total - 1)
    return base + (base >= eval_index).astype(base.dtype)


def make_eval_indices(n_total: int, offset: int, n_samples: int) -> jnp.ndarray:
    if n_samples is None or n_samples <= 0 or n_samples >= n_total:
        indices = np.arange(n_total, dtype=np.int32)
    else:
        indices = np.arange(offset, offset + n_samples, dtype=np.int32)
        indices = np.clip(indices, 0, n_total - 1)
    return jnp.asarray(indices, dtype=jnp.int32)


def make_metrics_fn(
    model,
    sim_type,
    acq,
    eval_indices,
    model_mask,
    lam: float,
    temperature: float,
    num_samples: int,
):
    acq_jnp = jax.tree_util.tree_map(jnp.asarray, acq)
    lam_arr = jnp.array([lam])
    sample_indices = jnp.arange(max(num_samples, 1))

    def train_mse_samples(rng, x):
        x = jnp.asarray(x)

        def body(sample_idx):
            rng_i = jax.random.fold_in(rng, sample_idx)
            if model_mask is None:
                rng_mask, rng_theta = jax.random.split(rng_i)
                mask = model.sample_mask(
                    rng_mask, acq_jnp, x, lam_arr, temperature=temperature
                )
            else:
                mask = model_mask
                rng_theta = rng_i
            theta = model.sample_theta(
                rng_theta, acq_jnp, x, mask, temperature=temperature
            )
            sim = sim_type.from_theta(theta, model_mask=mask)
            pred = sim.signal(acq_jnp)  # deterministic simulator
            return jnp.nanmean((pred - x) ** 2)

        return jax.lax.map(body, sample_indices)

    def pred_error_loo(rng, x, eval_index):
        x = jnp.asarray(x)
        n_total = x.shape[0]
        idx_train = _loo_indices(n_total, eval_index)

        x_train = jnp.take(x, idx_train, axis=0)
        acq_train = jax.tree_util.tree_map(
            lambda a: jnp.take(a, idx_train, axis=0), acq_jnp
        )
        x_eval = jax.lax.dynamic_index_in_dim(x, eval_index, axis=0, keepdims=True)
        acq_eval = jax.tree_util.tree_map(
            lambda a: jax.lax.dynamic_index_in_dim(
                a, eval_index, axis=0, keepdims=True
            ),
            acq_jnp,
        )

        if model_mask is None:
            rng_mask, rng_theta = jax.random.split(rng)
            mask = model.sample_mask(
                rng_mask, acq_train, x_train, lam_arr, temperature=temperature
            )
        else:
            mask = model_mask
            rng_theta = rng
        theta = model.sample_theta(
            rng_theta, acq_train, x_train, mask, temperature=temperature
        )
        sim = sim_type.from_theta(theta, model_mask=mask)
        pred = sim.signal(acq_eval)  # deterministic simulator
        mse = jnp.nanmean((pred - x_eval) ** 2)
        return mse

    def loo_mse_samples(rng, x):
        def sample_body(sample_idx):
            rng_sample = jax.random.fold_in(rng, sample_idx)

            def eval_body(eval_index):
                rng_i = jax.random.fold_in(rng_sample, eval_index)
                return pred_error_loo(rng_i, x, eval_index)

            errors = jax.lax.map(eval_body, eval_indices)
            return jnp.nanmean(errors)

        return jax.lax.map(sample_body, sample_indices)

    def metrics(rng, x):
        rng_train, rng_loo = jax.random.split(rng)
        return train_mse_samples(rng_train, x), loo_mse_samples(rng_loo, x)

    return jax.jit(jax.vmap(metrics, in_axes=(0, 0)))


def eval_in_batches(metrics_fn, data, batch_size: int, seed: int, label: str):
    train_out = []
    loo_out = []
    key = jax.random.PRNGKey(seed)
    for batch_start in range(0, data.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, data.shape[0])
        batch_data = data[batch_start:batch_end]
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_data.shape[0])
        train_batch, loo_batch = metrics_fn(keys, batch_data)
        train_out.append(np.asarray(train_batch))
        loo_out.append(np.asarray(loo_batch))
        LOGGER.info("%s seed=%d batch %d:%d", label, seed, batch_start, batch_end)
    return np.concatenate(train_out, axis=0), np.concatenate(loo_out, axis=0)


def summarize_errors(errors: np.ndarray) -> tuple[float, float]:
    errors = np.asarray(errors)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(errors)), float(np.std(errors))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    repo_root = Path(__file__).resolve().parents[2]
    args = parse_args(repo_root)

    model_cache: dict[str, tuple[object, object]] = {}
    if args.eval_simformer:
        _default_model, _simulator = load_model(args.checkpoint)
        model_cache[str(args.checkpoint)] = (
            _default_model,
            _default_model.tokenizer.simulator,
        )

    def _get_model(checkpoint_path: Path):
        key = str(checkpoint_path)
        if key not in model_cache:
            model, _simulator = load_model(checkpoint_path)
            model_cache[key] = (model, model.tokenizer.simulator)
        return model_cache[key]
    if args.eval_simformer:
        model, _simulator = load_model(args.checkpoint)
        sim_type = model.tokenizer.simulator

    _data_img, data_norm, brain_mask, bvals, bvecs = load_and_process_data(
        str(args.data_path),
        "nodif_brain_mask.nii.gz",
        "data.nii.gz",
        "bvals",
        "bvecs",
    )
    acq = acquisition_scheme(np.array(bvals), np.array(bvecs))

    slice_index = None if args.slice_index is None or args.slice_index < 0 else args.slice_index
    data_norm_slice, _brain_mask_slice, full_data_flat_in_brain = prepare_brain_data(
        data_norm,
        brain_mask,
        args.outlier_quantile,
        slice_index,
        args.max_voxels,
        args.subsample_voxels,
        args.subsample_seed,
        args.mask_erosion_iters,
    )
    if full_data_flat_in_brain.size == 0:
        raise ValueError("No voxels available after masking.")

    n_dirs = full_data_flat_in_brain.shape[-1]
    eval_indices = make_eval_indices(n_dirs, args.loo_offset, args.loo_samples)
    rumba_eval_indices = make_eval_indices(
        n_dirs, args.rumba_loo_offset, args.rumba_loo_samples
    )

    mask_names = args.mask or ["all"]
    results = {
        "checkpoint": str(args.checkpoint),
        "conv_checkpoint": str(args.conv_checkpoint),
        "data_path": str(args.data_path),
        "num_voxels": int(full_data_flat_in_brain.shape[0]),
        "num_directions": int(n_dirs),
        "loo_indices": [int(i) for i in np.asarray(eval_indices)],
        "rumba_loo_indices": [int(i) for i in np.asarray(rumba_eval_indices)],
        "lam": float(args.lam),
        "temperature": float(args.temperature),
        "num_seeds": int(args.num_seeds),
        "num_samples": int(args.num_samples),
        "mask_erosion_iters": int(args.mask_erosion_iters),
        "subsample_voxels": int(args.subsample_voxels),
        "subsample_seed": int(args.subsample_seed),
        "masks": {},
    }

    mse_outputs = {}
    eval_indices_np = np.asarray(eval_indices)
    rumba_eval_indices_np = np.asarray(rumba_eval_indices)

    if args.eval_simformer:
        for mask_name in mask_names:
            model_mask, mask_lam, mask_checkpoint = _resolve_mask_choice(
                mask_name, args.lam, args.conv_checkpoint
            )
            checkpoint_path = mask_checkpoint or args.checkpoint
            model, sim_type = _get_model(checkpoint_path)
            metrics_fn = make_metrics_fn(
                model,
                sim_type,
                acq,
                eval_indices,
                model_mask,
                mask_lam,
                args.temperature,
                args.num_samples,
            )

            train_samples = []
            loo_samples = []
            for seed in range(args.num_seeds):
                train_seed, loo_seed = eval_in_batches(
                    metrics_fn,
                    full_data_flat_in_brain,
                    args.batch_size,
                    seed,
                    mask_name,
                )
                train_samples.append(train_seed)
                loo_samples.append(loo_seed)

            train_samples = np.stack(train_samples, axis=0)
            loo_samples = np.stack(loo_samples, axis=0)
            train_mse = np.mean(train_samples, axis=(0, 2))
            loo_mse = np.mean(loo_samples, axis=(0, 2))
            train_rmse = np.sqrt(train_mse)
            loo_rmse = np.sqrt(loo_mse)

            train_mean, train_std = summarize_errors(train_rmse)
            loo_mean, loo_std = summarize_errors(loo_rmse)

            results["masks"][mask_name] = {
                "checkpoint": str(checkpoint_path),
                "lam": float(mask_lam),
                "train_rmse_mean": train_mean,
                "train_rmse_std": train_std,
                "loo_rmse_mean": loo_mean,
                "loo_rmse_std": loo_std,
            }
            if args.output_npz is not None:
                mse_outputs[f"{mask_name}_train_mse"] = train_mse
                mse_outputs[f"{mask_name}_loo_mse"] = loo_mse
                train_mse_samples = np.transpose(train_samples, (0, 2, 1)).reshape(
                    -1, train_samples.shape[1]
                )
                loo_mse_samples = np.transpose(loo_samples, (0, 2, 1)).reshape(
                    -1, loo_samples.shape[1]
                )
                mse_outputs[f"{mask_name}_train_mse_samples"] = train_mse_samples
                mse_outputs[f"{mask_name}_loo_mse_samples"] = loo_mse_samples

            LOGGER.info(
                "%s: train_rmse mean=%.6f std=%.6f | loo_rmse mean=%.6f std=%.6f",
                mask_name,
                train_mean,
                train_std,
                loo_mean,
                loo_std,
            )

    if args.eval_rumba:
        rumba_train_mse, rumba_loo_mse = eval_rumba_errors(
            data_norm_slice,
            np.asarray(bvals),
            np.asarray(bvecs),
            full_data_flat_in_brain,
            rumba_eval_indices_np,
            args.rumba_roi_radii,
            args.rumba_fa_thr,
        )
        rumba_train_rmse = np.sqrt(rumba_train_mse)
        rumba_loo_rmse = np.sqrt(rumba_loo_mse)
        train_mean, train_std = summarize_errors(rumba_train_rmse)
        loo_mean, loo_std = summarize_errors(rumba_loo_rmse)
        results["rumba"] = {
            "train_rmse_mean": train_mean,
            "train_rmse_std": train_std,
            "loo_rmse_mean": loo_mean,
            "loo_rmse_std": loo_std,
        }
        if args.output_npz is not None:
            mse_outputs["rumba_train_mse"] = rumba_train_mse
            mse_outputs["rumba_loo_mse"] = rumba_loo_mse
        LOGGER.info(
            "rumba: train_rmse mean=%.6f std=%.6f | loo_rmse mean=%.6f std=%.6f",
            train_mean,
            train_std,
            loo_mean,
            loo_std,
        )

    if args.eval_csd:
        csd_train_mse, csd_loo_mse = eval_csd_errors(
            data_norm_slice,
            np.asarray(bvals),
            np.asarray(bvecs),
            full_data_flat_in_brain,
            eval_indices_np,
            args.rumba_roi_radii,
            args.rumba_fa_thr,
        )
        csd_train_rmse = np.sqrt(csd_train_mse)
        csd_loo_rmse = np.sqrt(csd_loo_mse)
        train_mean, train_std = summarize_errors(csd_train_rmse)
        loo_mean, loo_std = summarize_errors(csd_loo_rmse)
        results["csd"] = {
            "train_rmse_mean": train_mean,
            "train_rmse_std": train_std,
            "loo_rmse_mean": loo_mean,
            "loo_rmse_std": loo_std,
        }
        if args.output_npz is not None:
            mse_outputs["csd_train_mse"] = csd_train_mse
            mse_outputs["csd_loo_mse"] = csd_loo_mse
        LOGGER.info(
            "csd: train_rmse mean=%.6f std=%.6f | loo_rmse mean=%.6f std=%.6f",
            train_mean,
            train_std,
            loo_mean,
            loo_std,
        )

    if args.eval_dti:
        dti_train_mse, dti_loo_mse = eval_dti_errors(
            np.asarray(bvals),
            np.asarray(bvecs),
            full_data_flat_in_brain,
            eval_indices_np,
        )
        dti_train_rmse = np.sqrt(dti_train_mse)
        dti_loo_rmse = np.sqrt(dti_loo_mse)
        train_mean, train_std = summarize_errors(dti_train_rmse)
        loo_mean, loo_std = summarize_errors(dti_loo_rmse)
        results["dti"] = {
            "train_rmse_mean": train_mean,
            "train_rmse_std": train_std,
            "loo_rmse_mean": loo_mean,
            "loo_rmse_std": loo_std,
        }
        if args.output_npz is not None:
            mse_outputs["dti_train_mse"] = dti_train_mse
            mse_outputs["dti_loo_mse"] = dti_loo_mse
        LOGGER.info(
            "dti: train_rmse mean=%.6f std=%.6f | loo_rmse mean=%.6f std=%.6f",
            train_mean,
            train_std,
            loo_mean,
            loo_std,
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_npz, **mse_outputs)


if __name__ == "__main__":
    main()
