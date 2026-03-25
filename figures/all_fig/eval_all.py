#!/usr/bin/env python3
"""Scripted version of eval_all.ipynb."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

from dmri.eval.export_theta import embed_in_full_brain_array
from dmri.eval.load_data import load_and_process_data
from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.train.utils import load_checkpoint
from dmri.utils.dmriutils import export_nifti
from dmri.utils.viz import orthoview_ultracompact, save_orthoview_html


def parse_args(repo_root: Path, default_output_dir: Path) -> argparse.Namespace:
    default_checkpoint = repo_root / "results/updated_all_3_6_8_128_model_prior"
    default_data_path = repo_root / "data/data"

    parser = argparse.ArgumentParser(
        description="Run eval_all pipeline and save outputs."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Path to the training run directory to load a checkpoint from.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path,
        help="Path to the diffusion dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to write outputs into.",
    )
    parser.add_argument(
        "--p-prior",
        type=float,
        default=1.0,
        help="Mask prior hyperparameter value.",
    )
    parser.add_argument(
        "--logprob-batch-size",
        type=int,
        default=10_000,
        help="Batch size for log-probability evaluation in brain.",
    )
    parser.add_argument(
        "--mask-sample-count",
        type=int,
        default=50,
        help="Number of mask samples per voxel for model selection.",
    )
    parser.add_argument(
        "--mask-sample-batch-size",
        type=int,
        default=5_000,
        help="Batch size for sampling masks in brain.",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.999,
        help="Upper quantile used to clip data outliers.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path):
    checkpoint, model, simulator = load_checkpoint(str(checkpoint_path))
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema", checkpoint.get("params", params))
    model = nnx.merge(graphdef, params, state)
    model.eval()
    return model, simulator


def mask_to_label(mask, model_types) -> str:
    name = ""
    for i, m in enumerate(mask):
        if i >= len(model_types):
            continue
        if m:
            name += str(model_types[i]).split(".")[-1].replace("'", "")
    return name[:-1]


def mask_to_bool_list(mask) -> list[bool]:
    return [bool(value) for value in np.asarray(mask)]


def mask_dof(mask, model_types, include_noise: bool) -> int:
    mask_arr = np.asarray(mask).astype(bool)
    if include_noise:
        mask_arr = np.concatenate([mask_arr, np.array([False, True], dtype=bool)])
    dof = int(mask_arr.sum() - 2)
    for m_type, m in zip(model_types, mask_arr):
        dof += int(m_type.theta_dim * m)
    return dof


def prepare_brain_data(data_norm, brain_mask, outlier_quantile):
    full_data_flat = data_norm.reshape(-1, data_norm.shape[-1])
    brain_mask_flat = brain_mask.reshape(-1).astype(bool)

    full_data_flat_in_brain = full_data_flat[brain_mask_flat, :]
    full_data_flat_in_brain = np.nan_to_num(
        full_data_flat_in_brain, nan=0.0, posinf=0.0, neginf=0.0
    )
    exclude_outliers = np.quantile(full_data_flat_in_brain, outlier_quantile)
    full_data_flat_in_brain = np.clip(full_data_flat_in_brain, 0, exclude_outliers)

    return brain_mask_flat, full_data_flat_in_brain


def eval_in_batches(fn, data, batch_size, rng, label):
    outputs = []
    key = rng
    for batch_start in range(0, data.shape[0], batch_size):
        print(f"{label} batch {batch_start}")
        key, subkey = jax.random.split(key)
        batch_end = min(batch_start + batch_size, data.shape[0])
        batch_data = data[batch_start:batch_end]
        batch_keys = jax.random.split(subkey, batch_data.shape[0])
        batch_out = fn(batch_keys, batch_data)
        outputs.append(np.asarray(batch_out))
    return np.concatenate(outputs, axis=0)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = Path(__file__).resolve().parent
    args = parse_args(repo_root, default_output_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, simulator = load_model(args.checkpoint)
    sim_type = model.tokenizer.simulator

    data_img, data_norm, brain_mask, bvals, bvecs = load_and_process_data(
        str(args.data_path),
        "nodif_brain_mask.nii.gz",
        "data.nii.gz",
        "bvals",
        "bvecs",
    )
    acq = acquisition_scheme(bvals, bvecs)
    brain_mask_flat, full_data_flat_in_brain = prepare_brain_data(
        data_norm, brain_mask, args.outlier_quantile
    )

    preselected_masks = [
        jnp.array([True] * 1 + [False] * 9),  # Ball
        jnp.array([True] * 2 + [False] * 8),  # Ball + 1 Stick
        jnp.array([True] * 3 + [False] * 7),  # Ball + 2 Sticks
        jnp.array([True] * 4 + [False] * 6),  # Ball + 3 Sticks
        jnp.array([True] + [False] * 3 + [True] * 1 + [False] * 5),  # Ball + Zeppelin
        jnp.array([True] + [False] * 3 + [True] * 2 + [False] * 4),  # Ball + 2 Zeppelins
        jnp.array([True] + [False] * 3 + [True] * 3 + [False] * 3),  # Ball + 3 Zeppelins
        jnp.array([True] + [False] * 6 + [True, False, False]),  # One Tensor
        jnp.array([True] + [False] * 6 + [True, True, False]),  # Two Tensors
        jnp.array([True] + [False] * 6 + [True, True, True]),  # Three Tensors
    ]
    labels_preselected = [
        mask_to_label(mask, sim_type.model_types) for mask in preselected_masks
    ]
    indexed_models_text = "".join(
        f"{i}: {label}\n" for i, label in enumerate(labels_preselected)
    )

    def log_prob_preselected(key, x):
        log_probs = []
        for mask in preselected_masks:
            key, _ = jax.random.split(key)
            full_mask = jnp.concatenate([mask, jnp.array([False, True])], axis=0)
            log_probs.append(
                model.log_prob_mask(
                    full_mask, acq=acq, x=x, mask_prior=jnp.array([args.p_prior])
                )
            )
        return jnp.stack(log_probs, axis=-1)

    log_prob_preselected = jax.jit(jax.vmap(log_prob_preselected, in_axes=(0, 0)))

    probs_brain_preselected = eval_in_batches(
        log_prob_preselected,
        full_data_flat_in_brain,
        args.logprob_batch_size,
        jax.random.PRNGKey(0),
        "logprob preselected",
    )

    full_probs_preselected = np.zeros(
        data_norm.shape[:-1] + (len(preselected_masks),), dtype=np.float32
    )
    full_probs_preselected[brain_mask.astype(bool), :] = np.asarray(
        jax.nn.softmax(probs_brain_preselected, axis=-1)
    )

    model_selected_preselected = np.argmax(full_probs_preselected, axis=-1)
    preselected_counts = np.bincount(
        model_selected_preselected[brain_mask.astype(bool)],
        minlength=len(preselected_masks),
    )
    preselected_dof = [
        mask_dof(mask, sim_type.model_types, include_noise=True)
        for mask in preselected_masks
    ]
    preselected_df = pd.DataFrame(
        {
            "Model": labels_preselected,
            "Count": preselected_counts.astype(int),
            "Mask": [mask_to_bool_list(mask) for mask in preselected_masks],
            "DOF": preselected_dof,
        }
    )
    preselected_df.to_csv(
        output_dir / f"preselected_models_summary_{args.p_prior}.csv", index=False
    )
    export_nifti(
        full_probs_preselected.astype(float),
        data_img,
        str(output_dir),
        f"model_probs_preselected_prior_{args.p_prior}.nii.gz",
    )

    fig = orthoview_ultracompact(model_selected_preselected, width=1200, height=800)
    fig.add_annotation(
        x=0,
        y=1,
        text="Model selection (preselected subset, no parameter penalty)",
        showarrow=False,
        font=dict(size=20, color="white"),
        xref="paper",
        yref="paper",
    )
    fig.add_annotation(
        x=0,
        y=0.95,
        text=indexed_models_text,
        showarrow=False,
        font=dict(size=12, color="white"),
        xref="paper",
        yref="paper",
    )
    save_orthoview_html(
        fig, output_dir / "model_selection_all_gauss_no_penalty.html"
    )

    fig = orthoview_ultracompact(
        full_probs_preselected,
        vmin=0.0,
        vmax=0.5,
        channel_names=labels_preselected,
        width=1000,
    )
    save_orthoview_html(
        fig, output_dir / f"model_probs_preselected_prior_{args.p_prior}.html"
    )

    def sample_best_mask(key, x):
        sample_fn = partial(
            model.sample_mask, acq=acq, x=x, mask_prior=jnp.array([args.p_prior])
        )
        masks = jax.vmap(sample_fn)(jax.random.split(key, args.mask_sample_count))
        log_probs = jax.vmap(
            partial(
                model.log_prob_mask,
                acq=acq,
                x=x,
                mask_prior=jnp.array([args.p_prior]),
            )
        )(masks)
        idx = jnp.argmax(log_probs)
        return masks[idx]

    sample_best_mask = jax.jit(jax.vmap(sample_best_mask, in_axes=(0, 0)))

    samples_in_brain = eval_in_batches(
        sample_best_mask,
        full_data_flat_in_brain,
        args.mask_sample_batch_size,
        jax.random.PRNGKey(0),
        "mask sampling",
    )

    map_models = embed_in_full_brain_array(
        samples_in_brain, brain_mask_flat, data_norm.shape[:-1]
    )
    export_nifti(
        map_models.astype(float),
        data_img,
        str(output_dir),
        f"model_map_all_prior_{args.p_prior}.nii.gz",
    )

    labels_per_voxel = [
        mask_to_label(samples_in_brain[i, :-2], sim_type.model_types)
        for i in range(samples_in_brain.shape[0])
    ]
    comb, counts = np.unique(labels_per_voxel, return_counts=True)
    idx = np.argsort(-counts)
    comb = comb[idx]
    counts = counts[idx]

    top10 = comb[:10]
    top10_maskref = []
    for top in top10:
        for i, ref in enumerate(labels_per_voxel):
            if ref == top:
                top10_maskref.append(samples_in_brain[i])
                break

    top10_dof = [
        mask_dof(mask, sim_type.model_types, include_noise=False)
        for mask in top10_maskref
    ]

    df = pd.DataFrame(
        {
            "Model": top10,
            "Count": counts[:10],
            "Mask": [mask_to_bool_list(m) for m in top10_maskref],
            "DOF": top10_dof,
        }
    )
    df.to_csv(output_dir / f"top10_models_summary_{args.p_prior}.csv", index=False)

    top10_masks = jnp.stack(top10_maskref)

    def log_prob_top10(key, x):
        log_probs = []
        for mask in top10_masks:
            key, _ = jax.random.split(key)
            log_probs.append(
                model.log_prob_mask(
                    mask, acq=acq, x=x, mask_prior=jnp.array([args.p_prior])
                )
            )
        return jnp.stack(log_probs, axis=-1)

    log_prob_top10 = jax.jit(jax.vmap(log_prob_top10, in_axes=(0, 0)))

    probs_brain_top10 = eval_in_batches(
        log_prob_top10,
        full_data_flat_in_brain,
        args.logprob_batch_size,
        jax.random.PRNGKey(0),
        "logprob top10",
    )

    full_probs_top10 = np.zeros(
        data_norm.shape[:-1] + (len(top10),), dtype=np.float32
    )
    full_probs_top10[brain_mask.astype(bool), :] = np.asarray(
        jax.nn.softmax(probs_brain_top10, axis=-1)
    )

    export_nifti(
        full_probs_top10.astype(float),
        data_img,
        str(output_dir),
        f"model_probs_top10_prior_{args.p_prior}.nii.gz",
    )

    fig = orthoview_ultracompact(
        full_probs_top10, vmin=0.0, vmax=0.5, channel_names=top10, width=1000
    )
    save_orthoview_html(
        fig, output_dir / f"model_probs_top10_prior_{args.p_prior}.html"
    )


if __name__ == "__main__":
    main()
