#!/usr/bin/env python3
"""Scripted version of eval_all_conv.ipynb."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from flax import nnx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dmri.eval.export_theta import embed_in_full_brain_array
from dmri.eval.load_data import load_and_process_data
from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.train.utils import load_checkpoint
from dmri.utils.dmriutils import export_nifti
from dmri.utils.viz import orthoview_ultracompact, save_orthoview_html


def parse_args(repo_root: Path, default_output_dir: Path) -> argparse.Namespace:
    default_checkpoint = (
        repo_root / "results/updated_all_3_6_8_128_model_prior_with_convs_2gpus"
    )
    default_data_path = repo_root / "data/data"

    parser = argparse.ArgumentParser(
        description="Run eval_all_conv pipeline and save outputs."
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
        "--synthetic-count",
        type=int,
        default=10_000,
        help="Number of synthetic samples for PIT coverage.",
    )
    parser.add_argument(
        "--coverage-samples",
        type=int,
        default=1000,
        help="Number of mask samples for PIT coverage.",
    )
    parser.add_argument(
        "--coverage-batch-size",
        type=int,
        default=100,
        help="Batch size for PIT coverage evaluation.",
    )
    parser.add_argument(
        "--logprob-batch-size",
        type=int,
        default=1_000,
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
        default=5_00,
        help="Batch size for sampling masks in brain.",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.999,
        help="Upper quantile used to clip data outliers.",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip PIT coverage evaluation.",
    )
    return parser.parse_args()


def apply_style(repo_root: Path) -> None:
    style_path = repo_root / "dmri/utils/pyloric.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


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


def log_prob_pit(model, data, rng, n=200):
    rng_samp, rng_tie, rng_jit = jax.random.split(rng, 3)

    keys = jax.random.split(rng_samp, n)
    sample_fn = lambda k: model.sample_mask(
        k, acq=data["acq"], x=data["x"], mask_prior=data["mask_prior"],
    )
    masks = jax.vmap(sample_fn)(keys)

    logprob_fn = lambda m: model.log_prob_mask(
        m, acq=data["acq"], x=data["x"], mask_prior=data["mask_prior"]
    )

    lp_true = jnp.reshape(logprob_fn(data["model_mask"]), (-1,)).sum()

    lp_samp = jax.vmap(logprob_fn)(masks)
    lp_samp = jnp.reshape(lp_samp, (n, -1)).sum(axis=1)

    eq = jnp.isclose(lp_samp, lp_true, rtol=1e-4, atol=1e-4)
    lt = (lp_samp < lp_true) & (~eq)

    n_lt = jnp.sum(lt)
    n_eq = jnp.sum(eq)

    tie = jax.random.randint(rng_tie, (), 0, n_eq + 1)
    r = n_lt + tie

    v = jax.random.uniform(rng_jit, (), minval=0.0, maxval=1.0)
    pit = (r + v) / (n + 1)

    return pit


def expected_coverage(model, data, rng, n=1000, batch_size=100):
    batch = data["x"].shape[0]
    keys = jax.random.split(rng, batch)

    pits = jax.lax.map(
        lambda args: log_prob_pit(model, *args, n=n),
        (data, keys),
        batch_size=batch_size,
    )

    alpha = jnp.linspace(0.0, 1.0, 101)
    coverage = jnp.mean(pits[:, None] <= alpha[None, :], axis=0)
    return alpha, coverage, pits


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
    repo_root = Path(__file__).resolve().parents[2]
    default_output_dir = Path(__file__).resolve().parent
    args = parse_args(repo_root, default_output_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style(repo_root)

    output_tag = "conv"

    model, simulator = load_model(args.checkpoint)
    sim_type = model.tokenizer.simulator

    if not args.skip_coverage:
        data = jax.vmap(simulator[0])(
            jax.random.split(jax.random.PRNGKey(0), args.synthetic_count)
        )
        data0 = jax.vmap(
            partial(simulator[0], mask_prior_hyperparameter=jnp.array([0.1]))
        )(jax.random.split(jax.random.PRNGKey(42), args.synthetic_count))
        data1 = jax.vmap(
            partial(simulator[0], mask_prior_hyperparameter=jnp.array([0.3]))
        )(jax.random.split(jax.random.PRNGKey(42), args.synthetic_count))
        data2 = jax.vmap(
            partial(simulator[0], mask_prior_hyperparameter=jnp.array([0.5]))
        )(jax.random.split(jax.random.PRNGKey(43), args.synthetic_count))
        data3 = jax.vmap(
            partial(simulator[0], mask_prior_hyperparameter=jnp.array([0.8]))
        )(jax.random.split(jax.random.PRNGKey(44), args.synthetic_count))

        coverage = expected_coverage(
            model,
            data,
            jax.random.PRNGKey(0),
            n=args.coverage_samples,
            batch_size=args.coverage_batch_size,
        )
        coverage0 = expected_coverage(
            model,
            data0,
            jax.random.PRNGKey(42),
            n=args.coverage_samples,
            batch_size=args.coverage_batch_size,
        )
        coverage1 = expected_coverage(
            model,
            data1,
            jax.random.PRNGKey(0),
            n=args.coverage_samples,
            batch_size=args.coverage_batch_size,
        )
        coverage2 = expected_coverage(
            model,
            data2,
            jax.random.PRNGKey(1),
            n=args.coverage_samples,
            batch_size=args.coverage_batch_size,
        )
        coverage3 = expected_coverage(
            model,
            data3,
            jax.random.PRNGKey(2),
            n=args.coverage_samples,
            batch_size=args.coverage_batch_size,
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(coverage[0], coverage[1], marker="o", label="Overall")
        ax.plot(coverage0[0], coverage0[1], marker="o", label="p=0.1")
        ax.plot(coverage1[0], coverage1[1], marker="o", label="p=0.3")
        ax.plot(coverage2[0], coverage2[1], marker="o", label="p=0.5")
        ax.plot(coverage3[0], coverage3[1], marker="o", label="p=0.8")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.legend()
        fig.tight_layout()
        fig_path = output_dir / f"pit_coverage_{output_tag}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        np.savez_compressed(
            output_dir / f"pit_coverage_{output_tag}.npz",
            alpha=np.asarray(coverage[0]),
            coverage=np.asarray(coverage[1]),
            coverage0=np.asarray(coverage0[1]),
            coverage1=np.asarray(coverage1[1]),
            coverage2=np.asarray(coverage2[1]),
            coverage3=np.asarray(coverage3[1]),
            pits=np.asarray(coverage[2]),
            pits0=np.asarray(coverage0[2]),
            pits1=np.asarray(coverage1[2]),
            pits2=np.asarray(coverage2[2]),
            pits3=np.asarray(coverage3[2]),
        )

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
        jnp.array([True] * 1 + [False] * 18),  # Ball
        jnp.array([True] * 2 + [False] * 17),  # Ball + 1 Stick
        jnp.array([True] * 3 + [False] * 16),  # Ball + 2 Sticks
        jnp.array([True] * 4 + [False] * 15),  # Ball + 3 Sticks
        jnp.array([True] + [False] * 3 + [True] * 1 + [False] * 14),  # Ball + Zeppelin
        jnp.array([True] + [False] * 3 + [True] * 2 + [False] * 13),  # Ball + 2 Zeppelins
        jnp.array([True] + [False] * 3 + [True] * 3 + [False] * 12),  # Ball + 3 Zeppelins
        jnp.array([True] + [False] * 6 + [True, False, False] + [False] * 9),  # One Tensor
        jnp.array([True] + [False] * 6 + [True, True, False] + [False] * 9),  # Two Tensors
        jnp.array([True] + [False] * 6 + [True, True, True] + [False] * 9),  # Three Tensors
        jnp.array([True] + [False] * 9 + [True, True] + [False] * 7),  # Ball Dot WStick
        jnp.array([True] + [False] * 9 + [False, False, False] + [True, True] + [False] * 4),  # Ball BStick BZeppelin
        jnp.array([False] * 15 + [True, False, False, False]),  # Noddi B
        jnp.array([False] * 15 + [False, True, False, False]),  # Noddi W
        jnp.array([False] * 15 + [False, False, True, False]),  # Sandi B
        jnp.array([False] * 15 + [False, False, False, True]),  # Sandi W
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
        output_dir / f"preselected_models_summary_{output_tag}_{args.p_prior}.csv",
        index=False,
    )
    export_nifti(
        full_probs_preselected.astype(float),
        data_img,
        str(output_dir),
        f"model_probs_preselected_{output_tag}_prior_{args.p_prior}.nii.gz",
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
        fig, output_dir / f"model_selection_all_{output_tag}_gauss_no_penalty.html"
    )

    fig = orthoview_ultracompact(
        full_probs_preselected,
        vmin=0.0,
        vmax=0.5,
        channel_names=labels_preselected,
        width=1000,
    )
    save_orthoview_html(
        fig,
        output_dir / f"model_probs_preselected_{output_tag}_prior_{args.p_prior}.html",
    )

    def sample_best_mask(key, x):
        sample_fn = partial(
            model.sample_mask, acq=acq, x=x, mask_prior=jnp.array([args.p_prior]), temperature=0.5
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
        f"model_map_all_{output_tag}_prior_{args.p_prior}.nii.gz",
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
    df.to_csv(
        output_dir / f"top10_models_summary_{output_tag}_{args.p_prior}.csv",
        index=False,
    )

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
        f"model_probs_top10_{output_tag}_prior_{args.p_prior}.nii.gz",
    )

    fig = orthoview_ultracompact(
        full_probs_top10, vmin=0.0, vmax=0.5, channel_names=top10, width=1000
    )
    save_orthoview_html(
        fig, output_dir / f"model_probs_top10_{output_tag}_prior_{args.p_prior}.html"
    )


if __name__ == "__main__":
    main()
