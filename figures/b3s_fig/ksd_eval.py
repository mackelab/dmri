#!/usr/bin/env python3
"""Evaluate KSDs for synthetic and real data across short/long sequences."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from flax import nnx
from scipy import ndimage

from dmri.eval.export_metrics import multiscale_preconditioned_ksd_and_pvalue
from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.train.utils import load_checkpoint

# Avoid forcing a non-interactive backend in notebooks.
if os.environ.get("MPLBACKEND") is None and "IPython" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.style.use("../../dmri/utils/pyloric.mplstyle")

LABELS_MASK = {
    "B": jnp.array([[True, False, False, False, True]]),
    "S": jnp.array(
        [
            [False, True, False, False, True],
            [False, False, True, False, True],
            [False, False, False, True, True],
        ]
    ),
    "S2": jnp.array(
        [
            [False, True, True, False, True],
            [False, False, True, True, True],
            [False, True, False, True, True],
        ]
    ),
    "S3": jnp.array([[False, True, True, True, True]]),
    "BS1": jnp.array(
        [
            [True, True, False, False, True],
            [True, False, True, False, True],
            [True, False, False, True, True],
        ]
    ),
    "BS2": jnp.array(
        [
            [True, True, True, False, True],
            [True, True, False, True, True],
            [True, False, True, True, True],
        ]
    ),
    "BS3": jnp.array([[True, True, True, True, True]]),
}


def maybe_apply_style(repo_root: Path) -> None:
    style_path = repo_root / "dmri" / "utils" / "pyloric.mplstyle"
    if style_path.exists():
        plt.style.use(style_path.as_posix())
        return
    print(f"Matplotlib style not found at {style_path}")


def load_real_data(data_root: Path, folder_name: str) -> dict[str, np.ndarray]:
    data_path = data_root / folder_name / "data.nii.gz"
    mask_path = data_root / folder_name / "nodif_brain_mask.nii.gz"
    bvals_path = data_root / folder_name / "bvals"
    bvecs_path = data_root / folder_name / "bvecs"

    data, _ = load_nifti(data_path.as_posix())
    brain_mask, _ = load_nifti(mask_path.as_posix())
    brain_mask = ndimage.binary_erosion(brain_mask.astype(bool), iterations=2)
    raw_bvals, bvecs = read_bvals_bvecs(bvals_path.as_posix(), bvecs_path.as_posix())

    bvals = np.round(raw_bvals / 1000.0) * 1000.0
    bvals = np.clip(bvals, 0.0, None)
    b0_mask = bvals == 0

    s0 = np.mean(data[..., b0_mask], axis=-1)
    data_norm = data / s0[..., None]
    data_norm = np.where(brain_mask[..., None], data_norm, 0.0)
    data_norm = np.where(np.isnan(data_norm), 0.0, data_norm)

    idx = np.argsort(bvals)
    bvals = bvals[idx]
    bvecs = bvecs[idx]
    data_norm = data_norm[..., idx]

    flat_data = data_norm.reshape(-1, data_norm.shape[-1])
    flat_mask = brain_mask.reshape(-1).astype(bool)
    in_brain = flat_data[flat_mask, :]
    in_brain = np.nan_to_num(in_brain, nan=0.0, posinf=0.0, neginf=0.0)

    acq = acquisition_scheme(np.asarray(bvals), np.asarray(bvecs))
    return {"x": in_brain, "acq": acq}


def build_mask_support(mask_prior, max_support: int) -> jnp.ndarray | None:
    num_models = mask_prior.num_model_components
    num_noise = mask_prior.num_noise_components
    min_active = int(getattr(mask_prior, "min_active_models", 0))

    total_models = 2**num_models
    total_noise = max(1, num_noise)
    total_support = total_models * total_noise
    if total_support > max_support:
        return None

    combos = list(itertools.product([False, True], repeat=num_models))
    if min_active > 0:
        combos = [combo for combo in combos if sum(combo) >= min_active]
    model_masks = np.array(combos, dtype=bool)

    if num_noise == 0:
        support = model_masks
    else:
        noise_masks = np.eye(num_noise, dtype=bool)
        expanded = []
        for model_mask in model_masks:
            for noise_mask in noise_masks:
                expanded.append(np.concatenate([model_mask, noise_mask], axis=0))
        support = np.stack(expanded, axis=0)
    return jnp.asarray(support, dtype=jnp.bool_)


def infer_support_from_masks(*mask_arrays: np.ndarray) -> jnp.ndarray:
    combined = np.concatenate(mask_arrays, axis=0)
    unique_masks = np.unique(combined, axis=0)
    return jnp.asarray(unique_masks, dtype=jnp.bool_)


def make_sample_fn(model, num_steps: int, t_min: float, last_euler_step: bool):
    def sample_one(key, acq, x, model_mask):
        return model.sample_theta(
            key,
            acq,
            x,
            model_mask,
            last_euler_step=last_euler_step,
            t_min=t_min,
            num_steps=num_steps,
        )

    return jax.jit(jax.vmap(sample_one, in_axes=(0, None, None, None)))


def eval_ksd_single(
    data: dict[str, jnp.ndarray],
    rng: jax.Array,
    sample_fn,
    sim_type,
    bandwidths: jnp.ndarray,
    theta_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng_sample, rng_eval = jax.random.split(rng)
    keys = jax.random.split(rng_sample, num=theta_samples)
    thetas = sample_fn(keys, data["acq"], data["x"], data["model_mask"])
    ksd, pval = multiscale_preconditioned_ksd_and_pvalue(
        sim_type,
        thetas=thetas,
        mask=data["model_mask"],
        acq=data["acq"],
        x=data["x"],
        key=rng_eval,
        bandwidths=bandwidths,
    )
    return ksd, pval


def eval_ksd_prior_single(
    data: dict[str, jnp.ndarray],
    rng: jax.Array,
    sim_type,
    bandwidths: jnp.ndarray,
    theta_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng_sample, rng_eval = jax.random.split(rng)
    thetas = jax.random.normal(rng_sample, (theta_samples, sim_type.theta_dim))
    ksd, pval = multiscale_preconditioned_ksd_and_pvalue(
        sim_type,
        thetas=thetas,
        mask=data["model_mask"],
        acq=data["acq"],
        x=data["x"],
        key=rng_eval,
        bandwidths=bandwidths,
    )
    return ksd, pval


def run_batched_eval(
    dataset, rng, eval_fn, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    total = int(dataset["x"].shape[0])
    keys = jax.random.split(rng, total)

    def eval_batch(batch, batch_keys):
        def eval_one(data_key):
            data, key = data_key
            return eval_fn(data, key)

        return jax.lax.map(eval_one, (batch, batch_keys), batch_size=batch_size)

    eval_batch_jit = jax.jit(eval_batch)
    print(f"Evaluating {total} samples (batch_size={batch_size})")
    ksd, pval = eval_batch_jit(dataset, keys)
    return np.asarray(ksd), np.asarray(pval)


def group_by_mask(values: np.ndarray, masks: np.ndarray, support: np.ndarray) -> list[np.ndarray]:
    grouped = []
    for mask in support:
        subset = np.all(masks == mask, axis=1)
        grouped.append(values[subset])
    return grouped


def pad_per_mask(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max((len(v) for v in values), default=0)
    counts = np.zeros(len(values), dtype=np.int32)
    padded = np.full((len(values), max_len), np.nan, dtype=np.float32)
    for idx, arr in enumerate(values):
        counts[idx] = len(arr)
        if counts[idx] > 0:
            padded[idx, : counts[idx]] = arr.astype(np.float32)
    return padded, counts


def save_mask_results(
    out_path: Path,
    support: np.ndarray,
    ksd: list[np.ndarray],
    ksd_prior: list[np.ndarray],
    pval: list[np.ndarray],
    pval_prior: list[np.ndarray],
) -> None:
    ksd_padded, ksd_counts = pad_per_mask(ksd)
    ksd_prior_padded, ksd_prior_counts = pad_per_mask(ksd_prior)
    pval_padded, pval_counts = pad_per_mask(pval)
    pval_prior_padded, pval_prior_counts = pad_per_mask(pval_prior)

    np.savez_compressed(
        out_path,
        mask_support=support.astype(bool),
        ksd=ksd_padded,
        ksd_counts=ksd_counts,
        ksd_prior=ksd_prior_padded,
        ksd_prior_counts=ksd_prior_counts,
        pval=pval_padded,
        pval_counts=pval_counts,
        pval_prior=pval_prior_padded,
        pval_prior_counts=pval_prior_counts,
    )


def normalize_by_prior(
    ksd: list[np.ndarray], ksd_prior: list[np.ndarray]
) -> list[np.ndarray]:
    normalized = []
    for values, prior_values in zip(ksd, ksd_prior):
        values = np.asarray(values)
        prior_values = np.asarray(prior_values)
        if values.size > 0:
            values = values[np.isfinite(values)]
        if prior_values.size > 0:
            prior_values = prior_values[np.isfinite(prior_values)]
        if len(values) == 0 or len(prior_values) == 0:
            normalized.append(np.array([], dtype=np.float32))
            continue
        denom = np.mean(prior_values)
        if not np.isfinite(denom) or denom == 0.0:
            normalized.append(np.array([], dtype=np.float32))
            continue
        normalized.append(values / denom)
    return normalized


def trim_values(values: list[np.ndarray], quantile: float | None) -> list[np.ndarray]:
    trimmed = []
    for arr in values:
        arr = np.asarray(arr)
        if arr.size > 0:
            arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            trimmed.append(arr)
            continue
        if quantile is None:
            trimmed.append(arr)
            continue
        cutoff = np.quantile(arr, quantile)
        trimmed.append(arr[arr < cutoff])
    return trimmed


def build_label_masks(label_map: dict, expected_dim: int) -> dict[str, np.ndarray]:
    label_masks = {}
    for label, mask_arr in label_map.items():
        arr = np.asarray(mask_arr, dtype=bool)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != expected_dim:
            print(
                f"Skipping label {label}: expected mask length {expected_dim}, got {arr.shape[1]}"
            )
            continue
        label_masks[label] = arr
    return label_masks


def expand_label_masks(
    label_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    masks = []
    labels = []
    for label, mask_list in label_masks.items():
        for mask in np.asarray(mask_list):
            masks.append(mask)
            labels.append(label)
    if not masks:
        return np.zeros((0, 0), dtype=bool), []
    return np.stack(masks, axis=0), labels


def group_by_mask_list(
    values: np.ndarray,
    masks: np.ndarray,
    mask_list: np.ndarray,
) -> list[np.ndarray]:
    grouped = []
    mask_arr = np.asarray(masks)
    for mask in mask_list:
        subset = np.all(mask_arr == mask, axis=1)
        grouped.append(values[subset])
    return grouped




def style_violin(violin, color: str) -> None:
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)
        body.set_alpha(0.7)
    if "cmins" in violin:
        violin["cmins"].set_color(color)
    if "cmaxes" in violin:
        violin["cmaxes"].set_color(color)
    for key in ("cbars", "cmeans"):
        if key in violin:
            violin[key].set_color("black")
            violin[key].set_linewidth(0.4)


def plot_violin(
    groups: list[list[np.ndarray]],
    labels: list[str],
    out_path: Path | None,
    title: str,
    ylabel: str,
    *,
    group_labels: list[str] | None = None,
    colors: list[str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = (4.2, 1.4),
    violin_width: float | None = None,
    group_span: float = 0.4,
    rotation: float = 35.0,
    xlabel: str | None = "Model configuration",
    xtick_fontsize: int = 9,
    ylabel_fontsize: int = 9,
    legend: bool = True,
    legend_loc: str = "upper right",
    legend_kwargs: dict[str, object] | None = None,
    tight_layout: bool = True,
    show: bool = False,
    save_kwargs: dict[str, object] | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
    if not groups:
        print(f"No data to plot for {out_path}")
        return

    group_arrays = [[np.asarray(values) for values in group] for group in groups]

    if not labels:
        print(f"No data to plot for {out_path}")
        return

    for group in group_arrays:
        if len(group) != len(labels):
            raise ValueError(
                f"Group length {len(group)} does not match labels length {len(labels)}."
            )

    num_groups = len(group_arrays)
    if group_labels is None:
        group_labels = [f"Group {idx + 1}" for idx in range(num_groups)]
    if len(group_labels) != num_groups:
        raise ValueError("Number of group_labels must match groups.")

    if colors is None:
        default_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#64B5CD"]
        colors = [default_colors[idx % len(default_colors)] for idx in range(num_groups)]
    if len(colors) != num_groups:
        raise ValueError("Number of colors must match groups.")

    positions = np.arange(len(labels))
    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if num_groups == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-group_span / 2, group_span / 2, num_groups)

    if violin_width is None:
        if num_groups == 1:
            violin_width = 0.35
        else:
            spacing = group_span / (num_groups - 1)
            violin_width = min(0.35, spacing * 0.9)

    handles = []
    for group, label, color, offset in zip(group_arrays, group_labels, colors, offsets):
        keep_idx = [idx for idx, values in enumerate(group) if len(values) > 0]
        if not keep_idx:
            print(f"No data selected for {label} in {out_path}")
            continue
        group_positions = positions[keep_idx] + offset
        violin = ax.violinplot(
            [group[idx] for idx in keep_idx],
            positions=group_positions,
            widths=violin_width,
        )
        style_violin(violin, color)
        handles.append((label, color))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=xtick_fontsize, rotation=rotation, ha="right")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if title:
        ax.set_title(title)

    from matplotlib.patches import Patch

    if handles and legend:
        ax.legend(
            handles=[
                Patch(facecolor=color, edgecolor="black", label=label)
                for label, color in handles
            ],
            frameon=False,
            loc=legend_loc,
            **(legend_kwargs or {}),
        )
    if tight_layout:
        fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, **(save_kwargs or {}))
    if show:
        plt.show()
    #plt.close(fig)
    return fig, ax


def plot_multi_violin(
    groups: list[list[np.ndarray]],
    labels: list[str],
    out_path: Path,
    title: str,
    ylabel: str,
    group_labels: list[str],
    colors: list[str],
) -> None:
    plot_violin(
        groups,
        labels,
        out_path,
        title,
        ylabel,
        group_labels=group_labels,
        colors=colors,
        figsize=(4.2, 2.8),
        violin_width=0.25,
        group_span=0.6,
    )


def build_real_dataset(
    data: dict[str, np.ndarray],
    rng: jax.Array,
    num_samples: int,
    model_masks: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    total = int(data["x"].shape[0])
    if num_samples > total:
        raise ValueError(f"Requested {num_samples} samples but only {total} available.")
    idx = jax.random.choice(rng, total, (num_samples,), replace=False)
    x = jnp.asarray(data["x"])[idx]
    acq = jax.tree_util.tree_map(
        lambda arr: jnp.repeat(jnp.asarray(arr)[None, ...], num_samples, axis=0),
        data["acq"],
    )
    return {"x": x, "acq": acq, "model_mask": model_masks}


def sample_random_masks(mask_prior, rng, num_samples: int, hyperparameter: float | None):
    keys = jax.random.split(rng, num_samples)
    if hyperparameter is None:
        samples = jax.vmap(mask_prior.sample)(keys)
        return samples.model_mask
    hyper = jnp.full((mask_prior.mask_prior_dim,), hyperparameter)
    return jax.vmap(mask_prior.sample_model_mask, in_axes=(0, None))(keys, hyper)


def sample_selected_masks(model, rng, dataset, mask_prior_value: float):
    def sample_one(key, data):
        return model.sample_mask(
            key, acq=data["acq"], x=data["x"], mask_prior=mask_prior_value
        )

    keys = jax.random.split(rng, dataset["x"].shape[0])
    return jax.vmap(sample_one)(keys, dataset)


def parse_bandwidths(text: str) -> jnp.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Bandwidth list cannot be empty.")
    return jnp.asarray(values)


def next_key(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    rng, key = jax.random.split(rng)
    return rng, key


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KSDs for B3S models.")
    parser.add_argument(
        "--checkpoint",
        default="/home/macke/mgloeckler90/dmri/results/updated5_b3s_2_4_6_128",
        help="Path to the training run directory with checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ksd_eval",
        help="Directory to write KSD results and plots.",
    )
    parser.add_argument("--synthetic-samples", type=int, default=50000)
    parser.add_argument("--real-samples", type=int, default=50000)
    parser.add_argument("--theta-samples", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--t-min", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bandwidths", type=str, default="0.1,0.5,1.0,10.0")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-support", type=int, default=10_000)
    parser.add_argument("--real-short-folder", default="data")
    parser.add_argument("--real-long-folder", default="new_data")
    parser.add_argument(
        "--random-mask-hyper",
        type=float,
        default=0.3,
        help="Hyperparameter for random masks; set to -1 to sample from prior.",
    )
    parser.add_argument(
        "--selected-mask-prior",
        type=float,
        default=0.2,
        help="Mask prior value when sampling selected masks from the model.",
    )
    parser.add_argument(
        "--real-trim-quantile",
        type=float,
        default=0.99,
        help="Quantile cutoff for trimming outliers in real-data violins.",
    )
    parser.add_argument(
        "--synthetic-trim-quantile",
        type=float,
        default=0.99,
        help="Quantile cutoff for trimming outliers in synthetic violins.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_apply_style(repo_root)

    checkpoint, model, simulator = load_checkpoint(args.checkpoint)
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema", checkpoint.get("params", params))
    state = checkpoint.get("model_state", state)
    model = nnx.merge(graphdef, params, state)
    model.eval()

    sim_type = model.tokenizer.simulator
    mask_prior = sim_type.create_mask_prior()
    support = build_mask_support(mask_prior, args.max_support)

    rng = jax.random.PRNGKey(args.seed)

    rng, key = next_key(rng)
    if not isinstance(simulator, (list, tuple)) or len(simulator) < 2:
        raise ValueError("Expected at least two simulators for short and long data.")
    print(f"Sampling synthetic short dataset ({args.synthetic_samples})")
    dataset_synth_short = jax.vmap(simulator[0])(
        jax.random.split(key, num=args.synthetic_samples)
    )
    rng, key = next_key(rng)
    print(f"Sampling synthetic long dataset ({args.synthetic_samples})")
    dataset_synth_long = jax.vmap(simulator[1])(
        jax.random.split(key, num=args.synthetic_samples)
    )

    if support is None:
        support = infer_support_from_masks(
            np.asarray(dataset_synth_short["model_mask"]),
            np.asarray(dataset_synth_long["model_mask"]),
        )
    support_np = np.asarray(support)
    num_models = mask_prior.num_model_components

    bandwidths = parse_bandwidths(args.bandwidths)
    sample_fn = make_sample_fn(model, args.num_steps, args.t_min, last_euler_step=False)

    print("Evaluating synthetic KSDs (short)")
    eval_ksd = lambda data, key: eval_ksd_single(
        data, key, sample_fn, sim_type, bandwidths, args.theta_samples
    )
    eval_ksd_prior = lambda data, key: eval_ksd_prior_single(
        data, key, sim_type, bandwidths, args.theta_samples
    )

    rng, key = next_key(rng)
    ksds_short, pvals_short = run_batched_eval(
        dataset_synth_short, key, eval_ksd, args.batch_size
    )
    rng, key = next_key(rng)
    print("Evaluating synthetic KSD priors (short)")
    ksds_short_prior, pvals_short_prior = run_batched_eval(
        dataset_synth_short, key, eval_ksd_prior, args.batch_size
    )

    masks_short = np.asarray(dataset_synth_short["model_mask"])
    ksd_short_per_mask = group_by_mask(ksds_short, masks_short, support_np)
    ksd_short_prior_per_mask = group_by_mask(
        ksds_short_prior, masks_short, support_np
    )
    pval_short_per_mask = group_by_mask(pvals_short, masks_short, support_np)
    pval_short_prior_per_mask = group_by_mask(
        pvals_short_prior, masks_short, support_np
    )

    save_mask_results(
        out_dir / "synthetic_short_ksd_per_mask.npz",
        support_np,
        ksd_short_per_mask,
        ksd_short_prior_per_mask,
        pval_short_per_mask,
        pval_short_prior_per_mask,
    )
    print("Saved synthetic short results.")

    rng, key = next_key(rng)
    print("Evaluating synthetic KSDs (long)")
    ksds_long, pvals_long = run_batched_eval(
        dataset_synth_long, key, eval_ksd, args.batch_size
    )
    rng, key = next_key(rng)
    print("Evaluating synthetic KSD priors (long)")
    ksds_long_prior, pvals_long_prior = run_batched_eval(
        dataset_synth_long, key, eval_ksd_prior, args.batch_size
    )

    masks_long = np.asarray(dataset_synth_long["model_mask"])
    ksd_long_per_mask = group_by_mask(ksds_long, masks_long, support_np)
    ksd_long_prior_per_mask = group_by_mask(ksds_long_prior, masks_long, support_np)
    pval_long_per_mask = group_by_mask(pvals_long, masks_long, support_np)
    pval_long_prior_per_mask = group_by_mask(pvals_long_prior, masks_long, support_np)

    save_mask_results(
        out_dir / "synthetic_long_ksd_per_mask.npz",
        support_np,
        ksd_long_per_mask,
        ksd_long_prior_per_mask,
        pval_long_per_mask,
        pval_long_prior_per_mask,
    )
    print("Saved synthetic long results.")

    label_masks = build_label_masks(
        LABELS_MASK, dataset_synth_short["model_mask"].shape[1]
    )
    mask_list, labels = expand_label_masks(label_masks)
    ksd_short_model = group_by_mask_list(ksds_short, masks_short, mask_list)
    ksd_short_prior_model = group_by_mask_list(
        ksds_short_prior, masks_short, mask_list
    )
    ksd_long_model = group_by_mask_list(ksds_long, masks_long, mask_list)
    ksd_long_prior_model = group_by_mask_list(
        ksds_long_prior, masks_long, mask_list
    )
    ksd_short_ratio = normalize_by_prior(ksd_short_model, ksd_short_prior_model)
    ksd_long_ratio = normalize_by_prior(ksd_long_model, ksd_long_prior_model)
    plot_violin(
        [
            trim_values(ksd_short_ratio, args.synthetic_trim_quantile),
            trim_values(ksd_long_ratio, args.synthetic_trim_quantile),
        ],
        labels,
        out_dir / "synthetic_violin.png",
        "Synthetic rKSD ratios by mask",
        "rKSD",
        group_labels=["Short", "Long"],
        colors=["#4C72B0", "#DD8452"],
    )
    plot_violin(
        [
            trim_values(ksd_short_ratio, args.synthetic_trim_quantile),
            trim_values(ksd_long_ratio, args.synthetic_trim_quantile),
        ],
        labels,
        out_dir / "synthetic_violin.svg",
        "Synthetic rKSD ratios by mask",
        "rKSD",
        group_labels=["Short", "Long"],
        colors=["#4C72B0", "#DD8452"],
    )
    print("Saved synthetic violin plot.")

    print(f"Loading real short data from {args.real_short_folder}")
    data_root = repo_root / "data"
    real_short = load_real_data(data_root, args.real_short_folder)
    print(f"Loading real long data from {args.real_long_folder}")
    real_long = load_real_data(data_root, args.real_long_folder)

    random_hyper = args.random_mask_hyper
    if random_hyper < 0:
        random_hyper = None

    rng, key = next_key(rng)
    print(f"Sampling random masks for real data ({args.real_samples})")
    model_masks_random = sample_random_masks(
        mask_prior, key, args.real_samples, random_hyper
    )

    rng, key = next_key(rng)
    real_short_dataset = build_real_dataset(
        real_short, key, args.real_samples, model_masks_random
    )
    rng, key = next_key(rng)
    real_long_dataset = build_real_dataset(
        real_long, key, args.real_samples, model_masks_random
    )

    rng, key = next_key(rng)
    print("Evaluating real KSDs (short, random masks)")
    ksds_real_short, pvals_real_short = run_batched_eval(
        real_short_dataset, key, eval_ksd, args.batch_size
    )
    rng, key = next_key(rng)
    print("Evaluating real KSD priors (short, random masks)")
    ksds_real_short_prior, pvals_real_short_prior = run_batched_eval(
        real_short_dataset, key, eval_ksd_prior, args.batch_size
    )
    masks_real_short = np.asarray(real_short_dataset["model_mask"])
    ksd_real_short_per_mask = group_by_mask(
        ksds_real_short, masks_real_short, support_np
    )
    ksd_real_short_prior_per_mask = group_by_mask(
        ksds_real_short_prior, masks_real_short, support_np
    )
    pval_real_short_per_mask = group_by_mask(
        pvals_real_short, masks_real_short, support_np
    )
    pval_real_short_prior_per_mask = group_by_mask(
        pvals_real_short_prior, masks_real_short, support_np
    )

    save_mask_results(
        out_dir / "real_random_short_ksd_per_mask.npz",
        support_np,
        ksd_real_short_per_mask,
        ksd_real_short_prior_per_mask,
        pval_real_short_per_mask,
        pval_real_short_prior_per_mask,
    )
    print("Saved real short random-mask results.")

    rng, key = next_key(rng)
    print("Evaluating real KSDs (long, random masks)")
    ksds_real_long, pvals_real_long = run_batched_eval(
        real_long_dataset, key, eval_ksd, args.batch_size
    )
    rng, key = next_key(rng)
    print("Evaluating real KSD priors (long, random masks)")
    ksds_real_long_prior, pvals_real_long_prior = run_batched_eval(
        real_long_dataset, key, eval_ksd_prior, args.batch_size
    )

    masks_real_long = np.asarray(real_long_dataset["model_mask"])
    ksd_real_long_per_mask = group_by_mask(
        ksds_real_long, masks_real_long, support_np
    )
    ksd_real_long_prior_per_mask = group_by_mask(
        ksds_real_long_prior, masks_real_long, support_np
    )
    pval_real_long_per_mask = group_by_mask(
        pvals_real_long, masks_real_long, support_np
    )
    pval_real_long_prior_per_mask = group_by_mask(
        pvals_real_long_prior, masks_real_long, support_np
    )

    save_mask_results(
        out_dir / "real_random_long_ksd_per_mask.npz",
        support_np,
        ksd_real_long_per_mask,
        ksd_real_long_prior_per_mask,
        pval_real_long_per_mask,
        pval_real_long_prior_per_mask,
    )
    print("Saved real long random-mask results.")

    ksd_real_short_model = group_by_mask_list(
        ksds_real_short, masks_real_short, mask_list
    )
    ksd_real_short_prior_model = group_by_mask_list(
        ksds_real_short_prior, masks_real_short, mask_list
    )
    ksd_real_long_model = group_by_mask_list(
        ksds_real_long, masks_real_long, mask_list
    )
    ksd_real_long_prior_model = group_by_mask_list(
        ksds_real_long_prior, masks_real_long, mask_list
    )
    ratio_real_short = normalize_by_prior(
        ksd_real_short_model, ksd_real_short_prior_model
    )
    ratio_real_long = normalize_by_prior(
        ksd_real_long_model, ksd_real_long_prior_model
    )
    plot_violin(
        [
            trim_values(ratio_real_short, args.real_trim_quantile),
            trim_values(ratio_real_long, args.real_trim_quantile),
        ],
        labels,
        out_dir / "real_random_violin.png",
        "Real data rKSD ratios by model (random)",
        "rKSD",
        group_labels=["Short", "Long"],
        colors=["#4C72B0", "#DD8452"],
    )
    plot_violin(
        [
            trim_values(ratio_real_short, args.real_trim_quantile),
            trim_values(ratio_real_long, args.real_trim_quantile),
        ],
        labels,
        out_dir / "real_random_violin.svg",
        "Real data rKSD ratios by model (random)",
        "rKSD",
        group_labels=["Short", "Long"],
        colors=["#4C72B0", "#DD8452"],
    )
    print("Saved real random-mask violin plot.")

    rng, key = next_key(rng)
    print("Sampling selected masks (short)")
    model_masks_selected_short = sample_selected_masks(
        model, key, real_short_dataset, args.selected_mask_prior
    )
    rng, key = next_key(rng)
    print("Sampling selected masks (long)")
    model_masks_selected_long = sample_selected_masks(
        model, key, real_long_dataset, args.selected_mask_prior
    )

    real_short_selected = dict(real_short_dataset)
    real_short_selected["model_mask"] = model_masks_selected_short
    real_long_selected = dict(real_long_dataset)
    real_long_selected["model_mask"] = model_masks_selected_long

    rng, key = next_key(rng)
    print("Evaluating real KSDs (short, selected masks)")
    ksds_real_short_sel, pvals_real_short_sel = run_batched_eval(
        real_short_selected, key, eval_ksd, args.batch_size
    )
    rng, key = next_key(rng)
    print("Evaluating real KSD priors (short, selected masks)")
    ksds_real_short_sel_prior, pvals_real_short_sel_prior = run_batched_eval(
        real_short_selected, key, eval_ksd_prior, args.batch_size
    )

    masks_real_short_sel = np.asarray(real_short_selected["model_mask"])

    ksd_real_short_sel_per_mask = group_by_mask(
        ksds_real_short_sel, masks_real_short_sel, support_np
    )
    ksd_real_short_sel_prior_per_mask = group_by_mask(
        ksds_real_short_sel_prior, masks_real_short_sel, support_np
    )
    pval_real_short_sel_per_mask = group_by_mask(
        pvals_real_short_sel, masks_real_short_sel, support_np
    )
    pval_real_short_sel_prior_per_mask = group_by_mask(
        pvals_real_short_sel_prior, masks_real_short_sel, support_np
    )
    save_mask_results(
        out_dir / "real_selected_short_ksd_per_mask.npz",
        support_np,
        ksd_real_short_sel_per_mask,
        ksd_real_short_sel_prior_per_mask,
        pval_real_short_sel_per_mask,
        pval_real_short_sel_prior_per_mask,
    )
    print("Saved real short selected-mask results.")

    rng, key = next_key(rng)
    print("Evaluating real KSDs (long, selected masks)")
    ksds_real_long_sel, pvals_real_long_sel = run_batched_eval(
        real_long_selected, key, eval_ksd, args.batch_size
    )
    rng, key = next_key(rng)
    print("Evaluating real KSD priors (long, selected masks)")
    ksds_real_long_sel_prior, pvals_real_long_sel_prior = run_batched_eval(
        real_long_selected, key, eval_ksd_prior, args.batch_size
    )

    masks_real_long_sel = np.asarray(real_long_selected["model_mask"])
    ksd_real_long_sel_per_mask = group_by_mask(
        ksds_real_long_sel, masks_real_long_sel, support_np
    )
    ksd_real_long_sel_prior_per_mask = group_by_mask(
        ksds_real_long_sel_prior, masks_real_long_sel, support_np
    )
    pval_real_long_sel_per_mask = group_by_mask(
        pvals_real_long_sel, masks_real_long_sel, support_np
    )
    pval_real_long_sel_prior_per_mask = group_by_mask(
        pvals_real_long_sel_prior, masks_real_long_sel, support_np
    )

    save_mask_results(
        out_dir / "real_selected_long_ksd_per_mask.npz",
        support_np,
        ksd_real_long_sel_per_mask,
        ksd_real_long_sel_prior_per_mask,
        pval_real_long_sel_per_mask,
        pval_real_long_sel_prior_per_mask,
    )
    print("Saved real long selected-mask results.")

    ksd_real_short_sel_model = group_by_mask_list(
        ksds_real_short_sel, masks_real_short_sel, mask_list
    )
    ksd_real_short_sel_prior_model = group_by_mask_list(
        ksds_real_short_sel_prior, masks_real_short_sel, mask_list
    )
    ksd_real_long_sel_model = group_by_mask_list(
        ksds_real_long_sel, masks_real_long_sel, mask_list
    )
    ksd_real_long_sel_prior_model = group_by_mask_list(
        ksds_real_long_sel_prior, masks_real_long_sel, mask_list
    )
    ratio_real_short_sel = normalize_by_prior(
        ksd_real_short_sel_model, ksd_real_short_sel_prior_model
    )
    ratio_real_long_sel = normalize_by_prior(
        ksd_real_long_sel_model, ksd_real_long_sel_prior_model
    )

    plot_violin(
        [
            trim_values(ratio_real_short_sel, args.real_trim_quantile),
            trim_values(ratio_real_long_sel, args.real_trim_quantile),
        ],
        labels,
        out_dir / "real_selected_violin.png",
        "Real data rKSD by model (selected)",
        "rKSD",
        group_labels=["Short", "Long"],
        colors=["#4C72B0", "#DD8452"],
    )
    plot_violin(
        [
            trim_values(ratio_real_short_sel, args.real_trim_quantile),
            trim_values(ratio_real_long_sel, args.real_trim_quantile),
        ],
        labels,
        out_dir / "real_selected_violin.svg",
        "Real data rKSD by model (selected)",
        "rKSD",
        group_labels=["Short", "Long"],
        colors=["#4C72B0", "#DD8452"],
    )
    print("Saved real selected-mask violin plot.")

    joint_short = [
        trim_values(ksd_short_ratio, args.synthetic_trim_quantile),
        trim_values(ratio_real_short, args.real_trim_quantile),
        trim_values(ratio_real_short_sel, args.real_trim_quantile),
    ]
    joint_long = [
        trim_values(ksd_long_ratio, args.synthetic_trim_quantile),
        trim_values(ratio_real_long, args.real_trim_quantile),
        trim_values(ratio_real_long_sel, args.real_trim_quantile),
    ]
    group_labels = ["Synthetic", "Real random", "Real selected"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    plot_violin(
        joint_short,
        labels,
        out_dir / "joint_violin_short.png",
        "rKSD by model (short sequence)",
        "rKSD",
        group_labels=group_labels,
        colors=colors,
        figsize=(4.2, 2.8),
        violin_width=0.25,
        group_span=0.6,
    )
    plot_violin(
        joint_short,
        labels,
        out_dir / "joint_violin_short.svg",
        "rKSD by model (short sequence)",
        "rKSD",
        group_labels=group_labels,
        colors=colors,
        figsize=(4.2, 2.8),
        violin_width=0.25,
        group_span=0.6,
    )
    plot_violin(
        joint_long,
        labels,
        out_dir / "joint_violin_long.png",
        "rKSD by model (long sequence)",
        "rKSD",
        group_labels=group_labels,
        colors=colors,
        figsize=(4.2, 2.8),
        violin_width=0.25,
        group_span=0.6,
    )
    plot_violin(
        joint_long,
        labels,
        out_dir / "joint_violin_long.svg",
        "rKSD by model (long sequence)",
        "rKSD",
        group_labels=group_labels,
        colors=colors,
        figsize=(4.2, 2.8),
        violin_width=0.25,
        group_span=0.6,
    )
    print("Saved joint violin plots.")

    print(f"Finished. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
