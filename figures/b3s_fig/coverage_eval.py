#!/usr/bin/env python3
"""Evaluate PIT coverage for masks and theta on synthetic data."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import nnx

from dmri.train.utils import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.style.use("../../dmri/utils/pyloric.mplstyle")


LOGGER = logging.getLogger(__name__)


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


def build_label_masks(label_map: dict, expected_dim: int) -> dict[str, np.ndarray]:
    label_masks = {}
    for label, mask_arr in label_map.items():
        arr = np.asarray(mask_arr, dtype=bool)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != expected_dim:
            LOGGER.info(
                "Skipping label %s: expected mask length %d, got %d",
                label,
                expected_dim,
                arr.shape[1],
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


def maybe_apply_style(repo_root: Path) -> None:
    style_path = repo_root / "dmri" / "utils" / "pyloric.mplstyle"
    if style_path.exists():
        plt.style.use(style_path.as_posix())
        return
    LOGGER.info("Matplotlib style not found at %s", style_path)


def parse_float_list(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        return []
    return [float(value) for value in values]


def load_model(checkpoint_path: str):
    checkpoint, model, simulator = load_checkpoint(checkpoint_path)
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema")
    state = checkpoint.get("model_state", state)
    model = nnx.merge(graphdef, params, state)
    model.eval()
    return model, simulator


def generate_synthetic(
    sim_fn,
    rng,
    count: int,
    mask_prior_hyper: float | None,
    model_mask: np.ndarray | None = None,
):
    keys = jax.random.split(rng, count)
    if model_mask is not None:
        mask = jnp.asarray(model_mask)
        if mask_prior_hyper is None:
            return jax.vmap(partial(sim_fn, model_mask=mask))(keys)
        hyper = jnp.array([mask_prior_hyper])
        return jax.vmap(
            partial(sim_fn, mask_prior_hyperparameter=hyper, model_mask=mask)
        )(keys)
    if mask_prior_hyper is None:
        return jax.vmap(sim_fn)(keys)
    hyper = jnp.array([mask_prior_hyper])
    return jax.vmap(partial(sim_fn, mask_prior_hyperparameter=hyper))(keys)


def build_mask_pit_fn(model, n: int):
    def log_prob_mask_pit(data, rng):
        rng_samp, rng_tie, rng_jit = jax.random.split(rng, 3)

        keys = jax.random.split(rng_samp, n)
        sample_fn = lambda k: model.sample_mask(
            k, acq=data["acq"], x=data["x"], mask_prior=data["mask_prior"]
        )
        masks = jax.vmap(sample_fn)(keys)

        logprob_fn = lambda m: model.log_prob_mask(
            m, acq=data["acq"], x=data["x"], mask_prior=data["mask_prior"]
        )

        lp_true = jnp.reshape(logprob_fn(data["model_mask"]), (-1,)).sum()

        lp_samp = jax.vmap(logprob_fn)(masks)
        lp_samp = jnp.reshape(lp_samp, (n, -1)).sum(axis=1)

        eq = jnp.isclose(lp_samp, lp_true, rtol=1e-3, atol=1e-3)
        lt = (lp_samp < lp_true) & (~eq)

        n_lt = jnp.sum(lt)
        n_eq = jnp.sum(eq)

        tie = jax.random.randint(rng_tie, (), 0, n_eq + 1)
        r = n_lt + tie

        v = jax.random.uniform(rng_jit, (), minval=0.0, maxval=1.0)
        pit = (r + v) / (n + 1)
        return pit

    return jax.jit(log_prob_mask_pit)


def build_theta_pit_fn(model, n: int):
    def log_prob_theta_pit(data, rng):
        rng_samp, rng_tie, rng_jit = jax.random.split(rng, 3)

        keys = jax.random.split(rng_samp, n)
        sample_fn = lambda k: model.sample_theta(
            k, acq=data["acq"], x=data["x"], model_mask=data["model_mask"]
        )
        theta_mask = model.tokenizer.simulator.theta_mask(data["model_mask"])
        samples = jax.vmap(sample_fn)(keys)

        # Use masked sum of theta (matches original notebook behavior).
        logprob_fn = lambda th: jnp.sum(th * theta_mask, axis=-1)

        lp_true = logprob_fn(data["theta"])
        lp_samp = jax.vmap(logprob_fn)(samples)

        eq = jnp.isclose(lp_samp, lp_true, rtol=1e-6, atol=1e-6)
        lt = (lp_samp < lp_true) & (~eq)

        n_lt = jnp.sum(lt)
        n_eq = jnp.sum(eq)

        tie = jax.random.randint(rng_tie, (), 0, n_eq + 1)
        r = n_lt + tie

        v = jax.random.uniform(rng_jit, (), minval=0.0, maxval=1.0)
        pit = (r + v) / (n + 1)
        return pit

    return jax.jit(log_prob_theta_pit)


def build_expected_coverage_fn(pit_fn, batch_size: int):
    def expected_coverage(data, rng):
        batch = data["x"].shape[0]
        keys = jax.random.split(rng, batch)

        def pit_map(args):
            return pit_fn(*args)

        pits = jax.lax.map(pit_map, (data, keys), batch_size=batch_size)

        alpha = jnp.linspace(0.0, 1.0, 101)
        coverage = jnp.mean(pits[:, None] <= alpha[None, :], axis=0)
        return alpha, coverage, pits

    return jax.jit(expected_coverage)


def _plot_coverage_on_ax(ax, alpha, coverages, labels, title: str) -> None:
    ax.step(alpha, alpha, where="post", linestyle="--", color="black", label="Ideal")
    for coverage, label in zip(coverages, labels):
        ax.step(alpha, coverage, where="post", linewidth=1.6, label=label)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def plot_coverage(alpha, coverages, labels, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    _plot_coverage_on_ax(ax, alpha, coverages, labels, title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_coverage_pair(
    short_npz: Path,
    long_npz: Path,
    out_path: Path,
    title_prefix: str,
) -> None:
    short = np.load(short_npz, allow_pickle=True)
    long = np.load(long_npz, allow_pickle=True)
    alpha_short = np.asarray(short["alpha"])
    alpha_long = np.asarray(long["alpha"])
    cov_short = np.asarray(short["coverage"])
    cov_long = np.asarray(long["coverage"])
    labels_short = [str(label) for label in np.atleast_1d(short["labels"])]
    labels_long = [str(label) for label in np.atleast_1d(long["labels"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    _plot_coverage_on_ax(
        axes[0], alpha_short, cov_short, labels_short, f"{title_prefix} (short)"
    )
    _plot_coverage_on_ax(
        axes[1], alpha_long, cov_long, labels_long, f"{title_prefix} (long)"
    )
    axes[1].set_ylabel("")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_mask_coverage(
    alpha: np.ndarray,
    coverages: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    _plot_coverage_on_ax(ax, alpha, coverages, labels, title)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_mask_coverage_pair(
    short_npz: Path,
    long_npz: Path,
    out_path: Path,
    title_prefix: str,
) -> None:
    short = np.load(short_npz, allow_pickle=True)
    long = np.load(long_npz, allow_pickle=True)
    if "theta_mask_coverage" not in short.files or "theta_mask_coverage" not in long.files:
        LOGGER.info("Skipping per-mask coverage pair; missing coverage arrays.")
        return

    alpha_short = np.asarray(short["alpha"])
    alpha_long = np.asarray(long["alpha"])
    cov_short = np.asarray(short["theta_mask_coverage"])
    cov_long = np.asarray(long["theta_mask_coverage"])
    labels_short = [str(label) for label in np.atleast_1d(short["theta_mask_labels"])]
    labels_long = [str(label) for label in np.atleast_1d(long["theta_mask_labels"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    _plot_coverage_on_ax(
        axes[0],
        alpha_short,
        cov_short,
        labels_short,
        f"{title_prefix} (short)",
    )
    _plot_coverage_on_ax(
        axes[1],
        alpha_long,
        cov_long,
        labels_long,
        f"{title_prefix} (long)",
    )
    axes[1].set_ylabel("")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration_error(
    hyperparameters: np.ndarray,
    errors: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(hyperparameters, errors, marker="o", linewidth=1.6, color="#4C72B0")
    ax.set_xlabel("Mask prior hyperparameter")
    ax.set_ylabel("Mean |coverage - nominal|")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_mask_calibration_error(
    errors: np.ndarray,
    counts: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    positions = np.arange(len(errors))
    max_count = np.max(counts) if counts.size > 0 else 0
    if max_count > 0:
        sizes = 30.0 + 70.0 * (counts / max_count)
    else:
        sizes = np.full_like(errors, 30.0, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.scatter(
        positions,
        errors,
        s=sizes,
        color="#4C72B0",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
    ax.set_xlabel("Model configuration")
    ax.set_ylabel("Mean |coverage - nominal|")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_npz_mode_tag(path: Path) -> tuple[str | None, str | None]:
    stem = path.stem
    prefix = "pit_coverage_"
    if not stem.startswith(prefix):
        return None, None
    remainder = stem[len(prefix) :]
    parts = remainder.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], None
    return None, None


def compute_calibration_error(
    alpha: np.ndarray,
    coverage: np.ndarray,
    hyperparameters: np.ndarray,
    is_overall: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if hyperparameters.size == 0:
        return None
    if is_overall is None:
        mask_idx = np.where(~np.isnan(hyperparameters))[0]
    else:
        mask_idx = np.where(~is_overall)[0]
    if mask_idx.size == 0:
        return None
    alpha_grid = alpha[None, :]
    errors = np.mean(np.abs(coverage[mask_idx] - alpha_grid), axis=1)
    return hyperparameters[mask_idx], errors


def compute_per_mask_errors(
    pits: np.ndarray,
    model_masks: np.ndarray,
    alpha: np.ndarray,
    mask_list: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    masks = np.asarray(model_masks)
    errors = []
    counts = []
    for mask in mask_list:
        subset = np.all(masks == mask, axis=1)
        subset_pits = pits[subset]
        counts.append(subset_pits.shape[0])
        if subset_pits.shape[0] == 0:
            errors.append(np.nan)
        else:
            coverage = np.mean(subset_pits[:, None] <= alpha[None, :], axis=0)
            errors.append(np.mean(np.abs(coverage - alpha)))
    return mask_list, np.asarray(errors), np.asarray(counts), labels


def compute_per_mask_coverage(
    pits: np.ndarray,
    model_masks: np.ndarray,
    alpha: np.ndarray,
    mask_list: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    masks = np.asarray(model_masks)
    coverages = []
    for mask in mask_list:
        subset = np.all(masks == mask, axis=1)
        subset_pits = pits[subset]
        if subset_pits.shape[0] == 0:
            coverages.append(np.full_like(alpha, np.nan, dtype=np.float32))
        else:
            coverage = np.mean(subset_pits[:, None] <= alpha[None, :], axis=0)
            coverages.append(np.asarray(coverage))
    return mask_list, np.stack(coverages, axis=0), labels


def _mask_to_code(mask: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in np.asarray(mask, dtype=bool))


def build_theta_mask_support(
    model_masks: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    masks = np.asarray(model_masks, dtype=bool)
    if masks.size == 0:
        return masks, []

    unique_masks = np.unique(masks, axis=0)
    if unique_masks.size == 0:
        return unique_masks, []

    label_masks = build_label_masks(LABELS_MASK, unique_masks.shape[1])
    mask_list, label_list = expand_label_masks(label_masks)
    if mask_list.size == 0:
        labels = [_mask_to_code(mask) for mask in unique_masks]
        return unique_masks, labels

    support_set = {tuple(mask.tolist()) for mask in unique_masks}
    ordered_masks = []
    ordered_labels = []
    for mask, label in zip(mask_list, label_list):
        key = tuple(mask.tolist())
        if key in support_set:
            ordered_masks.append(mask)
            ordered_labels.append(label)

    covered = {tuple(mask.tolist()) for mask in ordered_masks}
    for mask in unique_masks:
        key = tuple(mask.tolist())
        if key in covered:
            continue
        ordered_masks.append(mask)
        ordered_labels.append(_mask_to_code(mask))

    return np.stack(ordered_masks, axis=0), ordered_labels


def plot_from_npz(npz_path: Path, out_dir: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    alpha = np.asarray(data["alpha"])
    coverage = np.asarray(data["coverage"])
    labels_raw = data["labels"] if "labels" in data.files else []
    labels = [str(label) for label in np.atleast_1d(labels_raw)]
    hyperparameters = (
        np.asarray(data["hyperparameters"]) if "hyperparameters" in data.files else np.array([])
    )
    is_overall = (
        np.asarray(data["is_overall"]).astype(bool) if "is_overall" in data.files else None
    )

    if coverage.ndim == 1:
        coverage = coverage[None, :]

    mode, tag = parse_npz_mode_tag(npz_path)
    title = f"{mode} PIT coverage ({tag})" if mode and tag else npz_path.stem
    plot_name = (
        f"pit_coverage_{mode}_{tag}.png"
        if mode and tag
        else f"{npz_path.stem}.png"
    )
    plot_path = out_dir / plot_name
    plot_coverage(alpha, coverage, labels, plot_path, title)
    LOGGER.info("Saved plot from %s", npz_path)

    if mode == "mask":
        if "calibration_error" in data.files and "calibration_error_hyper" in data.files:
            calib_hyper = np.asarray(data["calibration_error_hyper"])
            calib_error = np.asarray(data["calibration_error"])
        else:
            computed = compute_calibration_error(
                alpha, coverage, hyperparameters, is_overall
            )
            if computed is None:
                return
            calib_hyper, calib_error = computed

        error_name = (
            f"pit_coverage_mask_error_{tag}.png"
            if tag
            else f"{npz_path.stem}_error.png"
        )
        plot_calibration_error(
            calib_hyper,
            calib_error,
            out_dir / error_name,
            f"Mask calibration error ({tag})" if tag else "Mask calibration error",
        )
        LOGGER.info("Saved calibration error plot from %s", npz_path)
    elif mode == "theta":
        if (
            "theta_mask_errors" in data.files
            and "theta_mask_counts" in data.files
            and "theta_mask_labels" in data.files
        ):
            errors = np.asarray(data["theta_mask_errors"])
            counts = np.asarray(data["theta_mask_counts"])
            labels_raw = np.atleast_1d(data["theta_mask_labels"])
            if errors.size == 0 or counts.size == 0 or labels_raw.size == 0:
                return
            if labels_raw.size == 1 and labels_raw[0] is None:
                return
            labels = [str(label) for label in labels_raw]
            error_name = (
                f"pit_coverage_theta_mask_error_{tag}.png"
                if tag
                else f"{npz_path.stem}_mask_error.png"
            )
            plot_per_mask_calibration_error(
                errors,
                counts,
                labels,
                out_dir / error_name,
                f"Theta calibration error by mask ({tag})"
                if tag
                else "Theta calibration error by mask",
            )
            LOGGER.info("Saved theta calibration error plot from %s", npz_path)

        if "theta_mask_coverage" in data.files and "theta_mask_labels" in data.files:
            coverages = np.asarray(data["theta_mask_coverage"])
            labels_raw = np.atleast_1d(data["theta_mask_labels"])
            labels = [str(label) for label in labels_raw]
            coverage_name = (
                f"pit_coverage_theta_masks_{tag}.png"
                if tag
                else f"{npz_path.stem}_masks.png"
            )
            plot_per_mask_coverage(
                alpha,
                coverages,
                labels,
                out_dir / coverage_name,
                f"Theta PIT coverage by mask ({tag})"
                if tag
                else "Theta PIT coverage by mask",
            )
            LOGGER.info("Saved theta mask coverage plot from %s", npz_path)

def save_coverage(
    out_path: Path,
    alpha: np.ndarray,
    coverages: np.ndarray,
    pits: np.ndarray,
    labels: list[str],
    hyperparameters: np.ndarray,
    is_overall: np.ndarray,
    calibration_error: np.ndarray | None = None,
    calibration_error_hyper: np.ndarray | None = None,
    theta_mask_errors: np.ndarray | None = None,
    theta_mask_counts: np.ndarray | None = None,
    theta_mask_labels: list[str] | None = None,
    theta_mask_support: np.ndarray | None = None,
    theta_mask_coverage: np.ndarray | None = None,
) -> None:
    np.savez_compressed(
        out_path,
        alpha=alpha,
        coverage=coverages,
        pits=pits,
        labels=np.asarray(labels),
        hyperparameters=hyperparameters,
        is_overall=is_overall,
        calibration_error=calibration_error,
        calibration_error_hyper=calibration_error_hyper,
        theta_mask_errors=theta_mask_errors,
        theta_mask_counts=theta_mask_counts,
        theta_mask_labels=np.asarray(theta_mask_labels)
        if theta_mask_labels is not None
        else None,
        theta_mask_support=theta_mask_support,
        theta_mask_coverage=theta_mask_coverage,
    )


def run_coverage(
    model,
    simulator,
    rng,
    sim_index: int,
    synthetic_samples: int,
    pit_samples: int,
    batch_size: int,
    hyperparameters: list[float],
    include_overall: bool,
    out_dir: Path,
    tag: str,
    mode: str,
) -> jax.Array:
    sim_fn = simulator[sim_index]
    dataset_specs = []
    labels = []
    hypers = []
    overall_flags = []

    if include_overall:
        dataset_specs.append(("overall", None, True))

    for hyper in hyperparameters:
        dataset_specs.append((f"p={hyper:g}", hyper, False))

    if mode == "mask":
        pit_fn = build_mask_pit_fn(model, pit_samples)
    elif mode == "theta":
        pit_fn = build_theta_pit_fn(model, pit_samples)
    else:
        raise ValueError(f"Unknown mode {mode}")

    expected_coverage = build_expected_coverage_fn(pit_fn, batch_size)

    total_datasets = len(dataset_specs)
    LOGGER.info(
        "Running %s coverage for %d dataset(s) (%s).",
        mode,
        total_datasets,
        tag,
    )
    all_coverages = []
    all_pits = []
    theta_mask_errors = None
    theta_mask_counts = None
    theta_mask_labels = None
    theta_mask_support = None
    theta_mask_coverage = None
    selected_mask_data = None
    selected_mask_hyper = None
    alpha = None
    for idx, (label, hyper, is_overall) in enumerate(dataset_specs, start=1):
        LOGGER.info("Dataset %d/%d: %s (%s)", idx, total_datasets, label, tag)
        labels.append(label)
        if is_overall:
            hypers.append(np.nan)
            overall_flags.append(True)
            rng, key = jax.random.split(rng)
            LOGGER.info("Sampling synthetic data for %s overall (%s)", mode, tag)
            dataset = generate_synthetic(sim_fn, key, synthetic_samples, None)
        else:
            hypers.append(hyper)
            overall_flags.append(False)
            rng, key = jax.random.split(rng)
            LOGGER.info("Sampling synthetic data for %s p=%.3f (%s)", mode, hyper, tag)
            dataset = generate_synthetic(sim_fn, key, synthetic_samples, hyper)

        rng, key = jax.random.split(rng)
        LOGGER.info("Computing %s coverage for %s (%s)", mode, label, tag)
        alpha, coverage, pits = expected_coverage(dataset, key)
        coverage_host = np.asarray(coverage)
        pits_host = np.asarray(pits)
        all_coverages.append(coverage_host)
        all_pits.append(pits_host)
        if mode == "theta" and selected_mask_data is None and (
            is_overall or not include_overall
        ):
            selected_mask_data = np.asarray(dataset["model_mask"])
            selected_mask_hyper = None if is_overall else hyper
        del dataset, coverage, pits
        LOGGER.info("Finished dataset %d/%d: %s (%s)", idx, total_datasets, label, tag)

    coverage_arr = np.stack(all_coverages, axis=0)
    pits_arr = np.stack(all_pits, axis=0)
    alpha_arr = np.asarray(alpha)
    hyper_arr = np.asarray(hypers, dtype=np.float32)
    overall_arr = np.asarray(overall_flags, dtype=bool)

    plot_path = out_dir / f"pit_coverage_{mode}_{tag}.png"
    plot_coverage(alpha_arr, coverage_arr, labels, plot_path, f"{mode} PIT coverage ({tag})")

    save_path = out_dir / f"pit_coverage_{mode}_{tag}.npz"
    calibration_error = None
    calibration_error_hyper = None
    if mode == "mask":
        mask_idx = np.where(~overall_arr)[0]
        if mask_idx.size > 0:
            alpha_grid = alpha_arr[None, :]
            calibration_error = np.mean(
                np.abs(coverage_arr[mask_idx] - alpha_grid), axis=1
            )
            calibration_error_hyper = hyper_arr[mask_idx]
            error_plot = out_dir / f"pit_coverage_mask_error_{tag}.png"
            plot_calibration_error(
                calibration_error_hyper,
                calibration_error,
                error_plot,
                f"Mask calibration error ({tag})",
            )
        else:
            LOGGER.info("Skipping calibration error plot; no hyperparameter datasets.")
    elif mode == "theta":
        if selected_mask_data is None:
            LOGGER.info("Skipping theta mask coverage; no mask dataset available.")
        else:
            masks = np.asarray(selected_mask_data)
            mask_list, label_list = build_theta_mask_support(masks)
            if mask_list.size == 0:
                LOGGER.info("Skipping theta mask coverage; no masks available.")
            else:
                theta_mask_support = mask_list
                theta_mask_labels = label_list
                mask_coverages = []
                mask_errors = []
                mask_counts = []
                mask_prior_hyper = selected_mask_hyper
                LOGGER.info(
                    "Computing theta coverage for %d mask(s) (%s).",
                    mask_list.shape[0],
                    tag,
                )
                for idx, (mask, label) in enumerate(
                    zip(mask_list, label_list), start=1
                ):
                    LOGGER.info(
                        "Theta mask %d/%d: %s (%s)",
                        idx,
                        mask_list.shape[0],
                        label,
                        tag,
                    )
                    rng, key = jax.random.split(rng)
                    dataset = generate_synthetic(
                        sim_fn,
                        key,
                        synthetic_samples,
                        mask_prior_hyper,
                        model_mask=mask,
                    )
                    rng, key = jax.random.split(rng)
                    _, coverage, pits = expected_coverage(dataset, key)
                    coverage_host = np.asarray(coverage)
                    pits_host = np.asarray(pits)
                    mask_coverages.append(coverage_host)
                    mask_counts.append(pits_host.shape[0])
                    mask_errors.append(np.mean(np.abs(coverage_host - alpha_arr)))
                    del dataset, coverage, pits

                theta_mask_coverage = np.stack(mask_coverages, axis=0)
                theta_mask_counts = np.asarray(mask_counts)
                theta_mask_errors = np.asarray(mask_errors)
                theta_plot = out_dir / f"pit_coverage_theta_mask_error_{tag}.png"
                plot_per_mask_calibration_error(
                    theta_mask_errors,
                    theta_mask_counts,
                    theta_mask_labels,
                    theta_plot,
                    f"Theta calibration error by mask ({tag})",
                )
                coverage_plot = out_dir / f"pit_coverage_theta_masks_{tag}.png"
                plot_per_mask_coverage(
                    alpha_arr,
                    theta_mask_coverage,
                    theta_mask_labels,
                    coverage_plot,
                    f"Theta PIT coverage by mask ({tag})",
                )
                LOGGER.info("Saved theta mask calibration error plot for %s.", tag)

    save_coverage(
        save_path,
        alpha_arr,
        coverage_arr,
        pits_arr,
        labels,
        hyper_arr,
        overall_arr,
        calibration_error=calibration_error,
        calibration_error_hyper=calibration_error_hyper,
        theta_mask_errors=theta_mask_errors,
        theta_mask_counts=theta_mask_counts,
        theta_mask_labels=theta_mask_labels,
        theta_mask_support=theta_mask_support,
        theta_mask_coverage=theta_mask_coverage,
    )
    return rng


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PIT coverage on synthetic data.")
    parser.add_argument(
        "--checkpoint",
        default="/home/macke/mgloeckler90/dmri/results/updated5_b3s_2_4_6_128",
        help="Path to the training run directory with checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/coverage_eval2",
        help="Directory to write coverage results and plots.",
    )
    parser.add_argument(
        "--sequence",
        choices=("short", "long", "both"),
        default="both",
        help="Which acquisition sequence to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=("mask", "theta", "both"),
        default="mask",
        help="Which PIT coverage to evaluate.",
    )
    parser.add_argument("--synthetic-samples", type=int, default=10000)
    parser.add_argument("--theta-synthetic-samples", type=int, default=10000)
    parser.add_argument("--pit-samples", type=int, default=200)
    parser.add_argument("--theta-pit-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mask-hypers",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9,1.0",
        help="Comma-separated mask hyperparameters for synthetic coverage.",
    )
    parser.add_argument(
        "--skip-overall",
        action="store_true",
        help="Skip overall synthetic samples (no fixed mask hyperparameter).",
    )
    parser.add_argument(
        "--from-npz",
        action="append",
        default=[],
        help="Load PIT coverage .npz file(s) and regenerate plots.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maybe_apply_style(repo_root)

    if args.from_npz:
        for npz_path in args.from_npz:
            plot_from_npz(Path(npz_path), out_dir)
        return

    model, simulator = load_model(args.checkpoint)
    if not isinstance(simulator, (list, tuple)) or len(simulator) < 2:
        raise ValueError("Expected at least two simulators for short and long data.")

    rng = jax.random.PRNGKey(args.seed)
    hyperparameters = parse_float_list(args.mask_hypers)
    if args.skip_overall and not hyperparameters:
        raise ValueError("No datasets requested; provide --mask-hypers or omit --skip-overall.")

    sequence_map = {"short": [0], "long": [1], "both": [0, 1]}
    mode_list = ["mask", "theta"] if args.mode == "both" else [args.mode]

    for sim_index in sequence_map[args.sequence]:
        tag = "short" if sim_index == 0 else "long"
        for mode in mode_list:
            if mode == "mask":
                rng = run_coverage(
                    model,
                    simulator,
                    rng,
                    sim_index,
                    args.synthetic_samples,
                    args.pit_samples,
                    args.batch_size,
                    hyperparameters,
                    not args.skip_overall,
                    out_dir,
                    tag,
                    mode,
                )
            else:
                rng = run_coverage(
                    model,
                    simulator,
                    rng,
                    sim_index,
                    args.theta_synthetic_samples,
                    args.theta_pit_samples,
                    args.batch_size,
                    hyperparameters,
                    not args.skip_overall,
                    out_dir,
                    tag,
                    mode,
                )

    if args.sequence == "both" and not args.from_npz:
        for mode in mode_list:
            short_npz = out_dir / f"pit_coverage_{mode}_short.npz"
            long_npz = out_dir / f"pit_coverage_{mode}_long.npz"
            if not short_npz.exists() or not long_npz.exists():
                LOGGER.info("Skipping combined plot for %s; missing npz files.", mode)
                continue
            combined_path = out_dir / f"pit_coverage_{mode}_short_long.png"
            plot_coverage_pair(short_npz, long_npz, combined_path, f"{mode} PIT coverage")
            LOGGER.info("Saved combined short/long plot for %s.", mode)
            if mode == "theta":
                coverage_path = out_dir / "pit_coverage_theta_masks_short_long.png"
                plot_per_mask_coverage_pair(
                    short_npz,
                    long_npz,
                    coverage_path,
                    "Theta PIT coverage by mask",
                )
                LOGGER.info("Saved combined short/long theta mask coverage plot.")

    LOGGER.info("Finished. Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
