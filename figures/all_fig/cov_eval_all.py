#!/usr/bin/env python3
"""Evaluate PIT coverage for the all-model experiment on synthetic data."""

from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from flax import nnx

from dmri.train.utils import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.style.use("../../dmri/utils/pyloric.mplstyle")


LOGGER = logging.getLogger(__name__)


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


def sequence_tag(base: str, sequence: str) -> str:
    if not base:
        return sequence
    if sequence == "short":
        return base
    suffix = f"_{sequence}"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


_special_token_abbr = {
    "NoddiW": r"No$_{\mathcal{W}}$",
    "NoddiB": r"No$_{\mathcal{B}}$",
    "SandiW": r"Sa$_W$",
    "SandiB": r"Sa$_B$",
    "WatsonStick": r"S$_{\mathcal{W}}$",
    "BinghamStick": r"S$_{\mathcal{B}}$",
    "WatsonZeppelin": r"Z$_{\mathcal{W}}$",
    "BinghamZeppelin": r"Z$_{\mathcal{B}}$",
}

_abbr_token = {
    "Ball": "B",
    "Zeppelin": "Z",
    "Dti": "T",
    "Stick": "S",
    **_special_token_abbr,
}


def _shorthand(model: str) -> str:
    toks = model.split(">")
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        j = i + 1
        while j < len(toks) and toks[j] == t:
            j += 1
        count = j - i
        base = _abbr_token.get(t, t[:1].upper())
        out.append(f"{count}{base}" if count > 1 else base)
        i = j
    return "".join(out)


def _is_preselected(models: list[str]) -> bool:
    allowed = {"Stick", "Zeppelin", "Dti"}
    for model in models:
        toks = model.split(">")
        if toks[0] != "Ball":
            return False
        if len(toks) == 1:
            continue
        tail = toks[1:]
        if any(t != tail[0] for t in tail):
            return False
        if tail[0] not in allowed:
            return False
    return True


def _preselected_name(model: str) -> str:
    if model == "Ball":
        return "Ball"
    toks = model.split(">")
    comp = toks[1]
    count = len(toks) - 1
    comp_abbr = {"Stick": "S", "Zeppelin": "Z", "Dti": "T"}
    return f"B{count}{comp_abbr.get(comp, comp[:1].upper())}"


def _model_type_name(model_type) -> str:
    if hasattr(model_type, "__name__"):
        return model_type.__name__
    name = str(model_type).split(".")[-1].replace("'", "")
    return name.rstrip(">")


def _model_type_names(model) -> list[str]:
    model_types = getattr(model.tokenizer.simulator, "model_types", [])
    names = []
    for model_type in model_types:
        names.append(_model_type_name(model_type))
    return names


def mask_to_label(mask, model_types) -> str:
    names = []
    for i, m in enumerate(mask):
        if i >= len(model_types):
            continue
        if m:
            names.append(_model_type_name(model_types[i]))
    if not names:
        return "None"
    return ">".join(names)


def mask_to_shorthand(mask, model_types) -> str:
    label = mask_to_label(mask, model_types)
    if label == "None":
        return label
    return _shorthand(label)


def _maybe_shorthand_label(label: str) -> str:
    if ">" in label:
        return _shorthand(label)
    return label


def _ensure_unique_labels(labels: list[str]) -> list[str]:
    seen = {}
    unique = []
    for label in labels:
        count = seen.get(label, 0) + 1
        seen[label] = count
        if count == 1:
            unique.append(label)
        else:
            unique.append(f"{label}#{count}")
    return unique


def mask_parameter_count(mask, model_types) -> int:
    mask_arr = np.asarray(mask).astype(bool)
    model_mask = mask_arr[: len(model_types)]
    active = int(model_mask.sum())
    theta_params = 0
    for model_type, is_active in zip(model_types, model_mask):
        if is_active:
            theta_params += int(getattr(model_type, "theta_dim", 0))
    frac_params = max(active - 1, 0)
    return int(theta_params + frac_params)


def _sort_by_param_count(labels, param_counts, *arrays):
    if param_counts is None:
        return labels, param_counts, *arrays
    param_counts = np.asarray(param_counts)
    if param_counts.size == 0:
        return labels, param_counts, *arrays
    label_arr = np.asarray(labels, dtype=object)
    order = np.lexsort((label_arr, param_counts))
    labels_sorted = [str(label_arr[i]) for i in order]
    sorted_arrays = []
    for arr in arrays:
        if arr is None:
            sorted_arrays.append(None)
        else:
            sorted_arrays.append(np.asarray(arr)[order])
    return labels_sorted, param_counts[order], *sorted_arrays


def build_mask_groups(model, model_masks: np.ndarray, max_groups: int | None):
    masks = np.asarray(model_masks)
    unique_masks, counts = np.unique(masks, axis=0, return_counts=True)
    if unique_masks.size == 0:
        return unique_masks, [], np.array([])
    if max_groups is not None and max_groups > 0 and unique_masks.shape[0] > max_groups:
        LOGGER.info(
            "Limiting mask groups to %d of %d unique masks.",
            max_groups,
            unique_masks.shape[0],
        )
        idx = np.argsort(-counts)[:max_groups]
        unique_masks = unique_masks[idx]
    num_models = model.tokenizer.num_models
    model_types = model.tokenizer.simulator.model_types
    full_labels = [mask_to_label(mask[:num_models], model_types) for mask in unique_masks]
    if _is_preselected(full_labels):
        labels = [_preselected_name(label) for label in full_labels]
    else:
        labels = [_shorthand(label) for label in full_labels]
    param_counts = np.array(
        [mask_parameter_count(mask[:num_models], model_types) for mask in unique_masks],
        dtype=int,
    )
    if param_counts.size > 0:
        label_arr = np.asarray(labels, dtype=object)
        order = np.lexsort((label_arr, param_counts))
        unique_masks = unique_masks[order]
        labels = [labels[i] for i in order]
        param_counts = param_counts[order]
    labels = _ensure_unique_labels(labels)
    return unique_masks, labels, param_counts


def load_model(checkpoint_path: Path):
    checkpoint, model, simulator = load_checkpoint(str(checkpoint_path))
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema", checkpoint.get("params", params))
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

        # Use masked sum of theta (matches coverage_eval behavior).
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


def plot_coverage(alpha, coverages, labels, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
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
    ax.legend(frameon=False, fontsize=8)
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
    param_counts: np.ndarray | None = None,
    cmap_name: str = "viridis",
) -> None:
    labels = [_maybe_shorthand_label(str(label)) for label in labels]
    if param_counts is not None and len(param_counts) != len(labels):
        LOGGER.warning(
            "Param count length (%d) does not match labels (%d); skipping colormap.",
            len(param_counts),
            len(labels),
        )
        param_counts = None
    labels, param_counts, errors, counts = _sort_by_param_count(
        labels, param_counts, errors, counts
    )
    positions = np.arange(len(errors))
    max_count = np.max(counts) if counts.size > 0 else 0
    if max_count > 0:
        sizes = 30.0 + 70.0 * (counts / max_count)
    else:
        sizes = np.full_like(errors, 30.0, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    if param_counts is not None and len(param_counts) > 0:
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(
            vmin=float(np.min(param_counts)), vmax=float(np.max(param_counts))
        )
        scatter = ax.scatter(
            positions,
            errors,
            s=sizes,
            c=param_counts,
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.5,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Parameter count")
    else:
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


def plot_per_mask_coverage(
    alpha: np.ndarray,
    coverages: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
    param_counts: np.ndarray | None = None,
    cmap_name: str = "viridis",
) -> None:
    labels = [_maybe_shorthand_label(str(label)) for label in labels]
    if param_counts is not None and len(param_counts) != len(labels):
        LOGGER.warning(
            "Param count length (%d) does not match labels (%d); skipping colormap.",
            len(param_counts),
            len(labels),
        )
        param_counts = None
    labels, param_counts, coverages = _sort_by_param_count(
        labels, param_counts, coverages
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.step(alpha, alpha, where="post", linestyle="--", color="black", label="Ideal")
    if param_counts is not None and len(param_counts) > 0:
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(
            vmin=float(np.min(param_counts)), vmax=float(np.max(param_counts))
        )
        colors = cmap(norm(param_counts))
        for coverage, label, color in zip(coverages, labels, colors):
            ax.step(alpha, coverage, where="post", linewidth=1.6, label=label, color=color)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Parameter count")
    else:
        for coverage, label in zip(coverages, labels):
            ax.step(alpha, coverage, where="post", linewidth=1.6, label=label)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, ncol=2)
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
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if mask_list.size == 0:
        return None
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
    theta_mask_params: np.ndarray | None = None,
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
        theta_mask_params=theta_mask_params,
        theta_mask_coverage=theta_mask_coverage,
    )


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
            computed = compute_calibration_error(alpha, coverage, hyperparameters, is_overall)
            if computed is None:
                return
            calib_hyper, calib_error = computed

        if calib_hyper.size > 0 and calib_error.size > 0:
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
            theta_mask_params = None
            if "theta_mask_params" in data.files:
                params_raw = np.atleast_1d(data["theta_mask_params"])
                if not (params_raw.size == 1 and params_raw[0] is None):
                    theta_mask_params = np.asarray(params_raw)
            if errors.size > 0 and counts.size > 0 and labels_raw.size > 0:
                if not (labels_raw.size == 1 and labels_raw[0] is None):
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
                        param_counts=theta_mask_params,
                    )
                    LOGGER.info("Saved theta calibration error plot from %s", npz_path)

        if "theta_mask_coverage" in data.files and "theta_mask_labels" in data.files:
            coverages = np.asarray(data["theta_mask_coverage"])
            labels_raw = np.atleast_1d(data["theta_mask_labels"])
            theta_mask_params = None
            if "theta_mask_params" in data.files:
                params_raw = np.atleast_1d(data["theta_mask_params"])
                if not (params_raw.size == 1 and params_raw[0] is None):
                    theta_mask_params = np.asarray(params_raw)
            if (
                coverages.size > 0
                and labels_raw.size > 0
                and not (labels_raw.size == 1 and labels_raw[0] is None)
                and not (coverages.size == 1 and coverages.item() is None)
            ):
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
                    param_counts=theta_mask_params,
                )
                LOGGER.info("Saved theta mask coverage plot from %s", npz_path)


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
    max_mask_groups: int | None,
) -> jax.Array:
    if isinstance(simulator, (list, tuple)):
        if sim_index >= len(simulator):
            raise ValueError("Requested sequence index is unavailable in simulator list.")
        sim_fn = simulator[sim_index]
    else:
        if sim_index != 0:
            raise ValueError("Long-sequence evaluation requires a simulator list.")
        sim_fn = simulator
    datasets = []
    labels = []
    hypers = []
    overall_flags = []

    if include_overall:
        rng, key = jax.random.split(rng)
        LOGGER.info("Sampling synthetic data for overall (%s)", tag)
        datasets.append(generate_synthetic(sim_fn, key, synthetic_samples, None))
        labels.append("overall")
        hypers.append(np.nan)
        overall_flags.append(True)

    for hyper in hyperparameters:
        rng, key = jax.random.split(rng)
        LOGGER.info("Sampling synthetic data for p=%.3f (%s)", hyper, tag)
        datasets.append(generate_synthetic(sim_fn, key, synthetic_samples, hyper))
        labels.append(f"p={hyper:g}")
        hypers.append(hyper)
        overall_flags.append(False)

    if mode == "mask":
        pit_fn = build_mask_pit_fn(model, pit_samples)
    elif mode == "theta":
        pit_fn = build_theta_pit_fn(model, pit_samples)
    else:
        raise ValueError(f"Unknown mode {mode}")
    expected_coverage = build_expected_coverage_fn(pit_fn, batch_size)

    all_coverages = []
    all_pits = []
    alpha = None
    for label, dataset in zip(labels, datasets):
        rng, key = jax.random.split(rng)
        LOGGER.info("Computing %s coverage for %s (%s)", mode, label, tag)
        alpha, coverage, pits = expected_coverage(dataset, key)
        all_coverages.append(np.asarray(coverage))
        all_pits.append(np.asarray(pits))

    coverage_arr = np.stack(all_coverages, axis=0)
    pits_arr = np.stack(all_pits, axis=0)
    alpha_arr = np.asarray(alpha)
    hyper_arr = np.asarray(hypers, dtype=np.float32)
    overall_arr = np.asarray(overall_flags, dtype=bool)

    calibration_error = None
    calibration_error_hyper = None
    theta_mask_errors = None
    theta_mask_counts = None
    theta_mask_labels = None
    theta_mask_support = None
    theta_mask_params = None
    theta_mask_coverage = None

    plot_path = out_dir / f"pit_coverage_{mode}_{tag}.png"
    plot_coverage(alpha_arr, coverage_arr, labels, plot_path, f"{mode} PIT coverage ({tag})")

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
        mask_idx = np.where(overall_arr)[0]
        if mask_idx.size == 0:
            mask_idx = np.array([0], dtype=int)
        selected_idx = int(mask_idx[0])
        mask_dataset = datasets[selected_idx]
        mask_list, label_list, param_counts = build_mask_groups(
            model, np.asarray(mask_dataset["model_mask"]), max_mask_groups
        )
        if mask_list.size == 0:
            LOGGER.info("Skipping theta mask coverage; no masks available.")
        else:
            theta_mask_support = mask_list
            theta_mask_labels = label_list
            theta_mask_params = param_counts
            mask_prior_hyper = None
            if hyper_arr.size > 0 and not np.isnan(hyper_arr[0]):
                mask_prior_hyper = float(hyper_arr[0])
            mask_coverages = []
            mask_errors = []
            mask_counts = []
            LOGGER.info(
                "Computing theta coverage for %d mask group(s) (%s).",
                mask_list.shape[0],
                tag,
            )
            for idx, (mask, label) in enumerate(zip(mask_list, label_list), start=1):
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
                param_counts=theta_mask_params,
            )
            if theta_mask_coverage is not None:
                coverage_plot = out_dir / f"pit_coverage_theta_masks_{tag}.png"
                plot_per_mask_coverage(
                    alpha_arr,
                    theta_mask_coverage,
                    theta_mask_labels,
                    coverage_plot,
                    f"Theta PIT coverage by mask ({tag})",
                    param_counts=theta_mask_params,
                )
            LOGGER.info("Saved theta mask calibration error plot for %s.", tag)

    save_coverage(
        out_dir / f"pit_coverage_{mode}_{tag}.npz",
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
        theta_mask_params=theta_mask_params,
        theta_mask_coverage=theta_mask_coverage,
    )
    return rng


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PIT coverage for the all-model experiment."
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
        "--output-dir",
        type=Path,
        default=Path("outputs/coverage_eval2"),
        help="Directory to write coverage results and plots.",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=10_000,
        help="Number of synthetic samples for PIT coverage.",
    )
    parser.add_argument(
        "--theta-synthetic-samples",
        type=int,
        default=10_000,
        help="Number of synthetic samples for theta PIT coverage.",
    )
    parser.add_argument(
        "--pit-samples",
        type=int,
        default=100,
        help="Number of mask samples for PIT coverage.",
    )
    parser.add_argument(
        "--theta-pit-samples",
        type=int,
        default=50,
        help="Number of theta samples for PIT coverage.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help="Batch size for PIT coverage evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--mask-hypers",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated mask hyperparameters for synthetic coverage.",
    )
    parser.add_argument(
        "--theta-mask-hyper",
        type=float,
        default=1.0,
        help="Single mask hyperparameter value for theta coverage.",
    )
    parser.add_argument(
        "--skip-overall",
        action="store_true",
        help="Skip overall synthetic samples (no fixed mask hyperparameter).",
    )
    parser.add_argument(
        "--tag",
        default="all",
        help="Base tag used in output filenames.",
    )
    parser.add_argument(
        "--max-mask-groups",
        type=int,
        default=20,
        help="Max unique masks to label for theta plots (<=0 means no limit).",
    )
    parser.add_argument(
        "--from-npz",
        action="append",
        default=[],
        help="Load PIT coverage .npz file(s) and regenerate plots.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    print("Starting coverage evaluation...", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Sequence: {args.sequence}", flush=True)
    print(f"Tag: {args.tag}", flush=True)
    print(f"Mask hypers: {args.mask_hypers}", flush=True)
    print(f"Theta mask hyper: {args.theta_mask_hyper}", flush=True)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maybe_apply_style(repo_root)

    if args.from_npz:
        for npz_path in args.from_npz:
            plot_from_npz(Path(npz_path), out_dir)
        return

    model, simulator = load_model(args.checkpoint)
    if args.sequence in ("long", "both"):
        if not isinstance(simulator, (list, tuple)) or len(simulator) < 2:
            raise ValueError("Expected at least two simulators for short and long data.")
    model_names = _model_type_names(model)
    if model_names:
        model_msg = ", ".join(model_names)
        print(f"Inference model types ({len(model_names)}): {model_msg}", flush=True)
        LOGGER.info(
            "Running inference with %d model types: %s",
            len(model_names),
            model_msg,
        )
    else:
        print("Inference model types: unknown", flush=True)
        LOGGER.info("Running inference with unknown model types.")

    rng = jax.random.PRNGKey(args.seed)
    hyperparameters = parse_float_list(args.mask_hypers)
    mode_list = ["mask", "theta"] if args.mode == "both" else [args.mode]
    if args.skip_overall and not hyperparameters and "mask" in mode_list:
        raise ValueError("No datasets requested; provide --mask-hypers or omit --skip-overall.")

    max_mask_groups = args.max_mask_groups
    if max_mask_groups is not None and max_mask_groups <= 0:
        max_mask_groups = None

    theta_hyperparameters = [args.theta_mask_hyper]
    theta_include_overall = False
    sequence_map = {"short": [0], "long": [1], "both": [0, 1]}
    for sim_index in sequence_map[args.sequence]:
        tag = sequence_tag(args.tag, "long" if sim_index == 1 else "short")
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
                    max_mask_groups,
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
                    theta_hyperparameters,
                    theta_include_overall,
                    out_dir,
                    tag,
                    mode,
                    max_mask_groups,
                )

    LOGGER.info("Finished. Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
