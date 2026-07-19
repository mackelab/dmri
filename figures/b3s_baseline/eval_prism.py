#!/usr/bin/env python3
"""Evaluate a PRISM checkpoint on Ball3Stick synthetic acquisitions."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from flax import nnx
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmri.simulators.acquisition_scheme import (  # noqa: E402
    acquisition_scheme,
    random_hcp_acquisition,
    random_hcp_large_acquisition,
)
from dmri.simulators.models import Ball3StickSharedDiffusivity  # noqa: E402
from dmri.train.utils import load_checkpoint  # noqa: E402
from dmri.eval.export_metrics import (  # noqa: E402
    multiscale_preconditioned_ksd_and_pvalue,
    _normalize_metric_specs,
    _summarize_metric,
)


SIM_TYPE = Ball3StickSharedDiffusivity
DEFAULT_KSD_CONFIG = REPO_ROOT / "conf_eval" / "export" / "metrics" / "ksd.yaml"


def log_progress(message: str) -> None:
    print(f"[eval_prism] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PRISM checkpoint for the Ball3Stick baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "results" / "updated5_b3s_2_4_6_128",
        help="Path to the PRISM training run directory.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of synthetic test problems per acquisition regime.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Posterior samples per test problem.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=64,
        help="Number of sampler steps for model.sample_theta.",
    )
    parser.add_argument(
        "--log-prob-steps",
        type=int,
        default=256,
        help="Number of steps for model.log_prob_theta.",
    )
    parser.add_argument(
        "--which",
        choices=("latest", "best"),
        default="latest",
        help="Checkpoint variant to restore.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the metrics as JSON.",
    )
    parser.add_argument(
        "--ksd-config",
        type=Path,
        default=DEFAULT_KSD_CONFIG,
        help="Path to the KSD metric config YAML.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional number of synthetic problems to evaluate per batch. Defaults to full batch.",
    )
    parser.add_argument(
        "--real-data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root directory containing the real-data folders.",
    )
    parser.add_argument(
        "--real-short-folder",
        type=str,
        default="data",
        help="Short real-data folder under --real-data-root.",
    )
    parser.add_argument(
        "--real-long-folder",
        type=str,
        default="new_data",
        help="Long real-data folder under --real-data-root.",
    )
    parser.add_argument(
        "--real-samples",
        type=int,
        default=None,
        help="Number of real-data voxels to evaluate. Defaults to --num-trials.",
    )
    parser.add_argument(
        "--real-subset-seed",
        type=int,
        default=0,
        help="Seed for deterministic real-data subset selection.",
    )
    return parser.parse_args()


def load_prism_model(checkpoint_dir: Path, which: str):
    checkpoint, model, _ = load_checkpoint(str(checkpoint_dir), which=which)
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema", checkpoint.get("params", params))
    model = nnx.merge(graphdef, params, state)
    model.eval()
    return model, params


def count_parameters(tree: Any) -> int:
    return int(
        jax.tree_util.tree_reduce(lambda total, leaf: total + leaf.size, tree, 0)
    )


def component_parameter_counts(params: Any) -> dict[str, int]:
    counts = {"total": count_parameters(params)}
    for key in ("encoder", "inference_decoder", "model_decoder", "tokenizer"):
        if key in params:
            counts[key] = count_parameters(params[key])
    return counts


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except ImportError:
        import yaml

        with path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file {path}.")
    return data


def load_ksd_spec(path: Path):
    data = load_yaml_config(path)
    specs = _normalize_metric_specs([data])
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one KSD spec in {path}.")
    return specs[0]


def build_sample_fn(model, num_steps: int, num_samples: int):
    model_mask = jnp.ones((SIM_TYPE.num_compartments(),), dtype=bool)

    @jax.jit
    def sample_fn(rng, acq, x_o):
        keys = jax.random.split(rng, num_samples)
        return jax.vmap(
            partial(model.sample_theta, num_steps=num_steps, last_euler_step=True, t_min=1e-3),
            in_axes=(0, None, None, None),
        )(keys, acq, x_o, model_mask)

    return sample_fn


def build_log_prob_fn(model, num_steps: int):
    model_mask = jnp.ones((SIM_TYPE.num_compartments(),), dtype=bool)

    @jax.jit
    def log_prob_fn(thetas, acq, x_o):
        return jax.vmap(
            partial(model.log_prob_theta, num_steps=num_steps),
            in_axes=(0, None, None, None),
        )(thetas, acq, x_o, model_mask)

    return log_prob_fn


def make_keys(num_trials: int, offset: int) -> jax.Array:
    trial_ids = jnp.arange(num_trials)
    return jax.vmap(lambda i: jax.random.key(i + offset))(trial_ids)


@jax.jit
def simulate_signal(theta, acq, rng):
    simulator = SIM_TYPE.from_theta(theta)
    return simulator.signal(acq, rng)


def simulate_dataset(acquisition_fn, num_trials: int):
    acq_keys = make_keys(num_trials, 0)
    theta_keys = make_keys(num_trials, 1)
    signal_keys = make_keys(num_trials, 2)
    sample_keys = make_keys(num_trials, 3)
    rmse_keys = make_keys(num_trials, 4)
    ksd_keys = make_keys(num_trials, 5)

    acqs = jax.vmap(acquisition_fn)(acq_keys)
    thetas = jax.vmap(
        lambda key: jax.random.normal(key, (SIM_TYPE.theta_dim,))
    )(theta_keys)
    x_o = jax.vmap(simulate_signal)(thetas, acqs, signal_keys)
    return acqs, thetas, x_o, sample_keys, rmse_keys, ksd_keys


def slice_batch(tree, start: int, stop: int):
    return jax.tree_util.tree_map(lambda x: x[start:stop], tree)


def repeat_acquisition(acq, count: int):
    return jax.tree_util.tree_map(
        lambda arr: jnp.repeat(jnp.asarray(arr)[None, ...], count, axis=0),
        acq,
    )


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


def build_real_subset(
    data: dict[str, np.ndarray],
    *,
    num_samples: int,
    rng: jax.Array,
):
    total = int(data["x"].shape[0])
    if num_samples > total:
        raise ValueError(f"Requested {num_samples} real samples but only {total} are available.")
    idx = jax.random.choice(rng, total, (num_samples,), replace=False)
    x = jnp.asarray(data["x"])[idx]
    acqs = repeat_acquisition(data["acq"], num_samples)
    sample_keys = make_keys(num_samples, 3)
    rmse_keys = make_keys(num_samples, 4)
    ksd_keys = make_keys(num_samples, 5)
    return acqs, x, sample_keys, rmse_keys, ksd_keys


def evaluate_dataset(
    sample_fn,
    log_prob_fn,
    *,
    regime_name: str,
    acqs,
    x_o,
    sample_keys,
    rmse_keys,
    ksd_keys,
    batch_size: int | None,
    ksd_spec,
) -> dict[str, float]:
    num_trials = int(x_o.shape[0])
    batched_sample_fn = jax.vmap(sample_fn, in_axes=(0, 0, 0))
    batched_log_prob_fn = jax.vmap(log_prob_fn, in_axes=(0, 0, 0))
    batched_posterior_log_likelihood = jax.vmap(
        posterior_log_likelihood,
        in_axes=(0, 0, 0),
    )
    batched_ess = jax.vmap(ess, in_axes=(0, 0))
    batched_signal_rmse = jax.vmap(signal_rmse, in_axes=(0, 0, 0, 0))
    batched_ksd_fn = jax.vmap(build_ksd_fn(ksd_spec), in_axes=(0, 0, 0, 0))
    effective_batch_size = num_trials if batch_size is None else max(1, batch_size)
    num_batches = (num_trials + effective_batch_size - 1) // effective_batch_size
    log_progress(
        f"{regime_name}: evaluating in {num_batches} batch(es) with batch_size={effective_batch_size}"
    )

    ess_batches = []
    rmse_batches = []
    ksd_batches = []
    ksd_pvalue_batches = []

    for batch_idx, start in enumerate(range(0, num_trials, effective_batch_size), start=1):
        stop = min(start + effective_batch_size, num_trials)
        log_progress(
            f"{regime_name}: batch {batch_idx}/{num_batches} ({start}:{stop}) drawing posterior samples"
        )
        acq_batch = slice_batch(acqs, start, stop)
        x_o_batch = x_o[start:stop]
        sample_key_batch = sample_keys[start:stop]
        rmse_key_batch = rmse_keys[start:stop]
        ksd_key_batch = ksd_keys[start:stop]

        theta_samples = batched_sample_fn(
            sample_key_batch,
            acq_batch,
            x_o_batch,
        ).block_until_ready()
        log_progress(
            f"{regime_name}: batch {batch_idx}/{num_batches} computing posterior likelihood terms"
        )
        log_p = batched_posterior_log_likelihood(theta_samples, acq_batch, x_o_batch)
        log_q = batched_log_prob_fn(theta_samples, acq_batch, x_o_batch).block_until_ready()
        log_progress(
            f"{regime_name}: batch {batch_idx}/{num_batches} computing ESS and RMSE"
        )
        ess_batches.append(
            np.asarray(
                batched_ess(log_p, log_q) / theta_samples.shape[1],
                dtype=float,
            )
        )
        rmse_batches.append(
            np.asarray(
                batched_signal_rmse(theta_samples, x_o_batch, acq_batch, rmse_key_batch),
                dtype=float,
            )
        )
        log_progress(
            f"{regime_name}: batch {batch_idx}/{num_batches} computing KSD and p-values"
        )
        ksd_values, ksd_pvalues = batched_ksd_fn(
            theta_samples,
            acq_batch,
            x_o_batch,
            ksd_key_batch,
        )
        ksd_batches.append(np.asarray(ksd_values, dtype=float))
        ksd_pvalue_batches.append(np.asarray(ksd_pvalues, dtype=float))

    ess_values = np.concatenate(ess_batches, axis=0)
    rmse_values = np.concatenate(rmse_batches, axis=0)
    ksd_values = np.concatenate(ksd_batches, axis=0)
    ksd_pvalues = np.concatenate(ksd_pvalue_batches, axis=0)

    ess_summary = summary_stats(ess_values)
    rmse_summary = summary_stats(rmse_values)
    ksd_summary = summarize_ksd(ksd_values, ksd_spec)
    log_progress(f"{regime_name}: summarizing metrics")
    return {
        "ess_mean": ess_summary["mean"],
        "ess_median": ess_summary["median"],
        "ess_q10": ess_summary["q10"],
        "ess_q90": ess_summary["q90"],
        "rmse_mean": rmse_summary["mean"],
        "rmse_median": rmse_summary["median"],
        "rmse_q10": rmse_summary["q10"],
        "rmse_q90": rmse_summary["q90"],
        "ksd": ksd_summary,
        "ksd_pvalue_mean": float(np.mean(ksd_pvalues)),
        "ksd_pvalue_median": float(np.median(ksd_pvalues)),
    }


@partial(jax.vmap, in_axes=(0, None, None))
def posterior_log_likelihood(theta, acq, x_o):
    simulator = SIM_TYPE.from_theta(theta)
    return simulator.log_likelihood(acq, x_o)


def ess(log_p, log_q):
    logw = log_p - log_q
    finite = jnp.isfinite(logw)
    lw = jnp.where(finite, logw, -jnp.inf)
    lw_max = jnp.max(lw)
    shifted = lw - lw_max
    w = jnp.where(finite, jnp.exp(shifted), 0.0)
    sum_w = jnp.sum(w)
    sum_w2 = jnp.sum(w * w)
    return (sum_w * sum_w) / jnp.maximum(sum_w2, 1e-30)


@jax.jit
def signal_rmse(theta_samples, x_o, acq, rng):
    @partial(jax.vmap, in_axes=(0, None, 0))
    def forward_model(theta, acq_inner, rng_inner):
        simulator = SIM_TYPE.from_theta(theta)
        return simulator.signal(acq_inner, rng_inner)

    rngs = jax.random.split(rng, theta_samples.shape[0])
    x_o_pred = forward_model(theta_samples, acq, rngs)
    return jnp.sqrt(jnp.mean((x_o_pred - x_o) ** 2, axis=-1))


def summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.1)),
        "q90": float(np.quantile(arr, 0.9)),
    }


def summarize_ksd(values: np.ndarray, spec) -> dict[str, float]:
    return _summarize_metric(spec, np.asarray(values, dtype=np.float32))


def full_model_mask() -> jax.Array:
    return jnp.ones((SIM_TYPE.num_compartments(),), dtype=bool)


def build_ksd_fn(spec):
    options = dict(spec.options or {})
    bandwidths = np.asarray(
        options.get("bandwidths", (0.05, 0.1, 0.5)),
        dtype=np.float32,
    ).reshape(-1)
    max_samples = options.get("max_samples", options.get("num_samples"))
    bandwidths_jax = jnp.asarray(bandwidths)
    n_bootstrap = int(options.get("n_bootstrap", 256))
    mask = full_model_mask()

    def ksd_fn(theta_samples, acq, x_o, key):
        if max_samples is not None:
            theta_samples = theta_samples[: int(max_samples)]
        return multiscale_preconditioned_ksd_and_pvalue(
            sim_type=SIM_TYPE,
            thetas=theta_samples,
            mask=mask,
            acq=acq,
            x=x_o,
            key=key,
            bandwidths=bandwidths_jax,
            n_bootstrap=n_bootstrap,
        )

    return ksd_fn


def evaluate_ksd(theta_samples, acq, x_o, key, spec) -> tuple[float, float]:
    ksd, p_value = build_ksd_fn(spec)(theta_samples, acq, x_o, key)
    return float(ksd), float(p_value)


def evaluate_regime(
    sample_fn,
    log_prob_fn,
    acquisition_fn,
    *,
    num_trials: int,
    batch_size: int | None,
    ksd_spec,
) -> dict[str, float]:
    regime_name = acquisition_fn.__name__
    log_progress(f"{regime_name}: simulating {num_trials} synthetic problems")
    acqs, _, x_o, sample_keys, rmse_keys, ksd_keys = simulate_dataset(
        acquisition_fn,
        num_trials,
    )
    return evaluate_dataset(
        sample_fn,
        log_prob_fn,
        regime_name=regime_name,
        acqs=acqs,
        x_o=x_o,
        sample_keys=sample_keys,
        rmse_keys=rmse_keys,
        ksd_keys=ksd_keys,
        batch_size=batch_size,
        ksd_spec=ksd_spec,
    )


def evaluate_real_regime(
    sample_fn,
    log_prob_fn,
    *,
    regime_name: str,
    data: dict[str, np.ndarray],
    num_trials: int,
    subset_rng: jax.Array,
    batch_size: int | None,
    ksd_spec,
) -> dict[str, float]:
    log_progress(f"{regime_name}: selecting deterministic subset of {num_trials} real voxels")
    acqs, x_o, sample_keys, rmse_keys, ksd_keys = build_real_subset(
        data,
        num_samples=num_trials,
        rng=subset_rng,
    )
    return evaluate_dataset(
        sample_fn,
        log_prob_fn,
        regime_name=regime_name,
        acqs=acqs,
        x_o=x_o,
        sample_keys=sample_keys,
        rmse_keys=rmse_keys,
        ksd_keys=ksd_keys,
        batch_size=batch_size,
        ksd_spec=ksd_spec,
    )


def evaluate_single_example(sample_fn, log_prob_fn, ksd_spec) -> dict[str, float]:
    acq = random_hcp_acquisition(jax.random.key(1))
    theta = jax.random.normal(jax.random.key(40), (SIM_TYPE.theta_dim,))
    x_o = SIM_TYPE.from_theta(theta).signal(acq, jax.random.key(3))
    theta_samples = sample_fn(jax.random.key(4), acq, x_o).block_until_ready()
    log_p = posterior_log_likelihood(theta_samples, acq, x_o)
    log_q = log_prob_fn(theta_samples, acq, x_o).block_until_ready()
    ksd_value, p_value = evaluate_ksd(
        theta_samples,
        acq,
        x_o,
        jax.random.key(5),
        ksd_spec,
    )
    return {
        "param_rmse": float(
            jnp.sqrt(jnp.mean((theta_samples.mean(axis=0) - theta) ** 2))
        ),
        "ess": float(ess(log_p, log_q)),
        "ksd": ksd_value,
        "ksd_pvalue": p_value,
        "num_samples": int(theta_samples.shape[0]),
    }


def main() -> None:
    args = parse_args()
    real_samples = args.num_trials if args.real_samples is None else args.real_samples
    log_progress(f"loading checkpoint from {args.checkpoint}")
    model, params = load_prism_model(args.checkpoint, args.which)
    log_progress("building sampling and log-prob functions")
    sample_fn = build_sample_fn(model, args.sample_steps, args.num_samples)
    log_prob_fn = build_log_prob_fn(model, args.log_prob_steps)
    ksd_spec = load_ksd_spec(args.ksd_config)
    log_progress(f"loading real short data from {args.real_data_root / args.real_short_folder}")
    real_short = load_real_data(args.real_data_root, args.real_short_folder)
    log_progress(f"loading real long data from {args.real_data_root / args.real_long_folder}")
    real_long = load_real_data(args.real_data_root, args.real_long_folder)
    log_progress("starting evaluation")

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "ksd_config": str(args.ksd_config.resolve()),
        "which": args.which,
        "real_data_root": str(args.real_data_root.resolve()),
        "real_short_folder": args.real_short_folder,
        "real_long_folder": args.real_long_folder,
        "real_subset_seed": int(args.real_subset_seed),
        "real_samples": int(real_samples),
        "parameter_counts": component_parameter_counts(params),
        "single_example": evaluate_single_example(sample_fn, log_prob_fn, ksd_spec),
        "short_acquisition": evaluate_regime(
            sample_fn,
            log_prob_fn,
            random_hcp_acquisition,
            num_trials=args.num_trials,
            batch_size=args.batch_size,
            ksd_spec=ksd_spec,
        ),
        "long_acquisition": evaluate_regime(
            sample_fn,
            log_prob_fn,
            random_hcp_large_acquisition,
            num_trials=args.num_trials,
            batch_size=args.batch_size,
            ksd_spec=ksd_spec,
        ),
        "real_short_acquisition": evaluate_real_regime(
            sample_fn,
            log_prob_fn,
            regime_name="real_short_acquisition",
            data=real_short,
            num_trials=real_samples,
            subset_rng=jax.random.key(args.real_subset_seed),
            batch_size=args.batch_size,
            ksd_spec=ksd_spec,
        ),
        "real_long_acquisition": evaluate_real_regime(
            sample_fn,
            log_prob_fn,
            regime_name="real_long_acquisition",
            data=real_long,
            num_trials=real_samples,
            subset_rng=jax.random.key(args.real_subset_seed + 1),
            batch_size=args.batch_size,
            ksd_spec=ksd_spec,
        ),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
