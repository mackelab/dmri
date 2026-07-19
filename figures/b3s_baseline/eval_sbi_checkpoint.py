#!/usr/bin/env python3
"""Evaluate an SBI baseline checkpoint on Ball3Stick synthetic acquisitions."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from scipy import ndimage

jax.config.update("jax_platforms", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmri.simulators.acquisition_scheme import (  # noqa: E402
    acquisition_scheme,
    random_hcp_acquisition,
    random_hcp_large_acquisition,
)
from dmri.eval.export_metrics import (  # noqa: E402
    multiscale_preconditioned_ksd_and_pvalue,
    _normalize_metric_specs,
    _summarize_metric,
)
from dmri.simulators.models import Ball3StickSharedDiffusivity  # noqa: E402
from figures.b3s_baseline.train_script_sbi import (  # noqa: E402
    TrainConfig,
    build_data_pipeline,
    build_flow_from_batch,
    resolve_device,
    slice_batch,
    split_batch,
)


SIM_TYPE = Ball3StickSharedDiffusivity
DEFAULT_KSD_CONFIG = REPO_ROOT / "conf_eval" / "export" / "metrics" / "ksd.yaml"


def log_progress(message: str) -> None:
    print(f"[eval_sbi_checkpoint] {message}", flush=True)


def clear_runtime_caches(label: str, device: torch.device | None = None) -> None:
    log_progress(f"clearing caches after {label}")
    jax.clear_caches()
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an SBI baseline checkpoint for the Ball3Stick baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "main_b3s" / "checkpoints" / "best.pt",
        help="Path to the SBI checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override. Defaults to CUDA when available.",
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


def load_flow(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**payload["config"])

    _, prefetch_loaders = build_data_pipeline(cfg, device)
    first_batch = next(iter(prefetch_loaders[0]))

    flow, _ = build_flow_from_batch(cfg, device, first_batch)
    flow.load_state_dict(payload["model_state_dict"])
    flow.eval()
    return flow, payload, cfg, first_batch


def parameter_counts(flow: torch.nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(p.numel() for p in flow.parameters())),
        "embedding_net": int(sum(p.numel() for p in flow.embedding_net.parameters())),
        "transform": int(sum(p.numel() for p in flow.net._transform.parameters())),
    }


def load_yaml_config(path: Path) -> dict:
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


def to_condition(x_o, acq, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "x": torch.from_numpy(np.array(x_o, copy=True)).to(device=device)[None, :],
        "acq": {
            "bvals": torch.from_numpy(np.array(acq.bvals, copy=True)).to(device=device)[
                None, :
            ],
            "bvecs": torch.from_numpy(np.array(acq.bvecs, copy=True)).to(device=device)[
                None, :
            ],
        },
    }


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
    rmse_keys = make_keys(num_trials, 3)
    ksd_keys = make_keys(num_trials, 4)

    acqs = jax.vmap(acquisition_fn)(acq_keys)
    thetas = jax.vmap(
        lambda key: jax.random.normal(key, (SIM_TYPE.theta_dim,))
    )(theta_keys)
    x_o = jax.vmap(simulate_signal)(thetas, acqs, signal_keys)
    return acqs, thetas, x_o, rmse_keys, ksd_keys


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
    rmse_keys = make_keys(num_samples, 3)
    ksd_keys = make_keys(num_samples, 4)
    return acqs, x, rmse_keys, ksd_keys


def select_batch_item(tree, i: int):
    return jax.tree_util.tree_map(lambda x: x[i], tree)


def full_model_mask() -> jax.Array:
    return jnp.ones((SIM_TYPE.num_compartments(),), dtype=bool)


def evaluate_ksd(theta_samples, acq, x_o, key, spec) -> tuple[float, float]:
    options = dict(spec.options or {})
    bandwidths = np.asarray(
        options.get("bandwidths", (0.05, 0.1, 0.5)),
        dtype=np.float32,
    ).reshape(-1)
    max_samples = options.get("max_samples", options.get("num_samples"))
    if max_samples is not None:
        theta_samples = theta_samples[: int(max_samples)]
    ksd, p_value = multiscale_preconditioned_ksd_and_pvalue(
        sim_type=SIM_TYPE,
        thetas=theta_samples,
        mask=full_model_mask(),
        acq=acq,
        x=x_o,
        key=key,
        bandwidths=jnp.asarray(bandwidths),
        n_bootstrap=int(options.get("n_bootstrap", 256)),
    )
    return float(ksd), float(p_value)


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


def evaluate_single_example(flow, first_batch, device: torch.device, ksd_spec) -> dict[str, float]:
    eval_batch = slice_batch(first_batch, 8)
    theta, condition = split_batch(eval_batch)
    with torch.no_grad():
        loss = flow.loss(theta, condition).mean()
        samples, log_probs = flow.sample_and_log_prob(torch.Size([4]), condition)
        rmse = torch.sqrt(torch.mean((samples.mean(dim=0) - theta) ** 2))

    acq = random_hcp_acquisition(jax.random.key(1))
    theta_true = jax.random.normal(jax.random.key(40), (SIM_TYPE.theta_dim,))
    x_o = SIM_TYPE.from_theta(theta_true).signal(acq, jax.random.key(3))
    condition = to_condition(x_o, acq, device)
    with torch.no_grad():
        theta_samples = flow.sample(sample_shape=(50,), condition=condition)
    ksd_value, p_value = evaluate_ksd(
        theta_samples.detach().cpu().numpy()[:, 0],
        acq,
        x_o,
        jax.random.key(5),
        ksd_spec,
    )
    return {
        "loss": float(loss.item()),
        "param_rmse": float(rmse.item()),
        "ksd": ksd_value,
        "ksd_pvalue": p_value,
        "samples_shape": list(samples.shape),
        "log_probs_shape": list(log_probs.shape),
    }


def evaluate_regime(
    flow,
    *,
    regime_name: str,
    acqs,
    x_o,
    device: torch.device,
    num_samples: int,
    rmse_keys,
    ksd_keys,
    ksd_spec,
) -> dict[str, float]:
    num_trials = int(x_o.shape[0])
    ess_values = []
    rmse_values = []
    ksd_values = []
    ksd_pvalues = []
    progress_interval = max(1, num_trials // 10)

    log_progress(f"{regime_name}: evaluating posterior samples and metrics")
    with torch.no_grad():
        for i in range(num_trials):
            if i == 0 or (i + 1) % progress_interval == 0 or i + 1 == num_trials:
                log_progress(f"{regime_name}: {i + 1}/{num_trials}")
            acq = select_batch_item(acqs, i)
            x_o_i = x_o[i]
            condition = to_condition(x_o_i, acq, device)
            theta_samples = flow.sample(sample_shape=(num_samples,), condition=condition)
            theta_samples_np = theta_samples.detach().cpu().numpy()[:, 0]
            log_p = posterior_log_likelihood(theta_samples_np, acq, x_o_i)
            log_q = (
                flow.log_prob(theta_samples, condition).detach().cpu().numpy().squeeze()
            )
            ess_values.append(float(ess(log_p, log_q)) / num_samples)
            rmse_values.append(
                np.asarray(signal_rmse(theta_samples_np, x_o_i, acq, rmse_keys[i]))
            )
            ksd_value, p_value = evaluate_ksd(
                theta_samples_np,
                acq,
                x_o_i,
                ksd_keys[i],
                ksd_spec,
            )
            ksd_values.append(ksd_value)
            ksd_pvalues.append(p_value)

    log_progress(f"{regime_name}: summarizing metrics")
    rmse_array = np.asarray(rmse_values, dtype=float)
    ess_summary = summary_stats(np.asarray(ess_values, dtype=float))
    rmse_summary = summary_stats(rmse_array)
    ksd_summary = summarize_ksd(np.asarray(ksd_values, dtype=float), ksd_spec)
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


def evaluate_synthetic_regime(
    flow,
    acquisition_fn,
    *,
    device: torch.device,
    num_trials: int,
    num_samples: int,
    ksd_spec,
) -> dict[str, float]:
    regime_name = acquisition_fn.__name__
    log_progress(f"{regime_name}: simulating {num_trials} synthetic problems")
    acqs, _, x_o, rmse_keys, ksd_keys = simulate_dataset(
        acquisition_fn,
        num_trials,
    )
    return evaluate_regime(
        flow,
        regime_name=regime_name,
        acqs=acqs,
        x_o=x_o,
        device=device,
        num_samples=num_samples,
        rmse_keys=rmse_keys,
        ksd_keys=ksd_keys,
        ksd_spec=ksd_spec,
    )


def evaluate_real_regime(
    flow,
    *,
    regime_name: str,
    data: dict[str, np.ndarray],
    num_trials: int,
    subset_rng: jax.Array,
    device: torch.device,
    num_samples: int,
    ksd_spec,
) -> dict[str, float]:
    log_progress(f"{regime_name}: selecting deterministic subset of {num_trials} real voxels")
    acqs, x_o, rmse_keys, ksd_keys = build_real_subset(
        data,
        num_samples=num_trials,
        rng=subset_rng,
    )
    return evaluate_regime(
        flow,
        regime_name=regime_name,
        acqs=acqs,
        x_o=x_o,
        device=device,
        num_samples=num_samples,
        rmse_keys=rmse_keys,
        ksd_keys=ksd_keys,
        ksd_spec=ksd_spec,
    )


def main() -> None:
    args = parse_args()
    real_samples = args.num_trials if args.real_samples is None else args.real_samples
    log_progress(f"loading checkpoint from {args.checkpoint}")
    log_progress("configured JAX for CPU-only metrics")
    device = resolve_device(args.device)
    log_progress(f"resolved device {device}")
    flow, payload, cfg, first_batch = load_flow(args.checkpoint, device)
    log_progress("loaded flow and checkpoint state")
    ksd_spec = load_ksd_spec(args.ksd_config)
    log_progress(f"loading real short data from {args.real_data_root / args.real_short_folder}")
    real_short = load_real_data(args.real_data_root, args.real_short_folder)
    log_progress(f"loading real long data from {args.real_data_root / args.real_long_folder}")
    real_long = load_real_data(args.real_data_root, args.real_long_folder)
    log_progress("starting evaluation")

    single_example = evaluate_single_example(flow, first_batch, device, ksd_spec)
    clear_runtime_caches("single_example", device)

    short_acquisition = evaluate_synthetic_regime(
        flow,
        random_hcp_acquisition,
        device=device,
        num_trials=args.num_trials,
        num_samples=args.num_samples,
        ksd_spec=ksd_spec,
    )
    clear_runtime_caches("short_acquisition", device)

    long_acquisition = evaluate_synthetic_regime(
        flow,
        random_hcp_large_acquisition,
        device=device,
        num_trials=args.num_trials,
        num_samples=args.num_samples,
        ksd_spec=ksd_spec,
    )
    clear_runtime_caches("long_acquisition", device)

    real_short_acquisition = evaluate_real_regime(
        flow,
        regime_name="real_short_acquisition",
        data=real_short,
        num_trials=real_samples,
        subset_rng=jax.random.key(args.real_subset_seed),
        device=device,
        num_samples=args.num_samples,
        ksd_spec=ksd_spec,
    )
    clear_runtime_caches("real_short_acquisition", device)

    real_long_acquisition = evaluate_real_regime(
        flow,
        regime_name="real_long_acquisition",
        data=real_long,
        num_trials=real_samples,
        subset_rng=jax.random.key(args.real_subset_seed + 1),
        device=device,
        num_samples=args.num_samples,
        ksd_spec=ksd_spec,
    )
    clear_runtime_caches("real_long_acquisition", device)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "ksd_config": str(args.ksd_config.resolve()),
        "real_data_root": str(args.real_data_root.resolve()),
        "real_short_folder": args.real_short_folder,
        "real_long_folder": args.real_long_folder,
        "real_subset_seed": int(args.real_subset_seed),
        "real_samples": int(real_samples),
        "step": int(payload["step"]),
        "best_eval_loss": float(payload["best_eval_loss"]),
        "sim_type": cfg.sim_type,
        "parameter_counts": parameter_counts(flow),
        "single_example": single_example,
        "short_acquisition": short_acquisition,
        "long_acquisition": long_acquisition,
        "real_short_acquisition": real_short_acquisition,
        "real_long_acquisition": real_long_acquisition,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
