#!/usr/bin/env python3
"""Benchmark Torch SBI checkpoint inference and training operations."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SBI checkpoint sample/log_prob/forward/backward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "main_b3s" / "checkpoints" / "latest.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup runs per benchmark.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Timed runs per benchmark.",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=2048,
        help="Requested training batch size for forward/backward benchmarks.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        choices=("none", "aot_eager"),
        default="aot_eager",
        help="Optional conservative torch.compile benchmark strategy.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save JSON results.",
    )
    return parser.parse_args()


def load_flow(
    checkpoint_path: Path,
    device: torch.device,
    *,
    training_batch_size: int,
) -> tuple[torch.nn.Module, TrainConfig, Any, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**payload["config"])
    cfg.batch_size = max(1, int(training_batch_size))

    _, prefetch_loaders = build_data_pipeline(cfg, device)
    short_batch = next(iter(prefetch_loaders[0]))
    long_batch = next(iter(prefetch_loaders[1]))

    flow, _ = build_flow_from_batch(cfg, device, short_batch)
    flow.load_state_dict(payload["model_state_dict"])
    flow.eval()
    return flow, cfg, short_batch, long_batch


def parameter_counts(flow: torch.nn.Module) -> dict[str, int]:
    counts = {
        "total": int(sum(p.numel() for p in flow.parameters())),
    }
    if hasattr(flow, "embedding_net"):
        counts["embedding_net"] = int(
            sum(p.numel() for p in flow.embedding_net.parameters())
        )
    if hasattr(flow, "net") and hasattr(flow.net, "_transform"):
        counts["transform"] = int(
            sum(p.numel() for p in flow.net._transform.parameters())
        )
    if hasattr(flow, "net") and hasattr(flow.net, "_distribution"):
        counts["distribution"] = int(
            sum(p.numel() for p in flow.net._distribution.parameters())
        )
    return counts


def to_condition(x_o: Any, acq: Any, device: torch.device) -> dict[str, Any]:
    return {
        "x": torch.from_numpy(np.asarray(x_o)).to(device=device)[None, :],
        "acq": {
            "bvals": torch.from_numpy(np.asarray(acq.bvals)).to(device=device)[None, :],
            "bvecs": torch.from_numpy(np.asarray(acq.bvecs)).to(device=device)[None, :],
        },
    }


def torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def allocated_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.memory_allocated(device) / (1024**2)


def reserved_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.memory_reserved(device) / (1024**2)


def peak_allocated_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def peak_reserved_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / (1024**2)


def benchmark_torch_op(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
    setup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    for _ in range(warmup):
        if setup is not None:
            setup()
            torch_sync(device)
        reset_peak_memory(device)
        out = fn()
        torch_sync(device)
        del out

    times_ms: list[float] = []
    baseline_allocated_mib: list[float] = []
    baseline_reserved_mib: list[float] = []
    peak_allocated_mib: list[float] = []
    peak_reserved_mib: list[float] = []

    for _ in range(repeat):
        gc.collect()
        if setup is not None:
            setup()
            torch_sync(device)
        reset_peak_memory(device)
        baseline_allocated = allocated_memory_mib(device)
        baseline_reserved = reserved_memory_mib(device)
        torch_sync(device)
        t0 = time.perf_counter()
        out = fn()
        torch_sync(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        peak_allocated = peak_allocated_memory_mib(device)
        peak_reserved = peak_reserved_memory_mib(device)
        if baseline_allocated is not None:
            baseline_allocated_mib.append(baseline_allocated)
        if baseline_reserved is not None:
            baseline_reserved_mib.append(baseline_reserved)
        if peak_allocated is not None:
            peak_allocated_mib.append(peak_allocated)
        if peak_reserved is not None:
            peak_reserved_mib.append(peak_reserved)
        del out

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "baseline_cuda_allocated_mib_mean": (
            statistics.mean(baseline_allocated_mib) if baseline_allocated_mib else None
        ),
        "baseline_cuda_allocated_mib_std": (
            statistics.pstdev(baseline_allocated_mib)
            if len(baseline_allocated_mib) > 1
            else 0.0
        )
        if baseline_allocated_mib
        else None,
        "baseline_cuda_reserved_mib_mean": (
            statistics.mean(baseline_reserved_mib) if baseline_reserved_mib else None
        ),
        "baseline_cuda_reserved_mib_std": (
            statistics.pstdev(baseline_reserved_mib)
            if len(baseline_reserved_mib) > 1
            else 0.0
        )
        if baseline_reserved_mib
        else None,
        "peak_cuda_allocated_mib_mean": (
            statistics.mean(peak_allocated_mib) if peak_allocated_mib else None
        ),
        "peak_cuda_allocated_mib_std": (
            statistics.pstdev(peak_allocated_mib)
            if len(peak_allocated_mib) > 1
            else 0.0
        )
        if peak_allocated_mib
        else None,
        "peak_cuda_reserved_mib_mean": (
            statistics.mean(peak_reserved_mib) if peak_reserved_mib else None
        ),
        "peak_cuda_reserved_mib_std": (
            statistics.pstdev(peak_reserved_mib)
            if len(peak_reserved_mib) > 1
            else 0.0
        )
        if peak_reserved_mib
        else None,
        "peak_cuda_incremental_mib_mean": (
            statistics.mean(
                peak - base
                for peak, base in zip(peak_allocated_mib, baseline_allocated_mib)
            )
            if peak_allocated_mib and baseline_allocated_mib
            else None
        ),
    }


def make_condition(
    sim_type: type,
    acquisition_fn: Callable[[jax.Array], Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    acq = acquisition_fn(jax.random.key(seed))
    theta = jax.random.normal(jax.random.key(seed + 1), (sim_type.theta_dim,))
    x_o = sim_type.from_theta(theta).signal(acq, jax.random.key(seed + 2))
    return to_condition(x_o, acq, device)


def benchmark_compiled_op(
    name: str,
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
    compile_mode: str,
    flow: torch.nn.Module,
    setup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if compile_mode != "aot_eager":
        raise ValueError(f"Unsupported compile mode: {compile_mode}")

    if not hasattr(torch, "compile"):
        return {
            "status": "compile_unavailable",
            "compile_strategy": compile_mode,
            "error": "torch.compile is not available in this PyTorch build.",
        }

    suppress_errors_previous = None
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
        suppress_errors_previous = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = True

    compile_kwargs = {
        "backend": "aot_eager",
        "fullgraph": False,
        "dynamic": None,
        "mode": "default",
    }

    try:
        compiled_fn = torch.compile(fn, **compile_kwargs)
        result = benchmark_torch_op(
            compiled_fn,
            device=device,
            warmup=warmup,
            repeat=repeat,
            setup=setup,
        )
        result["status"] = "ok"
        result["compile_strategy"] = "aot_eager_with_fallback"
    except Exception as exc:  # pragma: no cover - runtime dependent
        result = {
            "status": "compile_failed",
            "compile_strategy": "aot_eager_with_fallback",
            "error": repr(exc),
            "mean_ms": None,
            "std_ms": None,
            "baseline_cuda_allocated_mib_mean": None,
            "baseline_cuda_allocated_mib_std": None,
            "baseline_cuda_reserved_mib_mean": None,
            "baseline_cuda_reserved_mib_std": None,
            "peak_cuda_allocated_mib_mean": None,
            "peak_cuda_allocated_mib_std": None,
            "peak_cuda_reserved_mib_mean": None,
            "peak_cuda_reserved_mib_std": None,
            "peak_cuda_incremental_mib_mean": None,
        }
    finally:
        if hasattr(torch, "_dynamo") and suppress_errors_previous is not None:
            torch._dynamo.config.suppress_errors = suppress_errors_previous
        flow.zero_grad(set_to_none=True)
        flow.eval()

    return result


def main() -> None:
    args = parse_args()

    global random_hcp_acquisition
    global random_hcp_large_acquisition
    global TrainConfig
    global build_data_pipeline
    global build_flow_from_batch
    global resolve_device
    global resolve_sim_type
    global slice_batch
    global split_batch

    from dmri.simulators.acquisition_scheme import (
        random_hcp_acquisition,
        random_hcp_large_acquisition,
    )
    from figures.b3s_baseline.train_script_sbi import (
        TrainConfig,
        build_data_pipeline,
        build_flow_from_batch,
        resolve_device,
        resolve_sim_type,
        slice_batch,
        split_batch,
    )
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.resolve()

    flow, cfg, short_batch, long_batch = load_flow(
        checkpoint_path,
        device,
        training_batch_size=args.training_batch_size,
    )
    sim_type = resolve_sim_type(cfg.sim_type)

    short_condition = make_condition(
        sim_type,
        random_hcp_acquisition,
        seed=100,
        device=device,
    )
    long_condition = make_condition(
        sim_type,
        random_hcp_large_acquisition,
        seed=200,
        device=device,
    )

    short_train_batch = slice_batch(
        short_batch,
        min(args.training_batch_size, int(short_batch["theta"].shape[0])),
    )
    long_train_batch = slice_batch(
        long_batch,
        min(args.training_batch_size, int(long_batch["theta"].shape[0])),
    )

    short_theta_train, short_condition_train = split_batch(short_train_batch)
    long_theta_train, long_condition_train = split_batch(long_train_batch)

    sample_shape = torch.Size([1])

    flow.eval()
    with torch.inference_mode():
        short_theta_eval = flow.sample(sample_shape=sample_shape, condition=short_condition)
        long_theta_eval = flow.sample(sample_shape=sample_shape, condition=long_condition)

    def sample_short() -> torch.Tensor:
        flow.eval()
        with torch.inference_mode():
            return flow.sample(sample_shape=sample_shape, condition=short_condition)

    def log_prob_short() -> torch.Tensor:
        flow.eval()
        with torch.inference_mode():
            return flow.log_prob(short_theta_eval, short_condition)

    def sample_long() -> torch.Tensor:
        flow.eval()
        with torch.inference_mode():
            return flow.sample(sample_shape=sample_shape, condition=long_condition)

    def log_prob_long() -> torch.Tensor:
        flow.eval()
        with torch.inference_mode():
            return flow.log_prob(long_theta_eval, long_condition)

    def forward_short() -> torch.Tensor:
        flow.train()
        return flow.loss(short_theta_train, short_condition_train).mean()

    def backward_short() -> torch.Tensor:
        flow.train()
        flow.zero_grad(set_to_none=True)
        loss = flow.loss(short_theta_train, short_condition_train).mean()
        loss.backward()
        return loss.detach()

    def forward_long() -> torch.Tensor:
        flow.train()
        return flow.loss(long_theta_train, long_condition_train).mean()

    def backward_long() -> torch.Tensor:
        flow.train()
        flow.zero_grad(set_to_none=True)
        loss = flow.loss(long_theta_train, long_condition_train).mean()
        loss.backward()
        return loss.detach()

    def setup_inference() -> None:
        flow.eval()
        flow.zero_grad(set_to_none=True)

    def setup_forward() -> None:
        flow.train()
        flow.zero_grad(set_to_none=True)

    def setup_backward() -> None:
        flow.train()
        flow.zero_grad(set_to_none=True)

    ops = {
        "sample_short": {"fn": sample_short, "setup": setup_inference},
        "log_prob_short": {"fn": log_prob_short, "setup": setup_inference},
        "sample_long": {"fn": sample_long, "setup": setup_inference},
        "log_prob_long": {"fn": log_prob_long, "setup": setup_inference},
        "forward_short": {"fn": forward_short, "setup": setup_forward},
        "backward_short": {"fn": backward_short, "setup": setup_backward},
        "forward_long": {"fn": forward_long, "setup": setup_forward},
        "backward_long": {"fn": backward_long, "setup": setup_backward},
    }

    eager_results = {
        name: benchmark_torch_op(
            spec["fn"],
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
            setup=spec["setup"],
        )
        for name, spec in ops.items()
    }

    compiled_results: dict[str, Any] | None = None
    if args.compile_mode != "none":
        compiled_results = {
            name: benchmark_compiled_op(
                name,
                spec["fn"],
                device=device,
                warmup=args.warmup,
                repeat=args.repeat,
                compile_mode=args.compile_mode,
                flow=flow,
                setup=spec["setup"],
            )
            for name, spec in ops.items()
        }

    result = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "sim_type": sim_type.__name__,
        "parameter_counts": parameter_counts(flow),
        "benchmark_config": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "inference_batch_size": 1,
            "training_batch_size_short": int(short_theta_train.shape[0]),
            "training_batch_size_long": int(long_theta_train.shape[0]),
            "compile_mode": args.compile_mode,
            "memory_measurement": "full_peak_cuda_allocated_and_reserved_after_per_run_setup",
        },
        "eager": eager_results,
        "compiled": compiled_results,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
