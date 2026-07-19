#!/usr/bin/env python3
"""Helpers and CLI for profiling JAX workloads."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _normalize_device_override(device: str | None) -> str | None:
    if device is None:
        return None
    aliases = {
        "cuda": "gpu",
    }
    return aliases.get(device, device)


def _bootstrap_jax_platform(argv: list[str]) -> str | None:
    bootstrap_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap_parser.add_argument("--device", default=None)
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)
    device = _normalize_device_override(bootstrap_args.device)
    if device is not None:
        os.environ["JAX_PLATFORMS"] = device
    return device


_DEVICE_OVERRIDE = _bootstrap_jax_platform(sys.argv[1:]) if __name__ == "__main__" else None

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ProfileCase:
    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    jit: bool = True
    static_argnums: int | tuple[int, ...] | None = None
    static_argnames: str | tuple[str, ...] | None = None


def _tree_nbytes(tree: Any) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "nbytes"):
            total += int(leaf.nbytes)
    return total


def _block(tree: Any) -> Any:
    return jax.block_until_ready(tree)


def _live_arrays_nbytes() -> int | None:
    if not hasattr(jax, "live_arrays"):
        return None

    total = 0
    for arr in jax.live_arrays():
        if hasattr(arr, "nbytes"):
            total += int(arr.nbytes)
    return total


def _device_memory_stats() -> dict[str, int] | None:
    stats_list: list[dict[str, Any]] = []
    for device in jax.devices():
        if not hasattr(device, "memory_stats"):
            continue
        try:
            stats = device.memory_stats()
        except Exception:
            continue
        if isinstance(stats, dict):
            stats_list.append(stats)

    if not stats_list:
        return None

    totals: dict[str, int] = {}
    for stats in stats_list:
        for key, value in stats.items():
            if isinstance(value, (int, np.integer)):
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def _device_memory_delta(
    before: dict[str, int] | None, after: dict[str, int] | None
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in keys}


def _summarize_memory_deltas(
    deltas: list[dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    if not deltas:
        return {}

    keys = set().union(*(delta.keys() for delta in deltas))
    summary: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [int(delta.get(key, 0)) for delta in deltas]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _compiler_memory_stats_to_dict(stats: Any) -> dict[str, Any] | None:
    if stats is None:
        return None

    keys = (
        "generated_code_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if hasattr(stats, key):
            out[key] = int(getattr(stats, key))
    return out


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return str(value)


def save_profile_result(result: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    return output_path


def profile_jax_function(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    name: str | None = None,
    jit: bool = True,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: str | tuple[str, ...] | None = None,
    warmup: int = 1,
    repeat: int = 10,
    clear_caches: bool = False,
    capture_trace: bool = False,
    trace_dir: str | Path | None = None,
    capture_device_memory_profile: bool = False,
    memory_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    kwargs = kwargs or {}
    name = name or getattr(fn, "__name__", "anonymous_fn")

    if warmup < 0:
        raise ValueError("warmup must be >= 0.")
    if repeat < 1:
        raise ValueError("repeat must be >= 1.")

    if clear_caches:
        jax.clear_caches()

    wrapped = fn
    if jit:
        wrapped = jax.jit(
            fn,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    result: dict[str, Any] = {
        "name": name,
        "backend": jax.default_backend(),
        "devices": [device.platform for device in jax.devices()],
        "input_nbytes": _tree_nbytes((args, kwargs)),
        "jit": jit,
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
    }

    compiler_memory_estimate = None
    compile_time_s = None
    run_target = wrapped

    if jit:
        try:
            compile_start = time.perf_counter()
            lowered = wrapped.lower(*args, **kwargs)
            compiled = lowered.compile()
            compile_time_s = time.perf_counter() - compile_start
            compiler_memory_estimate = _compiler_memory_stats_to_dict(
                compiled.memory_analysis()
            )
            run_target = compiled
        except Exception as err:
            compiler_memory_estimate = {"error": repr(err)}

    result["compile_s"] = compile_time_s
    result["compiler_memory_estimate"] = compiler_memory_estimate

    device_memory_before_first = _device_memory_stats()
    first_exec_start = time.perf_counter()
    out = run_target(*args, **kwargs)
    out = _block(out)
    first_exec_s = time.perf_counter() - first_exec_start
    device_memory_after_first = _device_memory_stats()

    result["first_exec_s"] = first_exec_s
    result["compile_plus_first_exec_s"] = (
        compile_time_s + first_exec_s if compile_time_s is not None else first_exec_s
    )
    result["output_nbytes"] = _tree_nbytes(out)
    result["device_memory_before_first"] = device_memory_before_first
    result["device_memory_after_first"] = device_memory_after_first
    result["device_memory_first_delta"] = _device_memory_delta(
        device_memory_before_first, device_memory_after_first
    )

    for _ in range(warmup):
        _block(run_target(*args, **kwargs))

    live_before = _live_arrays_nbytes()
    device_memory_before_steady = _device_memory_stats()
    timings: list[float] = []
    steady_memory_deltas: list[dict[str, int]] = []

    def _timed_runs() -> None:
        for _ in range(repeat):
            device_before_run = _device_memory_stats()
            t0 = time.perf_counter()
            y = run_target(*args, **kwargs)
            _block(y)
            timings.append(time.perf_counter() - t0)
            device_after_run = _device_memory_stats()
            device_delta = _device_memory_delta(device_before_run, device_after_run)
            if device_delta is not None:
                steady_memory_deltas.append(device_delta)

    if capture_trace:
        trace_dir = Path(trace_dir or f"./profiles/{name}_trace").resolve()
        trace_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(trace_dir)):
            _timed_runs()
    else:
        _timed_runs()

    live_after = _live_arrays_nbytes()
    device_memory_after_steady = _device_memory_stats()

    result["steady_state"] = {
        "runs": repeat,
        "mean_s": statistics.mean(timings),
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "max_s": max(timings),
        "std_s": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        "all_runs_s": timings,
    }
    result["live_arrays_before_nbytes"] = live_before
    result["live_arrays_after_nbytes"] = live_after
    result["live_arrays_delta_nbytes"] = (
        None
        if live_before is None or live_after is None
        else live_after - live_before
    )
    result["device_memory_before_steady"] = device_memory_before_steady
    result["device_memory_after_steady"] = device_memory_after_steady
    result["device_memory_steady_delta"] = _device_memory_delta(
        device_memory_before_steady, device_memory_after_steady
    )
    result["device_memory_steady_run_summary"] = _summarize_memory_deltas(
        steady_memory_deltas
    )

    artifacts: dict[str, str] = {}

    if capture_trace and trace_dir is not None:
        artifacts["trace_dir"] = str(trace_dir)

    if capture_device_memory_profile:
        memory_profile_path = Path(
            memory_profile_path or f"./profiles/{name}_memory.prof"
        ).resolve()
        memory_profile_path.parent.mkdir(parents=True, exist_ok=True)
        _block(run_target(*args, **kwargs))
        jax.profiler.save_device_memory_profile(str(memory_profile_path))
        artifacts["device_memory_profile"] = str(memory_profile_path)

    result["artifacts"] = artifacts
    return result


def profile_case(
    case: ProfileCase,
    *,
    warmup: int = 1,
    repeat: int = 10,
    clear_caches: bool = False,
    capture_trace: bool = False,
    trace_dir: str | Path | None = None,
    capture_device_memory_profile: bool = False,
    memory_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    result = profile_jax_function(
        case.fn,
        args=case.args,
        kwargs=case.kwargs,
        name=case.name,
        jit=case.jit,
        static_argnums=case.static_argnums,
        static_argnames=case.static_argnames,
        warmup=warmup,
        repeat=repeat,
        clear_caches=clear_caches,
        capture_trace=capture_trace,
        trace_dir=trace_dir,
        capture_device_memory_profile=capture_device_memory_profile,
        memory_profile_path=memory_profile_path,
    )
    result.update(case.metadata)
    return result


def _count_parameters(tree: Any) -> int:
    return int(
        jax.tree_util.tree_reduce(lambda total, leaf: total + leaf.size, tree, 0)
    )


def _component_parameter_counts(params: Any) -> dict[str, int]:
    counts = {"total": _count_parameters(params)}
    for key in ("encoder", "inference_decoder", "model_decoder", "tokenizer"):
        try:
            if key in params:
                counts[key] = _count_parameters(params[key])
        except TypeError:
            continue
    return counts


def _broadcast_tree(tree: Any, batch_size: int) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (batch_size,) + jnp.shape(leaf)),
        tree,
    )


def _load_model_bundle(checkpoint_path: Path, which: str = "latest"):
    from dmri.train.utils import load_checkpoint

    checkpoint, model, _ = load_checkpoint(str(checkpoint_path), which=which)
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    params = checkpoint.get("params_ema", checkpoint.get("params", params))
    model = nnx.merge(graphdef, params, state)
    model.eval()
    graphdef, params, state = nnx.split(model, nnx.Param, ...)
    return model, graphdef, params, state


def _build_random_acquisition(kind: str, rng: jax.Array) -> Any:
    from dmri.simulators.acquisition_scheme import (
        random_hcp_acquisition,
        random_hcp_large_acquisition,
    )

    acquisition_fns = {
        "random_hcp_acq": random_hcp_acquisition,
        "random_large_hcp_acq": random_hcp_large_acquisition,
    }
    try:
        return acquisition_fns[kind](rng)
    except KeyError as err:
        raise ValueError(f"Unknown acquisition kind: {kind!r}") from err


def build_model_summary(
    checkpoint_path: str | Path,
    *,
    acquisition: str = "random_hcp_acq",
    mask_prior: float = 1.0,
    temperature: float = 1.0,
    theta_num_steps: int = 64,
    loss_batch_size: int = 2048,
    include_train_stats: bool = True,
    seed: int = 0,
    checkpoint_which: str = "latest",
) -> tuple[dict[str, Any], list[ProfileCase]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    model, graphdef, params, state = _load_model_bundle(
        checkpoint_path, which=checkpoint_which
    )
    rng = jax.random.PRNGKey(seed)
    rng_acq, rng_x, rng_mask, rng_theta, rng_loss_x = jax.random.split(rng, 5)
    acq = _build_random_acquisition(acquisition, rng_acq)
    num_acquisitions = int(acq.bvals.shape[0])
    if include_train_stats and loss_batch_size < 1:
        raise ValueError("loss_batch_size must be >= 1.")

    x_single = jax.random.uniform(
        rng_x,
        shape=(num_acquisitions,),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    mask_prior_arr = jnp.asarray([mask_prior], dtype=jnp.float32)

    model_mask_single = _block(
        model.sample_mask(
            rng=rng_mask,
            acq=acq,
            x=x_single,
            mask_prior=mask_prior_arr,
            temperature=temperature,
        )
    )
    theta_single = _block(
        model.sample_theta(
            rng=rng_theta,
            acq=acq,
            x=x_single,
            model_mask=model_mask_single,
            num_steps=theta_num_steps,
            temperature=temperature,
        )
    )

    def sample_mask_fn(rng_key: jax.Array) -> jax.Array:
        return model.sample_mask(
            rng=rng_key,
            acq=acq,
            x=x_single,
            mask_prior=mask_prior_arr,
            temperature=temperature,
        )

    def log_prob_mask_fn() -> jax.Array:
        return model.log_prob_mask(
            model_mask=model_mask_single,
            acq=acq,
            x=x_single,
            mask_prior=mask_prior_arr,
        )

    def sample_theta_fn(rng_key: jax.Array) -> jax.Array:
        return model.sample_theta(
            rng=rng_key,
            acq=acq,
            x=x_single,
            model_mask=model_mask_single,
            num_steps=theta_num_steps,
            temperature=temperature,
        )

    def log_prob_theta_fn() -> jax.Array:
        return model.log_prob_theta(
            theta=theta_single,
            acq=acq,
            x=x_single,
            model_mask=model_mask_single,
            num_steps=theta_num_steps,
        )

    summary = {
        "kind": "model_summary",
        "checkpoint": str(checkpoint_path),
        "checkpoint_which": checkpoint_which,
        "backend": jax.default_backend(),
        "devices": [device.platform for device in jax.devices()],
        "acquisition": acquisition,
        "seed": seed,
        "mask_prior": mask_prior,
        "temperature": temperature,
        "theta_num_steps": theta_num_steps,
        "loss_batch_size": loss_batch_size,
        "train_stats_included": include_train_stats,
        "num_acquisitions": num_acquisitions,
        "theta_dim": int(model.cfg.simulator.theta_dim),
        "num_models": int(model.tokenizer.num_models),
        "num_noises": int(model.tokenizer.num_noises),
        "model_type": type(model).__name__,
        "encoder_type": type(model.encoder).__name__,
        "inference_decoder_type": type(model.inference_decoder).__name__,
        "model_decoder_type": type(model.model_decoder).__name__,
        "tokenizer_type": type(model.tokenizer).__name__,
        "parameter_counts": _component_parameter_counts(params),
    }
    cases = [
        ProfileCase(
            fn=sample_mask_fn,
            args=(jax.random.PRNGKey(seed + 10),),
            name="sample_mask",
            metadata={"operation": "sample_mask", "batch_size": 1},
            jit=True,
        ),
        ProfileCase(
            fn=log_prob_mask_fn,
            name="log_prob_mask",
            metadata={"operation": "log_prob_mask", "batch_size": 1},
            jit=True,
        ),
        ProfileCase(
            fn=sample_theta_fn,
            args=(jax.random.PRNGKey(seed + 11),),
            name="sample_theta",
            metadata={"operation": "sample_theta", "batch_size": 1},
            jit=True,
        ),
        ProfileCase(
            fn=log_prob_theta_fn,
            name="log_prob_theta",
            metadata={"operation": "log_prob_theta", "batch_size": 1},
            jit=True,
        ),
    ]
    if include_train_stats:
        x_batch = jax.random.uniform(
            rng_loss_x,
            shape=(loss_batch_size, num_acquisitions),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        loss_batch = {
            "model_mask": _broadcast_tree(model_mask_single, loss_batch_size),
            "theta": _broadcast_tree(theta_single, loss_batch_size),
            "x": x_batch,
            "acq": _broadcast_tree(acq, loss_batch_size),
            "mask_prior": _broadcast_tree(mask_prior_arr, loss_batch_size),
        }

        def loss_components_fn(
            params_tree: Any, rng_key: jax.Array, batch: dict[str, Any]
        ) -> jax.Array:
            loss_model = nnx.merge(graphdef, params_tree, state, copy=True)
            loss_model.train()
            return loss_model.loss_fn(
                rng_key,
                model_mask=batch["model_mask"],
                theta=batch["theta"],
                x=batch["x"],
                acq=batch["acq"],
                mask_prior=batch["mask_prior"],
            )

        def loss_forward_fn(
            params_tree: Any, rng_key: jax.Array, batch: dict[str, Any]
        ) -> jax.Array:
            return loss_components_fn(params_tree, rng_key, batch)

        def loss_backward_fn(
            params_tree: Any, rng_key: jax.Array, batch: dict[str, Any]
        ) -> dict[str, jax.Array]:
            def scalar_loss_fn(p: Any) -> jax.Array:
                return jnp.sum(loss_components_fn(p, rng_key, batch))

            loss_value, grads = jax.value_and_grad(scalar_loss_fn)(params_tree)
            grad_sq_norm = jax.tree_util.tree_reduce(
                lambda total, leaf: total + jnp.vdot(leaf, leaf).real,
                grads,
                jnp.array(0.0, dtype=jnp.float32),
            )
            return {
                "loss": loss_value,
                "grad_l2": jnp.sqrt(grad_sq_norm),
            }

        cases.extend(
            [
                ProfileCase(
                    fn=loss_forward_fn,
                    args=(params, jax.random.PRNGKey(seed + 12), loss_batch),
                    name="loss_forward",
                    metadata={"operation": "loss_forward", "batch_size": loss_batch_size},
                    jit=True,
                ),
                ProfileCase(
                    fn=loss_backward_fn,
                    args=(params, jax.random.PRNGKey(seed + 13), loss_batch),
                    name="loss_backward",
                    metadata={"operation": "loss_backward", "batch_size": loss_batch_size},
                    jit=True,
                ),
            ]
        )
    return summary, cases


def _case_artifact_path(
    base_path: str | Path | None, case_name: str, *, directory: bool
) -> Path | None:
    if base_path is None:
        return None
    base_path = Path(base_path)
    if directory or base_path.suffix == "":
        return base_path / case_name
    return base_path.with_name(f"{base_path.stem}_{case_name}{base_path.suffix}")


def profile_model_summary(
    checkpoint_path: str | Path,
    *,
    acquisition: str = "random_hcp_acq",
    mask_prior: float = 1.0,
    temperature: float = 1.0,
    theta_num_steps: int = 64,
    loss_batch_size: int = 2048,
    include_train_stats: bool = True,
    seed: int = 0,
    checkpoint_which: str = "latest",
    warmup: int = 1,
    repeat: int = 10,
    clear_caches: bool = False,
    capture_trace: bool = False,
    trace_dir: str | Path | None = None,
    capture_device_memory_profile: bool = False,
    memory_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    summary, cases = build_model_summary(
        checkpoint_path=checkpoint_path,
        acquisition=acquisition,
        mask_prior=mask_prior,
        temperature=temperature,
        theta_num_steps=theta_num_steps,
        loss_batch_size=loss_batch_size,
        include_train_stats=include_train_stats,
        seed=seed,
        checkpoint_which=checkpoint_which,
    )
    operations: dict[str, Any] = {}
    for case in cases:
        case_name = case.name or "unnamed_case"
        operations[case_name] = profile_case(
            case,
            warmup=warmup,
            repeat=repeat,
            clear_caches=clear_caches,
            capture_trace=capture_trace,
            trace_dir=_case_artifact_path(trace_dir, case_name, directory=True),
            capture_device_memory_profile=capture_device_memory_profile,
            memory_profile_path=_case_artifact_path(
                memory_profile_path, case_name, directory=False
            ),
        )
    summary["operations"] = operations
    return summary


def build_demo_profile_case(size: int = 2048) -> ProfileCase:
    x = jax.random.normal(jax.random.PRNGKey(0), (size, size), dtype=jnp.float32)
    y = jax.random.normal(jax.random.PRNGKey(1), (size, size), dtype=jnp.float32)

    def demo_fn(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.tanh(a @ b)

    return ProfileCase(
        fn=demo_fn,
        args=(x, y),
        name=f"demo_matmul_{size}",
        jit=True,
    )


def _resolve_dotted_name(target: str) -> Any:
    if ":" not in target:
        raise ValueError(
            f"Expected 'module:attribute' target, received {target!r}."
        )
    module_name, attr_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def _coerce_profile_case(obj: Any) -> ProfileCase:
    if isinstance(obj, ProfileCase):
        return obj
    if callable(obj):
        return ProfileCase(fn=obj)
    if isinstance(obj, tuple) and len(obj) == 3:
        fn, args, kwargs = obj
        return ProfileCase(fn=fn, args=tuple(args), kwargs=dict(kwargs))
    raise TypeError(
        "Profile factory must return a ProfileCase, a callable, or a "
        "(fn, args, kwargs) tuple."
    )


def _run_factory(target: str, factory_kwargs_json: str | None) -> ProfileCase:
    factory = _resolve_dotted_name(target)
    if not callable(factory):
        return _coerce_profile_case(factory)

    factory_kwargs = (
        json.loads(factory_kwargs_json) if factory_kwargs_json is not None else {}
    )
    return _coerce_profile_case(factory(**factory_kwargs))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile a JAX workload.")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "tpu", "cuda"),
        default=None,
        help="JAX device platform override. 'cuda' is accepted as an alias for 'gpu'.",
    )
    parser.add_argument(
        "--exclude-train-stats",
        action="store_true",
        help="Skip loss_forward/loss_backward training benchmarks for model-summary.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--clear-caches",
        action="store_true",
        help="Call jax.clear_caches() before compiling/running.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Capture a JAX profiler trace directory.",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument(
        "--device-memory-profile",
        action="store_true",
        help="Save a device memory profile after the profiling run.",
    )
    parser.add_argument("--memory-profile-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    factory_parser = subparsers.add_parser(
        "factory",
        help="Resolve a ProfileCase factory via module:attribute.",
    )
    factory_parser.add_argument(
        "target",
        help="Factory or object import path, for example "
        "'my_module:build_profile_case'.",
    )
    factory_parser.add_argument(
        "--factory-kwargs-json",
        default=None,
        help="JSON object passed as keyword arguments to the factory.",
    )

    model_summary_parser = subparsers.add_parser(
        "model-summary",
        aliases=["sample-mask"],
        help="Profile and summarize a DMRI checkpoint across core model functions.",
    )
    model_summary_parser.add_argument("--checkpoint", type=Path, required=True)
    model_summary_parser.add_argument(
        "--acquisition",
        choices=("random_hcp_acq", "random_large_hcp_acq"),
        default="random_hcp_acq",
        help="Synthetic acquisition generator to use.",
    )
    model_summary_parser.add_argument("--mask-prior", type=float, default=1.0)
    model_summary_parser.add_argument("--temperature", type=float, default=1.0)
    model_summary_parser.add_argument(
        "--theta-num-steps",
        type=int,
        default=64,
        help="Number of diffusion steps for theta sampling/log-prob benchmarking.",
    )
    model_summary_parser.add_argument(
        "--loss-batch-size",
        type=int,
        default=2048,
        help="Synthetic batch size for loss forward/backward benchmarking.",
    )
    model_summary_parser.add_argument(
        "--exclude-train-stats",
        action="store_true",
        help="Skip loss_forward/loss_backward training benchmarks.",
    )
    model_summary_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to generate the synthetic summary inputs.",
    )
    model_summary_parser.add_argument(
        "--checkpoint-which",
        default="latest",
        help="Checkpoint selector passed through to load_checkpoint().",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Profile a simple matrix multiplication demo workload.",
    )
    demo_parser.add_argument("--size", type=int, default=2048)

    return parser


def _format_nbytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    sign = -1.0 if value < 0 else 1.0
    size = abs(float(value))
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{sign * size:.2f} {unit}"
        size /= 1024.0
    return f"{sign * size:.2f} TiB"


def _format_mean_std_nbytes(summary: dict[str, float | int] | None) -> str:
    if not summary:
        return "n/a"
    mean = summary.get("mean")
    std = summary.get("std")
    if mean is None or std is None:
        return "n/a"
    return f"{_format_nbytes(int(round(float(mean))))} ± {_format_nbytes(int(round(float(std))))}"


def _print_profile_summary(result: dict[str, Any]) -> None:
    steady = result["steady_state"]
    device_memory_first_delta = result.get("device_memory_first_delta") or {}
    device_memory_steady_delta = result.get("device_memory_steady_delta") or {}
    device_memory_steady_run_summary = result.get("device_memory_steady_run_summary") or {}
    print(f"name: {result['name']}")
    print(f"backend: {result['backend']}")
    print(f"jit: {result['jit']}")
    print(f"input_nbytes: {result['input_nbytes']}")
    print(f"output_nbytes: {result['output_nbytes']}")
    if result["compile_s"] is not None:
        print(f"compile_s: {result['compile_s']:.6f}")
    print(f"first_exec_s: {result['first_exec_s']:.6f}")
    print(f"steady_mean_s: {steady['mean_s']:.6f}")
    print(f"steady_median_s: {steady['median_s']:.6f}")
    print(f"steady_std_s: {steady['std_s']:.6f}")
    print(f"steady_min_s: {steady['min_s']:.6f}")
    print(f"steady_max_s: {steady['max_s']:.6f}")
    print(
        "actual_first_mem: "
        f"in_use={_format_nbytes(device_memory_first_delta.get('bytes_in_use'))} "
        f"reserved={_format_nbytes(device_memory_first_delta.get('bytes_reserved'))} "
        f"peak_in_use={_format_nbytes(device_memory_first_delta.get('peak_bytes_in_use'))}"
    )
    print(
        "actual_steady_mem: "
        f"in_use={_format_nbytes(device_memory_steady_delta.get('bytes_in_use'))} "
        f"reserved={_format_nbytes(device_memory_steady_delta.get('bytes_reserved'))} "
        f"peak_in_use={_format_nbytes(device_memory_steady_delta.get('peak_bytes_in_use'))}"
    )
    print(
        "actual_steady_mem_std: "
        f"in_use={_format_mean_std_nbytes(device_memory_steady_run_summary.get('bytes_in_use'))} "
        f"reserved={_format_mean_std_nbytes(device_memory_steady_run_summary.get('bytes_reserved'))} "
        f"peak_in_use={_format_mean_std_nbytes(device_memory_steady_run_summary.get('peak_bytes_in_use'))}"
    )
    if "acquisition" in result:
        print(f"acquisition: {result['acquisition']}")
    if "num_acquisitions" in result:
        print(f"num_acquisitions: {result['num_acquisitions']}")
    if result["artifacts"]:
        print("artifacts:")
        for key, value in result["artifacts"].items():
            print(f"  {key}: {value}")


def _print_model_summary(summary: dict[str, Any]) -> None:
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"checkpoint_which: {summary['checkpoint_which']}")
    print(f"backend: {summary['backend']}")
    print(f"devices: {summary['devices']}")
    print(f"acquisition: {summary['acquisition']}")
    print(f"num_acquisitions: {summary['num_acquisitions']}")
    print(f"theta_dim: {summary['theta_dim']}")
    print(f"num_models: {summary['num_models']}")
    print(f"num_noises: {summary['num_noises']}")
    print(f"model_type: {summary['model_type']}")
    print(f"encoder_type: {summary['encoder_type']}")
    print(f"inference_decoder_type: {summary['inference_decoder_type']}")
    print(f"model_decoder_type: {summary['model_decoder_type']}")
    print(f"tokenizer_type: {summary['tokenizer_type']}")
    print(f"theta_num_steps: {summary['theta_num_steps']}")
    print(f"loss_batch_size: {summary['loss_batch_size']}")
    print(f"train_stats_included: {summary['train_stats_included']}")
    print("parameter_counts:")
    for key, value in summary["parameter_counts"].items():
        print(f"  {key}: {value:,}")
    print("operations:")
    for name, result in summary["operations"].items():
        steady = result["steady_state"]
        memory = result.get("compiler_memory_estimate") or {}
        device_memory_first_delta = result.get("device_memory_first_delta") or {}
        device_memory_steady_delta = result.get("device_memory_steady_delta") or {}
        device_memory_steady_run_summary = (
            result.get("device_memory_steady_run_summary") or {}
        )
        compile_s = result.get("compile_s")
        compile_str = "n/a" if compile_s is None else f"{compile_s:.6f}s"
        print(
            "  "
            f"{name}: "
            f"batch={result.get('batch_size', 'n/a')} "
            f"compile={compile_str} "
            f"first={result['first_exec_s']:.6f}s "
            f"mean={steady['mean_s']:.6f}s "
            f"median={steady['median_s']:.6f}s "
            f"std={steady['std_s']:.6f}s "
            f"temp_mem={_format_nbytes(memory.get('temp_size_in_bytes'))} "
            f"args_mem={_format_nbytes(memory.get('argument_size_in_bytes'))} "
            f"out_mem={_format_nbytes(memory.get('output_size_in_bytes'))} "
            f"live_delta={_format_nbytes(result.get('live_arrays_delta_nbytes'))}"
        )
        print(
            "    "
            "actual_first_mem: "
            f"in_use={_format_nbytes(device_memory_first_delta.get('bytes_in_use'))} "
            f"reserved={_format_nbytes(device_memory_first_delta.get('bytes_reserved'))} "
            f"peak_in_use={_format_nbytes(device_memory_first_delta.get('peak_bytes_in_use'))}"
        )
        print(
            "    "
            "actual_steady_mem: "
            f"in_use={_format_nbytes(device_memory_steady_delta.get('bytes_in_use'))} "
            f"reserved={_format_nbytes(device_memory_steady_delta.get('bytes_reserved'))} "
            f"peak_in_use={_format_nbytes(device_memory_steady_delta.get('peak_bytes_in_use'))}"
        )
        print(
            "    "
            "actual_steady_mem_std: "
            f"in_use={_format_mean_std_nbytes(device_memory_steady_run_summary.get('bytes_in_use'))} "
            f"reserved={_format_mean_std_nbytes(device_memory_steady_run_summary.get('bytes_reserved'))} "
            f"peak_in_use={_format_mean_std_nbytes(device_memory_steady_run_summary.get('peak_bytes_in_use'))}"
        )
        if result["artifacts"]:
            print(f"    artifacts: {result['artifacts']}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "factory":
        case = _run_factory(args.target, args.factory_kwargs_json)
        result = profile_case(
            case,
            warmup=args.warmup,
            repeat=args.repeat,
            clear_caches=args.clear_caches,
            capture_trace=args.trace,
            trace_dir=args.trace_dir,
            capture_device_memory_profile=args.device_memory_profile,
            memory_profile_path=args.memory_profile_path,
        )
    elif args.command in {"model-summary", "sample-mask"}:
        result = profile_model_summary(
            checkpoint_path=args.checkpoint,
            acquisition=args.acquisition,
            mask_prior=args.mask_prior,
            temperature=args.temperature,
            theta_num_steps=args.theta_num_steps,
            loss_batch_size=args.loss_batch_size,
            include_train_stats=not args.exclude_train_stats,
            seed=args.seed,
            checkpoint_which=args.checkpoint_which,
            warmup=args.warmup,
            repeat=args.repeat,
            clear_caches=args.clear_caches,
            capture_trace=args.trace,
            trace_dir=args.trace_dir,
            capture_device_memory_profile=args.device_memory_profile,
            memory_profile_path=args.memory_profile_path,
        )
    elif args.command == "demo":
        case = build_demo_profile_case(size=args.size)
        result = profile_case(
            case,
            warmup=args.warmup,
            repeat=args.repeat,
            clear_caches=args.clear_caches,
            capture_trace=args.trace,
            trace_dir=args.trace_dir,
            capture_device_memory_profile=args.device_memory_profile,
            memory_profile_path=args.memory_profile_path,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        if "operations" in result:
            result["result_json"] = str(output_path)
        else:
            result["artifacts"]["result_json"] = str(output_path)
        save_profile_result(result, output_path)

    if "operations" in result:
        _print_model_summary(result)
    else:
        _print_profile_summary(result)


if __name__ == "__main__":
    main()
