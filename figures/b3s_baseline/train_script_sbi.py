from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency at runtime
    wandb = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmri.simulators.acquisition_scheme import (
    random_hcp_acquisition,
    random_hcp_large_acquisition,
)
from dmri.simulators.models import MultiShellBall3StickSharedDiffusivity, Ball3StickSharedDiffusivity
from dmri.train.dataset import SimulationDataset

from sbi_embedding_net import SBIEmbeddingNet
from torch_sbi_baseline import build_nsf


LOG = logging.getLogger("train_script_sbi")
DEFAULT_SIM_TYPE = MultiShellBall3StickSharedDiffusivity
SIM_TYPE_REGISTRY = {
    MultiShellBall3StickSharedDiffusivity.__name__: MultiShellBall3StickSharedDiffusivity,
    Ball3StickSharedDiffusivity.__name__: Ball3StickSharedDiffusivity,
}
SIM_TYPE_CHOICES = tuple(SIM_TYPE_REGISTRY)
SIM_TYPE = DEFAULT_SIM_TYPE


@dataclass
class TrainConfig:
    output_dir: str | None
    run_name: str | None
    seed: int
    device: str | None
    num_steps: int
    batch_size: int
    simulation_batch_size: int
    buffer_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    log_every: int
    eval_every: int
    checkpoint_every: int
    keep_last_n: int
    eval_batch_size: int
    num_eval_samples: int
    embedding_model_dim: int
    embedding_output_dim: int
    embedding_num_cls_tokens: int
    embedding_num_layers: int
    hidden_features: int
    num_transforms: int
    num_bins: int
    tail_bound: float
    wandb_project: str | None
    wandb_entity: str | None
    wandb_mode: str
    resume: str | None
    sim_type: str = DEFAULT_SIM_TYPE.__name__


class TorchReadySimulationDataset(Dataset):
    def __init__(self, base_dataset: SimulationDataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        acq = sample["acq"]
        sample["acq"] = {
            "bvals": np.asarray(acq.bvals),
            "bvecs": np.asarray(acq.bvecs),
        }
        for key, value in sample.items():
            if key == "acq":
                continue
            sample[key] = np.asarray(value)
        return sample


def resolve_sim_type(sim_type_name: str | None) -> type:
    resolved_name = sim_type_name or DEFAULT_SIM_TYPE.__name__
    try:
        return SIM_TYPE_REGISTRY[resolved_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported sim_type {resolved_name!r}. Expected one of {SIM_TYPE_CHOICES}."
        ) from exc


def build_simulator(
    sim_type_name: str,
    *,
    large_acquisition: bool,
) -> Any:
    sim_type = resolve_sim_type(sim_type_name)
    model_mask_dim = len(sim_type.model_types) + len(sim_type.noise_types)

    def simulator(
        rng: jax.Array,
        mask_prior_hyperparameter: Any = None,
    ) -> dict[str, Any]:
        del mask_prior_hyperparameter
        rng0, rng1, _, _, _, rng5 = jax.random.split(rng, 6)
        if large_acquisition:
            acq = random_hcp_large_acquisition(
                rng0,
                num_acquisitions=297,
                typical_prob=0.8,
                random_prob=0.2,
            )
        else:
            acq = random_hcp_acquisition(rng0)
        theta = jax.random.normal(rng1, shape=(sim_type.theta_dim,))
        model_mask = jnp.ones((model_mask_dim,), dtype=bool)
        ball_stick = sim_type.from_theta(theta, model_mask=model_mask)
        x_o = ball_stick.signal(acq, rng=rng5)
        return {
            "acq": acq,
            "model_mask": model_mask,
            "prior_mask": jnp.ones((1,)),
            "theta": theta,
            "x": x_o,
        }

    return jax.jit(simulator)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a dict-conditioned SBI NSF baseline for B3S.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--simulation-batch-size", type=int, default=4096)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--keep-last-n", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-eval-samples", type=int, default=4)
    parser.add_argument("--embedding-model-dim", type=int, default=128)
    parser.add_argument("--embedding-output-dim", type=int, default=512)
    parser.add_argument("--embedding-num-cls-tokens", type=int, default=4)
    parser.add_argument("--embedding-num-layers", type=int, default=4)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--num-transforms", type=int, default=8)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--tail-bound", type=float, default=3.0)
    parser.add_argument("--wandb-project", type=str, default="dmri-sbi")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--sim-type",
        type=str,
        choices=SIM_TYPE_CHOICES,
        default=DEFAULT_SIM_TYPE.__name__,
    )
    return TrainConfig(**vars(parser.parse_args()))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runs_root() -> Path:
    return (Path(__file__).resolve().parent / "runs").resolve()


def resolve_resume_path(resume: str) -> Path:
    raw_path = Path(resume).expanduser()
    search_paths: list[Path] = []

    if raw_path.is_absolute():
        search_paths.append(raw_path)
    else:
        search_paths.extend([
            (Path.cwd() / raw_path),
            (Path(__file__).resolve().parent / raw_path),
            (runs_root() / raw_path),
        ])

    for candidate in search_paths:
        if candidate.exists():
            return candidate.resolve()

    if raw_path.is_absolute():
        return raw_path.resolve()
    return (runs_root() / raw_path).resolve()


def resolve_run_dir(cfg: TrainConfig) -> Path:
    if cfg.resume is not None:
        resume_path = resolve_resume_path(cfg.resume)
        if resume_path.is_file():
            return (
                resume_path.parent.parent
                if resume_path.parent.name == "checkpoints"
                else resume_path.parent
            )
        if resume_path.name == "checkpoints":
            return resume_path.parent
        return resume_path

    if cfg.output_dir is not None:
        return Path(cfg.output_dir).expanduser().resolve()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"sbi_{timestamp}"
    return (runs_root() / run_name).resolve()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(resume_path: str, run_dir: Path) -> Path:
    candidate = resolve_resume_path(resume_path)
    if candidate.is_dir() and candidate.name == "checkpoints":
        candidate = candidate / "latest.pt"
    elif candidate.is_dir():
        candidate = candidate / "checkpoints" / "latest.pt"
    if candidate.exists():
        return candidate

    fallback = run_dir / "checkpoints" / "latest.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")


def checkpoint_sim_type(payload: Mapping[str, Any]) -> str:
    checkpoint_cfg = payload.get("config", {})
    sim_type_name = payload.get("sim_type")
    if sim_type_name is None and isinstance(checkpoint_cfg, Mapping):
        sim_type_name = checkpoint_cfg.get("sim_type")
    return resolve_sim_type(sim_type_name).__name__


def maybe_override_sim_type_from_resume(cfg: TrainConfig, run_dir: Path) -> TrainConfig:
    if cfg.resume is None:
        return cfg

    checkpoint_path = resolve_checkpoint_path(cfg.resume, run_dir)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resume_sim_type = checkpoint_sim_type(payload)
    if resume_sim_type == cfg.sim_type:
        return cfg
    if cfg.sim_type != DEFAULT_SIM_TYPE.__name__:
        raise ValueError(
            "Checkpoint simulator type mismatch: "
            f"checkpoint uses {resume_sim_type}, current config uses {cfg.sim_type}. "
            "Resume with a matching --sim-type."
        )
    LOG.info(
        "Using simulator type %s from resume checkpoint %s",
        resume_sim_type,
        checkpoint_path,
    )
    cfg.sim_type = resume_sim_type
    return cfg


def save_config(cfg: TrainConfig, run_dir: Path) -> None:
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))


def init_wandb(cfg: TrainConfig, run_dir: Path) -> Any:
    if cfg.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed but wandb logging was requested.")
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name or run_dir.name,
        dir=str(run_dir),
        config=asdict(cfg),
        mode=cfg.wandb_mode,
    )


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Mapping):
        return {key: to_device(value, device) for key, value in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )

    def __iter__(self):
        if self.stream is None:
            for batch in self.loader:
                yield batch
            return

        loader_iter = iter(self.loader)
        next_batch = None

        def preload() -> None:
            nonlocal next_batch
            try:
                batch = next(loader_iter)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = to_device(batch, self.device)

        preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = next_batch
            preload()
            yield batch


def split_batch(batch: Mapping[str, Any]) -> tuple[Tensor, dict[str, Any]]:
    theta = batch["theta"].float()
    condition = {
        "x": batch["x"].float(),
        "acq": {
            "bvals": batch["acq"]["bvals"].float(),
            "bvecs": batch["acq"]["bvecs"].float(),
        },
    }
    return theta, condition


def slice_batch(batch: Any, stop: int) -> Any:
    if torch.is_tensor(batch):
        return batch[:stop]
    if isinstance(batch, Mapping):
        return {key: slice_batch(value, stop) for key, value in batch.items()}
    raise TypeError(f"Unsupported batch leaf type: {type(batch)}")


def build_data_pipeline(
    cfg: TrainConfig, device: torch.device
) -> tuple[list[SimulationDataset], list[CUDAPrefetcher]]:
    simulators = [
        build_simulator(cfg.sim_type, large_acquisition=False),
        build_simulator(cfg.sim_type, large_acquisition=True),
    ]
    datasets: list[SimulationDataset] = []
    prefetch_loaders: list[CUDAPrefetcher] = []

    for index, simulator_fn in enumerate(simulators):
        dataset = SimulationDataset(
            simulator_fn,
            rng=jax.random.PRNGKey(cfg.seed + 1 + index),
            simulation_device=jax.devices("cpu")[0],
            simulation_batch_size=cfg.simulation_batch_size,
            buffer_size=cfg.buffer_size,
            return_numpy=True,
        )
        torch_dataset = TorchReadySimulationDataset(dataset)
        dataloader = DataLoader(
            torch_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
            num_workers=cfg.num_workers,
        )
        datasets.append(dataset)
        prefetch_loaders.append(CUDAPrefetcher(dataloader, device))

    return datasets, prefetch_loaders


def build_flow_from_batch(
    cfg: TrainConfig,
    device: torch.device,
    batch: Mapping[str, Any],
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    embedding_net = SBIEmbeddingNet(
        model_dim=cfg.embedding_model_dim,
        output_dim=cfg.embedding_output_dim,
        num_cls_tokens=cfg.embedding_num_cls_tokens,
        num_layers=cfg.embedding_num_layers,
    ).to(device)
    theta, condition = split_batch(batch)
    flow = build_nsf(
        batch_x=theta,
        batch_y=condition,
        embedding_net=embedding_net,
        z_score_y="none",
        hidden_features=cfg.hidden_features,
        num_transforms=cfg.num_transforms,
        num_bins=cfg.num_bins,
        tail_bound=cfg.tail_bound,
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    return flow, optimizer


def total_loss(flow: torch.nn.Module, *batches: Mapping[str, Any]) -> Tensor:
    losses = [flow.loss(*split_batch(batch)).mean() for batch in batches]
    return torch.stack(losses).mean()


def next_batch(
    prefetch_loader: CUDAPrefetcher,
    iterator: Any | None,
) -> tuple[Mapping[str, Any], Any]:
    if iterator is None:
        iterator = iter(prefetch_loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(prefetch_loader)
        batch = next(iterator)
    return batch, iterator


def evaluate(
    flow: torch.nn.Module,
    prefetch_loaders: list[CUDAPrefetcher],
    eval_batch_size: int,
    num_eval_samples: int,
) -> dict[str, float]:
    flow.eval()
    loss_values: list[float] = []
    sample_means: list[float] = []
    sample_stds: list[float] = []
    log_prob_means: list[float] = []
    with torch.no_grad():
        metrics: dict[str, float] = {}
        for index, prefetch_loader in enumerate(prefetch_loaders):
            eval_batch = slice_batch(next(iter(prefetch_loader)), eval_batch_size)
            eval_theta, eval_condition = split_batch(eval_batch)
            eval_loss = flow.loss(eval_theta, eval_condition).mean()
            samples, log_probs = flow.sample_and_log_prob(
                torch.Size([num_eval_samples]),
                eval_condition,
            )
            loss_value = float(eval_loss.detach().cpu())
            sample_mean = float(samples.mean().detach().cpu())
            sample_std = float(samples.std().detach().cpu())
            log_prob_mean = float(log_probs.mean().detach().cpu())
            metrics[f"eval_loss_seq_{index}"] = loss_value
            metrics[f"sample_mean_seq_{index}"] = sample_mean
            metrics[f"sample_std_seq_{index}"] = sample_std
            metrics[f"log_prob_mean_seq_{index}"] = log_prob_mean
            loss_values.append(loss_value)
            sample_means.append(sample_mean)
            sample_stds.append(sample_std)
            log_prob_means.append(log_prob_mean)
    flow.train()
    metrics["eval_loss"] = float(np.mean(loss_values))
    metrics["sample_mean"] = float(np.mean(sample_means))
    metrics["sample_std"] = float(np.mean(sample_stds))
    metrics["log_prob_mean"] = float(np.mean(log_prob_means))
    return metrics


def checkpoint_payload(
    flow: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    best_eval_loss: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_eval_loss": best_eval_loss,
        "sim_type": cfg.sim_type,
        "config": asdict(cfg),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def cleanup_old_checkpoints(step_dir: Path, keep_last_n: int) -> None:
    checkpoints = sorted(step_dir.glob("step_*.pt"))
    if len(checkpoints) <= keep_last_n:
        return
    for path in checkpoints[: len(checkpoints) - keep_last_n]:
        path.unlink(missing_ok=True)


def save_checkpoint(
    run_dir: Path,
    flow: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    best_eval_loss: float,
    is_best: bool,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(flow, optimizer, cfg, step, best_eval_loss)

    latest_path = checkpoint_dir / "latest.pt"
    step_path = checkpoint_dir / f"step_{step:08d}.pt"
    torch.save(payload, latest_path)
    torch.save(payload, step_path)
    cleanup_old_checkpoints(checkpoint_dir, cfg.keep_last_n)
    LOG.info("Saved checkpoint: %s", step_path)

    if is_best:
        torch.save(payload, checkpoint_dir / "best.pt")
        LOG.info("Updated best checkpoint at step %d", step)


def load_checkpoint(
    resume_path: str | None,
    run_dir: Path,
    cfg: TrainConfig,
    flow: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    if resume_path is None:
        return 0, float("inf")

    candidate = resolve_checkpoint_path(resume_path, run_dir)
    LOG.info("Resuming from %s", candidate)
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    resume_sim_type = checkpoint_sim_type(payload)
    if resume_sim_type != cfg.sim_type:
        raise ValueError(
            "Checkpoint simulator type mismatch: "
            f"checkpoint uses {resume_sim_type}, current config uses {cfg.sim_type}. "
            "Resume with a matching --sim-type."
        )
    flow.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    optimizer_to_device(optimizer, device)
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return int(payload["step"]), float(payload.get("best_eval_loss", float("inf")))


def maybe_log_wandb(metrics: dict[str, float], step: int) -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.log(metrics, step=step)


def optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main() -> None:
    configure_logging()
    cfg = parse_args()
    cfg.sim_type = resolve_sim_type(cfg.sim_type).__name__
    seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(cfg.device)
    run_dir = resolve_run_dir(cfg)
    cfg = maybe_override_sim_type_from_resume(cfg, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir)

    LOG.info("Run directory: %s", run_dir)
    LOG.info("Device: %s", device)
    LOG.info("Simulator type: %s", cfg.sim_type)
    LOG.info("Config: %s", json.dumps(asdict(cfg), indent=2, sort_keys=True))

    run = init_wandb(cfg, run_dir)
    datasets: list[SimulationDataset] = []

    try:
        LOG.info("Building simulation datasets and dataloaders...")
        datasets, prefetch_loaders = build_data_pipeline(cfg, device)
        LOG.info("Initialising flow from first batch...")
        first_batch = next(iter(prefetch_loaders[0]))
        flow, optimizer = build_flow_from_batch(cfg, device, first_batch)
        start_step, best_eval_loss = load_checkpoint(
            cfg.resume,
            run_dir,
            cfg,
            flow,
            optimizer,
            device,
        )

        flow.train()
        train_start_time = time.perf_counter()
        loss_window: list[float] = []
        loader_iters: list[Any | None] = [None for _ in prefetch_loaders]
        LOG.info("Starting training loop at step %d / %d", start_step, cfg.num_steps)

        for step in range(start_step + 1, cfg.num_steps + 1):
            if step > cfg.num_steps:
                break

            optimizer.zero_grad(set_to_none=True)
            step_batches = []
            total_examples = 0
            for index, prefetch_loader in enumerate(prefetch_loaders):
                batch, loader_iters[index] = next_batch(
                    prefetch_loader,
                    loader_iters[index],
                )
                step_batches.append(batch)
                total_examples += int(batch["theta"].shape[0])

            loss = total_loss(flow, *step_batches)
            loss.backward()
            grad_norm = clip_grad_norm_(flow.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            loss_window.append(loss_value)
            if len(loss_window) > cfg.log_every:
                loss_window.pop(0)

            if step % cfg.log_every == 0 or step == start_step + 1:
                elapsed = time.perf_counter() - train_start_time
                examples_per_second = (
                    (step - start_step) * total_examples / max(elapsed, 1e-6)
                )
                train_metrics = {
                    "train/loss": loss_value,
                    "train/avg_loss": sum(loss_window) / len(loss_window),
                    "train/grad_norm": float(grad_norm),
                    "train/examples_per_second": examples_per_second,
                    "train/step": float(step),
                }
                LOG.info(
                    "[%d/%d] loss=%.4f avg_loss=%.4f grad_norm=%.2f examples_per_second=%.1f",
                    step,
                    cfg.num_steps,
                    train_metrics["train/loss"],
                    train_metrics["train/avg_loss"],
                    train_metrics["train/grad_norm"],
                    train_metrics["train/examples_per_second"],
                )
                maybe_log_wandb(train_metrics, step)

            should_eval = step % cfg.eval_every == 0 or step == cfg.num_steps
            is_best = False
            if should_eval:
                eval_metrics = evaluate(
                    flow,
                    prefetch_loaders,
                    cfg.eval_batch_size,
                    cfg.num_eval_samples,
                )
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                LOG.info(
                    "[%d/%d] eval_loss=%.4f sample_mean=%.4f sample_std=%.4f log_prob_mean=%.4f",
                    step,
                    cfg.num_steps,
                    eval_metrics["eval/eval_loss"],
                    eval_metrics["eval/sample_mean"],
                    eval_metrics["eval/sample_std"],
                    eval_metrics["eval/log_prob_mean"],
                )
                maybe_log_wandb(eval_metrics, step)
                if eval_metrics["eval/eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval/eval_loss"]
                    is_best = True

            should_checkpoint = (
                step % cfg.checkpoint_every == 0
                or step == cfg.num_steps
                or is_best
            )
            if should_checkpoint:
                save_checkpoint(
                    run_dir,
                    flow,
                    optimizer,
                    cfg,
                    step,
                    best_eval_loss,
                    is_best=is_best,
                )

        LOG.info("Training finished. Best eval loss: %.4f", best_eval_loss)
        save_checkpoint(
            run_dir,
            flow,
            optimizer,
            cfg,
            cfg.num_steps,
            best_eval_loss,
            is_best=False,
        )
    finally:
        for dataset in datasets:
            dataset.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
