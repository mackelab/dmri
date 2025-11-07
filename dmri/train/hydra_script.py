import importlib
import logging
import os
import random
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

# memory_fraction = 0.9  # Use 98% of available memory
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
import hydra
import jax
import numpy as np
import optax
import wandb
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from dmri.train.build_model import build_model
from dmri.train.build_simulator import build_simulator
from dmri.train.checkpointing import CheckpointManager
from dmri.train.dataset import instantiate_dataloader
from dmri.train.evaluator import build_pure_eval_fns

# Backends


logo = r"""

 /$$$$$$$  /$$      /$$ /$$$$$$$  /$$$$$$
| $$__  $$| $$$    /$$$| $$__  $$|_  $$_/
| $$  \ $$| $$$$  /$$$$| $$  \ $$  | $$
| $$  | $$| $$ $$/$$ $$| $$$$$$$/  | $$
| $$  | $$| $$  $$$| $$| $$__  $$  | $$
| $$  | $$| $$\  $ | $$| $$  \ $$  | $$
| $$$$$$$/| $$ \/  | $$| $$  | $$ /$$$$$$
|_______/ |__/     |__/|__/  |__/|______/
"""


@dataclass
class TrainState:
    params: Any
    model_state: Any
    opt_state: Any
    ema_state: Any
    rng: Any
    step: int = 0


def tree_copy(pytree: Any) -> Any:
    """Create a shallow copy of a pytree."""
    return jax.tree_util.tree_map(lambda x: x, pytree)


def configure_environment(cfg: DictConfig) -> tuple[logging.Logger, str, str]:
    """Configure logging and resolve Hydra output directories."""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    output_super_dir = os.path.dirname(output_dir)
    while os.path.basename(output_super_dir) != cfg.name:
        parent = os.path.dirname(output_super_dir)
        if parent == output_super_dir:
            break
        output_super_dir = parent
    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info(f"Output super directory: {output_super_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")
    return log, output_dir, output_super_dir


def init_wandb_if_needed(cfg: DictConfig) -> bool:
    """Initialise wandb if requested."""
    if not cfg.use_wandb:
        return False
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.name)
    return True


def seed_everything(seed: int) -> jax.Array:
    """Seed Python, numpy and JAX RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.key(seed)


def _normalise_loader_params(params_cfg: Any, count: int) -> list[dict[Any, Any]]:
    """Convert Hydra loader params into a list of dictionaries."""
    if params_cfg is None:
        return [dict() for _ in range(count)]
    if isinstance(params_cfg, (DictConfig, ListConfig)):
        container = OmegaConf.to_container(params_cfg, resolve=True)
    else:
        container = params_cfg
    if container is None:
        return [dict() for _ in range(count)]
    if isinstance(container, list):
        entries = [dict(entry) for entry in container]
        if len(entries) < count:
            last_entry = entries[-1] if entries else {}
            for _ in range(count - len(entries)):
                entries.append(dict(last_entry))
        return entries[:count]
    if isinstance(container, dict):
        return [dict(container) for _ in range(count)]
    raise TypeError(f"Unsupported loader parameter type: {type(container)}")


def _resolve_device_spec(
    device_spec: Any, *, default: Optional[jax.Device] = None
) -> Optional[jax.Device]:
    """Resolve a device specification into a concrete ``jax.Device``."""
    if device_spec is None:
        return default
    if isinstance(device_spec, str):
        kind_part, _, index_part = device_spec.strip().partition(":")
        kind = kind_part or None
        index = int(index_part) if index_part else 0
    elif isinstance(device_spec, dict):
        kind = device_spec.get("kind")
        index = device_spec.get("index", 0)
    else:
        raise TypeError(f"Unsupported device specification type: {type(device_spec)}")
    devices = jax.devices(kind) if kind else jax.devices()
    if not devices:
        raise ValueError(f"No devices available for specification {device_spec!r}")
    index = int(index)
    if index < 0 or index >= len(devices):
        raise ValueError(
            f"Device index {index} out of range for specification {device_spec!r}"
        )
    return devices[index]


def _ensure_prng_key(value: Any) -> jax.Array:
    """Normalise user-provided RNG representations to a JAX PRNGKey."""
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, (np.ndarray, list, tuple)):
        raise TypeError(
            "Provide integer seeds for RNG configuration, not array-like values."
        )
    return jax.random.PRNGKey(int(value))


def create_dataloaders(
    cfg: DictConfig, simulators: Sequence[Any]
) -> tuple[list[Any], Any]:
    """Instantiate training and evaluation dataloaders."""
    dataset_module = importlib.import_module("dmri.train.dataset")
    loader_name = cfg.train.dataloader.name
    dataset_type = getattr(dataset_module, loader_name)
    simulators = list(simulators)
    if not simulators:
        raise ValueError("Expected at least one simulator callable.")

    loader_cfg_dict = OmegaConf.to_container(cfg.train.dataloader, resolve=True)
    if not isinstance(loader_cfg_dict, dict):
        raise TypeError("cfg.train.dataloader must resolve to a mapping.")

    dataset_base_cfg = loader_cfg_dict.get("dataset")
    if not isinstance(dataset_base_cfg, dict):
        raise ValueError(
            "cfg.train.dataloader.dataset must be defined and resolve to a mapping."
        )

    train_dataset_overrides = _normalise_loader_params(
        loader_cfg_dict.get("train_dataset"), len(simulators)
    )
    val_dataset_override = _normalise_loader_params(
        loader_cfg_dict.get("val_dataset"), 1
    )[0]

    train_loader_section = loader_cfg_dict.get("train_loader")
    if train_loader_section is None:
        train_loader_section = loader_cfg_dict.get("train_params")
    val_loader_section = loader_cfg_dict.get("val_loader")
    if val_loader_section is None:
        val_loader_section = loader_cfg_dict.get("val_params")

    train_loader_overrides = _normalise_loader_params(
        train_loader_section, len(simulators)
    )

    if not any(train_loader_overrides):
        default_train_loader = {
            key: loader_cfg_dict[key]
            for key in ("batch_size", "shuffle", "drop_last")
            if key in loader_cfg_dict
        }
        if not default_train_loader:
            raise ValueError(
                "cfg.train.dataloader.train_loader.batch_size (or legacy keys) must be provided."
            )
        train_loader_overrides = [
            dict(default_train_loader) for _ in range(len(simulators))
        ]

    for override in train_loader_overrides:
        if override.get("batch_size") is None:
            raise ValueError(
                "Each train dataloader configuration must specify batch_size."
            )

    val_loader_override = (
        _normalise_loader_params(val_loader_section, 1)[0]
        if val_loader_section is not None
        else {}
    )

    def build_dataset_params(
        base: dict[str, Any], override: dict[str, Any], rng
    ) -> dict[str, Any]:
        params = dict(base)
        params.update(override)

        batch_size = params.get("simulation_batch_size")
        if batch_size is None:
            raise ValueError(
                "cfg.train.dataloader.dataset.batch_size must be specified."
            )
        buffer_size = params.get("buffer_size", 4096)
        jit_simulator = params.get("jit_simulator", True)

        params["simulation_batch_size"] = int(batch_size)
        params["buffer_size"] = max(1, int(buffer_size))
        params["jit_simulator"] = bool(jit_simulator)

        params["return_numpy"] = bool(params.get("return_numpy", False))

        simulation_device_spec = params.pop("simulation_device", None)
        default_device = jax.devices()[0]
        params["simulation_device"] = _resolve_device_spec(
            simulation_device_spec, default=default_device
        )

        provided_rng = params.pop("rng", None)
        provided_seed = params.pop("seed", None)
        if provided_rng is not None:
            params["rng"] = _ensure_prng_key(provided_rng)
        elif provided_seed is not None:
            params["rng"] = jax.random.PRNGKey(int(provided_seed))
        else:
            params["rng"] = rng
        return params

    def build_loader_params(base_cfg: dict[str, Any], *, seed: int) -> dict[str, Any]:
        params = dict(base_cfg)
        batch_size = params.get("batch_size")
        if batch_size is None:
            raise ValueError("Each dataloader configuration must specify batch_size.")
        params["batch_size"] = int(batch_size)
        if params.get("seed") is None:
            params["seed"] = seed
        return params

    base_seed = int(cfg.seed)
    rngs = jax.random.split(jax.random.PRNGKey(base_seed), len(simulators) + 1)
    train_rngs = rngs[: len(simulators)]
    val_rng = rngs[-1]

    train_datasets = []
    for simulator, override, rng in zip(
        simulators, train_dataset_overrides, train_rngs
    ):
        dataset_params = build_dataset_params(dataset_base_cfg, override, rng)
        init = jax.vmap(simulator)(jax.random.split(rng, 2**14))
        dataset = dataset_type(simulator, **dataset_params)
        dataset.set_data(init)
        train_datasets.append(dataset)

    val_dataset_params = build_dataset_params(
        dataset_base_cfg, val_dataset_override, val_rng
    )
    val_dataset = dataset_type(simulators[0], **val_dataset_params)

    train_loaders = []
    for index, (dataset, loader_cfg) in enumerate(
        zip(train_datasets, train_loader_overrides)
    ):
        loader_params = build_loader_params(loader_cfg, seed=base_seed + index)
        train_loaders.append(
            instantiate_dataloader(
                dataset,
                loader_params,
                seed=base_seed + index,
            )
        )

    if val_loader_override.get("batch_size") is None:
        val_loader_override["batch_size"] = train_loader_overrides[0]["batch_size"]
    val_loader_params = build_loader_params(
        val_loader_override, seed=base_seed + len(simulators)
    )
    val_loader = instantiate_dataloader(
        val_dataset,
        val_loader_params,
        seed=base_seed + len(simulators),
        default_shuffle=False,
        default_drop_last=False,
    )
    return train_loaders, val_loader


def align_to_inner_steps(
    value: Optional[int], inner_steps: int, name: str, log: logging.Logger
) -> Optional[int]:
    """Align cadence values to multiples of inner_steps; warn if adjustment is needed."""
    if value is None:
        return None
    if inner_steps <= 0:
        return value
    aligned = (value // inner_steps) * inner_steps
    if aligned == 0 and value > 0:
        log.warning(
            f"{name} ({value}) is smaller than inner_steps ({inner_steps}); using inner_steps instead."
        )
        aligned = inner_steps
    return aligned


def initialize_ema_state(
    track_ema: bool, ema_transform: Optional[optax.GradientTransformation], params: Any
) -> Optional[Any]:
    """Initialise EMA state if requested."""
    if not track_ema or ema_transform is None:
        return None
    ema_state = ema_transform.init(params)
    return ema_state


def get_ema_params(ema_state: Optional[Any]) -> Optional[Any]:
    """Extract EMA parameters from the EMA state."""
    if ema_state is None:
        return None
    ema_value = getattr(ema_state, "ema", None)
    if ema_value is None:
        return None
    return ema_value


def apply_checkpoint_to_state(
    train_state: TrainState,
    checkpoint: Any | dict[str, Any],
    cfg: DictConfig,
    optimizer: optax.GradientTransformation,
    ema_transform: Optional[optax.GradientTransformation],
    log: logging.Logger,
    rebuild_optimizer: bool,
) -> TrainState:
    """Populate the in-memory train_state with data from a checkpoint payload."""
    train_state.params = checkpoint["params"]
    train_state.step = checkpoint.get("step", train_state.step)

    if "model_state" in checkpoint:
        train_state.model_state = checkpoint["model_state"]
    else:
        log.warning(
            "Checkpoint does not include model_state; continuing with existing state."
        )

    if rebuild_optimizer:
        reference = (
            checkpoint.get("params_ema")
            if cfg.train.track_ema and "params_ema" in checkpoint
            else train_state.params
        )
        train_state.opt_state = optimizer.init(reference)  # type: ignore
    else:
        train_state.opt_state = checkpoint["optimizer_state"]

    if cfg.train.track_ema:
        stored_ema_state = checkpoint.get("ema_state")
        stored_ema_params = checkpoint.get("params_ema")

        if stored_ema_state is not None:
            train_state.ema_state = stored_ema_state
        elif stored_ema_params is not None and ema_transform is not None:
            train_state.ema_state = initialize_ema_state(
                True,
                ema_transform,
                stored_ema_params,
            )
            log.warning(
                "EMA state missing from checkpoint; reinitialising EMA statistics from stored EMA params."
            )
        elif train_state.ema_state is None and ema_transform is not None:
            train_state.ema_state = initialize_ema_state(
                True,
                ema_transform,
                train_state.params,
            )
            log.warning(
                "EMA information missing from checkpoint; reinitialising EMA statistics from current params."
            )

    if "rng" in checkpoint:
        train_state.rng = checkpoint["rng"]
    else:
        log.warning("Checkpoint missing RNG key; stochastic components will reset.")

    return train_state


def save_training_checkpoint(
    checkpoint_manager: CheckpointManager,
    train_state: TrainState,
    loss_value: float,
    track_ema: bool,
) -> None:
    """Persist the full training state via the checkpoint manager."""
    checkpoint_manager.save(
        step=train_state.step,
        params=train_state.params,
        optimizer_state=train_state.opt_state,
        loss=loss_value,
        params_ema=get_ema_params(train_state.ema_state) if track_ema else None,
        model_state=train_state.model_state,
        ema_state=train_state.ema_state if track_ema else None,
        rng=train_state.rng,
    )


def resume_from_checkpoint(
    cfg: DictConfig,
    checkpoint_manager: CheckpointManager,
    optimizer: optax.GradientTransformation,
    train_state: TrainState,
    ema_transform: Optional[optax.GradientTransformation],
    log: logging.Logger,
) -> TrainState:
    """Load the latest checkpoint if continue_training is enabled."""
    if not cfg.train.get("continue_training", False):
        return train_state

    latest_step = checkpoint_manager.get_latest_step()
    if latest_step is None or latest_step <= 0:
        log.warning("No valid checkpoint found. Starting from scratch.")
        return train_state

    log.info(f"Restoring checkpoint at step {latest_step}")
    reference_ema = (
        get_ema_params(train_state.ema_state) if cfg.train.track_ema else None
    )
    checkpoint = checkpoint_manager.restore(
        step=latest_step,
        params=train_state.params,
        optimizer_state=train_state.opt_state,
        params_ema=reference_ema,
        model_state=train_state.model_state,
        ema_state=train_state.ema_state if cfg.train.track_ema else None,
        rng=train_state.rng,
    )
    if checkpoint is None:
        log.warning("Failed to restore checkpoint. Starting from scratch.")
        return train_state

    train_state = apply_checkpoint_to_state(
        train_state=train_state,
        checkpoint=checkpoint,
        cfg=cfg,
        optimizer=optimizer,
        ema_transform=ema_transform,
        log=log,
        rebuild_optimizer=cfg.train.restart_optimizer,
    )
    log.info(f"Resumed training from step {train_state.step}")
    return train_state


def build_loss_fn(cfg: DictConfig, graphdef: Any, static: Any):
    """Create the loss function closure."""

    def loss_fn(params, state, data, rng):
        if isinstance(data, (tuple, list)):
            batches = tuple(data)
        else:
            batches = (data,)
        num_batches = len(batches)
        if num_batches == 0:
            raise ValueError("Expected at least one batch for loss computation.")

        rngs = jax.random.split(rng, num_batches)
        model = nnx.merge(graphdef, params, static, state, copy=True)
        model.train()

        loss1 = []
        loss2 = []

        for batch, subkey in zip(batches, rngs):
            losses = model.loss_fn(
                subkey,
                **batch,
                weight_by_complexity=cfg.train.weight_by_complexity,
                cut_off_tsm=cfg.train.cut_off_tsm,
            )
            loss1.append(losses[0])
            loss2.append(losses[1])

        normaliser = 1.0 / num_batches
        loss1 = cfg.train.model_selection_weight * normaliser * sum(loss1)
        loss2 = cfg.train.model_inference_loss_weight * normaliser * sum(loss2)
        total_loss = loss1 + loss2
        _, _, _, new_state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)
        return total_loss, (loss1, loss2, new_state)

    return loss_fn


def build_update_fn(
    optimizer: optax.GradientTransformation,
    loss_fn,
    use_ema: bool,
    ema_transform: Optional[optax.GradientTransformation],
):
    """Create the JIT-compiled update step including optional EMA maintenance."""
    if use_ema and ema_transform is None:
        raise ValueError("EMA requested but no ema_transform provided.")

    @jax.jit
    def update(params, state, opt_state, ema_state, data, rng):
        (total_loss, (loss1, loss2, new_state)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, state, data, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        if use_ema:
            _, ema_state = ema_transform.update(new_params, ema_state)  # type: ignore
        return new_params, new_state, opt_state, ema_state, (loss1, loss2), total_loss

    return update


def get_queue_size(loader: Any) -> int:
    """Fetch loader queue size if available."""
    if hasattr(loader, "queue") and hasattr(loader.queue, "qsize"):
        return int(loader.queue.qsize())
    return 0


def train_loop(
    cfg: DictConfig,
    log: logging.Logger,
    train_state: TrainState,
    optimizer: optax.GradientTransformation,
    update_step,
    checkpoint_manager: CheckpointManager,
    evaluator,
    loaders: Sequence[Any],
    eval_loader: Any,
    track_ema: bool,
    ema_transform: Optional[optax.GradientTransformation],
    checkpoint_freq: Optional[int],
    eval_freq: Optional[int],
    restart_every: Optional[int],
    max_train_hours: float,
    wandb_active: bool,
):
    """Main training loop."""
    loaders = list(loaders)
    if not loaders:
        raise ValueError("At least one training dataloader is required.")
    inner_steps = cfg.train.inner_steps
    datastreams = [iter(loader) for loader in loaders]
    start_time = time.time()

    log.info("Compiling training and evaluation step...")
    _ = update_step(
        train_state.params,
        train_state.model_state,
        train_state.opt_state,
        train_state.ema_state,
        next(datastreams[0]),
        train_state.rng,
    )
    _ = evaluator.eval_nnl_mask(
                train_state.params,
                train_state.model_state,
                eval_loader,
                iters=cfg.train.eval.nnl_mask.iters,
            )
    _ = evaluator.eval_nnl_theta(
                train_state.params,
                train_state.model_state,
                eval_loader,
                iters=cfg.train.eval.nnl_theta.iters,
            )

    log.info(f"Maximum training time: {max_train_hours} hours")
    while True:
        loss_mask = []
        loss_theta = []
        for _ in range(inner_steps):
            train_state.rng, subkey = jax.random.split(train_state.rng)
            batches = tuple(next(stream) for stream in datastreams)
            data = batches if len(batches) > 1 else batches[0]
            (
                train_state.params,
                train_state.model_state,
                train_state.opt_state,
                train_state.ema_state,
                losses,
                _,
            ) = update_step(
                train_state.params,
                train_state.model_state,
                train_state.opt_state,
                train_state.ema_state,
                data,
                subkey,
            )
            train_state.step += 1
            loss_mask.append(losses[0])
            loss_theta.append(losses[1])
        loss_mask = float(sum(loss_mask) / len(loss_mask))
        loss_theta = float(sum(loss_theta) / len(loss_theta))
        total_loss_value = float(loss_mask + loss_theta)
        queue_size = sum(get_queue_size(loader) for loader in loaders)

        # Collect and average dataset stats from training loaders
        dataset_stats = {}
        train_datasets = [loader._ds for loader in loaders if hasattr(loader, '_ds')]
        if train_datasets:
            # Collect stats from all training datasets
            all_stats = []
            for dataset in train_datasets:
                if hasattr(dataset, 'get_stats'):
                    all_stats.append(dataset.get_stats())

            # Compute averaged production_time and samples_written/samples_requested ratio
            if all_stats:
                production_times = [stats.get('production_time', 0) for stats in all_stats]
                samples_written = [stats.get('samples_written', 0) for stats in all_stats]
                samples_requested = [stats.get('samples_requested', 1) for stats in all_stats]

                avg_production_time = sum(production_times) / len(production_times)
                total_written = sum(samples_written)
                total_requested = sum(samples_requested)

                dataset_stats['train_dataset/production_time'] = avg_production_time
                if total_requested > 0:
                    dataset_stats['train_dataset/samples_written_requested_ratio'] = total_written / total_requested

        # Log dataset stats locally
        if dataset_stats:
            stats_str = ", ".join(f"{k}: {v:.4f}" for k, v in dataset_stats.items())
            log.info(f"Dataset stats - {stats_str}")

        log.info(
            f"Step {train_state.step}, Loss mask: {loss_mask}, Loss theta: {loss_theta}, data_queue_size: {queue_size}"
        )

        if restart_every and train_state.step % restart_every == 0:
            log.info(f"Restarting optimizer at step {train_state.step}")
            reference = get_ema_params(train_state.ema_state) if track_ema else None
            if reference is None:
                reference = train_state.params
            train_state.opt_state = optimizer.init(reference)

        if wandb_active:
            wandb_dict = {
                "loss mask": loss_mask,
                "loss theta": loss_theta,
                "queue_size": queue_size,
                "step": train_state.step,
            }
            # Add dataset stats to wandb
            wandb_dict.update(dataset_stats)
            wandb.log(wandb_dict)

        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= max_train_hours:
            log.info(
                f"Reached maximum training time of {max_train_hours} hours. Stopping."
            )
            save_training_checkpoint(
                checkpoint_manager,
                train_state,
                total_loss_value,
                track_ema,
            )
            checkpoint_manager.wait_until_finished()
            break

        if eval_freq and train_state.step > 0 and train_state.step % eval_freq == 0:
            log.info(f"Evaluating model at step {train_state.step}")
            params_eval = get_ema_params(train_state.ema_state) if track_ema else None
            if params_eval is None:
                params_eval = train_state.params
            mask_nnl = evaluator.eval_nnl_mask(
                params_eval,
                train_state.model_state,
                eval_loader,
                iters=cfg.train.eval.nnl_mask.iters,
            )
            theta_nnl = evaluator.eval_nnl_theta(
                params_eval,
                train_state.model_state,
                eval_loader,
                iters=cfg.train.eval.nnl_theta.iters,
            )
            log.info(f"Mask NLL: {mask_nnl}, Theta NLL: {theta_nnl}")
            if wandb_active:
                wandb.log({
                    "mask_negative_log_likelihood": float(mask_nnl),
                    "theta_negative_log_likelihood": float(theta_nnl),
                    "step": train_state.step,
                })

        if (
            checkpoint_freq
            and train_state.step > 0
            and train_state.step % checkpoint_freq == 0
        ):
            log.info(f"Saving checkpoint at step {train_state.step}")
            save_training_checkpoint(
                checkpoint_manager,
                train_state,
                total_loss_value,
                track_ema,
            )

        if checkpoint_manager.should_recover(total_loss_value):
            log.warning(
                f"Recovery triggered at step {train_state.step}. Restoring from checkpoint."
            )
            try:
                latest_step = checkpoint_manager.get_latest_step()
                if latest_step is None:
                    log.warning(
                        "No checkpoints available for recovery. Continuing without recovery."
                    )
                else:
                    checkpoint = checkpoint_manager.restore(
                        step=latest_step,
                        params=train_state.params,
                        optimizer_state=train_state.opt_state,
                        params_ema=get_ema_params(train_state.ema_state)
                        if track_ema
                        else None,
                        model_state=train_state.model_state,
                        ema_state=train_state.ema_state if track_ema else None,
                        rng=train_state.rng,
                    )
                    if checkpoint is None:
                        log.warning(
                            "Failed to restore recovery checkpoint. Continuing without recovery."
                        )
                    else:
                        train_state = apply_checkpoint_to_state(
                            train_state=train_state,
                            checkpoint=checkpoint,
                            cfg=cfg,
                            optimizer=optimizer,
                            ema_transform=ema_transform,
                            log=log,
                            rebuild_optimizer=False,
                        )
                        log.info(f"Recovered to step {train_state.step}")
                        datastreams = [iter(loader) for loader in loaders]
            except Exception as err:  # pragma: no cover - defensive logging
                log.error(f"Error during recovery: {err}")
                log.warning("Continuing without recovery.")

    log.info("Training complete")
    checkpoint_manager.wait_until_finished()


def main():
    """Main script function."""
    print(logo)
    _main()


def build_optimizer(optimizer_cfg):
    optimizer_type = getattr(optax, optimizer_cfg.optimizer)
    if optimizer_cfg.scheduler:
        scheduler_type = getattr(optax, optimizer_cfg.scheduler)
    else:
        scheduler_type = None
    use_ema = optimizer_cfg.get("ema", False)
    use_adaptive_clip = optimizer_cfg.get("adaptive_gradient_clipping", False)
    grad_transforms = []
    if use_adaptive_clip:
        grad_clip = optax.adaptive_grad_clip(
            optimizer_cfg.get("gradient_clip_value", 10.0)
        )
        grad_transforms.append(grad_clip)
    if scheduler_type:
        scheduler = scheduler_type(**optimizer_cfg.scheduler_params)
        optimizer = optimizer_type(scheduler)
    else:
        lr = optimizer_cfg.get("learning_rate", 1e-4)
        optimizer = optimizer_type(learning_rate=lr)
    grad_transforms.append(optimizer)
    if use_ema:
        grad_transforms.append(optax.ema(optimizer_cfg.get("ema_decay", 0.8)))

    return optax.chain(*grad_transforms)


@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def _main(cfg: DictConfig):
    log, output_dir, output_super_dir = configure_environment(cfg)
    wandb_active = init_wandb_if_needed(cfg)

    rng_key = seed_everything(cfg.seed)
    log.info(f"Seed: {cfg.seed}")

    log.info("Building simulator")
    log.info(f"Simulator cfg: {cfg.simulator}")
    sim_type, simulators = build_simulator(cfg)
    log.info(f"Simulator type: {sim_type}")

    model = build_model(cfg, sim_type)
    model.train()
    log.info(f"Model cfg: {model.cfg}")
    graphdef, params, static, state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)

    evaluator = build_pure_eval_fns(graphdef, static, sim_type)

    checkpoint_dir = os.path.join(output_super_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    checkpoint_manager = CheckpointManager(
        ckpt_dir=checkpoint_dir,
        max_to_keep=cfg.get("max_checkpoints", 3),
        keep_best=cfg.get("keep_best_checkpoint", True),
        recovery_threshold=cfg.get("recovery_threshold", float("inf")),
        continue_training=cfg.train.get("continue_training", False),
    )

    optimizer = build_optimizer(cfg.train.optimizer)
    opt_state = optimizer.init(params)

    ema_transform = (
        optax.ema(cfg.train.ema_decay, debias=False) if cfg.train.track_ema else None
    )
    ema_state = initialize_ema_state(cfg.train.track_ema, ema_transform, params)

    train_state = TrainState(
        params=params,
        model_state=state,
        opt_state=opt_state,
        ema_state=ema_state,
        rng=rng_key,
        step=0,
    )
    train_state = resume_from_checkpoint(
        cfg, checkpoint_manager, optimizer, train_state, ema_transform, log
    )

    loss_fn = build_loss_fn(cfg, graphdef, static)
    update_step = build_update_fn(
        optimizer, loss_fn, cfg.train.track_ema, ema_transform
    )

    train_loaders, eval_loader = create_dataloaders(cfg, simulators)

    inner_steps = cfg.train.inner_steps
    checkpoint_freq = align_to_inner_steps(
        cfg.train.checkpoint_freq, inner_steps, "checkpoint frequency", log
    )
    eval_freq = align_to_inner_steps(
        cfg.train.eval_freq, inner_steps, "evaluation frequency", log
    )
    restart_every = align_to_inner_steps(
        cfg.train.get("restart_every"), inner_steps, "restart frequency", log
    )

    max_train_hours = cfg.train.get("max_train_hours", float("inf"))

    train_loop(
        cfg=cfg,
        log=log,
        train_state=train_state,
        optimizer=optimizer,
        update_step=update_step,
        checkpoint_manager=checkpoint_manager,
        evaluator=evaluator,
        loaders=train_loaders,
        eval_loader=eval_loader,
        track_ema=cfg.train.track_ema,
        ema_transform=ema_transform,
        checkpoint_freq=checkpoint_freq,
        eval_freq=eval_freq,
        restart_every=restart_every,
        max_train_hours=max_train_hours,
        wandb_active=wandb_active,
    )
