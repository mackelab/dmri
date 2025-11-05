import importlib
import logging
import os
import random
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

memory_fraction = 0.9  # Use 98% of available memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

import hydra
import jax
import numpy as np
import optax
from flax import nnx
from omegaconf import DictConfig, OmegaConf

import wandb
from dmri.train.build_model import build_model
from dmri.train.build_simulator import build_simulator
from dmri.train.checkpointing import CheckpointManager
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

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
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


def _normalise_loader_params(params_cfg: Any, count: int) -> list[dict[str, Any]]:
    """Convert Hydra loader params into a list of dictionaries."""
    container = OmegaConf.to_container(params_cfg, resolve=True)
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


def _instantiate_loader(loader_type: Any, simulator_fn: Any, params: dict[str, Any]) -> Any:
    params = dict(params)  # shallow copy
    dataloader_kwargs = params.pop("dataloader_kwargs", None)
    if dataloader_kwargs:
        params.update(dataloader_kwargs)
    loader = loader_type(simulator_fn=simulator_fn, **params)
    if hasattr(loader, "reset"):
        loader.reset()
    return loader


def create_dataloaders(cfg: DictConfig, simulators: Sequence[Any]) -> tuple[list[Any], Any]:
    """Instantiate training and evaluation dataloaders."""
    loader_module = importlib.import_module("dmri.train.dataloader")
    loader_type = getattr(loader_module, cfg.train.dataloader.name)
    simulators = list(simulators)
    if not simulators:
        raise ValueError("Expected at least one simulator callable.")

    train_loader_params_list = _normalise_loader_params(
        cfg.train.dataloader.train_params, max(1, len(simulators))
    )
    train_simulators = simulators[: len(train_loader_params_list)]
    train_loaders = [
        _instantiate_loader(loader_type, sim_fn, params)
        for sim_fn, params in zip(train_simulators, train_loader_params_list)
    ]

    eval_params = _normalise_loader_params(cfg.train.dataloader.val_params, 1)[0]
    eval_loader = _instantiate_loader(loader_type, simulators[0], eval_params)
    return train_loaders, eval_loader


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
    if hasattr(ema_state, "_replace"):
        ema_state = ema_state._replace(ema=tree_copy(params))
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
    checkpoint: dict[str, Any],
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
        train_state.opt_state = optimizer.init(reference)
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

    def _unpack_batch(batch):
        if cfg.simulator.with_posterior_score:
            p_mask, model_mask, thetas, xs, acq, target_score = batch
        else:
            p_mask, model_mask, thetas, xs, acq = batch
            target_score = None
        return p_mask, model_mask, thetas, xs, acq, target_score

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

        total_loss = 0.0
        accumulated_losses = None

        for batch, subkey in zip(batches, rngs):
            p_mask, model_mask, thetas, xs, acq, target_score = _unpack_batch(batch)
            losses = tuple(
                model.loss_fn(
                    subkey,
                    model_mask=model_mask,
                    theta=thetas,
                    x=xs,
                    acq=acq,
                    mask_prior=p_mask,
                    target_score=target_score,
                    weight_by_complexity=cfg.train.weight_by_complexity,
                    cut_off_tsm=cfg.train.cut_off_tsm,
                )
            )
            if accumulated_losses is None:
                accumulated_losses = losses
            else:
                accumulated_losses = tuple(acc + loss for acc, loss in zip(accumulated_losses, losses))
            loss1 = cfg.train.model_selection_weight * losses[0]
            loss2 = cfg.train.model_inference_loss_weight * losses[1]
            total_loss = total_loss + loss1 + loss2

        normaliser = 1.0 / num_batches
        total_loss = total_loss * normaliser
        averaged_losses = tuple(loss * normaliser for loss in accumulated_losses)
        _, _, _, new_state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)
        return total_loss, (averaged_losses, new_state)

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
        (total_loss, (losses, new_state)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, state, data, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        if use_ema:
            _, ema_state = ema_transform.update(new_params, ema_state)
        return new_params, new_state, opt_state, ema_state, losses, total_loss

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
                total_loss,
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
            wandb.log(
                {
                    "loss mask": loss_mask,
                    "loss theta": loss_theta,
                    "queue_size": queue_size,
                    "step": train_state.step,
                }
            )

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
            log.info(
                f"Mask NLL: {mask_nnl}, Theta NLL: {theta_nnl}"
            )
            if wandb_active:
                wandb.log(
                    {
                        "mask_negative_log_likelihood": float(mask_nnl),
                        "theta_negative_log_likelihood": float(theta_nnl),
                        "step": train_state.step,
                    }
                )

        if checkpoint_freq and train_state.step > 0 and train_state.step % checkpoint_freq == 0:
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
                        params_ema=get_ema_params(train_state.ema_state) if track_ema else None,
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
