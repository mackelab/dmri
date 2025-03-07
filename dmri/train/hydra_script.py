import importlib
import logging
import os
import random
import socket
import time

from dmri.train.build_simulator import build_simulator
from dmri.train.build_model import build_model
from dmri.train.dataloader import StreamDataLoader
from dmri.train.checkpointing import CheckpointManager
from dmri.train.evaluator import build_pure_eval_fns
import hydra
import numpy as np
import jax
from omegaconf import DictConfig, OmegaConf
import optax

import wandb

# Backends


logo = """

 /$$$$$$$  /$$      /$$ /$$$$$$$  /$$$$$$
| $$__  $$| $$$    /$$$| $$__  $$|_  $$_/
| $$  \ $$| $$$$  /$$$$| $$  \ $$  | $$
| $$  | $$| $$ $$/$$ $$| $$$$$$$/  | $$
| $$  | $$| $$  $$$| $$| $$__  $$  | $$
| $$  | $$| $$\  $ | $$| $$  \ $$  | $$
| $$$$$$$/| $$ \/  | $$| $$  | $$ /$$$$$$
|_______/ |__/     |__/|__/  |__/|______/
"""


def main():
    """Main script function"""
    print(logo)
    _main()


@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def _main(cfg: DictConfig):
    """Evaluate score based inference"""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Go back to the folder named "cfg.name"
    output_super_dir = os.path.dirname(output_dir)
    while os.path.basename(output_super_dir) != cfg.name:
        output_super_dir = os.path.dirname(output_super_dir)

    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info("Output super directory: {}".format(output_super_dir))
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")

    # Init wandb
    if cfg.use_wandb:
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    rng_key = jax.random.key(seed)
    log.info(f"Seed: {seed}")

    sim_type, simulator = build_simulator(cfg)

    model, params = build_model(cfg, sim_type)
    log.info(f"Model cfg: {model.cfg}")

    # Create evaluator for online model performance metrics
    evaluator = build_pure_eval_fns(model, sim_type)

    # Train model
    log.info("Training")

    # Set up checkpoint manager
    # Make sure checkpoint_dir is in results/{name}/checkpoints
    checkpoint_dir = os.path.join("results", cfg.name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log.info(f"Checkpoint directory: {checkpoint_dir}")

    continue_training = cfg.train.get("continue_training", False)
    checkpoint_manager = CheckpointManager(
        ckpt_dir=checkpoint_dir,
        max_to_keep=cfg.get("max_checkpoints", 3),
        keep_best=cfg.get("keep_best_checkpoint", True),
        recovery_threshold=cfg.get("recovery_threshold", float("inf")),
        continue_training=continue_training,
    )

    # For resuming training
    start_step = 0
    if continue_training:
        latest_step = checkpoint_manager.get_latest_step()
        if latest_step > 0:
            log.info(f"Restoring checkpoint at step {latest_step}")
            checkpoint = checkpoint_manager.restore()
            if checkpoint is not None:
                params = checkpoint["params"]
                opt_state = checkpoint["optimizer_state"]
                start_step = checkpoint["step"] + 1
                log.info(f"Resumed training from step {start_step}")
            else:
                log.warning("Failed to restore checkpoint. Starting from scratch.")

    # Learning rate scheduler and optimizer
    optimizer_cfg = cfg.train.optimizer
    optimizer_type = getattr(optax, optimizer_cfg.optimizer)
    scheduler_type = getattr(optax, optimizer_cfg.scheduler)
    use_ema = optimizer_cfg.get("ema", False)
    use_adaptive_clip = optimizer_cfg.get("adaptive_gradient_clipping", False)
    grad_transforms = []
    if use_adaptive_clip:
        grad_clip = optax.adaptive_grad_clip(
            optimizer_cfg.get("gradient_clip_value", 10.0)
        )
        grad_transforms.append(grad_clip)
    scheduler = scheduler_type(**optimizer_cfg.scheduler_params)
    optimizer = optimizer_type(scheduler)

    grad_transforms.append(optimizer)
    if use_ema:
        grad_transforms.append(optax.ema(optimizer_cfg.get("ema_decay", 0.8)))
    optimizer = optax.chain(*grad_transforms)

    # Initialize optimizer state if not restored from checkpoint
    if not continue_training or start_step == 0:
        opt_state = optimizer.init(params)

    def loss_fn(params, data, rng):
        p_mask, model_mask, thetas, xs, acq = data
        losses = model.loss_fn(
            params,
            rng,
            model_mask=model_mask,
            theta=thetas,
            x=xs,
            bvals=acq.bvals,
            bvecs=acq.bvecs,
            mask_prior=p_mask,
        )
        return sum(losses), losses

    @jax.jit
    def update(params, opt_state, data, rng):
        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, data, rng
        )
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, losses

    # Create a data loader
    loader_name = cfg.train.dataloader.name
    loader_module = importlib.import_module("dmri.train.dataloader")
    loader_type = getattr(loader_module, loader_name)
    loader_train_params = cfg.train.dataloader.train_params
    loader_eval_params = cfg.train.dataloader.val_params

    loader = loader_type(
        simulator,
        **loader_train_params,
    )

    # Create a separate loader for evaluation
    eval_loader = loader_type(
        simulator,
        **loader_eval_params,
    )

    key = rng_key
    inner_steps = cfg.train.inner_steps
    checkpoint_freq = (cfg.train.checkpoint_freq // inner_steps) * inner_steps
    eval_freq = (cfg.train.eval_freq // inner_steps) * inner_steps

    step = start_step
    datastream = iter(loader)

    # Get maximum training time in hours (default: run forever)
    max_train_hours = cfg.train.get("max_train_hours", float("inf"))
    log.info(f"Maximum training time: {max_train_hours} hours")
    start_time = time.time()

    while True:
        key, subkey = jax.random.split(key)
        for _ in range(50):
            data = next(datastream)
            params, opt_state, loss = update(params, opt_state, data, subkey)
            step += 1
        total_loss = float(sum(loss))
        queue_size = int(loader.queue.qsize())
        log.info(
            f"Step {step}, Loss mask: {loss[0]}, Loss theta: {loss[1]}, data_queue_size: {queue_size}"
        )

        # Log elapsed time
        elapsed_hours = (time.time() - start_time) / 3600
        if cfg.use_wandb:
            wandb.log(
                {
                    "loss mask": float(loss[0]),
                    "loss theta": float(loss[1]),
                    "queue_size": queue_size,
                    "step": step,
                    "elapsed_hours": elapsed_hours,
                }
            )

        # Check if we've exceeded maximum training time
        if elapsed_hours >= max_train_hours:
            log.info(
                f"Reached maximum training time of {max_train_hours} hours. Stopping."
            )
            # Save final checkpoint
            checkpoint_manager.save(
                step=step,
                model=model,
                params=params,
                optimizer_state=opt_state,
                metrics={"loss_mask": float(loss[0]), "loss_theta": float(loss[1])},
            )
            break

        # Evaluate model periodically
        if step > 0 and step % eval_freq == 0:
            log.info(f"Evaluating model at step {step}")

            # Evaluate negative log-likelihood for masks
            key, eval_key = jax.random.split(key)
            mask_nnl = evaluator.eval_nnl_mask(params, eval_loader, iters=5)

            # Evaluate negative log-likelihood for thetas
            theta_nnl = evaluator.eval_nnl_theta(params, eval_loader, iters=5)

            # Evaluate ess
            ess = evaluator.eval_effective_sample_size(
                params, eval_loader, eval_key, K=5, iters=1
            )

            log.info(f"Mask NLL: {mask_nnl}, Theta NLL: {theta_nnl}, ESS: {ess}")

            if cfg.use_wandb:
                wandb.log(
                    {
                        "mask_negative_log_likelihood": float(mask_nnl),
                        "theta_negative_log_likelihood": float(theta_nnl),
                        "step": step,
                    }
                )

            # Add these metrics to checkpoint
            metrics = {
                "loss_mask": float(loss[0]),
                "loss_theta": float(loss[1]),
                "mask_nnl": float(mask_nnl),
                "theta_nnl": float(theta_nnl),
            }
        else:
            metrics = {
                "loss_mask": float(loss[0]),
                "loss_theta": float(loss[1]),
            }

        # Save checkpoint periodically
        if step > 0 and step % checkpoint_freq == 0:
            log.info(f"Saving checkpoint at step {step}")
            checkpoint_manager.save(
                step=step,
                model=model,
                params=params,
                optimizer_state=opt_state,
                metrics=metrics,
            )

        # Check if we need to recover from a bad update
        if checkpoint_manager.should_recover(total_loss):
            log.warning(
                f"Recovery triggered at step {step}. Restoring from checkpoint."
            )
            checkpoint = checkpoint_manager.restore()
            if checkpoint is not None:
                params = checkpoint["params"]
                opt_state = checkpoint["optimizer_state"]
                step = checkpoint["step"]
                log.info(f"Recovered to step {step}")

    log.info("Training complete")
