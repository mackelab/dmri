import logging
import os
import random
import socket
import sys
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

    # Create evaluator for model performance metrics
    evaluator = build_pure_eval_fns(model, sim_type)

    # Train model
    log.info("Training")

    # Set up checkpoint manager
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    continue_training = cfg.get("continue_training", False)
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
    max_steps = cfg.get("max_steps", 100 * 500)
    scheduler = optax.cosine_onecycle_schedule(max_steps, 5e-4, final_div_factor=5)
    optimizer = optax.chain(
        optax.adaptive_grad_clip(50.0), optax.ema(0.01), optax.adamw(scheduler)
    )

    # Initialize optimizer state if not restored from checkpoint
    if not continue_training or start_step == 0:
        opt_state = optimizer.init(params)

    def loss_fn(params, data, rng):
        p_mask, model_mask, thetas, xs, acq = data
        return model.loss_fn(
            params,
            rng,
            model_mask=model_mask,
            theta=thetas,
            x=xs,
            bvals=acq.bvals,
            bvecs=acq.bvecs,
            mask_prior=p_mask,
        ).sum()

    @jax.jit
    def update(params, opt_state, data, rng):
        loss, grads = jax.value_and_grad(loss_fn)(params, data, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    loader = StreamDataLoader(
        simulator,
        batch_size=64,
        max_queue_size=10_000,
        num_producers=4,
    )

    # Create a separate loader for evaluation
    eval_loader = StreamDataLoader(
        simulator,
        batch_size=32,
        max_queue_size=1000,
        num_producers=2,
    )

    key = rng_key
    checkpoint_freq = cfg.get("checkpoint_freq", 2000)  # Save checkpoint every N steps
    eval_freq = cfg.get("eval_freq", 500)  # Evaluate model every N steps
    step = start_step

    for data in loader:
        key, subkey = jax.random.split(key)
        params, opt_state, loss = update(params, opt_state, data, subkey)

        if step % 10 == 0:  # Log every 10 steps
            log.info(f"Step {step}, Loss: {loss}")
            if cfg.use_wandb:
                wandb.log({"loss": loss, "step": step})

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
                params, eval_loader, eval_key, K=5
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
                "loss": float(loss),
                "mask_nnl": float(mask_nnl),
                "theta_nnl": float(theta_nnl),
            }
        else:
            metrics = {"loss": float(loss)}

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
        if checkpoint_manager.should_recover(loss):
            log.warning(
                f"Recovery triggered at step {step}. Restoring from checkpoint."
            )
            checkpoint = checkpoint_manager.restore()
            if checkpoint is not None:
                params = checkpoint["params"]
                opt_state = checkpoint["optimizer_state"]
                step = checkpoint["step"]
                log.info(f"Recovered to step {step}")

        step += 1

        # Check if we've reached the maximum steps
        if step >= max_steps:
            log.info(f"Reached maximum steps {max_steps}")
            # Save final checkpoint with evaluation metrics
            key, eval_key = jax.random.split(key)
            mask_nnl = evaluator.eval_nnl_mask(params, eval_loader, iters=10)
            theta_nnl = evaluator.eval_nnl_theta(params, eval_loader, iters=10)

            final_metrics = {
                "loss": float(loss),
                "final_mask_nnl": float(mask_nnl),
                "final_theta_nnl": float(theta_nnl),
            }

            checkpoint_manager.save(
                step=step,
                model=model,
                params=params,
                optimizer_state=opt_state,
                metrics=final_metrics,
            )

            if cfg.use_wandb:
                wandb.log(final_metrics)
            break

    log.info("Training complete")
