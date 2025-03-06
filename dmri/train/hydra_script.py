import logging
import os
import random
import socket
import sys
import time

from dmri.train.build_simulator import build_simulator
from dmri.train.build_model import build_model
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


    # Train model
    log.info("Training")

    scheduler = optax.cosine_onecycle_schedule(100 * 500, 5e-4)
    optimizer = optax.chain(
        optax.adaptive_grad_clip(50.0), optax.ema(0.01), optax.adamw(scheduler)
    )
    opt_state = optimizer.init(params)

    def loss_fn(params, data, rng):
        p_mask, model_mask, thetas, xs, bvals, bvecs = data
        return model.loss_fn(
            params,
            rng,
            model_mask=model_mask,
            theta=thetas,
            x=xs,
            bvals=bvals,
            bvecs=bvecs,
            mask_prior=p_mask,
        ).sum()

    @jax.jit
    def update(params, opt_state, data, rng):
        loss, grads = jax.value_and_grad(loss_fn)(params, data, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    batch_simulator = jax.jit(jax.vmap(simulator))

    key = rng_key
    for i in range(10):
        key, subkey1 = jax.random.split(key, 2)
        p_mask, masks, thetas, x_os, acq = batch_simulator(
            jax.random.split(subkey1, 2**16)
        )
        l = 0
        for j in range(500):
            key, subkey2 = jax.random.split(key, 2)
            idx = jax.random.randint(subkey2, (128,), 0, 2**16)
            (
                p_mask_batch,
                masks_batch,
                thetas_batch,
                x_os_batch,
                bvals_batch,
                bvecs_batch,
            ) = (
                p_mask[idx],
                masks[idx],
                thetas[idx],
                x_os[idx],
                acq.bvals[idx],
                acq.bvecs[idx],
            )
            params, opt_state, loss = update(
                params,
                opt_state,
                (
                    p_mask_batch,
                    masks_batch,
                    thetas_batch,
                    x_os_batch,
                    bvals_batch,
                    bvecs_batch,
                ),
                subkey2,
            )
            l += loss
        print(l / 500)
