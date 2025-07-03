import os
from datetime import datetime

from flax import nnx
from omegaconf import OmegaConf

from dmri.train.build_model import build_model
from dmri.train.build_simulator import build_simulator
from dmri.train.checkpointing import CheckpointManager
from dmri.train.hydra_script import build_optimizer


def load_cfg(path):
    # Find most recent run directory (format: YYYY-MM-DD_HH-MM-SS)
    dirs = [
        d
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
        and len(d.split("_")) == 2
        and len(d.split("_")[0].split("-")) == 3
    ]

    if not dirs:
        raise ValueError(f"No run directories found in {path}")

    # Parse timestamps and find the most recent one
    timestamp_dirs = []
    for d in dirs:
        try:
            timestamp = datetime.strptime(d, "%Y-%m-%d_%H-%M-%S")
            timestamp_dirs.append((timestamp, d))
        except ValueError:
            continue

    if not timestamp_dirs:
        raise ValueError(f"No valid timestamped directories found in {res_folder}")

    # Sort by timestamp (newest first)
    timestamp_dirs.sort(reverse=True)
    most_recent_dir = timestamp_dirs[0][1]

    # Try to load config from .hydra first, then fall back to 0/.hydra
    base_path = os.path.join(path, most_recent_dir)
    config_path = os.path.join(base_path, ".hydra", "config.yaml")

    if not os.path.exists(config_path):
        config_path = os.path.join(base_path, "0", ".hydra", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Could not find config.yaml in either .hydra or 0/.hydra directories in {base_path}"
            )

    cfg = OmegaConf.load(config_path)
    return cfg


def load_checkpoint(path, which="latest"):
    cfg = load_cfg(path)
    sim_type, simulator = build_simulator(cfg)
    model = build_model(cfg, sim_type)
    model.eval()

    graphdef, params, static, state = nnx.split(model, nnx.Param, nnx.Intermediate, ...)

    checkpoint_dir = os.path.join(path, "checkpoints")
    continue_training = True
    checkpoint_manager = CheckpointManager(
        ckpt_dir=checkpoint_dir,
        max_to_keep=cfg.get("max_checkpoints", 5),
        keep_best=cfg.get("keep_best_checkpoint", True),
        recovery_threshold=cfg.get("recovery_threshold", float("inf")),
        continue_training=continue_training,
        use_async=cfg.get(
            "use_async_checkpointing", True
        ),  # Enable async checkpointing
    )

    optimizer = build_optimizer(cfg.train.optimizer)
    opt_state = optimizer.init(params)

    if which == "latest":
        latest_step = checkpoint_manager.get_latest_step()
    elif which == "best":
        latest_step = 0
    else:
        raise ValueError(f"Invalid checkpoint type: {which}")

    if cfg.train.track_ema:
        checkpoint = checkpoint_manager.restore(
            latest_step, params=params, optimizer_state=opt_state, params_ema=params
        )

    else:
        checkpoint = checkpoint_manager.restore(
            latest_step, params=params, optimizer_state=opt_state
        )

    return checkpoint, model, simulator
