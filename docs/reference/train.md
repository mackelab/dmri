# Training API

Core training helpers are exposed here directly from their docstrings. Start with `build_model`, `build_simulator`, and `load_checkpoint` when composing a workflow outside the CLI.

See the [training guide](../guides/training.md) for experiment configuration and checkpoint layout.

## Dataset utilities

::: dmri.train.dataset
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - SimulationDataset
        - instantiate_dataloader

## Hydra entry point

::: dmri.train.train_script
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "^TrainState$"
        - "^tree_copy$"
        - "^configure_environment$"
        - "^init_wandb_if_needed$"
        - "^seed_everything$"
        - "^create_dataloaders$"
        - "^align_to_inner_steps$"
        - "^initialize_ema_state$"
        - "^get_ema_params$"
        - "^apply_checkpoint_to_state$"
        - "^resume_from_checkpoint$"
        - "^save_training_checkpoint$"
        - "^build_optimizer$"
        - "^build_loss_fn$"
        - "^build_update_fn$"
        - "^train_loop$"
        - "^main$"

## Checkpointing

::: dmri.train.checkpointing
    options:
      show_root_heading: true
      show_root_full_path: false

## Evaluation loop

::: dmri.train.evaluator
    options:
      show_root_heading: true
      show_root_full_path: false

## Builders and utilities

::: dmri.train.build_simulator
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.train.build_model
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.train.utils
    options:
      show_root_heading: true
      show_root_full_path: false
