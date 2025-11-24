# Training API

Core training helpers are exposed here directly from their docstrings. Use them inside notebooks or when wiring custom CLIs.

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
