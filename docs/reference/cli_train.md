# Training CLI

The `dmri` command is Hydra-based and pulls defaults from the packaged `conf_train` configuration module. Use this page to discover the main groups and override them on the command line.

## Essential commands

- List overrides for a preset: `dmri +experiment=b3s_2_4_6_128 --help`
- Dry-run the resolved config: `dmri +experiment=b3s_2_4_6_128 --cfg job`
- Override inline: `dmri +experiment=b3s_2_4_6_128 train.dataloader.train_loader.batch_size=128 train.optimizer.learning_rate=1e-3`

## Configuration map (`conf_train/`)

- `config.yaml`: run metadata (`name`, `seed`, `use_wandb`, output directories) plus default groups.
- `experiment/`: ready-made recipes that combine simulator, model, optimizer, and partition choices.
- `simulator/`: acquisition schemes and signal simulators for synthetic training data.
- `model/`: neural architectures for selection/inference networks (dmri and ssfp variants).
- `train/`: loop controls (inner steps, EMA, checkpoint cadence, loss weights) and dataloader/optimizer subgroups.
- `partition/`: resource presets for local vs. cluster runs.
- `launcher/`: Hydra launcher wiring (local or Slurm).

## Common overrides

- Batch size / optimizer: `train.dataloader.train_loader.batch_size`, `train.optimizer.learning_rate`, `train.optimizer.scheduler`.
- Model architecture groups: `model/embedding_net`, `model/model_selection_net`, `model/inference_net`.
- Simulator recipe: `simulator=ball3stick_shared` (or any file in `conf_train/simulator/`).
- Partition and launcher: `partition=<profile>`, `launcher=slurm`.

## Outputs

Runs write to `results/<run_name>/<timestamp>/` with:
- `checkpoints/` (numeric steps and optionally `best/`), `.hydra/` (frozen config), logs/metrics, and W&B metadata when enabled.
