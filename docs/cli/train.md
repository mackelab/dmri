# Training CLI

The `dmri train` command is Hydra-based and pulls defaults from the shared, versioned `conf` configuration module. Hydra overrides follow the subcommand.

## Essential commands

- List overrides for a preset: `dmri train +experiment/train=b3s_2_4_6_128 --help`
- Dry-run the resolved config: `dmri train +experiment/train=b3s_2_4_6_128 --cfg job`
- Override inline: `dmri train +experiment/train=b3s_2_4_6_128 training.dataloader.train_loader.batch_size=128 training.optimizer.learning_rate=1e-3`

## Configuration map (`conf/`)

- `train.yaml`: versioned run metadata, tracking settings, output directories, and defaults.
- `experiment/train/`: ready-made recipes that combine simulator, model, optimizer, and partition choices.
- `simulator/*.yaml`: one-line associations between names and importable `MultiCompartment` classes.
- `simulator/acquisition/`: directly importable acquisition factories used for synthetic training data.
- `model/`: neural architectures for selection/inference networks (dmri and ssfp variants).
- `training/`: loop controls (inner steps, EMA, checkpoint cadence, loss weights) and dataloader/optimizer subgroups.
- `infrastructure/`: resource presets and Hydra launcher wiring.

## Common overrides

- Batch size / optimizer: `training.dataloader.train_loader.batch_size`, `training.optimizer.learning_rate`, `training.optimizer.scheduler`.
- Model architecture groups: `model/embedding_net`, `model/model_selection_net`, `model/inference_net`.
- Simulator recipe: `simulator=ball3stick_shared`.
- Acquisition and score targets: `simulator/acquisition=multi`, `simulator.posterior_score=true`.
- External simulator class: `simulator.model_class=my_package.models.MyModel` (no repository config or registry change required).
- Partition and launcher: `infrastructure/partition=<profile>`, `infrastructure/launcher=slurm`.

## Outputs

Runs write to `results/<run_name>/<timestamp>/` with:
- `checkpoints/` (numeric steps and optionally `best/`), `artifact.yaml`, `.hydra/` (frozen config), logs/metrics, and W&B metadata when enabled.

## Adding a simulator

Define a `MultiCompartment` subclass in any importable package and set its class path:

```bash
dmri train simulator.model_class=my_package.models.MyModel
```

For a reusable short name, add one file such as `conf/simulator/my_model.yaml`:

```yaml
model_class: my_package.models.MyModel
```

No Python registry, `sim_type` file, or duplicate acquisition/score recipe is required.
