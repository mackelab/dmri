# `dmri train`

Train a model from the Hydra configuration in `conf/train.yaml`.

## Syntax

```text
dmri train [HYDRA_OPTIONS] [OVERRIDE ...]
dmri train --multirun [HYDRA_OPTIONS] [OVERRIDE ...]
```

Use `--cfg job` to print the composed application config without training and
`--info defaults` to inspect the selected config groups.

## Local single run

```bash
dmri train infrastructure/launcher=local infrastructure/partition=none \
  run.name=test tracking.enabled=false training.max_train_hours=1
```

A single run executes in the current process. Hydra does not use the configured
Submitit launcher unless the run is a multirun. Values under an experiment
preset's `hydra.sweeper.params` also do not apply to a single run.

## Local multirun

```bash
dmri train --multirun +experiment/train=b3s_2_4_6_128 \
  infrastructure/launcher=local infrastructure/partition=none
```

`--multirun` applies `hydra.sweeper.params` and submits each job through the
local Submitit launcher. A comma-separated override creates a sweep:

```bash
dmri train -m infrastructure/launcher=local infrastructure/partition=none \
  training.optimizer.learning_rate=1e-4,3e-4
```

## SLURM multirun

```bash
dmri train --multirun +experiment/train=b3s_2_4_6_128 \
  infrastructure/launcher=slurm infrastructure/partition=h100_long
```

The supplied cluster partition presets set `hydra.mode=MULTIRUN`, so `-m` is
optional when one of those presets is selected. Site-specific partition names
and resources are defined in `conf/infrastructure/partition/`.

## Config groups

| Group | Purpose |
| --- | --- |
| `experiment/train` | Named combinations and sweep parameters |
| `simulator` | `MultiCompartment` model class |
| `simulator/acquisition` | One or more acquisition factory targets |
| `model` | Model and network config defaults |
| `model/embedding_net` | Signal embedding |
| `model/model_selection_net` | Model-mask network |
| `model/inference_net` | Parameter inference network |
| `training` | Loop, loss, evaluation, EMA, and checkpoint cadence |
| `training/dataloader` | Simulation buffer and loader settings |
| `training/optimizer` | Optimizer and scheduler settings |
| `infrastructure/launcher` | Local or SLURM Submitit launcher |
| `infrastructure/partition` | Launcher resources and queue |

## Common overrides

```bash
# Select a simulator and acquisitions using the current schema.
dmri train simulator=ball3stick_shared simulator/acquisition=multi

# Use an importable custom MultiCompartment subclass.
dmri train simulator.model_class=my_package.models.MyModel

# Change training settings.
dmri train training.dataloader.train_loader.batch_size=1024 \
  training.optimizer.learning_rate=3e-4 training.track_ema=false

# Print the application config and the preset's Hydra sweep parameters.
dmri train +experiment/train=b3s_2_4_6_128 --cfg all
```

The canonical simulator fields are `simulator.model_class`,
`simulator.acquisitions`, `simulator.mask_prior`, and
`simulator.posterior_score`. Select `simulator/acquisition=<name>` instead of
writing the `acquisitions` list inline when a supplied group is suitable.

## Stopping

`training.max_train_hours` is the implemented training limit. It is checked
after each block of `training.inner_steps`; the run can therefore exceed the
limit by one block and then writes a regular checkpoint before exiting.

There is no training `max_steps` option. Step values configure checkpoint,
validation, and optimizer-restart cadence, not a final step. External process or
SLURM termination does not trigger a final checkpoint, so set
`training.max_train_hours` below `hydra.launcher.timeout_min / 60` when a final
save is required.

## Output

Single runs use:

```text
results/<run.name>/<YYYY-MM-DD_HH-MM-SS>/
```

Multiruns add the Hydra job number:

```text
results/<run.name>/<YYYY-MM-DD_HH-MM-SS>/<hydra.job.num>/
```

Each job contains `.hydra/`, `artifact.yaml`, the Hydra log, and
`checkpoints/`. See the [training guide](../guides/training.md) for checkpoint,
resume, and evaluation behavior.
