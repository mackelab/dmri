# Training

`dmri train` composes `conf/train.yaml`, generates training and validation data
from the configured simulator, builds the model, and runs the JAX training
loop. Use Hydra overrides to change a run; the saved Hydra config records the
resolved settings but is not a guarantee of exact reproducibility.

## Choose how the run executes

### One local process

```bash
dmri train infrastructure/launcher=local infrastructure/partition=none \
  run.name=local_test tracking.enabled=false training.max_train_hours=1
```

Without `--multirun` or `hydra.mode=MULTIRUN`, training runs directly in the
calling process. The launcher config is not invoked. This is the simplest mode
for development and for one-machine training.

Experiment files commonly put architecture values under
`hydra.sweeper.params`. Those values are sweep parameters and are ignored in a
single run. Either provide the same values as direct overrides or run the
experiment as a multirun.

### Local Hydra multirun

```bash
dmri train --multirun +experiment/train=b3s_2_4_6_128 \
  infrastructure/launcher=local infrastructure/partition=none
```

Hydra expands the preset's sweep parameters and Submitit starts each job
locally. Comma-separated values add sweep dimensions. Each job receives a
numbered output subdirectory.

### SLURM

```bash
dmri train --multirun +experiment/train=b3s_2_4_6_128 \
  infrastructure/launcher=slurm infrastructure/partition=h100_long
```

The SLURM launcher maps `hydra.launcher` values to a submitted job. Supplied
partition presets set queue, time, CPU, memory, and GPU fields and normally set
`hydra.mode=MULTIRUN`; inspect and select a preset that exists at your site.
`hydra.launcher.timeout_min` is the scheduler limit, while
`training.max_train_hours` is enforced inside training. They are independent.

## Compose a training config

The base config selects these groups:

```text
conf/
|-- train.yaml
|-- experiment/train/
|-- simulator/
|   `-- acquisition/
|-- model/
|   |-- embedding_net/
|   |-- model_selection_net/
|   `-- inference_net/
|-- training/
|   |-- dataloader/
|   `-- optimizer/
`-- infrastructure/
    |-- launcher/
    `-- partition/
```

`experiment/train` presets select groups and define sweep parameters. The other
groups can also be selected directly:

```bash
dmri train simulator=ball3stick_shared simulator/acquisition=multi \
  model=dmri training/dataloader=datastream \
  training/optimizer=adamw infrastructure/launcher=local \
  infrastructure/partition=none
```

Common scalar overrides are:

```bash
dmri train run.seed=7 run.name=my_run \
  tracking.enabled=false \
  training.dataloader.train_loader.batch_size=1024 \
  training.optimizer.learning_rate=3e-4 \
  training.max_train_hours=12
```

The current simulator schema separates the structural signal model from data
generation:

```yaml
simulator:
  model_class: my_package.models.MyModel
  posterior_score: false
  mask_prior: {}
  acquisitions:
    - _target_: my_package.acquisition.random_acquisition
      _partial_: true
      num_acquisitions: 128
```

`model_class` must resolve to a `MultiCompartment` subclass. Every acquisition
entry must instantiate to a callable, which is why function targets use
`_partial_: true`. A supplied `simulator/acquisition` group writes the same
`simulator.acquisitions` list.

## Steps, validation, and stopping

One optimizer update increments the training step by one. The loop performs
`training.inner_steps` updates before logging, checking elapsed time, validating,
or saving by cadence. `training.checkpoint_freq`, `training.eval_freq`, and the
optional `training.restart_every` are rounded down to a multiple of
`inner_steps`; a positive value smaller than `inner_steps` becomes one full
block. Add the last setting with `+training.restart_every=<steps>` because it is
absent from the base config.

`training.max_train_hours` is the only configured completion limit. Elapsed time
is checked after a block, and the loop writes a regular checkpoint before
stopping on that limit. There is no supported maximum-step setting. The loop can
also stop when a non-finite loss cannot be recovered from a regular checkpoint.
Signals and scheduler termination do not cause a final save.

Periodic validation computes mask and parameter negative log likelihood and,
when configured, KSD. With EMA enabled, validation uses EMA parameters. The sum
of the two negative log likelihood values selects the best checkpoint; KSD does
not select it.

## Outputs

A single job writes to:

```text
results/<run.name>/<timestamp>/
|-- .hydra/
|   |-- config.yaml
|   |-- hydra.yaml
|   `-- overrides.yaml
|-- artifact.yaml
|-- checkpoints/
|   |-- <step>/
|   `-- best/
|       `-- <step>/
`-- train_script.log
```

A multirun writes jobs below
`results/<run.name>/<timestamp>/<hydra.job.num>/`. `artifact.yaml` contains only
the model config and the simulator model class needed to rebuild the model. The
full `.hydra/config.yaml` also contains acquisitions, training, and tracking
settings.

Regular checkpoints are numeric directories immediately under `checkpoints/`.
Orbax retains three by default; set
`+training.checkpoint.max_checkpoints=<count>` to change this. The latest
checkpoint is the highest retained regular step. When best-checkpoint tracking
is enabled, `checkpoints/best/` retains one validation-selected step.

Every regular or best checkpoint contains raw parameters, optimizer state,
model state, step, loss, and RNG state. When `training.track_ema=true`, it also
contains `params_ema` and EMA state. EMA does not replace raw parameters.

## Resume training

Resume searches only the current job's `checkpoints/` directory and restores
the latest regular checkpoint, not `checkpoints/best/`. A new command normally
creates a new timestamped directory, so `training.continue_training=true` alone
does not locate an earlier run. Point Hydra at the existing job directory:

```bash
dmri train infrastructure/launcher=local infrastructure/partition=none \
  hydra.run.dir=/absolute/path/to/existing/run \
  training.continue_training=true
```

For a resumed multirun, preserve the original mapping of jobs to directories by
setting `hydra.sweep.dir` and `hydra.sweep.subdir` to the existing layout.

Normal resume restores parameters, model state, optimizer state, EMA state, RNG,
and step when present. `training.restart_optimizer=true` rebuilds optimizer
state after loading. `training.partial_restore=true` requests an Orbax partial
parameter restore and also rebuilds optimizer and EMA state. Simulation buffers,
loader position, and background producer state are not checkpointed, so a
resumed run should not be described as an exact continuation of sample order.

## Load a checkpoint in Python

```python
from flax import nnx
from dmri.train.utils import load_checkpoint

checkpoint, model, simulator_model = load_checkpoint(
    "/absolute/path/to/results/my_run/2026-08-02_12-00-00",
    which="best",
)
params = checkpoint.get("params_ema", checkpoint["params"])
nnx.update(model, params)
model.eval()
```

`which` accepts `"latest"`, `"best"`, or an integer regular step. Loading
returns the checkpoint payload and a rebuilt model; it does not apply the
returned parameters to that model. Choose `params_ema` explicitly when wanted,
with `params` as the fallback.

## Hand off to evaluation

For a run at
`/work/results/my_run/2026-08-02_12-00-00`, evaluation expects the parent as
`checkpoint.path` and the run directory name as `checkpoint.model_name`:

```bash
dmri eval \
  checkpoint.path=/work/results/my_run \
  checkpoint.model_name=2026-08-02_12-00-00 \
  checkpoint.which=best \
  evaluation.input.path=/work/data/subject
```

For a multirun, use the numbered job directory in the same way: its parent is
`checkpoint.path` and its job number is `checkpoint.model_name`. Evaluation uses
`checkpoint.params_name=params_ema` by default and falls back to `params` if the
selected checkpoint has no EMA parameters. See the
[evaluation guide](evaluation.md) for input, sampling, metrics, and export
settings.

## Bundle or publish a checkpoint

```python
from dmri.train.utils import bundle_checkpoint, upload_checkpoint_to_hub

bundle_checkpoint(run_dir, "bundles", model_name="my_model")
upload_checkpoint_to_hub(
    run_dir,
    "owner/models",
    model_name="my_model",
    private=True,
)
```

Both functions include best and latest by default. `which` also accepts
`"best"`, `"latest"`, `"all"`, or an integer regular step. A bundle stores the
model under `<output_dir>/<model_name>/` with `config.yaml`, `artifact.yaml`, and
the selected checkpoint tree.
