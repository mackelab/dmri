# Configuration

`dmri train` and `dmri eval` use Hydra. Configuration files are under `conf/`.
Command-line overrides use dotted keys, and config groups use `group=name`.

## Inspect a configuration

Print a composed configuration without running it:

```bash
dmri train --cfg job --resolve infrastructure/launcher=local infrastructure/partition=none
dmri eval --cfg job --resolve
```

List Hydra help and available groups:

```bash
dmri train --help
dmri eval --help
```

Use `+experiment/train=NAME` or `+experiment/eval=NAME` because experiment
groups are not selected by the base defaults.

## Override syntax

Set scalar values with dotted keys:

```bash
dmri train run.name=test training.max_train_hours=1 tracking.enabled=false
dmri eval checkpoint.model_name=my_run/2026-08-02_12-00-00
```

Select a group:

```bash
dmri train simulator=ball3stick_shared simulator/acquisition=multi
dmri eval evaluation/selection=ball3stick_best
```

Replace a nested group at a package path:

```bash
dmri eval \
  evaluation/theta/corrector@evaluation.sampling.theta.corrector=none
```

Quote list values and removal overrides in a shell:

```bash
dmri eval \
  evaluation.pipeline.default_mask='[true,true,true,false,true]' \
  '~evaluation.export.theta.metrics'
```

Hydra rejects keys that are not present in a structured config. Prefix a new
optional key with `+`, for example `+training.restart_every=100000`.

## Training tree

The base training config is `conf/train.yaml`:

```text
run
tracking
simulator
model
training
infrastructure
```

Important groups are:

- `conf/simulator/`: importable `MultiCompartment` classes.
- `conf/simulator/acquisition/`: acquisition generator partials.
- `conf/model/`: network and embedding configuration.
- `conf/training/`: loop, dataloader, and optimizer settings.
- `conf/infrastructure/`: Hydra launcher and resource settings.
- `conf/experiment/train/`: named combinations and sweep parameters.

The default output directories are:

```yaml
hydra:
  run:
    dir: results/${run.name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: results/${run.name}/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
```

Values under `hydra.sweeper.params` apply only to a multirun. Use `--multirun`
when launching a named training experiment that relies on them.

## Evaluation tree

The base evaluation config is `conf/eval.yaml`:

```text
run
checkpoint
evaluation
  input
  pipeline
  sampling.mask
  sampling.theta
  selection
  export.theta
  export.model_selection
```

Important groups are:

- `conf/evaluation/input/`: file or synthetic data.
- `conf/evaluation/mask/`: model-mask sampling.
- `conf/evaluation/theta/`: theta sampling and correctors.
- `conf/evaluation/selection/`: average, best, or no model selection.
- `conf/evaluation/export/`: exporters and metrics.
- `conf/experiment/eval/`: named evaluation configurations.

The default evaluation uses file input, samples masks and theta, uses no
selection transform, and exports Ball3Stick theta and model-selection results.

## Prediction tree

`dmri predict` composes `conf/predict.yaml`, which lists `eval` in its defaults —
so it inherits the whole evaluation tree above — and then layers the
prediction-specific policy:

```text
conf/predict.yaml
conf/predict/
  quality/       very-fast, fast, balanced, high
  model_mode/    per-sample, best, fixed
  fixed_model/   B1S, B2S, B3S
  viewer/        default, none
```

The CLI is a front end over this: `--quality fast` becomes
`predict/quality=fast`, `--fixed-model B2S` becomes `predict/fixed_model=B2S`,
and `--set key=value` passes anything else straight through. Because each preset
is a group choice rather than a set of inlined numbers, the run's
`.hydra/hydra.yaml` records *which* preset was used, not only its resolved
values.

A quality option sets the sampling keys and selects a corrector. It needs
`override` on the corrector because `conf/evaluation/theta/default.yaml` already
chooses one:

```yaml
# conf/predict/quality/fast.yaml
# @package _global_
defaults:
  - override /evaluation/theta/corrector@evaluation.sampling.theta.corrector: none

evaluation:
  precision: fp16
  sampling:
    theta: {num_samples: 25, params: {num_steps: 20}}
    mask: {n_samples: 25}
```

Prediction also selects `evaluation/export/theta: ball3stick_predict`, which is
`ball3stick` without the metrics exporters — there is no ground truth at
prediction time to score against.

To run the same configuration through `dmri_eval` instead, compose `predict`
directly:

```bash
dmri_eval --config-name predict predict/quality=fast --cfg job   # inspect
dmri_eval --config-name predict predict/quality=fast             # run
```

## Saved configuration

Hydra writes these files in each run directory:

```text
.hydra/
|-- config.yaml
|-- hydra.yaml
`-- overrides.yaml
```

They record the resolved application configuration, Hydra configuration, and
command-line overrides. Reproducing a run can still depend on package versions,
source revisions, devices, external data, and random seeds that are not embedded
in those files.

Training also writes `artifact.yaml`. This is a small model-construction
manifest used by portable checkpoints. It is not a replacement for the full
saved training configuration.

## Legacy configurations

The loader normalizes older schema layouts when reading saved runs. New configs
should use the current schema shown above, including `simulator.model_class`,
`simulator.acquisitions`, `simulator.mask_prior`, and
`simulator.posterior_score`.
