# `dmri predict`

## Syntax

```text
dmri predict FOLDER [options]
dmri predict --list-models [--repo-id REPOSITORY] [--revision REVISION]
```

`FOLDER` must contain `data.nii.gz`, `nodif_brain_mask.nii.gz`, `bvals`, and
`bvecs`.

## Checkpoint

- `--model NAME`: model subfolder in the repository. Default:
  `msb3s_2_4_6_128`.
- `--repo-id OWNER/REPOSITORY`: Hugging Face model repository. Default:
  `manugloeck/dmri-pretrained`.
- `--revision REVISION`: Hub branch, tag, or commit.
- `--cache-dir PATH`: Hugging Face cache location.
- `--local-files-only`: do not contact the Hub.
- `--local-checkpoint PATH`: local portable model bundle; bypasses the Hub.
- `--list-models`: list repository model subfolders and exit.

## Sampling

- `--quality {very-fast,fast,balanced,high}`: named sampling settings. Default:
  `balanced`. Each name selects a `conf/predict/quality/` group, so the numbers
  behind it are visible in the tree and recorded in the run's `.hydra` output.
- `--num-steps INTEGER`: ODE steps per theta sample.
- `--mask-samples INTEGER`: posterior mask samples.
- `--theta-samples INTEGER`: posterior parameter samples.
- `--corrector {auto,none}`: parameter correction policy.
- `--precision {fp32,bf16,fp16}`: network-forward precision. CPU fp16 falls
  back to fp32.
- `--seed INTEGER`: random seed. Default: `1`.

## Model selection

- `--model-mode {per-sample,best,fixed}`: sampled model per theta draw, one
  selected model per voxel, or one fixed model.
- `--fixed-model {B1S,B2S,B3S}`: fixed Ball3Stick-family model; implies fixed
  mode.

In per-sample mode, mask samples must be at least theta samples.

## Hardware

- `--batch-size INTEGER`: fixed voxels per batch; skips automatic sizing.
- `--memory-fraction FLOAT`: fraction of device memory JAX may preallocate.

CPU prediction prints a warning because even a few thousand voxels can take a
long time. Use a supported GPU accelerator when possible.

## Output and interaction

- `--output-subdir NAME`: directory inside `FOLDER`. Default: `dmri_output`.
- `--overwrite`: replace an existing output directory.
- `--no-viewer`: do not write `view_results.html`.
- `--non-interactive`: do not prompt for missing choices.
- `--verbose`: include detailed evaluation logs.

Interactive selection is used only when stdin and stdout are terminals. Use
arrow keys or `j`/`k`, Enter, or number keys.

## Checkpoint and configuration

- `--checkpoint-which {latest,best,STEP}`: which checkpoint to load. Default:
  `latest`. The pretrained bundles on the Hub currently ship only `best`, so the
  Hub path needs `--checkpoint-which best`.
- `--set KEY=VALUE`: set any configuration key directly. Repeatable, and applied
  last so it overrides both `--quality` and the other flags.

`--set` is the escape hatch for anything without a dedicated flag:

```bash
dmri predict FOLDER --quality fast \
  --set evaluation.sampling.theta.params.t_max=60 \
  --set evaluation.export.theta.export_stds=true
```

Prediction composes `conf/predict.yaml`, which inherits the whole evaluation tree
and layers the prediction presets on top. See
[Configuration](../reference/configuration.md#prediction-tree).

See the [prediction guide](../guides/prediction.md) for workflow and output
details.
