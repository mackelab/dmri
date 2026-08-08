# Prediction

`dmri predict` applies a pretrained checkpoint to diffusion MRI data in the
standard FSL folder layout. It downloads or opens a checkpoint, samples model
masks and parameters, and writes NIfTI maps. Use [`dmri eval`](evaluation.md)
when you need custom Hydra configuration, synthetic data, or metrics.

## Input

The input folder must contain:

```text
FOLDER/
|-- data.nii.gz
|-- nodif_brain_mask.nii.gz
|-- bvals
`-- bvecs
```

The image and mask dimensions must agree. The signal volume must have one
measurement per b-value and b-vector.

> **Check acquisition compatibility.** A checkpoint was trained for a
> particular acquisition distribution. Matching filenames and image dimensions
> does not establish compatibility.

## Run prediction

For an interactive terminal:

```bash
dmri predict FOLDER
```

The selector asks for a model, quality, model mode, and a fixed model when that
mode is selected. Use arrow keys or `j`/`k`, then Enter. Number keys select an
entry directly. Command-line values are not prompted again.

For scripts:

```bash
dmri predict FOLDER \
  --model msb3s_2_4_6_128 \
  --quality balanced \
  --model-mode best \
  --non-interactive
```

The default repository is `manugloeck/dmri-pretrained`, and the default model is
`msb3s_2_4_6_128`. Hugging Face downloads are cached locally.

## Quality

| Quality | ODE steps | Samples | Network precision | Corrector |
| --- | ---: | ---: | --- | --- |
| `very-fast` | 8 | 10 | fp16 on supported accelerators | none |
| `fast` | 20 | 25 | fp16 on supported accelerators | none |
| `balanced` | 40 | 50 | fp32 | auto |
| `high` | 60 | 100 | fp32 | auto |

An explicit `--num-steps`, `--theta-samples`, `--precision`, or `--corrector`
overrides the selected preset. Runtime grows approximately with voxels x theta
samples x network evaluations. Validate lower-cost settings on representative
data before using them for a study.

On CPU, fp16 falls back to fp32. CPU prediction is supported but can be very
slow even for a few thousand voxels. Use a supported GPU accelerator for normal
volume-sized inputs.

## Model mode

`per-sample` retains model uncertainty. Each parameter sample is conditioned on
a sampled model mask:

```bash
dmri predict FOLDER --model-mode per-sample
```

`best` selects one highest-probability feasible model per voxel:

```bash
dmri predict FOLDER --model-mode best
```

`fixed` uses one Ball-and-Stick model throughout the mask:

```bash
dmri predict FOLDER --fixed-model B2S
```

The current Ball3Stick checkpoints support `B1S`, `B2S`, and `B3S`. These names
are checkpoint-family specific.

In `per-sample` mode, `--mask-samples` must be at least
`--theta-samples`.

## Checkpoint source

List models or choose another Hub revision:

```bash
dmri predict --list-models
dmri predict FOLDER --model MODEL --repo-id OWNER/REPOSITORY
dmri predict FOLDER --revision COMMIT_OR_TAG
```

Use cached files without a network request:

```bash
dmri predict FOLDER --local-files-only
```

Use a local portable bundle:

```bash
dmri predict FOLDER --local-checkpoint /models/my_model
```

See [Checkpoints and outputs](../reference/outputs.md) for the bundle layout.

## Output

The default output is `FOLDER/dmri_output/`:

```text
dmri_output/
|-- ball3stick_inference_results/
|-- ball3stick_model_selection_results/
`-- view_results.html
```

The HTML viewer contains selected scalar maps. It is written after successful
inference; use `--no-viewer` to skip it. The NIfTI files remain the complete
output and can be opened in another viewer.

Common inference maps include:

- `mean_f0samples.nii.gz`: mean isotropic fraction.
- `mean_f1samples.nii.gz` through `mean_f3samples.nii.gz`: mean stick fractions.
- `mean_fsumsamples.nii.gz`: mean total anisotropic fraction, conditioned on
  samples that retain the ball when that export option is enabled.
- `mean_fsumsamples_all.nii.gz`: unconditioned mean total anisotropic fraction.
- `frac_ball_active.nii.gz`: fraction of posterior masks containing the ball.
- `mean_dsamples.nii.gz`: mean diffusivity summary.
- `mean_num_fib_predsamples.nii.gz`: mean predicted number of fibers.

Outside-mask voxels are zero. See [Checkpoints and outputs](../reference/outputs.md)
for directories and naming conventions.

Choose a different subdirectory or replace an earlier result:

```bash
dmri predict FOLDER --output-subdir analysis --overwrite
```

## Batching and logs

If `--batch-size` is omitted, DMRI probes stage-specific batch sizes and caches
them in `.jax_cache/dmri_batch_size.json`. A first run on a device therefore
includes compilation and probing time. Later runs can reuse both batch and JAX
compilation caches. Runtime out-of-memory errors reduce the current batch and
retry it.

Use `--batch-size` to set a fixed voxel batch. Use `--memory-fraction` before
backend initialization when sharing a GPU. All visible supported accelerators
are used.

Normal output shows a summary and progress. `--verbose` adds evaluation logs.
Set `DMRI_SHOW_NATIVE_LOGS=1` only when debugging JAX/XLA diagnostics.

## Research use

DMRI has not been clinically validated and must not be used for clinical
decisions.
