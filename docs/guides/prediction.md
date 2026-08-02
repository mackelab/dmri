# Pretrained Prediction

`dmri predict` is the direct interface for applying a pretrained model to diffusion MRI data. It runs inference and model selection, but omits metrics that require ground truth. Use the [full evaluation workflow](evaluation.md) only when its advanced Hydra configuration and metric exports are needed.

## Prepare the folder

The input folder must use this layout:

```text
FOLDER/
├── data.nii.gz
├── nodif_brain_mask.nii.gz
├── bvals
└── bvecs
```

Run the default model with:

```bash
dmri predict FOLDER
```

The default model is `msb3s_2_4_6_128`, a multi-shell Ball3Stick model from the public Hugging Face repository `manugloeck/dmri-pretrained`. Downloads are cached by `huggingface_hub`.

!!! warning "Acquisition compatibility"
    A pretrained checkpoint expects an acquisition scheme compatible with its training configuration. Required filenames and valid NIfTI inputs do not by themselves establish compatibility. Confirm the selected model's expected acquisition before interpreting results.

!!! danger "Research use only"
    DMRI has not been clinically validated and must not be used for clinical decisions.

## Outputs

By default, results are written below `FOLDER/dmri_output/`. Inference output is organized under `ball3stick_inference_results/`, and model-selection output under `ball3stick_model_selection_results/`. The command refuses to replace an existing output folder unless `--overwrite` is supplied.

```bash
dmri predict FOLDER --output-subdir analysis --overwrite
```

`--output-subdir` accepts a single directory name and keeps output inside `FOLDER`.

### Fiber fraction maps

`mean_f0samples.nii.gz` is the isotropic (ball) fraction and `mean_f1..f3samples.nii.gz`
are the three stick fractions. Within any one posterior sample the four sum to 1.

`mean_fsumsamples.nii.gz` is the total anisotropic fraction, averaged over the
posterior samples **whose model mask contains the ball**. This conditioning
matters: prediction averages over sampled model masks, and a sample that drops
the ball renormalizes to `f0 = 0`, which pins that sample's `f_sum` at exactly 1
and biases the map upward — most visibly in CSF. Two companion maps let you see
the effect:

- `mean_fsumsamples_all.nii.gz` — the unconditioned average, identically `1 - f0`.
- `frac_ball_active.nii.gz` — the fraction of samples that kept the ball. Where
  this is 1 the two maps agree; where it is lower they diverge.

Set `export.condition_fsum_on_ball=false` to make `mean_fsumsamples.nii.gz`
unconditioned.

## Guided run

With no options beyond the folder, prediction asks which model and which quality
preset to use, defaulting to the recommended choice at every prompt:

```bash
dmri predict FOLDER
```

It only prompts on a terminal, so scripts and CI keep working unchanged; pass
`--non-interactive` to be explicit. Choose directly to skip the questions:

```bash
dmri predict FOLDER --model b3s_2_4_6_128 --quality fast
```

`--quality` sets the sampling pipeline: `fast` uses 20 steps x 25 samples, fp16,
and no theta corrector; `balanced` uses 40 x 50, fp32, and automatic correction;
`high` uses 60 x 100, fp32, and automatic correction. Network sampling cost is
linear in `steps x samples`, while fast saves additional time by skipping the
corrector. The accuracy cost is data-dependent, so validate fast against a
higher-quality setting on your own scans.

During the run, compilation stages show spinners and sampling stages show voxel
progress bars. `--verbose` adds the detailed evaluation log without removing the
summary or progress display. Redirected output automatically uses stable text
lines instead of terminal animation.

Native XLA compiler chatter is hidden in normal and verbose modes. Set
`DMRI_SHOW_NATIVE_LOGS=1` only when debugging the JAX/XLA backend itself.

### Choose how models vary

By default, model uncertainty is retained: every parameter draw in every voxel
uses a posterior model-mask draw.

```bash
dmri predict FOLDER --model-mode per-sample
```

Choose one highest-probability feasible model per voxel instead:

```bash
dmri predict FOLDER --model-mode best
```

Or use one fixed Ball-and-Stick model throughout the brain:

```bash
dmri predict FOLDER --fixed-model B1S
dmri predict FOLDER --fixed-model B2S
dmri predict FOLDER --fixed-model B3S
```

Named fixed choices depend on the checkpoint's simulator family. The current
Ball3Stick-family checkpoints provide B1S, B2S, and B3S.

## Select a checkpoint

Choose another model or repository:

```bash
dmri predict FOLDER --model MODEL_NAME
dmri predict FOLDER --repo-id OWNER/REPOSITORY --revision REVISION
dmri predict --list-models
```

Use `--cache-dir PATH` to choose the Hugging Face cache. Use `--local-files-only` to disable downloads and require cached files.

For a local model bundle, bypass the Hub:

```bash
dmri predict FOLDER --local-checkpoint /path/to/model_bundle
```

## Sampling controls

Reproduce or adjust inference with `--seed`, `--mask-samples`, and `--theta-samples`:

```bash
dmri predict FOLDER --seed 1 --mask-samples 50 --theta-samples 50
```

The number of mask samples must be at least the number of theta samples.

Larger sample counts change the amount of sampling work. See the [prediction CLI reference](../cli/predict.md) for the complete option summary.

## Hardware

Prediction sizes itself to the machine. The batch size is the largest one that
fits: measured from free device memory, refined against XLA's memory analysis of
the compiled sampler. Every visible GPU is used — parameters are replicated once
and each batch is sharded across the devices, so the batch grows with the device
count. On CPU the budget comes from available host memory and never counts swap.

The first run on a new machine spends a few minutes autotuning, reported as a
single step up front. It is cached in `.jax_cache/dmri_batch_size.json`, so later
runs skip it entirely — on the reference run, 120 s first and 27 s thereafter.

Theta sampling runs in two passes — the network sampler and then the corrector —
each with its own batch size, because the corrector operates on parameters rather
than network activations and fits a far larger batch.

### Trading accuracy for speed

```bash
dmri predict FOLDER --num-steps 20 --theta-samples 25
```

Sampler cost scales as `num_steps x theta-samples x voxels`. The default 40 steps
performs 41 network evaluations per sample (the solver initialises once and takes
a final Euler correction); the run logs the true count. Fewer steps is
proportionally faster and less accurate — check a reduced setting against a
full-step run on your own data before adopting it.

Sampling depends only on `--seed`: each voxel's random key is fixed before
batching, so the same seed gives the same maps whatever batch size or GPU count
the machine resolves to.

Pin the batch size yourself when sharing a GPU or reproducing a previous run:

```bash
dmri predict FOLDER --batch-size 8192 --memory-fraction 0.5
```

### Half precision

```bash
dmri predict FOLDER --precision bf16
```

`--precision` accepts `fp32`, `bf16` and `fp16`. It defaults to `fp16` for fast
quality and `fp32` otherwise; an explicit flag overrides the preset. Half
precision halves the activation memory — which by itself allows a roughly 25%
larger batch — and is faster on tensor-core GPUs.

Only the network's matmul inputs are half precision. Weights stay float32,
results accumulate in float32, and the diffusion sampler, the correctors and
every exported map are computed in float32, so the parts of the pipeline that
are sensitive to dynamic range (the stick-breaking bijection, the likelihood with
its ~1e-3 mm²/s diffusivities) are unaffected. On the reference run, the
half-precision maps differ from float32 by less than float32's own seed-to-seed
Monte Carlo variation.

If half precision ever does produce non-finite samples, the run logs a warning
naming the affected fraction and suggests `fp32`.
