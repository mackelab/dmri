# Evaluation And Export

`dmri predict` and `dmri eval` use the same inference implementation but expose
it differently:

- `dmri predict FOLDER` is the routine inference command. It validates the four
  standard input files, uses a pretrained Hugging Face model unless a local
  bundle is supplied, offers named quality and model modes, disables metrics,
  and writes below `FOLDER/dmri_output/` by default.
- `dmri eval` is the Hydra entry point. It supports local training runs, local
  portable bundles, Hugging Face bundles, generated data, configurable metrics,
  pipeline-stage switches, and exporter selection. It does not perform the
  input and output safety checks provided by `dmri predict`.

Use `dmri eval` for the workflows below. Run with `--cfg job --resolve` to print
the composed configuration without evaluating it.

## Evaluate A Local Training Run

A normal training result has this layout:

```text
results/<run>/<timestamp>/
  .hydra/config.yaml
  checkpoints/
```

Point `checkpoint.model_name` at the path relative to `checkpoint.results_root`,
which defaults to `results/` below the command's starting directory:

```bash
dmri eval +experiment/eval=eval_b3s_best_model_selection \
  checkpoint.model_name=my_run/2026-08-02_12-00-00 \
  checkpoint.which=best \
  evaluation.input.path=/data/sub-001/dwi
```

Set `checkpoint.results_root=/other/results` when the run is not below the
default `results/` directory. `checkpoint.results_folder` is an optional path
inserted between `results_root` and `model_name`.

## Evaluate A Portable Checkpoint

A portable bundle has a root `config.yaml`, optional `artifact.yaml`, and a
`checkpoints/` tree. `checkpoint.path` must name its parent directory and
`checkpoint.model_name` its directory name:

```bash
dmri eval +experiment/eval=eval_b3s_no_selection \
  checkpoint.path=/models \
  checkpoint.model_name=msb3s_2_4_6_128 \
  checkpoint.which=latest \
  evaluation.input.path=/data/sub-001/dwi
```

This resolves the checkpoint as `/models/msb3s_2_4_6_128`.

## Evaluate A Hugging Face Checkpoint

Set a repository ID to switch from local loading to the Hugging Face loader:

```bash
dmri eval +experiment/eval=eval_b3s_best_model_selection \
  checkpoint.pretrained.repo_id=manugloeck/dmri-pretrained \
  checkpoint.model_name=msb3s_2_4_6_128 \
  checkpoint.which=latest \
  evaluation.input.path=/data/sub-001/dwi \
  run.output_dir=/data/sub-001/dwi/eval-results
```

Optional settings are `checkpoint.pretrained.revision`,
`checkpoint.pretrained.cache_dir`, and
`checkpoint.pretrained.local_files_only=true`. A Hub repository must contain the
same portable model subfolder layout used by `bundle_checkpoint`. When
`run.output_dir` is omitted, Hub-backed evaluation still writes to
`results/<checkpoint.model_name>/`; it does not write into the Hub cache.

## Real Input

The file input config expects these names by default:

```text
<input>/data.nii.gz
<input>/nodif_brain_mask.nii.gz
<input>/bvals
<input>/bvecs
```

Use either input form:

```bash
# Direct path. This takes precedence if both forms are set.
evaluation.input.path=/data/sub-001/dwi

# Path assembled as /data/sub-001/dwi.
evaluation.input.data_root=/data evaluation.input.data_folder=sub-001/dwi
```

Override `evaluation.input.mri_data`, `brain_mask`, `bvals_data`, or
`bvecs_data` for other filenames.

The real-data loader performs the following steps:

1. Converts mask values greater than zero to `true`.
2. Rounds b-values to the nearest multiple of 1000 and clips them at zero when
   `evaluation.input.round_bvals=true`, which is the default.
3. Stably sorts b-values and applies the same order to b-vectors and signal
   volumes.
4. Reads only in-mask voxels, in chunks along the gradient axis.
5. Divides each voxel's signal by its mean b=0 signal. A missing b=0 shell is an
   error; voxels with a non-positive b=0 mean become zero.
6. Replaces NaN and infinite signal values with zero.
7. Clips the normalized in-mask signal to its global 99.9th percentile when
   `evaluation.input.clip_outliers=true`, which is the default for file input.

Restrict work to a spatial range with `evaluation.input.slice.x`, `.y`, and
`.z`. Each value accepts an index such as `52` or a slice string such as
`20:80`, `::2`, or `:`. The mask is restricted before signal volumes are read.

## Synthetic Input

Synthetic input samples model masks, parameters, and signals from the simulator
stored with the checkpoint. It therefore provides true parameters and true
model masks for SBC and model-selection metrics.

```bash
dmri eval evaluation/input=synthetic \
  evaluation/export/theta=raw \
  checkpoint.pretrained.repo_id=manugloeck/dmri-pretrained \
  checkpoint.model_name=msb3s_2_4_6_128 \
  evaluation.input.num_voxels=512 \
  evaluation.pipeline.sample_mask=false \
  evaluation.pipeline.select_models=false \
  run.output_dir=/tmp/dmri-synthetic-eval
```

The default synthetic acquisition is `hcp`. Other named builders are
`clinical`, `hardi`, `advanced_research`, `hcp_large`, and `ssfp`:

```bash
evaluation.input.acquisition_scheme=clinical
```

Alternatively, set `evaluation.input.acquisition_scheme=null` and supply
`evaluation.input.bvals=[0,1000,2000]` plus matching `bvecs`. If `bvecs` is
`null`, random unit vectors are generated. Synthetic signals are already in the
simulator's normalized scale; the real-data b=0 normalization and b-value
rounding are not applied.

With `evaluation.input.use_true_model_mask_for_synthetic=true`, the generated
mask conditions parameter sampling unless a selected mask is produced by the
model-selection stage.

## Pipeline And Model Modes

The three stage switches are:

- `evaluation.pipeline.sample_mask`: draw posterior model-mask samples.
- `evaluation.pipeline.select_models`: turn mask samples or feasible-model
  probabilities into masks used for parameter sampling.
- `evaluation.pipeline.sample_theta`: draw posterior parameter samples.

Configure them together with an `evaluation/selection` group. These are the
three useful parameter-inference modes.

### Per-Sample Models

Each parameter draw uses a sampled model mask:

```bash
dmri eval evaluation/selection=average \
  evaluation.pipeline.sample_mask=true \
  evaluation.pipeline.select_models=true \
  evaluation.pipeline.default_mask=null \
  evaluation.sampling.mask.n_samples=50 \
  evaluation.sampling.theta.num_samples=50 \
  checkpoint.model_name=my_run/2026-08-02_12-00-00 \
  evaluation.input.path=/data/sub-001/dwi
```

The number of mask samples must be at least the number of theta samples.

### Best Feasible Model

For every voxel, score the masks listed in the selection config and use the
highest-probability mask:

```bash
dmri eval evaluation/selection=ball3stick_best \
  evaluation.pipeline.sample_mask=false \
  evaluation.pipeline.select_models=true \
  evaluation.pipeline.default_mask=null \
  checkpoint.model_name=my_run/2026-08-02_12-00-00 \
  evaluation.input.path=/data/sub-001/dwi
```

The supplied `ball3stick_best` group considers B1S, B2S, and B3S masks. Its mask
length and component order are specific to Ball3Stick-family simulators.

### Fixed Model

Disable mask sampling and selection, then provide one component mask for every
voxel. This example is the Ball3Stick B2S mask:

```bash
dmri eval evaluation/selection=none \
  evaluation.pipeline.sample_mask=false \
  evaluation.pipeline.select_models=false \
  evaluation.pipeline.default_mask='[true,true,true,false,true]' \
  checkpoint.model_name=my_run/2026-08-02_12-00-00 \
  evaluation.input.path=/data/sub-001/dwi
```

The mask must contain exactly one value per simulator component. If
`default_mask=null`, parameter sampling enables all components. Mask-only runs
are also supported by setting `sample_theta=false`.

## Parameter Sampling And Correctors

The default theta config draws 50 samples with 40 ODE steps. Change those costs
directly:

```bash
evaluation.sampling.theta.num_samples=25 \
evaluation.sampling.theta.params.num_steps=20
```

Select a corrector through its Hydra group:

```bash
evaluation/theta/corrector@evaluation.sampling.theta.corrector=auto
```

Available configs include `none`, `auto`, `smc`, `smc_high`,
`smc_from_scratch`, and `mcmc`. `none` returns the network samples unchanged.
`auto` uses SMC for one fixed mask per voxel and MCMC when masks vary across
posterior samples. SMC does not accept per-sample masks. Correctors run after
network sampling in a separate batch.

## Devices, Precision, And Batching

Evaluation uses all visible GPUs, otherwise all visible TPUs, otherwise the JAX
default devices. With multiple accelerators, model parameters are replicated and
voxel batches are split across devices.

CPU execution is supported but posterior sampling and KSD can be very slow for
whole volumes. Use a GPU or TPU for normal evaluation. On CPU, requested `fp16`
falls back to `fp32`.

Set network-forward precision with:

```bash
evaluation.precision=fp32  # fp32, bf16, or fp16
```

For `bf16` and `fp16`, model parameters, accumulation, the sampler, correctors,
metrics, and exports remain float32; only network inputs use the lower
precision. If half-precision sampling produces non-finite theta values, rerun in
`fp32`.

When batch sizes are `null`, evaluation probes a workable voxel batch for each
stage and caches the result in `.jax_cache/dmri_batch_size.json`. Separate sizes
are chosen for mask sampling, theta sampling, the corrector, and feasible-model
scoring. Override every stage with `evaluation.batch_size=<voxels>`, or set
`evaluation.sampling.mask.eval_batch_size` and
`evaluation.sampling.theta.eval_batch_size` separately. Runtime out-of-memory
errors cause the current batch to be halved and retried.

## Metrics And Prerequisites

Theta exporters compose metrics below `evaluation.export.theta.metrics`.
Disable all of them with:

```bash
dmri eval ... '~evaluation.export.theta.metrics'
```

Metric requirements are:

| Metric type | Required data |
| --- | --- |
| `reconstruction_mse` | Theta samples, observed signal, and acquisition; despite its key, the implementation computes mean absolute signal error |
| `posterior_nll` | Theta samples, observed signal, acquisition, and the simulator's standard-normal theta prior |
| `ksd` | At least two theta samples, observed signal, acquisition, and the simulator posterior |
| `sliced_wasserstein` | Theta samples plus `options.reference_samples_path`; the reference sample array must have the same shape |
| `sbc_marginal_coverage` | Synthetic true parameters and posterior theta samples; skipped for real input |
| `sbc_model_mask` | Synthetic true masks and posterior mask samples |
| `model_selection_calibration` | Synthetic true masks and model-selection predictions or probabilities |
| `model_selection_classification` | Synthetic true masks and model-selection predictions or probabilities |

Metrics marked as not requiring theta can run when `sample_theta=false`. Metrics
that require theta are skipped by the CLI when no theta samples are available.
The model-selection metrics raise an error without synthetic true masks.

When theta samples exist and the configured theta metric set is non-empty, the
evaluator adds KSD if the set does not already contain it, except when every
configured metric is a model-selection metric. Remove the complete metrics node
to avoid this automatic KSD run.

Each metric normally writes its configured NIfTI map and a JSON summary in the
theta or model-selection export directory. KSD can also write a p-value NIfTI;
SBC and model-selection metrics can write additional JSON diagnostics. A metric
that is not applicable and returns no values writes no files.

## Export Locations

Without an output override, the export root for a normal run or Hub checkpoint
is:

```text
<checkpoint.results_root>/<checkpoint.results_folder>/<checkpoint.model_name>/
```

The optional `results_folder` segment is omitted when it is `null`, so the
defaults reduce to `results/<checkpoint.model_name>/`. For a local portable
bundle configured with `checkpoint.path=<parent>`, the export root is
`<parent>/<checkpoint.model_name>/`. Each exporter creates its own named
subdirectory, for example:

```text
results/my_run/2026-08-02_12-00-00/
  ball3stick_inference_results_best_model_selection/
  ball3stick_model_selection_results/
  ball3stick_model_selection_mask_samples/
```

`evaluation.export.theta.name` and
`evaluation.export.model_selection.name` set the subdirectory names. Set
`run.output_dir=/absolute/output` to replace the export root. This is separate
from `hydra.run.dir`, which stores Hydra configuration and log metadata.

The `ball3stick` theta exporter always writes `raw_thetas.nii.gz` and derives
model-specific parameter maps. The `raw` exporter writes `raw_thetas.nii.gz`,
`theta_samples.npz`, `true_theta.npz`, and `true_model_mask.npz`; the true-value
archives contain `None` for real input. Model-selection exporters write either
Ball3Stick NIfTI maps or raw mask samples, depending on their configured type.

Set `evaluation.pipeline.reuse_theta_samples=true` to reuse a single existing
`raw_thetas.nii.gz` found below the current theta export directory. Its voxel,
sample, and parameter dimensions must match the current run.
