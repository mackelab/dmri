# Prediction CLI

## Synopsis

```text
dmri predict FOLDER [options]
dmri predict --list-models [--repo-id REPO_ID] [--revision REVISION]
```

`FOLDER` must contain `data.nii.gz`, `nodif_brain_mask.nii.gz`, `bvals`, and `bvecs`. Prediction writes `ball3stick_inference_results/` and `ball3stick_model_selection_results/` below `FOLDER/dmri_output/` by default and omits ground-truth metrics.

## Model source

- `--model NAME`: pretrained model name; default `b3s_2_4_6_128`.
- `--repo-id ID`: Hugging Face model repository; default `manugloeck/dmri-pretrained`.
- `--revision REVISION`: repository branch, tag, or commit.
- `--cache-dir PATH`: Hugging Face cache directory.
- `--local-checkpoint PATH`: use a local model bundle instead of the Hub.
- `--local-files-only`: disable downloads and use cached Hub files only.
- `--list-models`: list models in the selected repository and exit; `FOLDER` is not required.

## Guided run

Run `dmri predict FOLDER` with nothing else and it asks for what it needs:

```text
Model  (from manugloeck/dmri-pretrained)
  1) b3s_2_4_6_128    recommended default <-
  2) b3s_2_4_6_64
Choose [1]:

Quality
  1) fast             20 steps x 25 samples  (~4x faster than balanced)
  2) balanced         40 steps x 50 samples  (default) <-
  3) high             60 steps x 100 samples  (~3x slower than balanced)
Choose [2]:
```

Pressing enter takes the default at every step. Anything given on the command
line is not asked about. Prompts appear only on a terminal, so scripts and CI
never block; `--non-interactive` suppresses them explicitly.

## Output and sampling

- `--quality {fast,balanced,high}`: sampling preset; default `balanced`, which is
  the same 40 steps x 50 samples that earlier releases used.
- `--output-subdir NAME`: output directory name inside `FOLDER`; default `dmri_output`.
- `--overwrite`: replace an existing output directory.
- `--seed INTEGER`: sampling seed; default `1`.
- `--mask-samples INTEGER`: number of model-mask samples. Overrides `--quality`.
- `--theta-samples INTEGER`: number of parameter samples. Overrides `--quality`.
- `--verbose`: show the full evaluation log instead of the summary and progress bars.
- `--non-interactive`: never prompt.

`--mask-samples` must be greater than or equal to `--theta-samples` because
each parameter sample is conditioned on a sampled model mask.

The presets differ only in cost, which is exactly linear in
`num_steps x samples`: `fast` is 4x cheaper than `balanced` and `high` is 3x
dearer. How much accuracy that costs depends on the data, so check a reduced
setting against a `high` run before relying on it.

## Hardware and batching

The batch size defaults to the largest the machine can run. It is derived from
the device's free memory, refined against XLA's memory analysis of the compiled
sampler, then validated by a compile at the chosen size. If a batch still runs
out of memory the size is halved and retried.

This autotuning happens once, up front, in a single labelled step:

```
Autotuning batch sizes for this device (one-time; cached in .jax_cache)
  [1/3] mask sampling + model selection    4800 voxels/batch    22.1s
  [2/3] theta sampling                     4800 voxels/batch    37.9s
  [3/3] corrector                          4800 voxels/batch     8.0s
Autotuning done in 68.0s - later runs on this machine reuse it and skip this step.
```

It is genuinely a one-time cost: the result is cached in
`.jax_cache/dmri_batch_size.json` alongside the compilation cache. On the
reference run the first pass took 120 s and every later one 27 s.

A measured batch size is only valid for the computation it was measured on, so
the cache key covers everything that changes a stage's memory footprint — the
model, the signal length, sample counts, solver settings (`num_steps`,
`last_euler_step`, corrector parameters), precision, and the device set — plus a
fingerprint of the package source, so editing the model or the sampler
invalidates it too. Keys are per stage, so changing `--num-steps` re-tunes the
theta stages while mask sampling keeps its cached size, and switching back reuses
the earlier measurement. Anything that still slips through is caught at runtime by
the out-of-memory halving.

On CPU, where there is no allocator limit to read, the budget comes from the
host's actually-available memory (`MemAvailable`), deliberately excluding swap —
swapping is far slower than simply using a smaller batch.

All visible accelerators are used automatically: the model parameters are
replicated once and each batch is sharded across the devices, so the batch scales
with the device count.

Theta sampling runs as two passes with independent batch sizes. The network
sampler is memory-bound; the corrector works on parameters rather than network
activations and fits a batch roughly 7x larger, so fusing them would hold it
back.

Sampling is reproducible from the seed alone: each voxel's random key is derived
before batching, so the resolved batch size and the number of devices do not
change the draws.

The arithmetic can still differ in the last bits. XLA selects GPU kernels by
benchmarking them on the first compile and caches that choice, so a cold-cache run
may use different kernels — and therefore different summation orders — than later
ones. Measured effect: ~2e-8 mean relative difference, roughly eight orders of
magnitude below the run-to-run Monte Carlo variation of the sampler itself. Runs
sharing a warm cache are byte-identical.

- `--batch-size INTEGER`: pin the number of voxels per batch, skipping the
  automatic selection entirely. Also settable via `DMRI_BATCH_SIZE`, or by
  setting a per-stage `eval_batch_size` in the config.
- `--memory-fraction FLOAT`: fraction of device memory JAX may preallocate
  (default `0.75`). Lower it when sharing a GPU with another process.

## Sampling cost

- `--num-steps INTEGER`: ODE steps per posterior sample (default 40).

Cost is linear in the number of network evaluations, which is
`num_steps + 1` (the solver initialises once, then evaluates once per interval)
plus one more for the final Euler correction — so the default 40 steps performs
**41** evaluations per sample. The run logs the real count. Setting
`evaluation.sampling.theta.params.last_euler_step=false` drops the correction and saves one
evaluation per sample.

Total sampler work scales as `num_steps x --theta-samples x voxels`, so those two
flags are the knobs that matter. Reducing steps trades accuracy for speed; verify
against a full-step run on your own data before relying on a lower setting.

## Precision

- `--precision {fp32,bf16,fp16}`: numeric precision for the network forward pass;
  default `fp32`.

Half precision halves the activation memory (which admits a larger batch) and is
faster on tensor-core GPUs. It applies to the network's matmul inputs only:
parameters stay float32, results accumulate in float32, and the diffusion
sampler, the correctors and the exported maps are always computed in float32.
Measured against a float32 run, the half-precision maps differ by less than the
run-to-run Monte Carlo noise of float32 itself.

## Scope

Use `dmri predict` for pretrained inference without ground-truth metrics. Use [`dmri eval`](cli_eval.md) for the full advanced Hydra evaluation pipeline.

Pretrained checkpoints are acquisition-specific: verify compatibility with the input acquisition scheme. DMRI is research-only software and has not been clinically validated.
