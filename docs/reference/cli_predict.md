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

## Output and sampling

- `--output-subdir NAME`: output directory name inside `FOLDER`; default `dmri_output`.
- `--overwrite`: replace an existing output directory.
- `--seed INTEGER`: sampling seed; default `1`.
- `--mask-samples INTEGER`: number of model-mask samples; default `50`.
- `--theta-samples INTEGER`: number of parameter samples; default `50`.

`--mask-samples` must be greater than or equal to `--theta-samples` because
each parameter sample is conditioned on a sampled model mask.

## Scope

Use `dmri predict` for pretrained inference without ground-truth metrics. Use [`dmri eval`](cli_eval.md) for the full advanced Hydra evaluation pipeline.

Pretrained checkpoints are acquisition-specific: verify compatibility with the input acquisition scheme. DMRI is research-only software and has not been clinically validated.
