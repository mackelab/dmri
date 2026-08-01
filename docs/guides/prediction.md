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

The default model is `b3s_2_4_6_128` from the public Hugging Face repository `manugloeck/dmri-pretrained`. Downloads are cached by `huggingface_hub`.

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

Larger sample counts change the amount of sampling work. See the [prediction CLI reference](../reference/cli_predict.md) for the complete option summary.
