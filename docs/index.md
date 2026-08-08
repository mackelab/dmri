# DMRI

![DMRI logo: a stylized brain crossed by diffusion pathways](assets/logo.svg){ width="144" }

DMRI provides diffusion MRI simulators and tools for model selection and
parameter inference. Use it to apply a pretrained model, train a model on
simulated data, or evaluate a trained run.

> **Research use only.** DMRI has not been clinically validated and must not be
> used for clinical decisions. Pretrained checkpoints are acquisition-specific.
> Confirm that a checkpoint's training acquisition is compatible with your data
> before interpreting its output; matching filenames is not enough.

## Install

DMRI requires Python 3.11 or newer. From a clone of the repository:

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e .
```

See [Installation and quickstart](getting-started.md) for GPU options and a
complete prediction example.

## Predict

Put `data.nii.gz`, `nodif_brain_mask.nii.gz`, `bvals`, and `bvecs` in one
folder, then run:

```bash
dmri predict FOLDER --model msb3s_2_4_6_128 --quality balanced --non-interactive
```

Results and an HTML viewer are written below `FOLDER/dmri_output/`. See the
[prediction guide](guides/prediction.md) for checkpoint selection, output maps,
and performance options.

## Other workflows

- [Train a model](guides/training.md)
- [Evaluate a trained model](guides/evaluation.md)
- [Build and use simulators](guides/simulators.md)
- [Run the examples](examples/index.md)
- [Look up command options](cli/predict.md)
- [Use the Python API](reference/index.md)
- [Check configuration, outputs, and troubleshooting](reference/configuration.md)
