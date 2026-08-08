# Installation and Quickstart

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) or pip
- Git and a clone of this repository
- A supported GPU is strongly recommended for prediction

## Install the runtime

Create an environment and install DMRI from the repository:

```bash
git clone https://github.com/mackelab/dmri.git
cd dmri
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e .
```

With pip, replace the last command with `pip install -e .`.

For a CUDA 12 installation, use `uv pip install -e '.[cuda]'`. For CUDA 13,
use `uv pip install -e '.[cuda13]'`. Check that your driver and JAX CUDA
requirements are compatible before selecting an extra.

Development tools are not part of the default installation. Install them only
when working on DMRI with `uv pip install -e '.[dev]'`.

## Prepare input data

Prediction currently expects one folder containing exactly these named inputs:

```text
FOLDER/
├── data.nii.gz
├── nodif_brain_mask.nii.gz
├── bvals
└── bvecs
```

- `data.nii.gz` contains the diffusion-weighted volumes.
- `nodif_brain_mask.nii.gz` identifies the voxels to process.
- `bvals` and `bvecs` describe the acquisition gradients.

> **Check the acquisition.** A pretrained checkpoint is valid only for
> compatible acquisition schemes. The four required filenames do not establish
> compatibility. DMRI is research software and has not been clinically
> validated.

## Run prediction

Specify the model and quality preset so the command does not prompt:

```bash
dmri predict FOLDER \
  --model msb3s_2_4_6_128 \
  --quality balanced \
  --non-interactive
```

The model is downloaded from `manugloeck/dmri-pretrained` on Hugging Face and
cached locally. Use the [prediction guide](guides/prediction.md) to choose a
different model, use a local checkpoint, or change sampling settings.

> **CPU runtime.** Prediction on CPU is very slow, even for a few thousand
> voxels. Use a supported GPU accelerator when possible.

## View the output

By default, prediction writes to `FOLDER/dmri_output/`:

```text
FOLDER/dmri_output/
├── view_results.html
├── ball3stick_inference_results/
│   ├── mean_f0samples.nii.gz
│   ├── mean_f1samples.nii.gz
│   ├── mean_f2samples.nii.gz
│   ├── mean_f3samples.nii.gz
│   └── ...
└── ball3stick_model_selection_results/
    └── ...
```

Open `view_results.html` in a web browser to inspect the exported maps. The
viewer loads Plotly from a CDN. Pass `--no-viewer` to skip it,
`--output-subdir NAME` to choose another folder name inside `FOLDER`, or
`--overwrite` to replace an existing output folder.

See the [prediction CLI reference](cli/predict.md) for all options and the
[outputs reference](reference/outputs.md) for exported files.

## Continue

- [Train a model](guides/training.md) with `dmri train`.
- [Evaluate a trained model](guides/evaluation.md) with `dmri eval`.
- [Run the examples](examples/index.md) for Python workflows.
