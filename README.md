# DMRI

[![CI](https://github.com/mackelab/dmri/actions/workflows/ci.yml/badge.svg)](https://github.com/mackelab/dmri/actions/workflows/ci.yml)
[![Docs](https://github.com/mackelab/dmri/actions/workflows/docs.yml/badge.svg)](https://github.com/mackelab/dmri/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)

DMRI provides diffusion MRI simulators and tools for model selection and
parameter inference. The project is pre-release software, and its API and CLI
may change.

> **Research use only.** DMRI has not been clinically validated and must not be
> used for clinical decisions. Pretrained checkpoints are acquisition-specific;
> verify compatibility with the input acquisition before interpreting results.

## Install

DMRI requires Python 3.11 or newer. From a clone of the repository:

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e .
```

Supported Linux installations include JAX's bundled CUDA 13 runtime by default.
See the [installation and quickstart](https://www.mackelab.org/dmri/getting-started/)
for driver requirements and the required input layout.

## Predict

<video src="https://github.com/mackelab/dmri/raw/main/docs/assets/predict-demo.mp4"
       poster="https://github.com/mackelab/dmri/raw/main/docs/assets/predict-demo-poster.jpg"
       controls muted loop playsinline width="100%"></video>

[Watch the interactive prediction run](https://github.com/mackelab/dmri/raw/main/docs/assets/predict-demo.mp4)
if the video above does not play.

```bash
dmri predict FOLDER --model msb3s_2_4_6_128 --quality balanced --non-interactive
```

`FOLDER` must contain `data.nii.gz`, `nodif_brain_mask.nii.gz`, `bvals`, and
`bvecs`. Prediction writes results below `FOLDER/dmri_output/`.

- [Prediction documentation](https://www.mackelab.org/dmri/guides/prediction/)
- [Training documentation](https://www.mackelab.org/dmri/guides/training/)
- [Evaluation documentation](https://www.mackelab.org/dmri/guides/evaluation/)

## Development checks

Install development dependencies with `uv pip install -e '.[dev]'`, then run:

```bash
pytest
ruff check .
ruff format --check .
```

Documentation: <https://www.mackelab.org/dmri/>

License: [MIT](LICENCE)
