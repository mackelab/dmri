# DMRI: Diffusion MRI Model Selection

[![CI](https://github.com/mackelab/dmri/actions/workflows/ci.yml/badge.svg)](https://github.com/mackelab/dmri/actions/workflows/ci.yml)
[![Docs](https://github.com/mackelab/dmri/actions/workflows/docs.yml/badge.svg)](https://github.com/mackelab/dmri/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Made with JAX](https://img.shields.io/badge/Made%20with-JAX-007acc.svg)](https://github.com/google/jax)

This package provides simulation-based inference and model selection for fibre reconstruction models in diffusion MRI. It combines common microstructural components into configurable multicompartment models and infers both model composition and continuous parameters.

**Docs:** https://www.mackelab.org/dmri/

## Quickstart

```bash
# create a fresh env (uv is fast; pip/conda work too)
uv venv -p 3.11
source .venv/bin/activate

# install editable package + dev tools; add --extra cuda for GPUs
uv pip install -e '.[dev]'

# run the default pretrained model on a standard dMRI folder
dmri predict /path/to/dmri_folder
```

Prefer pip? Use `pip install -e '.[dev]'`. CUDA users can opt into `.[cuda]`.

## CLI in one glance

### Prediction
```bash
dmri predict FOLDER
```

`FOLDER` must contain `data.nii.gz`, `nodif_brain_mask.nii.gz`, `bvals`, and `bvecs`. The default `b3s_2_4_6_128` model is downloaded from the public [`manugloeck/dmri-pretrained`](https://huggingface.co/manugloeck/dmri-pretrained) repository and cached by `huggingface_hub`. Results are written below `FOLDER/dmri_output/` in `ball3stick_inference_results/` and `ball3stick_model_selection_results/`.

Pretrained models are acquisition-specific. Confirm that the selected checkpoint is compatible with the input acquisition scheme; the standard filenames alone do not establish compatibility. DMRI is research software and has not been clinically validated.

### Training
```bash
dmri train --help                      # discover overrides
dmri train +experiment/train=b3s_2_4_6_128 infrastructure/launcher=local infrastructure/partition=none tracking.enabled=false
dmri train training.optimizer.learning_rate=1e-3
```

Runs write to `results/<name>/<timestamp>/` with checkpoints and the frozen `.hydra/` config. Use `dmri.train.utils.load_checkpoint(...)` to restore in notebooks.

### Evaluation
```bash
dmri eval --help
dmri eval +experiment/eval=eval_b3s_no_selection \
  checkpoint.model_name=<run_name>/<timestamp> \
  evaluation.input.data_folder=<data_folder>
```

Full evaluation is the advanced Hydra interface for trained runs, custom samplers, exports, and ground-truth metrics. For routine inference without ground-truth metrics, use `dmri predict`. The deprecated `dmri_eval` command remains as a compatibility alias.

## Configuration map

Training and evaluation share the versioned Hydra tree in `conf/`.

```
conf/
├── train.yaml            # training entry config
├── eval.yaml             # evaluation entry config
├── experiment/           # train/eval presets
├── infrastructure/       # local/slurm and resource profiles
├── model/                # dmri/ssfp architectures
├── simulator/            # signal simulation recipes
├── training/             # loop knobs
    ├── dataloader/       # buffer sizes, batching, prefetch
    ├── optimizer/        # optax configs, schedulers, EMA
    └── default*.yaml     # inner steps, cadence, weights
└── evaluation/           # inputs, sampling, selection, exports
```

Evaluation metrics are composed under `conf/evaluation/export/theta/metrics/` and pulled into exporter presets.

## Usage

### Simulators by example

Define acquisition schemes, build compartment models, and synthesize signals directly:

```python
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from dmri.simulators import Ball, Stick, Zeppelin, MultiCompartment
from dmri.simulators.acquisition_scheme import acquisition_scheme

# Example acquisition scheme dMRI
bvals = jnp.linspace(0, 4000, 100)
bvecs = jax.random.normal(jax.random.key(0), (100, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
acq = acquisition_scheme(bvals, bvecs)

# Example simulator for a single ball
theta = np.random.randn(Ball.theta_dim) # Theta will always be normal
ball = Ball.from_theta(theta) # Deterministically maps theta to physical parameters
signal = ball.signal(acq) # Simulate signal

plt.plot(bvals, signal) # Plot signal

# But you can also combine multiple models
class BallStickZeppelin(MultiCompartment):
    model_types=[Ball, Stick, Zeppelin]
    noise_types = []

theta = np.random.randn(BallStickZeppelin.theta_dim)
# With all models
ball_stick_zeppelin = BallStickZeppelin.from_theta(theta)
# With only ball and stick
ball_stick = BallStickZeppelin.from_theta(theta, model_mask=jnp.array([True, True, False]))

# Simulate signal
signal = ball_stick_zeppelin.signal(acq)
signal_ball_stick = ball_stick.signal(acq)

plt.plot(bvals, signal)
plt.plot(bvals, signal_ball_stick)
```

You can also have a look at the notebooks in `docs/examples/` (rendered under **Examples** on the docs site) for more examples.

## JAX and PyTorch notes

- JAX grabs most GPU memory up front; if multiple jobs share a card, expect OOMs until you free memory (`nvidia-smi` is your friend).
- JIT warmup makes the first steps slower; steady-state throughput improves after compilation.
- Need PyTorch for notebooks? Install CPU-only wheels to avoid CUDA conflicts with JAX:
  `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`

## Development

- Tests: `pytest`
- Lint/format: `ruff check --fix` and `ruff format`
- Docs preview: `uv pip install -e '.[docs]' && uv run python scripts/render_docs_notebooks.py && uv run zensical serve`

## CI / docs deploy

- GitHub Actions runs the test suite on Python 3.11.
- Zensical deploys through the GitHub Pages workflow (see `zensical.toml`).


## License

This project is licensed under the MIT License; see `LICENCE` for details.
