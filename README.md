# DMRI: Diffusion MRI Model Selection

[![CI](https://github.com/mackelab/dmri/actions/workflows/ci.yml/badge.svg)](https://github.com/mackelab/dmri/actions/workflows/ci.yml)
[![Docs](https://github.com/mackelab/dmri/actions/workflows/docs.yml/badge.svg)](https://github.com/mackelab/dmri/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with JAX](https://img.shields.io/badge/Made%20with-JAX-007acc.svg)](https://github.com/google/jax)

Fast, reproducible diffusion MRI model selection powered by JAX, Hydra, and a library of simulators and neural architectures.

**Docs:** https://www.mackelab.org/dmri/

## Why use this repo?
- Hydra-driven experiments and sweeps with sensible defaults
- JAX-first training and evaluation pipelines with checkpointing and EMA support
- Rich simulator library for synthetic data and calibration
- API docs and notebooks for quick prototyping

## Quickstart

```bash
# create a fresh env (uv is fast; pip/conda work too)
uv venv -p 3.11
source .venv/bin/activate

# install editable package + dev tools; add --extra cuda for GPUs
uv pip install -e '.[dev]'

# run a first training job
dmri +experiment=ball3stick
```

Prefer pip? Use `pip install -e .[dev]` (quote extras in zsh). CUDA users can opt into `.[cuda]`.

## CLI in one glance

### Training
```bash
dmri --help                      # discover overrides
dmri +experiment=ball3stick      # run a preset
dmri train.optimizer.lr=1e-3     # inline override example
```

Runs write to `results/<name>/<timestamp>/` with checkpoints and the frozen `.hydra/` config. Use `dmri.train.utils.load_checkpoint(...)` to restore in notebooks.

### Evaluation
```bash
dmri_eval --help
dmri_eval +experiment=eval_no_selection model_name=<run_folder>
```

Outputs land next to the training run (e.g. `ball3stick_model_selection_results/`). Adapt data locations in `conf_eval/config.yaml`.

## Configuration map

Hydra configs live in `conf_train/` for training and `conf_eval/` for evaluation.

```
conf_train/
├── config.yaml           # run metadata + defaults
├── experiment/           # ready-made presets
├── launcher/             # local/slurm launchers
├── model/                # dmri/ssfp architectures
├── partition/            # resource profiles
├── simulator/            # signal simulation recipes
└── train/                # loop knobs
    ├── dataloader/       # buffer sizes, batching, prefetch
    ├── optimizer/        # optax configs, schedulers, EMA
    └── default*.yaml     # inner steps, cadence, weights
```

Evaluation metrics are composed under `conf_eval/export/metrics/` and pulled into exporter presets (e.g. `conf_eval/export/ball3stick.yaml`). Add SBC via `conf_eval/export/metrics/sbc_model_mask.yaml`.

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
ball = Ball.from_theta(theta) # Maps theta to random feasible parameters (diffusivity)
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

You can also have a look at the notebooks in the `notebooks/dmri_simulators.ipynb` for more examples.

## JAX and PyTorch notes

- JAX grabs most GPU memory up front; if multiple jobs share a card, expect OOMs until you free memory (`nvidia-smi` is your friend).
- JIT warmup makes the first steps slower; steady-state throughput improves after compilation.
- Need PyTorch for notebooks? Install CPU-only wheels to avoid CUDA conflicts with JAX:
  `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`

## Development

- Tests: `pytest`
- Lint/format: `ruff check --fix` and `ruff format`
- Docs preview: `uv pip install -e '.[docs]' && uv run mkdocs serve`

## CI / docs deploy

- GitHub Actions runs lint + tests across Python 3.10–3.12.
- MkDocs deploys cleanly with `mkdocs gh-deploy --force` (see `mkdocs.yml`).


## License

This project is licensed under the MIT License - see the LICENSE file for details.
