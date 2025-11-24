# Getting Started

Use this guide to bootstrap a working environment and run your first diffusion MRI model-selection job.

## Install

```bash
# recommended: in a fresh virtual environment
uv venv -p 3.11
source .venv/bin/activate

# install the project and dev tooling
uv pip install -e '.[dev]'
```

GPU users can opt into CUDA-enabled JAX:

```bash
uv pip install -e '.[cuda]'
```

If you prefer pip directly, quote extras in zsh: `pip install -e '.[dev]'`.

## First training run

1. Pick or edit a configuration under `conf/` (e.g. `conf/experiment/` presets).
2. Launch training:

   ```bash
   dmri +experiment=ball3stick
   ```

3. Inspect outputs in `results/<name>/<timestamp>/` (checkpoints, configs, logs).

## Evaluate a trained model

```bash
dmri_eval +experiment=eval_b3s_best_model_selection model_name=<run_folder>
```

Results land next to the training run (e.g. `ball3stick_model_selection_results/`).

## Editing configurations

- Hydra drives configuration. Use `dmri --help` or `dmri +experiment=<...> --help` to discover overrides.
- Common knobs: `train.optimizer`, `model.*` (embedding, selection, inference networks), and simulator definitions under `conf/simulator/`.

## Documentation feedback loop

Docstrings from the codebase render automatically in the API Reference. Add or improve docstrings in `dmri/*` modules and re-build the site to keep this page authoritative:

```bash
mkdocs serve
```
