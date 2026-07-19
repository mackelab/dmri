# Getting Started

Bootstrap an environment, launch a first training run, and learn where to look when you want to change configurations.

## Install

```bash
# recommended: in a fresh virtual environment
uv venv -p 3.11
source .venv/bin/activate

# install the project and dev tooling
uv pip install -e '.[dev]'
```

- CUDA-enabled JAX: `uv pip install -e '.[cuda]'`.
- pip without uv: `pip install -e '.[dev]'` (quote extras in zsh).
- conda: create an env, then install the same extras.

## Run a training job

1. Pick an experiment preset in `conf_train/experiment/` or compose your own by editing `conf_train/*`.
2. Launch locally (no SLURM cluster or W&B login required):

   ```bash
   dmri +experiment=b3s_2_4_6_128 launcher=local partition=none use_wandb=false
   ```

   Drop `launcher=local partition=none` on a SLURM cluster, and drop `use_wandb=false` once you are logged into Weights & Biases.
3. Monitor `results/<run_name>/<timestamp>/` for checkpoints, Hydra configs (`.hydra/`), and logs. If you are logged into Weights & Biases and left `use_wandb=true`, online logging starts automatically.

## Evaluate a trained model

```bash
dmri_eval +experiment=eval_b3s_best_model_selection \
  model_name=<run_folder> \
  data.data_folder=<data_folder>
```

- `model_name` points at a training run under `results/`.
- Data defaults to `data/`; override with `data.data_folder`.
- Exports land next to the run (e.g. `results/<run_folder>/ball3stick_model_selection_results/`).

## Know your configs

Hydra powers both CLIs. Use `--help` to see overrides for a specific experiment, and refer to the CLI reference pages for curated config maps:

- Training CLI reference: `reference/cli_train.md`
- Evaluation CLI reference: `reference/cli_eval.md`

Common tweaks live in `train.optimizer`, `model.*`, and simulator definitions under `conf_train/simulator/`.

## Documentation feedback loop

Docstrings render automatically in the API Reference. Improve docstrings in `dmri/*` and rebuild the site with:

```bash
mkdocs serve
```
