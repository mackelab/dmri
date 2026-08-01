# Getting Started

This page describes environment setup, pretrained prediction, training, and evaluation.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Git and a cloned copy of this repository

## Install

```bash
git clone https://github.com/mackelab/dmri.git
cd dmri

uv venv -p 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
```

- CUDA-enabled JAX: `uv pip install -e '.[cuda]'`.
- pip without uv: `pip install -e '.[dev]'` (quote extras in zsh).
- conda: create an env, then install the same extras.

## Run a pretrained prediction

Prepare a folder containing these four files:

```text
FOLDER/
├── data.nii.gz
├── nodif_brain_mask.nii.gz
├── bvals
└── bvecs
```

Then run:

```bash
dmri predict FOLDER
```

The default model is `b3s_2_4_6_128` from the public `manugloeck/dmri-pretrained` Hugging Face repository. `huggingface_hub` downloads and caches it automatically. Outputs are placed below `FOLDER/dmri_output/` in `ball3stick_inference_results/` and `ball3stick_model_selection_results/`. Prediction does not compute metrics that require ground truth.

See the [prediction guide](guides/prediction.md) for model, cache, offline, local-checkpoint, output, and sampling options.

!!! warning
    Pretrained models expect acquisition schemes compatible with their training configuration. Check compatibility before inference. DMRI is intended for research and has not been clinically validated.

## Run a training job

1. Pick an experiment preset in `conf_train/experiment/` or compose your own by editing `conf_train/*`.
2. Launch on a local machine with a GPU, without SLURM or W&B:

   ```bash
   dmri train +experiment=b3s_2_4_6_128 launcher=local partition=none use_wandb=false
   ```

   The local launcher requests one GPU. For a cluster, select launcher and partition profiles that match your site instead of relying on the repository defaults. Remove `use_wandb=false` after logging into Weights & Biases.
3. Monitor `results/<run_name>/<timestamp>/` for checkpoints, Hydra configs (`.hydra/`), and logs. If you are logged into Weights & Biases and left `use_wandb=true`, online logging starts automatically.

## Evaluate a trained model

```bash
dmri eval +experiment=eval_b3s_best_model_selection \
  model_name=<run_name>/<timestamp> \
  data.data_folder=<data_folder>
```

- `model_name` is the timestamped training path relative to `results/`.
- Evaluation requires an input path. Set `data.data_folder` or the preset's `data.path`.
- Full evaluation is intended for advanced workflows, including configurable exports and ground-truth metrics. Use `dmri predict` for inference without ground-truth metrics.
- The deprecated `dmri_eval` executable remains as a compatibility alias.

## Know your configs

Hydra powers `dmri train` and `dmri eval`. Use `--help` to see overrides for a specific experiment, and refer to the CLI reference pages for curated config maps:

- [Prediction CLI reference](reference/cli_predict.md)
- [Training CLI reference](reference/cli_train.md)
- [Evaluation CLI reference](reference/cli_eval.md)

Common tweaks live in `train.optimizer`, `model.*`, and simulator definitions under `conf_train/simulator/`.

The [example notebooks](examples/) provide executable demonstrations of the simulation, training, and evaluation interfaces.
