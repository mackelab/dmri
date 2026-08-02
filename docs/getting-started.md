# Getting Started

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

## Apply a pretrained model

`dmri predict` is the easiest way to apply a pretrained model to diffusion MRI data.

### Step 1: Prepare the folder

The input folder must contain these four files:

```text
FOLDER/
├── data.nii.gz
├── nodif_brain_mask.nii.gz
├── bvals
└── bvecs
```

### Step 2: Run with defaults

```bash
dmri predict FOLDER
```

With no other options, the command asks which model and quality preset to use, defaulting to the recommended choice at every prompt:

```text
Model  (from manugloeck/dmri-pretrained)
  1) b3s_2_4_6_128     Ball3Stick model family - single-shell dMRI
  2) b3s_2_4_6_64      Ball3Stick model family - compact single-shell model
  3) msb3s_2_4_6_128   Multi-shell Ball3Stick model family - multi-shell dMRI  <-
Choose [1]:

Quality
  1) fast             20 steps x 25 samples, fp16, no corrector
  2) balanced         40 steps x 50 samples, fp32, corrected  (default) <-
  3) high             60 steps x 100 samples, fp32, corrected
Choose [2]:
```

Pressing enter accepts the default at every step. Prompts only appear on a terminal, so scripts and CI are not blocked. Pass `--non-interactive` to suppress them explicitly.

The default model is `msb3s_2_4_6_128`, a multi-shell Ball3Stick model from the public `manugloeck/dmri-pretrained` Hugging Face repository. `huggingface_hub` downloads and caches it automatically.

### Step 3: Output layout

Results are placed below `FOLDER/dmri_output/` by default:

```text
FOLDER/dmri_output/
├── ball3stick_inference_results/
│   ├── mean_f0samples.nii.gz          # ball fraction
│   ├── mean_f1samples.nii.gz          # stick 1 fraction
│   ├── mean_f2samples.nii.gz          # stick 2 fraction
│   ├── mean_f3samples.nii.gz          # stick 3 fraction
│   ├── mean_fsumsamples.nii.gz        # total anisotropic fraction
│   ├── mean_dsamples.nii.gz           # diffusivity
│   └── ...
└── ball3stick_model_selection_results/
    └── ...
```

Use `--output-subdir NAME` to choose a different directory name inside `FOLDER`. Use `--overwrite` to replace an existing output folder.

### Step 4: Common options

**Quality presets** control speed and accuracy:

```bash
dmri predict FOLDER --quality fast      # 20 steps x 25 samples, fp16, no corrector
dmri predict FOLDER --quality balanced  # 40 steps x 50 samples, fp32, corrected (default)
dmri predict FOLDER --quality high      # 60 steps x 100 samples, fp32, corrected
```

**Model modes** control how the model composition varies across voxels:

```bash
dmri predict FOLDER --model-mode per-sample   # posterior model per voxel/sample (default)
dmri predict FOLDER --model-mode best          # one best model per voxel
dmri predict FOLDER --fixed-model B2S          # fixed B2S everywhere
```

**Hardware and reproducibility:**

```bash
dmri predict FOLDER --batch-size 8192 --memory-fraction 0.5   # shared GPU
dmri predict FOLDER --seed 42 --overwrite                   # reproducible re-run
```

### Step 5: Non-interactive and scripted usage

Pass every option on the command line so the command never prompts:

```bash
dmri predict FOLDER \
  --model msb3s_2_4_6_128 \
  --quality balanced \
  --model-mode best \
  --non-interactive \
  --overwrite
```

### Step 6: Offline and local checkpoints

Disable downloads and use only cached files:

```bash
dmri predict FOLDER --local-files-only
```

Use a local model bundle instead of the Hugging Face Hub:

```bash
dmri predict FOLDER --local-checkpoint /path/to/model_bundle
```

See the [prediction guide](guides/prediction.md) and the [CLI reference](cli/predict.md) for the full option list.

!!! warning
    Pretrained models expect acquisition schemes compatible with their training configuration. Check compatibility before inference. DMRI is intended for research and has not been clinically validated.

## Other entry points

| Command / API | Purpose | Documentation |
|---|---|---|
| `dmri predict` | Apply pretrained models to dMRI data | This page, [prediction guide](guides/prediction.md), [CLI reference](cli/predict.md) |
| `dmri train` | Train custom models via Hydra presets | [Training guide](guides/training.md), [CLI reference](cli/train.md) |
| `dmri eval` | Evaluate trained models with full metrics | [Evaluation guide](guides/evaluation.md), [CLI reference](cli/eval.md) |
| `dmri.hub.load_pretrained` | Load models in Python notebooks | [API reference](reference/hub.md) |

## Know your configs

Hydra powers `dmri train` and `dmri eval`. Use `--help` to see overrides for a specific experiment. Common tweak locations:

- `training.optimizer` — learning rate, scheduler
- `model.*` — architecture groups
- `conf/simulator/` — simulator definitions

The [example notebooks](examples/) provide executable demonstrations.
