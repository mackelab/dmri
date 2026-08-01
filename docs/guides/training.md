# Training Runs

Training is orchestrated via Hydra. Every run captures its config alongside checkpoints so you can reproduce results exactly.

## Launch a run

```bash
# local machine with one GPU
dmri train +experiment/train=b3s_2_4_6_128 infrastructure/launcher=local infrastructure/partition=none tracking.enabled=false

# SLURM cluster; choose profiles for your site
dmri train +experiment/train=b3s_2_4_6_128 infrastructure/launcher=slurm infrastructure/partition=<profile>
```

Key output folders under `results/<run_name>/<timestamp>/`:
- `checkpoints/`: parameter snapshots (optionally EMA-smoothed)
- `.hydra/`: frozen config used for the run
- logs and metrics emitted by the trainer

## Tuning experiments

- Override inline: `dmri train training.dataloader.train_loader.batch_size=128 training.optimizer.learning_rate=1e-3`
- Switch simulator presets: `dmri train +experiment/train=msb3s_2_4_6_128`
- Target specific hardware profiles via `conf/infrastructure/partition/*.yaml`.

## Config map

Training and evaluation share the versioned `conf/` tree:

```text
conf/
├── train.yaml
├── eval.yaml
├── experiment/train/
├── infrastructure/
├── model/
├── simulator/
└── training/
    ├── dataloader/
    ├── optimizer/
    ├── default*.yaml
```

- `train.yaml`: schema version, run metadata, tracking, output directories, and training defaults.
- `training/default*.yaml`: training loop knobs such as `inner_steps`, checkpoint/eval cadence, max wall-clock hours, EMA tracking, loss weights, label smoothing, and recovery thresholds.
- `training/dataloader/*.yaml`: simulation and loader settings with CPU, GPU, and multi-GPU variants.
- `training/optimizer/*.yaml`: optimizer choice and hyperparameters.
- `simulator/*.yaml`: signal simulation recipes and acquisition schemes to drive synthetic training data.
- `model/*.yaml`: neural architecture definitions for dmri/ssfp variants.
- `experiment/train/*.yaml`: ready-made presets combining simulator/model choices with training overrides.
- `infrastructure/partition/*.yaml`: cluster/queue presets.
- `infrastructure/launcher/*.yaml`: Hydra launcher wiring.

## Checkpoint handling

- Latest checkpoint: the highest numeric step directory under `results/<run>/<timestamp>/checkpoints/`
- Best checkpoint (if tracked): `results/<run>/<timestamp>/checkpoints/best/`
- Use `dmri.train.utils.load_checkpoint(path, which="best")` to restore models inside notebooks.

### Publish pretrained checkpoints

Closely related variants can share one Hugging Face model repository. Each upload
is stored in its own subfolder with a portable `config.yaml`, a minimal
`artifact.yaml` model manifest, and the selected Orbax checkpoint:

```python
from dmri.train.utils import upload_checkpoint_to_hub

upload_checkpoint_to_hub(
    "results/b3s_2_4_6_64/<timestamp>",
    "manugloeck/dmri-pretrained",
    model_name="b3s_2_4_6_64",
    which="best",
    private=False,
)
```

Authenticate once before uploading with `uv run hf auth login`. Repeat the call
with another `model_name` to add variants to the same repository. `which` accepts
`"best"` (recommended), `"latest"`, a training-step integer, or `"all"`.

Load only the requested model subfolder from the Hub cache:

```python
from flax import nnx
from dmri.train.utils import load_checkpoint

checkpoint, model, simulators = load_checkpoint(
    repo_id="manugloeck/dmri-pretrained",
    model_name="b3s_2_4_6_64",
    which="best",
)
params = checkpoint.get("params_ema", checkpoint["params"])
nnx.update(model, params)
model.eval()
```

Use `revision="<commit-or-tag>"` for reproducible downloads, `token=...` for a
private repository, or `local_files_only=True` after the snapshot is cached.

## Debugging slow starts

- JAX JIT warmup can make the first epoch slower—let it complete before judging throughput.
- If GPU memory is tight, ensure no other JAX processes are running (`nvidia-smi`) and reduce batch size.

## Recording more metrics

Metrics are configured under `conf/training/` and `conf/experiment/train/`. Add your own callbacks or exporters, then document them with docstrings so they appear in the API reference.
