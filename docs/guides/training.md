# Training Runs

Training is orchestrated via Hydra. Every run captures its config alongside checkpoints so you can reproduce results exactly.

## Launch a run

```bash
dmri +experiment=ball3stick
```

Key output folders under `results/<run_name>/<timestamp>/`:
- `checkpoints/`: parameter snapshots (optionally EMA-smoothed)
- `.hydra/`: frozen config used for the run
- logs and metrics emitted by the trainer

## Tuning experiments

- Override inline: `dmri train.batch_size=128 train.optimizer.lr=1e-3`
- Switch simulator presets: `dmri +experiment=msb3s_2_4_6_128`
- Target specific hardware profiles via `conf_train/partition/*.yaml` (e.g. GPU vs CPU).

## Config map

Training configs live in `conf_train/`:

```text
conf_train/
├── config.yaml
├── experiment/
├── launcher/
├── model/
├── partition/
├── simulator/
└── train/
    ├── dataloader/
    ├── optimizer/
    ├── default*.yaml
```

- `config.yaml`: run metadata (`name`, `seed`, `use_wandb`, output dirs) plus defaults pointing to the simulator, model, train, launcher, and partition groups.
- `train/default*.yaml`: training loop knobs such as `inner_steps`, checkpoint/eval cadence, max wall-clock hours, EMA tracking (`track_ema`, `ema_decay`), loss weights (`model_selection_weight`, `model_inference_loss_weight`), label smoothing, and recovery thresholds.
- `train/dataloader/*.yaml`: simulation and loader settings (buffer sizes, batch sizes, shuffle/drop-last, prefetch, async workers) with CPU, GPU, and multi-GPU variants.
- `train/optimizer/*.yaml`: optimizer choice and hyperparameters (learning rate, scheduler params, adaptive clipping, gradient clip values, EMA toggles).
- `simulator/*.yaml`: signal simulation recipes and acquisition schemes to drive synthetic training data.
- `model/*.yaml`: neural architecture definitions for dmri/ssfp variants.
- `experiment/*.yaml`: ready-made presets combining simulator/model choices with train overrides.
- `partition/*.yaml`: cluster/queue presets for resource requests and time limits.
- `launcher/*.yaml`: Hydra launcher wiring for local or Slurm execution.

## Checkpoint handling

- Latest checkpoint: `results/<run>/checkpoints/latest`
- Best checkpoint (if tracked): `results/<run>/checkpoints/best`
- Use `dmri.train.utils.load_checkpoint(path, which="best")` to restore models inside notebooks.

## Debugging slow starts

- JAX JIT warmup can make the first epoch slower—let it complete before judging throughput.
- If GPU memory is tight, ensure no other JAX processes are running (`nvidia-smi`) and reduce batch size.

## Recording more metrics

Metrics are configured under `conf_train/train/` and `conf_train/experiment/`. Add your own callbacks or exporters, then document them with docstrings so they appear in the API reference.
