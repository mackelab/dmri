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
- Target specific hardware profiles via `conf/partition/*.yaml` (e.g. GPU vs CPU).

## Checkpoint handling

- Latest checkpoint: `results/<run>/checkpoints/latest`
- Best checkpoint (if tracked): `results/<run>/checkpoints/best`
- Use `dmri.train.utils.load_checkpoint(path, which="best")` to restore models inside notebooks.

## Debugging slow starts

- JAX JIT warmup can make the first epoch slower—let it complete before judging throughput.
- If GPU memory is tight, ensure no other JAX processes are running (`nvidia-smi`) and reduce batch size.

## Recording more metrics

Metrics are configured under `conf/train/` and `conf/experiment/`. Add your own callbacks or exporters, then document them with docstrings so they appear in the API reference.
