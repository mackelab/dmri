# Examples

The notebooks are small demonstrations of specific APIs. They are not complete
training or evaluation pipelines.

1. [Simulator components](01_dmri_simulator_components.md) plots signals from
   individual compartments.
2. [Multi-compartment model](02_dmri_multicompartment_model.md) combines
   compartments, masks, fractions, and noise.
3. [Using pretrained models](03_pretrained_models.md) loads a checkpoint and
   calls model sampling methods on synthetic data.
4. [Train a small model](04_train_standalone.md) demonstrates model building
   and a short in-process training loop. It does not replace `dmri train` for
   checkpointed runs.

To run them locally:

```bash
uv pip install -e '.[dev]'
uv run jupyter lab docs/examples
```

The pretrained example downloads a checkpoint. CPU execution can be slow; a
supported GPU is recommended for posterior sampling.
