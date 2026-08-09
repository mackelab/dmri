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
5. [Getting example data](05_example_data.md) downloads an open dataset,
   converts it into the four-file FSL layout `dmri predict` reads, and runs a
   prediction.

To run them locally:

```bash
uv pip install -e '.[dev]'
uv run jupyter lab docs/examples
```

The pretrained example downloads a checkpoint. CPU execution can be slow; a
supported GPU is recommended for posterior sampling.

Notebooks 1-4 are re-executed on every documentation build, so their figures
always match the code. "Getting example data" is not: it downloads 174 MB and
runs a full prediction, so it ships with committed outputs and is meant to be
run by hand.
