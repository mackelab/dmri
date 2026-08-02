# Evaluation & Export

Use the full evaluation CLI to run advanced posterior inference, model selection, configurable metrics, and exports. For routine pretrained inference without ground-truth metrics, use [`dmri predict`](prediction.md).

## Typical workflow

```bash
dmri eval +experiment/eval=eval_b3s_best_model_selection \
  checkpoint.model_name=<run_name>/<timestamp> \
  evaluation.input.data_folder=<data_folder>
```

- `checkpoint.model_name` points at the timestamped training directory below `results/`.
- File-based presets require either `evaluation.input.data_folder` or `evaluation.input.path`.
- Exports (NIfTI volumes plus JSON summaries) are written next to the run under folders like `ball3stick_model_selection_results/`.

## Switching evaluation presets

Prebaked experiments live in `conf/experiment/eval/`:

- `eval_b3s_no_selection`: run inference with all model components enabled.
- `eval_b3s_average_model_selection`: use averaging over model posteriors.
- `eval_mask_test`: evaluate with mask priors.

Override any Hydra parameter inline:

```bash
dmri eval +experiment/eval=eval_ground_truth_b3s evaluation/theta/corrector@evaluation.sampling.theta.corrector=smc
```

The deprecated `dmri_eval` command remains available as a compatibility alias for `dmri eval`.

## Metric exports

Metric sets are composed under `conf/evaluation/export/theta/metrics/` and referenced by exporter configs such as `conf/evaluation/export/theta/ball3stick.yaml`.
Each metric writes a NIfTI volume plus a JSON summary of configured aggregations (mean, median, percentiles).

## Loading results programmatically

```python
from pathlib import Path
import json

export_dir = Path("results/my_run/2026-07-19_12-00-00/ball3stick_model_selection_results")
with open(export_dir / "reconstruction_mse_summary.json") as f:
    summary = json.load(f)
```

Summary files are emitted per configured metric. Their names follow `<metric>_summary.json` unless a metric overrides the filename.
