# Evaluation & Export

Use the evaluation CLI to run posterior inference, model selection, and export derived maps for downstream analysis.

## Typical workflow

```bash
dmri_eval +experiment=eval_b3s_best_model_selection \
  model_name=<run_name>/<timestamp> \
  data.data_folder=<data_folder>
```

- `model_name` points at the timestamped training directory below `results/`.
- File-based presets require either `data.data_folder` or `data.path`.
- Exports (NIfTI volumes plus JSON summaries) are written next to the run under folders like `ball3stick_model_selection_results/`.

## Switching evaluation presets

Prebaked experiments live in `conf_eval/experiment/`:

- `eval_b3s_no_selection`: run inference with all model components enabled.
- `eval_b3s_average_model_selection`: use averaging over model posteriors.
- `eval_mask_test`: evaluate with mask priors.

Override any Hydra parameter inline:

```bash
dmri_eval +experiment=eval_ground_truth_b3s theta_sample/corrector=smc
```

## Metric exports

Metric sets are composed under `conf_eval/export/metrics/` and referenced by exporter configs such as `conf_eval/export/ball3stick.yaml`.
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

For exploratory plots that are not part of the supported documentation, see the `experimental_notebooks/` directory in the repository. Those notebooks may change without notice.
