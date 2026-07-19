# CLI Reference: Evaluation (`dmri_eval`, `@conf_eval`)

The `dmri_eval` CLI mirrors the training interface and pulls defaults from the `@conf_eval` package. Use these configs to run posterior inference, model selection, and exports for trained runs.

## Essential commands

- List overrides for a preset: `dmri_eval +experiment=eval_b3s_best_model_selection --help`
- Dry-run the resolved config: `dmri_eval +experiment=eval_b3s_best_model_selection --cfg job`
- Override inline: `dmri_eval +experiment=eval_ground_truth_b3s theta_sample.corrector=smc`

## Configuration map (`conf_eval/`)

- `experiment/`: evaluation workflows (with/without model selection, mask tests, ground-truth recovery).
- `export/metrics/`: metric bundles (FA, MD, SNR, selection probabilities) referenced by exporter configs.
- `export/*.yaml`: export recipes that pair metric bundles with output formatting.

## Common overrides

- Target run: `model_name=<results_subdir>` (points to a training run under `results/`).
- Data location: `data.data_folder=<path_to_data>`.
- Correctors/samplers: `theta_sample.corrector`, `theta_sample.num_samples`.
- Export set: `export=ball3stick` (swap to any `conf_eval/export/*.yaml` recipe).

## Outputs

Exports live next to the training run, e.g. `results/<model_name>/ball3stick_model_selection_results/`, containing NIfTI volumes plus JSON summaries.
