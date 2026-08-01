# Evaluation CLI

The `dmri eval` command is the advanced Hydra interface. It pulls defaults from the packaged `conf_eval` configuration module and supports posterior inference, model selection, configurable metrics, and exports for trained runs. Use [`dmri predict`](cli_predict.md) for the simpler pretrained workflow, which omits ground-truth metrics.

## Essential commands

- List overrides for a preset: `dmri eval +experiment=eval_b3s_best_model_selection --help`
- Dry-run the resolved config: `dmri eval +experiment=eval_b3s_best_model_selection --cfg job`
- Override inline: `dmri eval +experiment=eval_ground_truth_b3s theta_sample/corrector=smc`

The deprecated `dmri_eval` executable remains as a compatibility alias.

## Configuration map (`conf_eval/`)

- `experiment/`: evaluation workflows (with/without model selection, mask tests, ground-truth recovery).
- `export/metrics/`: reconstruction MSE, NLL, KSD, SWD, SBC, calibration, and model-selection metrics.
- `export/*.yaml`: export recipes that pair metric bundles with output formatting.

## Common overrides

- Target run: `model_name=<run_name>/<timestamp>` (relative to `results/`).
- Data location: `data.data_folder=<path_to_data>`.
- Corrector group / sample count: `theta_sample/corrector`, `theta_sample.num_samples`.
- Export set: `export=ball3stick` (swap to any `conf_eval/export/*.yaml` recipe).

## Outputs

Exports live next to the training run, e.g. `results/<model_name>/ball3stick_model_selection_results/`, containing NIfTI volumes plus JSON summaries.
