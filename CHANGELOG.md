# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added the `dmri predict`, `dmri train`, and `dmri eval` command structure.
- Added a predict-first workflow that downloads cached pretrained models from
  Hugging Face and writes results beside the input data.
- Added `load_pretrained` and `list_pretrained_models` as the public Hub API.
- Fixed checkpoint saving, model-selection export, theta reuse, and mask-aware
  parameter export.
- Improved package metadata, wheel resources, documentation, contribution, and
  citation guidance.
- Improved `dmri predict` output with readable summaries, autotuning spinners,
  voxel progress, non-interactive text output, and quieter XLA diagnostics.
- Changed `--verbose` to add diagnostic detail without removing progress output.
- Changed undefined noise-only and malformed in-brain signal fractions to export
  as `NaN` instead of aborting after sampling; outside-brain voxels remain zero.
- Added prediction model modes for posterior per-sample masks, best model per
  voxel, and fixed B1S/B2S/B3S models.

## 0.1.0

- Added configurable diffusion MRI simulators and multicompartment models.
- Added training and evaluation workflows for simulation-based model selection.
- Added pretrained-model support, examples, and API documentation.
