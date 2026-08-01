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

## 0.1.0

- Added configurable diffusion MRI simulators and multicompartment models.
- Added training and evaluation workflows for simulation-based model selection.
- Added pretrained-model support, examples, and API documentation.
