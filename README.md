# DMRI: Diffusion MRI Model Selection

A Python package for selecting and evaluating models for diffusion MRI data analysis, built with JAX for high-performance computing.

## Overview

This project provides tools for:
- Processing diffusion MRI data
- Implementing and evaluating diffusion models
- Hyperparameter optimization for model selection
- Visualization of diffusion properties and model performance

## Installation

### Prerequisites
- Python 3.9+
- Poetry for dependency management

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/dmri.git
cd dmri

# Install dependencies with Poetry
poetry install

# For development installation
poetry install --with dev
```

## Usage

### Basic Example
```python
import dmri

# Example code will depend on specific implementation
```

### Command-line Interface
The package provides a command-line interface via Hydra:

```bash
# Run with default configuration
poetry run dmri

# Override configuration parameters
poetry run dmri hydra.job.name=experiment_name model=tensor dataset=hcp
```

## Project Structure

```
dmri/
├── dmri/                  # Main package
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Diffusion models implementation
│   ├── train/             # Training and evaluation code
│   │   └── hydra_script.py  # Hydra entry point
│   └── viz/               # Visualization utilities
├── conf/                  # Hydra configuration files
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks for examples
└── scripts/               # Utility scripts
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are located in the `conf/` directory.

Key configuration components:
- Model parameters
- Dataset specifications
- Training parameters
- Evaluation metrics

## Features

- **JAX-based Implementations**: Leveraging JAX for auto-differentiation and fast numerical computing
- **Hydra Integration**: Flexible configuration management
- **Experiment Tracking**: Integration with Weights & Biases for experiment tracking
- **Hyperparameter Optimization**: Using Optuna via Hydra's sweeper
- **Parallel Processing**: Distributed training with Submitit

## Development

### Testing
```bash
# Run tests
poetry run pytest

# With coverage report
poetry run pytest --cov
```

### Code Quality
```bash
# Run linting
poetry run ruff check .

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy dmri
```

### Pre-commit Hooks
This project uses pre-commit hooks for code quality checks:

```bash
poetry run pre-commit install
```

### Continuous Integration

This project uses GitHub Actions for continuous integration. The following workflows are available:

- **CI**: Runs tests and linting on multiple Python versions (3.9, 3.10, 3.11)
- **JAX GPU Tests**: Runs JAX-specific tests on GPU infrastructure
- **Publish**: Publishes the package to PyPI when a new release is created

Status badges:
![CI](https://github.com/your-username/dmri/actions/workflows/ci.yml/badge.svg)
## Citation

If you use this code in your research, please cite:

```
@misc{dmri2023,
  author = {Your Name},
  title = {DMRI: A package for diffusion MRI model selection},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/your-username/dmri}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
