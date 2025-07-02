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

The following are required to run the code:
- Python 3.10–3.12
- pip for dependency management
- optional: conda for environment management and CUDA 12.1+ for GPU acceleration

To avoid conflicts, it is recommended to use a virtual environment, e.g.:
```bash
conda create -n dmri python=3.11
conda activate dmri
```

### Install the package

To install the package, first clone the repository:
```bash
# Clone the repository
git clone https://github.com/your-username/dmri.git
cd dmri

# Install the package and main dependencies
pip install -e .

# If you use a GPU, install the GPU dependencies
pip install -e .[cuda]

# For development dependencies (testing, linting, formatting, etc.)
pip install -e .[dev]
```

If you installed with `[dev]`, you can check if the installation was successful by running:
```bash
pytest
```

## Extra dependencies
There might be some extra dependencies that are not included in the package and only
used in e.g. the notebooks. These need to be installed manually.

NOTE: If you want to use pytorch, you shoud install the CPU-only version of pytorch to avoid conflicts with the JAX version.

## Usage

### Basic Example
```python
import dmri

# Example code will depend on specific implementation
```

### Command-line Interface

The package provides a command-line interface via Hydra. This mainly allows to configure
training and evaluation of models. For training API, you can use:

```bash
# For a list of available configurations
dmri --help
```

For evaluation/application of models to data, you can use:

```bash
dmri_eval --help
```



## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are located in the `conf/` directory.

Key configuration components:
- Model parameters
- Dataset specifications
- Training parameters
- Evaluation parameters

### Continuous Integration

This project uses GitHub Actions for continuous integration. The following workflows are available:

- **CI**: Runs tests and linting on multiple Python versions (3.10, 3.11, 3.12)

Status badges:
![CI](https://github.com/your-username/dmri/actions/workflows/ci.yml/badge.svg)


## License

This project is licensed under the MIT License - see the LICENSE file for details.
