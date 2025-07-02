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

To install with CUDA support, run:
```bash
pip install --upgrade "jax[cuda12]"
```

### Setup

To install the package, first clone the repository:
```bash
# Clone the repository
git clone https://github.com/your-username/dmri.git
cd dmri

# Install the package and main dependencies
pip install -e .

# For development dependencies (testing, linting, formatting, etc.)
pip install -e .[dev]

# For notebook dependencies (e.g., PyTorch, sbi)
pip install -e .[notebook]

# You can also install both sets of extras at once:
pip install -e .[dev,notebook]
```

## Standard installation (CPU only)
```bash
pip install -e .
```

## For CUDA-enabled JAX (GPU support)
After installing the base dependencies, run:
```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Replace `cuda12` with your CUDA version if needed. See the [JAX CUDA releases page](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html) for more options.

## Usage

### Basic Example
```python
import dmri

# Example code will depend on specific implementation
```

### Command-line Interface
The package provides a command-line interface via Hydra. This mainly allows to configure
training and evaluation of models.

```bash
# For a list of available commands
dmri --help
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
