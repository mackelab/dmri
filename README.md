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
# Alternatively, you can install the package using pip:
pip install -e .
pip install -e .[dev]
```

```

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

- **CI**: Runs tests and linting on multiple Python versions (3.9, 3.10, 3.11)

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
