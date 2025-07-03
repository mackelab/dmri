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

## Extra dependencies/gotchas

### JAX

JAX is a library for high-performance numerical computing with automatic differentiation.

JAX is used for all numerical computations. It is installed as a dependency of the package.
Some gotchas:
- JAX will pre-allocate most of your GPU memory by default. If you run multiple notebooks/scripts in parallel, this will likely raise memory errors. You can check whether a job is running with `nvidia-smi`.
- JAX uses JIT compilation for all operations. This can make some operations appear slower than expected **on the first run**. Once compiled, the operation will be much faster.

For all practical purposes, it's just like NumPy/SciPy and closely mirrors their API.

### PyTorch conflicts

There might be some extra dependencies that are not included in the package and only
used in e.g. the notebooks. These need to be installed manually.

NOTE: If you want to use PyTorch, you should install the CPU-only version to avoid conflicts with the JAX version.

For example, to install torch and sbi for the notebooks, you can run:
```bash
pip install torch==2.5.1  --index-url https://download.pytorch.org/whl/cpu
pip install sbi
```

## Usage

### Simulators

Supported building simulators from fundamental compartments. Any model defined like this
can be used to build and train a inference/selection model automatically.

```python
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from dmri.simulators import Ball, Stick, Zeppelin, MultiCompartment
from dmri.simulators.acquisition_scheme import acquisition_scheme

# Example acquisition scheme dMRI
bvals = jnp.linspace(0, 4000, 100)
bvecs = jax.random.normal(jax.random.key(0), (100, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
acq = acquisition_scheme(bvals, bvecs)

# Example simulator for a single ball
theta = np.random.randn(Ball.theta_dim) # Theta will always be normal
ball = Ball.from_theta(theta) # Maps theta to random feasible parameters (diffusivity)
signal = ball.signal(acq) # Simulate signal

plt.plot(bvals, signal) # Plot signal

# But you can also combine multiple models
class BallStickZeppelin(MultiCompartment):
    model_types=[Ball, Stick, Zeppelin]
    noise_types = []

theta = np.random.randn(BallStickZeppelin.theta_dim)
# With all models
ball_stick_zeppelin = BallStickZeppelin.from_theta(theta)
# With only ball and stick
ball_stick = BallStickZeppelin.from_theta(theta, model_mask=jnp.array([True, True, False]))

# Simulate signal
signal = ball_stick_zeppelin.signal(acq)
signal_ball_stick = ball_stick.signal(acq)

plt.plot(bvals, signal)
plt.plot(bvals, signal_ball_stick)
```

You can also have a look at the notebooks in the `notebooks/dmri_simulators.ipynb` for more examples.

### Command-line Interface

Notably the command-line interface supports the following modes:
- Local training and evaluation – runs jobs in your current environment
- Slurm training and evaluation – submits jobs to a Slurm cluster
- W&B runtime statistics – logs running statistics to Weights & Biases (wandb)

However, these features need to be configured by the user; e.g. you must set up and log in to `wandb`.

#### Training

The package provides a command-line interface via Hydra, which allows you to configure training and evaluation of models. For the training API, run:

```bash
# For a list of available configurations
dmri --help
```

This will create a "results" folder in the current working directory with the following structure:
```
results/{name}            # Name of the run (default dmri)
├── 2025-06-30_15-19-06   # Date and time of the run containing config and logs
├── checkpoints/          # Checkpoints of the model (parameters over time)
```

After training you can already load the model and use it i.e. in a notebook e.g. `notebooks/eval_parameter_inference.ipynb` for more examples.

#### Evaluation

For evaluation/application of models to data, you can use:

```bash
dmri_eval --help
```

Depending on configuration, this will create additional folders in the results folder containing the evaluation results.

NOTE: Currently, only exporting ball3stick models is implemented.
NOTE: The evaluation needs to know where the data is located. This needs to be adapted in `conf_eval/config.yaml` accordingly.

```
results/{name}            # Name of the run (default dmri)
├── 2025-06-30_15-19-06   # Date and time of the run containing config and logs
├── checkpoints/          # Checkpoints of the model (parameters over time)
├── ball3stick_inference_results/                 # Inference results for ball3stick models
├── ball3stick_model_selection_results/           # Model selection results for ball3stick models
├── ...                                           # Additional depending on configuration
```

Notably you can modify the name of the e.g. folder in `conf_eval/export` to avoid overwriting existing results.

Certain evaluation runs—e.g. with or without specific model-selection steps—are pre-configured in the `conf_eval/experiments` folder.
For example to just run inference with all model components, you can use:
```bash
dmri_eval +experiment=eval_no_selection model_name=$NAME_OF_FOLDER_IN_RESULTS
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are located in the `conf/` directory for training and `conf_eval/` for evaluation.

Key configuration components:
- Model parameters
- Simulator specifications
- Training parameters
- Evaluation metrics

Just using the command-line interface, you can use the following command to see the available configurations: `dmri` will run the training with the default configuration. But you can also use some other predefined configurations using `dmri +experiment=ball3stick` for example.

### Continuous Integration

This project uses GitHub Actions for continuous integration. The following workflows are available:

- **CI**: Runs tests and linting on multiple Python versions (3.10, 3.11, 3.12)

Status badges:
![CI](https://github.com/your-username/dmri/actions/workflows/ci.yml/badge.svg)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
