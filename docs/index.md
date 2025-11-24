# DMRI: Package for simulations-based model selection and inference for diffusion MRI

<div class="hero">
  <div class="hero__text">
    <p class="eyebrow">High-performance Bayesian diffusion MRI</p>
    <h1>Fast simulators, robust inference, reproducible runs.</h1>
    <p class="lede">Train and deploy diffusion MRI model-selection pipelines powered by JAX, Hydra, and lightweight simulators.</p>
    <div class="hero__actions">
      <a class="button primary" href="getting-started/">Get started</a>
      <a class="button ghost" href="guides/training/">Schedule a training run</a>
    </div>
  </div>
  <div class="hero__card">
    <p class="hero__label">CLI quickstart</p>
    <pre><code class="language-bash">pip install -e '.[dev]'
dmri +experiment=ball3stick
dmri_eval +experiment=eval_b3s_best_model_selection</code></pre>
  </div>
</div>

## Why this package

- Simulator building blocks (ball, stick, zeppelin, SSFP, semi-global models) ready to combine implemented in native `jax` fully `jit` and `vmap` compliance allowing efficient and parallel simulation from combinatorial model families via automatic vectorizations and compilation.
- Hydra-powered configuration makes runs reproducible and easy to sweep i.e. via `SLURM`.
- Well-typed utilities for sampling, exporting NIfTIs, and composing acquisition schemes.

## How the docs are organized

- **Getting Started**: installation, environment setup, and the quickest way to run a training job.
- **Guides**: opinionated walkthroughs for building simulators, orchestrating training, and exporting evaluation artefacts.
- **API Reference**: pulled straight from docstrings so you always have the authoritative signatures and parameter notes.

## Next steps

- Prefer installing with `uv sync --dev` for faster environments.
- Skim the [Simulators guide](guides/simulators.md) to understand the primitives before composing new models.
- Add your own docstrings—this site will surface them automatically.
