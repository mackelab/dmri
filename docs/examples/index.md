# Example notebooks

The notebooks progress from individual simulator components to complete training and evaluation workflows. Saved outputs are rendered directly, and every page links to its source notebook.

<div class="card-grid">
  <a class="doc-card" href="01_dmri_simulator_components/">
    <span class="doc-card__number">01 · Components</span>
    <h3>Simulator model gallery</h3>
    <p>Inspect signals from a representative set of diffusion models.</p>
    <span class="example-meta">CPU · 5 min</span>
  </a>
  <a class="doc-card" href="02_dmri_multicompartment_model/">
    <span class="doc-card__number">02 · Composition</span>
    <h3>Multi-compartment simulator</h3>
    <p>Create an acquisition scheme and combine compartments step by step.</p>
    <span class="example-meta">CPU · 5 min</span>
  </a>
  <a class="doc-card" href="03_pretrained_models/">
    <span class="doc-card__number">03 · Evaluation</span>
    <h3>Using pretrained models</h3>
    <p>Load a published checkpoint and assess model and parameter predictions.</p>
    <span class="example-meta">CPU/GPU · download required</span>
  </a>
  <a class="doc-card" href="04_train_standalone/">
    <span class="doc-card__number">04 · Training</span>
    <h3>Train a small model</h3>
    <p>Run a complete simulation, training, and posterior-sampling workflow.</p>
    <span class="example-meta">CPU · 2 min</span>
  </a>
</div>

!!! tip "Run the notebooks locally"
    From a cloned repository, install the development environment with `uv pip install -e '.[dev]'`, then open the files in `docs/examples/` with Jupyter.
