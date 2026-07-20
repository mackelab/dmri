---
hide:
  - navigation
  - toc
---

<div class="landing-hero">
  <div class="landing-hero__copy">
    <p class="landing-kicker"><span></span> Simulation-based inference for diffusion MRI</p>
    <h1>Model selection and parameter inference for diffusion MRI.</h1>
    <p class="landing-lede">DMRI provides JAX implementations of composable tissue models and amortized inference methods for estimating model structure and biophysical parameters across acquisition protocols.</p>
    <div class="landing-actions">
      <a class="md-button md-button--primary" href="getting-started/">Installation and usage</a>
      <a class="landing-text-link" href="examples/">Example notebooks <span aria-hidden="true">→</span></a>
    </div>
    <ul class="landing-tags" aria-label="Key capabilities">
      <li>JIT + VMAP</li>
      <li>Hydra experiments</li>
      <li>Model selection</li>
    </ul>
  </div>

</div>

<section class="landing-section">
  <div class="section-heading">
    <p>Methodological workflow</p>
    <h2>A unified pipeline for simulation and inference.</h2>
  </div>

  <div class="workflow-rail">
    <a href="examples/01_dmri_simulator_components/" class="workflow-step">
      <span class="workflow-step__number">01</span>
      <div><h3>Characterize</h3><p>Compare signal responses across tissue compartments.</p></div>
    </a>
    <a href="examples/02_dmri_multicompartment_model/" class="workflow-step">
      <span class="workflow-step__number">02</span>
      <div><h3>Construct</h3><p>Specify compartments, fractions, model masks, and noise.</p></div>
    </a>
    <a href="examples/04_train_standalone/" class="workflow-step">
      <span class="workflow-step__number">03</span>
      <div><h3>Estimate</h3><p>Train amortized inference networks on simulated observations.</p></div>
    </a>
    <a href="examples/03_pretrained_models/" class="workflow-step">
      <span class="workflow-step__number">04</span>
      <div><h3>Assess</h3><p>Quantify model selection, uncertainty, and coverage.</p></div>
    </a>
  </div>
</section>

<section class="landing-section landing-section--split">
  <div class="capability-intro">
    <p class="landing-kicker"><span></span> Joint model and parameter inference</p>
    <h2>Inference over discrete model structure and continuous parameters.</h2>
    <p>The active compartment model is treated as a latent variable, allowing uncertainty over model structure to be represented alongside uncertainty in continuous parameters.</p>
    <a class="landing-text-link" href="guides/simulators/">Simulator methodology <span aria-hidden="true">→</span></a>
  </div>
  <div class="capability-list">
    <div class="capability-row"><span>01</span><div><h3>Signal models</h3><p>Ball, stick, zeppelin, tensor, restricted, NODDI, and SANDI compartments.</p></div></div>
    <div class="capability-row"><span>02</span><div><h3>Acquisition conditioning</h3><p>Inference may be conditioned on b-values, gradient directions, and acquisition metadata.</p></div></div>
    <div class="capability-row"><span>03</span><div><h3>Experiment configuration</h3><p>Hydra configurations support local execution, parameter sweeps, and SLURM environments.</p></div></div>
  </div>
</section>

<section class="quickstart-band" markdown>
<div class="quickstart-band__copy">
  <p>INSTALLATION</p>
  <h2>Install from source.</h2>
</div>

```bash
git clone https://github.com/mackelab/dmri.git
cd dmri && uv pip install -e '.[dev]'
```

<a class="md-button md-button--primary" href="getting-started/">Installation instructions</a>
</section>
