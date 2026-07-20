# Simulators & Priors

DMRI ships composable simulators so you can swap diffusion components, plug in priors, and keep optimization in a stable Gaussian parameter space. This page walks through the design and how to wire the pieces together.

## What this page covers
- Core closed-form compartments and how mixtures are assembled.
- Moving between physical parameters and `theta` space with priors and masks.
- Adding magnitude-noise models, SSFP acquisitions, and extension points.

## Model math (cheat sheet)

We follow the Stejskal–Tanner convention, writing the diffusion-weighted signal for component \(k\) as \(S_k(b, \mathbf{g})\):

- **Ball (isotropic Gaussian):** \(S(b) = \exp(-b D)\).
- **Stick (zero-radius cylinder):** \(S(b, \mathbf{g}) = \exp\big(-b\,D_{\parallel}(\mathbf{g}\cdot\boldsymbol{\mu})^2\big)\).
- **Zeppelin (axially symmetric Gaussian):** \(S(b, \mathbf{g}) = \exp\big(-b[ D_{\perp} + (D_{\parallel}-D_{\perp})(\mathbf{g}\cdot\boldsymbol{\mu})^2 ]\big)\).
- **Sphere (restricted, narrow-pulse limit):** uses a closed-form spherical form factor for radius \(R\), with radius expressed consistently with q-values in inverse millimetres.

A mixture with fractions \(f_k\) produces \(S(b, \mathbf{g}) = \sum_k f_k\, S_k(b, \mathbf{g})\) with \(\sum_k f_k = 1\); noise compartments (bounded Rician/Gaussian) are applied after the base signal.

## Pieces to combine

- **Acquisition schemes:** `dmri.simulators.acquisition_scheme` for pulsed-gradient experiments; `ssfp_acquisition_scheme` adds flip angles, T1/T2, B1, and gradient timing for SSFP.
- **Signal compartments:** classes in `dmri.simulators.local_signal_models` (Ball, Stick, Zeppelin, Dti, Sphere, Cylinder, NODDI*, SANDI*) expose `theta_dim`, `to_theta`, `to_params`, `signal`, and optional `to_fod`.
- **MultiCompartment mixer:** `dmri.simulators.MultiCompartment` mixes `model_types` with Dirichlet fractions (`fraction_prior`), optional shared parameters (`shared_parameter_type`), a Beta–Bernoulli mask prior (`create_mask_prior`), and convenience `theta_mask` helpers for gating parameters.
- **Noise:** `dmri.simulators.noise_compartments` provides Rician/Gaussian likelihoods plus bounded SNR presets; they operate on the already-mixed signal and require `rng` for sampling.

## Example: Ball+Stick+Zeppelin mixture

```python
import jax
import jax.numpy as jnp
from dmri.simulators import Ball, Stick, Zeppelin, MultiCompartment
from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.simulators.noise_compartments import BoundedRicianNoise

# 1) Acquisition
bvals = jnp.linspace(0, 4000, 100)
bvecs = jax.random.normal(jax.random.key(0), (100, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
acq = acquisition_scheme(bvals, bvecs)

# 2) Model definition
class BallStickZeppelin(MultiCompartment):
    model_types = [Ball, Stick, Zeppelin]
    noise_types = [BoundedRicianNoise]  # remove or swap for Gaussian if needed
    fraction_prior = jnp.array([1.0, 1.0, 1.0])
    mask_prior_kwargs = {"alpha": 2.0, "beta": 2.0, "min_active_models": 1}

# 3) Sample theta + mask and simulate
mask_prior = BallStickZeppelin.create_mask_prior()
mask = mask_prior.sample(jax.random.key(1)).model_mask
theta = jax.random.normal(jax.random.key(42), (BallStickZeppelin.theta_dim,))
model = BallStickZeppelin.from_theta(theta, model_mask=mask)

signal = model.signal(acq, rng=jax.random.key(7))  # full mixture + noise
params = model.get_all_params()  # inspect fractions, per-compartment params, mask
```

- To change active components, reconstruct the model with the desired mask before calling `signal`.
- For shared diffusivity across compartments, set `shared_parameter_type` on the subclass (see `SharedDiffusivity` in `dmri.simulators.multi_compartment`).

## Work in theta space and priors

- `to_params` / `to_theta` keep optimizer space Gaussian: fractions live in a Dirichlet (`fraction_prior`), per-compartment parameters follow their own CDF transforms, and masks can be enforced via `theta_mask(model_mask)`.
- Mask priors (`BetaBernoulliMaskPrior`) default to at least one active model; override `alpha`, `beta`, or `min_active_models` via `mask_prior_kwargs`.

```python
mask = jnp.array([True, False, True, True])  # 3 models + 1 noise
theta = jax.random.normal(jax.random.key(0), (BallStickZeppelin.theta_dim,))

# Map theta to physical parameters conditioned on the mask
fractions, comps, noises, _, shared = BallStickZeppelin.to_params(theta, model_mask=mask)

# Round-trip back to theta (useful for diagnostics)
theta_roundtrip = BallStickZeppelin.to_theta(
    fractions,
    comps,
    noises,
    model_mask=mask,
    shared_parameter=shared,
)

# Build a boolean selector for masked optimization variables
active_theta = BallStickZeppelin.theta_mask(mask)
```

```python
new_mask = jnp.array([True, True, False, True])
masked_model = BallStickZeppelin.from_theta(theta, model_mask=new_mask)
masked_signal = masked_model.signal(acq, rng=jax.random.key(8))
```

## Noise and likelihoods

- Choose bounded presets such as `RicianNoiseSNR310` or `GaussianNoiseSNR2030`, or use `BoundedRicianNoise`/`BoundedGaussianNoise` for custom ranges.
- Passing `rng` to `signal` returns a noisy observation. Evaluate an explicitly supplied observation separately with `model.log_likelihood(acq, observed_signal)`.

## SSFP acquisitions

The `ssfp_acquisition_scheme` class handles unit conversions for SSFP experiments (flip angles, diffusion gradients, T1/T2). Combine it with SSFP-aware compartments and the numerically stable `ssfp_signal_fn` from `dmri.utils.dmriutils`.

```python
from dmri.simulators.acquisition_scheme import ssfp_acquisition_scheme
from dmri.utils.dmriutils import ssfp_signal_fn

acq_ssfp = ssfp_acquisition_scheme(
    bvecs=bvecs,
    T1_raw=1200.0,
    T2_raw=85.0,
    B1=0.95,
    diffGradAmps_raw=35e-3,
    flipAngles_raw=14.0,
)

signal = ssfp_signal_fn(
    adc=0.002,
    qval=acq_ssfp.qvals,
    E1=acq_ssfp.E1,
    E2=acq_ssfp.E2,
    sa=acq_ssfp.sa,
    ca=acq_ssfp.ca,
    TR=acq_ssfp.TRs,
    diff_grad_dur=acq_ssfp.diffGradDur,
)
```

## Extend the library

- Start from existing compartments in `dmri.simulators.local_signal_models` and mirror the `to_theta` / `to_params` pair.
- Reuse spherical distributions in `dmri.simulators.sphereical_distributions` for orientation priors or `SharedParameterState` for tied diffusivities.
- Add rich docstrings to new components—the API Reference will surface them automatically.

Key references: Stejskal & Tanner (1965) for pulsed-gradient signals; Callaghan (1991) for restricted diffusion; Behrens et al. (2003) for Ball–Stick mixtures; and Gudbjartsson & Patz (1995) for magnitude-noise models.

Continue with the [multi-compartment notebook](../examples/02_dmri_multicompartment_model.md), or look up exact signatures in the [simulator API](../reference/simulators.md).
