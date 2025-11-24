# Simulators & Priors

DMRI ships composable simulators so you can explore competing diffusion models without rewriting kernels. The classes documented below expose friendly constructors and priors through their docstrings—this guide shows how to use them together.

## Modeling assumptions (quick math)

We follow the Stejskal–Tanner convention, writing the diffusion-weighted signal for
component \(k\) as \(S_k(b, \mathbf{g})\). Typical closed forms:

- **Ball (isotropic Gaussian):** \(S(b) = \exp(-b D)\).
- **Stick (zero-radius cylinder):** \(S(b, \mathbf{g}) = \exp\big(-b\,D_{\parallel}(\mathbf{g}\cdot\boldsymbol{\mu})^2\big)\).
- **Zeppelin (axially symmetric Gaussian):** \(S(b, \mathbf{g}) = \exp\big(-b[ D_{\perp} + (D_{\parallel}-D_{\perp})(\mathbf{g}\cdot\boldsymbol{\mu})^2 ]\big)\).
- **Sphere (restricted, narrow-pulse limit):** uses the Balinov et al. (1993) series for attenuation in a sphere of radius \(R\).

A multi-compartment mixture with fractions \(f_k\) produces

$$
S(b, \mathbf{g}) = \sum_k f_k\, S_k(b, \mathbf{g}), \quad \sum_k f_k = 1.
$$

Noise compartments (e.g., bounded Rician/Gaussian) are applied after the base signal.

### Noise and mask priors

- Rician/Gaussian noise follow the Gudbjartsson–Patz (1995) magnitude distribution with bounded SNR priors for stability.
- Mask priors (Beta–Bernoulli) select which compartments are active, letting you toggle, e.g., isotropic pools during inference.

Key references

- Stejskal & Tanner (1965) for the pulsed-gradient spin-echo signal model.
- Callaghan (1991) for restricted diffusion formalisms.
- Behrens et al. (2003) for the Ball–Stick mixture used in tractography.
- Balinov et al. (1993) for the spherical attenuation series.

## Build an acquisition scheme

```python
import jax
import jax.numpy as jnp
from dmri.simulators.acquisition_scheme import acquisition_scheme

bvals = jnp.linspace(0, 4000, 100)
bvecs = jax.random.normal(jax.random.key(0), (100, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
acq = acquisition_scheme(bvals, bvecs)
```

## Compose compartments

Signal compartments inherit from `SignalCompartment` and expose `from_theta` / `to_theta` helpers to map between priors and physical parameters. Noise compartments plug in independently.

```python
from dmri.simulators import Ball, Stick, Zeppelin, MultiCompartment

class BallStickZeppelin(MultiCompartment):
    model_types = [Ball, Stick, Zeppelin]
    noise_types = []  # add Rician or Gaussian later if needed

theta = jax.random.normal(jax.random.key(42), (BallStickZeppelin.theta_dim,))
model = BallStickZeppelin.from_theta(theta)
signal = model.signal(acq)  # simulated diffusion signal
```

Use `model_mask` to turn components on/off without retraining the parameterizer:

```python
signal_ball_stick = model.signal(acq, model_mask=jnp.array([True, True, False]))
```

## Working with SSFP acquisitions

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

## Tips for extending

- Start from existing compartments in `dmri.simulators.local_signal_models` and mirror the `to_theta` / `to_params` pair.
- Reuse spherical distributions in `dmri.simulators.sphereical_distributions` for orientation priors.
- Add rich docstrings to new components—the API Reference will surface them automatically.
