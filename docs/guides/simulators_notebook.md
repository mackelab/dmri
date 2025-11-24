# Simulator Tutorial (Notebook)

This page renders the key steps from `examples/dmri_simulators.ipynb` so you can skim the outputs without running code. Download and run the notebook for an interactive version.

- **Notebook:** `examples/dmri_simulators.ipynb`
- **What you’ll learn:** build acquisition schemes, simulate Ball/Stick signals, and combine them with `MultiCompartment`.

## 1. Build an acquisition scheme

```python
bvals = jnp.linspace(0, 3000, 100)
bvecs = jax.random.normal(jax.random.key(0), (100, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
acq = acquisition_scheme(bvals, bvecs)
```

## 2. Single-compartment models

**Ball (isotropic Gaussian)** — signal decays as \(S = \exp(-b \lambda)\).

![Ball signal](../assets/sim_ball.svg)

**Stick (zero-radius cylinder)** — attenuation depends on gradient alignment with \(\boldsymbol{\mu}\).

![Stick signal](../assets/sim_stick.svg)

## 3. Multi-compartment mixture

Subclass `MultiCompartment` to mix components with learnable fractions.

```python
class BallStick(MultiCompartment):
    model_types = [Ball, Stick]
    noise_types = []
    fraction_prior = jnp.ones(2)

theta = jax.random.normal(jax.random.key(123), (BallStick.theta_dim,))
model = BallStick.from_theta(theta)
signal_mix = model.signal(acq)
```

Resulting mixture (50% Ball, 50% Stick in this demo):

![Ball+Stick mixture](../assets/sim_multi.svg)

## 4. Next steps

- Swap in `Sphere` or `Cylinder` for restricted diffusion.
- Add noise compartments (e.g., `BoundedGaussianNoise`) for magnitude data.
- Use `model_mask` in `MultiCompartment.from_theta` to toggle components without changing the parameter layout.
