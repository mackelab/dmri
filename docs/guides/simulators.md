# Simulators

DMRI signal models map unconstrained parameter vectors (`theta`) to physical
parameters and simulated measurements. Training uses these models to generate
data. Evaluation also uses them for likelihoods, reconstruction metrics, and
synthetic inputs.

## Acquisition data

`acquisition_scheme` stores b-values and b-vectors:

```python
import jax.numpy as jnp

from dmri.simulators.acquisition_scheme import acquisition_scheme

bvals = jnp.array([0.0, 1000.0, 1000.0])
bvecs = jnp.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
acq = acquisition_scheme(bvals, bvecs)
```

The package also defines acquisition generators used by training configs and an
`ssfp_acquisition_scheme` for SSFP measurements.

## Compartments

Signal compartment classes are under `dmri.simulators.local_signal_models`.
Examples include `Ball`, `Stick`, `Zeppelin`, `Dti`, `Sphere`, `Cylinder`, and
models used by NODDI and SANDI.

Each compartment defines:

- `theta_dim`: number of unconstrained parameters.
- `from_theta`: construct physical parameters from theta.
- `to_theta` and `to_params`: convert between representations.
- `signal`: simulate measurements for an acquisition.

## Multi-compartment models

A `MultiCompartment` subclass declares signal and noise components:

```python
import jax
import jax.numpy as jnp

from dmri.simulators import Ball, MultiCompartment, Stick, Zeppelin
from dmri.simulators.noise_compartments import BoundedRicianNoise


class BallStickZeppelin(MultiCompartment):
    model_types = [Ball, Stick, Zeppelin]
    noise_types = [BoundedRicianNoise]
    fraction_prior = jnp.ones(3)


mask = jnp.array([True, True, False, True])
theta = jax.random.normal(jax.random.key(0), (BallStickZeppelin.theta_dim,))
model = BallStickZeppelin.from_theta(theta, model_mask=mask)
signal = model.signal(acq, rng=jax.random.key(1))
```

The model mask has one entry per signal and noise component. Fractions are
normalized over active signal components. Noise is applied after mixing the
signal components.

Use `theta_mask(model_mask)` to identify theta entries associated with active
components. Use `create_mask_prior()` to construct the model's mask prior.

## Training configuration

The canonical simulator config separates the model class from acquisition
generation:

```yaml
simulator:
  model_class: my_package.models.BallStickZeppelin
  posterior_score: false
  mask_prior: {}
  acquisitions:
    - _target_: dmri.simulators.random_hcp_acquisition
      _partial_: true
      num_acquisitions: 105
```

Repository presets split these fields across `conf/simulator/*.yaml` and
`conf/simulator/acquisition/*.yaml`:

```bash
dmri train simulator=ball3stick_shared simulator/acquisition=multi
```

`model_class` must be an importable `MultiCompartment` subclass. Acquisition
targets must instantiate to callables; function targets therefore use
`_partial_: true`.

`simulator.posterior_score=true` adds score targets to generated training data.
`simulator.mask_prior` passes keyword arguments to the model's
`create_mask_prior` method.

## Physical parameters and theta

Models train in theta space. `to_params` maps theta to fractions, compartment
parameters, noise parameters, and shared parameters. `to_theta` performs the
reverse conversion where supported. This keeps constraints such as positive
diffusivities and normalized fractions out of the optimizer interface.

```python
values = BallStickZeppelin.to_params(theta, model_mask=mask)
fractions, components, noises, _, shared = values

theta_again = BallStickZeppelin.to_theta(
    fractions,
    components,
    noises,
    model_mask=mask,
    shared_parameter=shared,
)
```

## Likelihoods and noise

Passing an RNG to `signal` samples configured noise. Evaluate an observed signal
with:

```python
log_prob = model.log_likelihood(acq, observed_signal)
```

Available noise classes include Gaussian, Rician, and noncentral-chi variants,
including bounded SNR presets. See the [simulator API](../reference/simulators.md)
for exact classes and signatures.

## Add a model

1. Implement or reuse compartment classes.
2. Define a `MultiCompartment` subclass.
3. Make the class importable.
4. Set `simulator.model_class` to its full import path.
5. Add a short YAML preset only if the model will be selected by name often.

No Python registry is required.

The [simulator examples](../examples/index.md) show signals and model masks with
small arrays.
