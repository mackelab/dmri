# Simulators API

Exact signatures for acquisition schemes, compartments, mixtures, priors, and
noise models. Read the [simulator guide](../guides/simulators.md) first, or open
the [component example](../examples/01_dmri_simulator_components.md).

## Acquisition schemes

::: dmri.simulators.acquisition_scheme.acquisition_scheme
    options:
      show_root_heading: true
      show_root_full_path: false
      separate_signature: true

::: dmri.simulators.acquisition_scheme.ssfp_acquisition_scheme
    options:
      show_root_heading: true
      show_root_full_path: false
      separate_signature: true

## Abstract simulator classes

::: dmri.simulators.base.Compartment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - to_theta
        - to_params
        - from_theta
        - tree_flatten
        - tree_unflatten

::: dmri.simulators.base.SignalCompartment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - signal_fn
        - log_signal_fn
        - signal
        - log_signal
        - fit
        - to_fod

::: dmri.simulators.base.NoiseCompartment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - noise
        - log_likelihood


::: dmri.simulators.local_signal_models.signal_response_kernels
    options:
        show_root_heading: true
        show_root_full_path: false


::: dmri.simulators.multi_compartment.MultiCompartment
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "^signal_fn$"
        - "^from_theta$"
        - "^to_theta$"
        - "^num_compartments$"
        - "^get_all_params$"

## Local signal models

A selection of voxel-wise dMRI signal compartments is implemented below.

### Isotropic compartments

::: dmri.simulators.local_signal_models.ball.Ball
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.ball.StaticBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.ball.MultiShellBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.ball.MultiShellStaticBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.ball.SSFPBall
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.ball.SSFPStaticBall
    options:
        show_root_heading: true
        show_root_full_path: false


::: dmri.simulators.local_signal_models.sphere.Sphere
    options:
        show_root_heading: true
        show_root_full_path: false

### Anisotropic compartments

::: dmri.simulators.local_signal_models.zeppelin.Zeppelin
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.zeppelin.StaticZeppelin
    options:
        show_root_heading: true
        show_root_full_path: false


::: dmri.simulators.local_signal_models.cylinder.Cylinder
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.convolved_zeppelin.WatsonZeppelin
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.convolved_zeppelin.BinghamZeppelin
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.stick.Stick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.stick.StaticStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.stick.MultiShellStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.stick.MultiShellStaticStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.stick.SSFPStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.stick.SSFPStaticStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.convolved_stick.WatsonStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.convolved_stick.BinghamStick
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.dot.Dot
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.dti.Dti
    options:
        show_root_heading: true
        show_root_full_path: false

### Spherical convolution based simulators

These compartments convolve a kernel with a fiber orientation distribution (FOD)
on the sphere. The shared base classes live in `dmri.simulators.convolved_models`.

::: dmri.simulators.convolved_models
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.noddi.NoddiW
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.noddi.NoddiB
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.sandi.SandiW
    options:
        show_root_heading: true
        show_root_full_path: false

::: dmri.simulators.local_signal_models.sandi.SandiB
    options:
        show_root_heading: true
        show_root_full_path: false

## Spherical distributions

Orientation distributions (FODs, Watson, Bingham, ...) used by convolved
compartments and as orientation priors.

::: dmri.simulators.sphereical_distributions
    options:
        show_root_heading: true
        show_root_full_path: false

## Predefined model collections

Ready-made `MultiCompartment` families mirroring common literature mixtures
(Ball–Stick, Ball–3-Stick, multi-shell variants, ...).

::: dmri.simulators.models
    options:
        show_root_heading: true
        show_root_full_path: false

## Noise models

::: dmri.simulators.noise_compartments
    options:
      show_root_heading: true
      show_root_full_path: false

## Mask priors

::: dmri.simulators.mask_prior
    options:
      show_root_heading: true
      show_root_full_path: false
