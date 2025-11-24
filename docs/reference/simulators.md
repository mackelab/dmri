# Simulators API

Docstrings power these references. Sections are grouped to keep the table of contents tidy; local signal models are collapsible for quick scanning.

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

## Compartments and mixing

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

???+ info "Isotropic and anisotropic"

    ::: dmri.simulators.local_signal_models.ball.Ball
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.ball.StaticBall
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.ball.MultiShellStaticBall
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.zeppelin.Zeppelin
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.convolved_zeppelin.ConvolvedZeppelin
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.cylinder.Cylinder
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.sphere.Sphere
        options:
          show_root_heading: true
          show_root_full_path: false

???+ info "Stick-based"

    ::: dmri.simulators.local_signal_models.stick.Stick
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.stick.StaticStick
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.convolved_stick.ConvolvedStick
        options:
          show_root_heading: true
          show_root_full_path: false

???+ info "Microstructure & tensor models"

    ::: dmri.simulators.local_signal_models.dot.Dot
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.dti.DTIModel
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.noddi.NoddiModel
        options:
          show_root_heading: true
          show_root_full_path: false

    ::: dmri.simulators.local_signal_models.sandi.SANDIModel
        options:
          show_root_heading: true
          show_root_full_path: false

???+ info "Kernels and helpers"

    ::: dmri.simulators.local_signal_models.signal_response_kernels
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
