"""Diffusion MRI simulators and priors.

The package exposes single-compartment signal models (Ball, Stick, Zeppelin, Cylinder,
Sphere, etc.), multi-compartment mixtures, acquisition helpers, and noise models.
Signal simulators follow the Stejskal–Tanner convention :math:`S = e^{-b D}` for
Gaussian compartments and extend to restricted geometries as in Callaghan (1991).

Key references
--------------
* Stejskal & Tanner, 1965. Spin diffusion measurements: spin echoes in the presence of a time‐dependent field gradient.
* Callaghan, 1991. Principles of Nuclear Magnetic Resonance Microscopy.
* Behrens et al., 2003. Characterization and propagation of uncertainty in diffusion-weighted MR imaging (Ball–Stick).
"""

from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.simulators.base import NoiseCompartment, SignalCompartment
from dmri.simulators.local_signal_models import *
from dmri.simulators.models import *
from dmri.simulators.multi_compartment import MultiCompartment
from dmri.simulators.noise_compartments import *
