import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment


class Sphere(SignalCompartment):
    r"""
    The Stejskal Tanner signal approximation of a sphere model. It assumes
    that pulse length is infinitessimally small and diffusion time large enough
    so that the diffusion is completely restricted. Only depends on q-value.

    Parameters
    ----------
    diameter : float,
        sphere diameter in meters.

    References
    ----------
    .. [1] Balinov, Balin, et al. "The NMR self-diffusion method applied to
        restricted diffusion. Simulation of echo attenuation from molecules in
        spheres and between planes." Journal of Magnetic Resonance, Series A
        104.1 (1993): 17-25.
    """

    theta_dim = 1
    radius_mean = math.log(0.01)
    radius_scale = 0.5

    def __init__(self, radius: float):
        self.radius = radius

    @classmethod
    def log_signal_fn(cls, aquisition_scheme, radius: float, rng=None):
        q = aquisition_scheme.q_values  # 1/mm
        E_sphere = jnp.ones_like(q)
        factor = 2 * jnp.pi * q * radius
        E_sphere_attenuation = (
            3 / (factor**2) * (jnp.sin(factor) / factor - jnp.cos(factor))
        ) ** 2
        q_nonzero = q > 0
        E_sphere = jnp.where(q_nonzero, E_sphere_attenuation, E_sphere)
        return jnp.log(E_sphere)

    @classmethod
    def to_theta(cls, radius: float) -> ArrayLike:
        """Convert the sphere radius to the parameter space theta."""
        return (jnp.log(radius) - cls.radius_mean) / cls.radius_scale

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the sphere radius."""
        radius = jnp.exp(theta * cls.radius_scale + cls.radius_mean)
        return (radius,)
