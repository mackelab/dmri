import math

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
    def log_signal_fn(cls, acq, radius: float, rng=None):
        q = acq.qvals  # 1/mm
        factor = 2 * jnp.pi * q * radius
        factor_sq = factor**2

        # Stable evaluation for small factors to avoid NaNs when q ~ 0
        small_factor = factor_sq < 1e-8
        safe_factor = jnp.where(small_factor, 1.0, factor)

        attenuation = (
            3.0
            / (safe_factor**2)
            * (jnp.sin(safe_factor) / safe_factor - jnp.cos(safe_factor))
        )
        attenuation = attenuation**2

        # Second-order series approximation around 0: (1 - x^2 / 10)^2
        attenuation_small = 1.0 - factor_sq / 5.0 + factor_sq**2 / 100.0

        E_sphere = jnp.where(small_factor, attenuation_small, attenuation)
        E_sphere = jnp.clip(E_sphere, a_min=1e-12)  # Guard against log(0)

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
