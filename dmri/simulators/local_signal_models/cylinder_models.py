import math
from dmri.simulators.local_signal_models.gaussian_models import Stick
import jax.numpy as jnp
import jax

from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment
from dmri.utils.dmriutils import (
    unitsphere_to_cartesian,
)

import numpy as np
from scipy.special import j1 as j1_scipy

x_l = np.linspace(0, 5, 1000)
y_l = j1_scipy(x_l)


def j1(x):
    return jnp.interp(x, x_l, y_l)


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


class Cylinder(SignalCompartment):
    r"""
    The Stejskal-Tanner approximation of the cylinder model with finite
    radius. Assumes short gradient pulse approximation and long diffusion
    time limit, where perpendicular diffusion depends only on q-value.
    """

    theta_dim = 4
    lam_par_max: float = 0.1
    radius_mean = math.log(0.01)
    radius_scale = 0.5

    def __init__(self, mu: ArrayLike, lam_par: float, radius: float):
        self.mu = mu
        self.lam_par = lam_par
        self.radius = radius

    @classmethod
    def log_signal_fn(
        cls,
        aquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        radius: float,
        rng=None,
    ):
        """Compute the log signal attenuation."""
        q = aquisition_scheme.q_values
        bvecs = aquisition_scheme.bvecs

        mu_cart = unitsphere_to_cartesian(mu)
        mu_perpendicular_plane = jnp.eye(3) - jnp.outer(mu_cart, mu_cart)
        magnitude_perpendicular = jnp.linalg.norm(
            mu_perpendicular_plane @ bvecs.T, axis=0
        )
        log_signal_parallel = Stick.log_signal_fn(
            aquisition_scheme,
            mu=mu,
            lam_par=lam_par,
        )
        log_signal_perpendicular = 2 * jnp.log(
            j1(2 * jnp.pi * q * radius)
        ) - 2 * jnp.log(2 * jnp.pi * q * radius)
        q_perp = q * magnitude_perpendicular
        log_signal_perpendicular = jnp.where(q_perp > 0, log_signal_perpendicular, 0.0)
        log_signal = log_signal_parallel + log_signal_perpendicular
        return log_signal

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float, radius: float) -> ArrayLike:
        """Convert parameters to the parameter space theta."""
        lambda_par_theta = jax.scipy.stats.norm.ppf(lam_par / cls.lam_par_max)
        radius_theta = (jnp.log(radius) - cls.radius_mean) / cls.radius_scale
        theta_mu0 = jax.scipy.stats.norm.ppf(mu[0] / jnp.pi)
        theta_mu1 = jax.scipy.stats.norm.ppf((mu[1] + jnp.pi) / (2 * jnp.pi))
        return jnp.array([lambda_par_theta, radius_theta, theta_mu0, theta_mu1])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the model parameters."""
        lambda_par = jax.scipy.stats.norm.cdf(theta[0]) * cls.lam_par_max
        radius = jnp.exp(theta[1] * cls.radius_scale + cls.radius_mean)
        theta_mu = jax.scipy.stats.norm.cdf(theta[2:])
        mu0 = theta_mu[0] * jnp.pi
        mu1 = theta_mu[1] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu0, mu1])
        return mu, lambda_par, radius
