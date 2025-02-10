from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike
import scipy.special as sp_special  # Add this import for special functions

from dmri.simulators.base import ModelCompartment
from dmri.simulators.local_signal_models.utils import (
    fit_diffusion_tensor_linearized,
    unitsphere_to_cartesian,
)


DIFFUSIVITY_SCALING = 1e-9
DIAMETER_SCALING = 1e-6


class C2Cylinder(ModelCompartment):
    r"""
    The Stejskal-Tanner approximation of the cylinder model with finite
    radius. Assumes short gradient pulse approximation and long diffusion
    time limit, where perpendicular diffusion depends only on q-value.
    """

    theta_dim = 4

    def __init__(self, mu=None, lambda_par=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter

    def perpendicular_attenuation(self, q, diameter):
        """Compute the cylinder's perpendicular signal attenuation."""
        radius = diameter / 2
        E = (2 * sp_special.j1(2 * jnp.pi * q * radius)) ** 2 / (
            2 * jnp.pi * q * radius
        ) ** 2
        return E

    def log_signal(self, bvals, bvecs, **kwargs):
        """Compute the log signal attenuation."""
        diameter = kwargs.get("diameter", self.diameter)
        lambda_par = kwargs.get("lambda_par", self.lambda_par)
        mu = kwargs.get("mu", self.mu)
        mu_cartesian = unitsphere_to_cartesian(mu)

        mu_perpendicular_plane = jnp.eye(3) - jnp.outer(mu_cartesian, mu_cartesian)
        magnitude_perpendicular = jnp.linalg.norm(
            jnp.dot(mu_perpendicular_plane, bvecs.T), axis=0
        )
        qvalues = jnp.sqrt(bvals / (4 * jnp.pi**2))
        E_parallel = jnp.exp(-bvals * lambda_par * jnp.dot(bvecs, mu_cartesian) ** 2)
        E_perpendicular = jnp.ones_like(qvalues)
        q_perpendicular = qvalues * magnitude_perpendicular
        q_nonzero = q_perpendicular > 0
        E_perpendicular = E_perpendicular.at[q_nonzero].set(
            self.perpendicular_attenuation(q_perpendicular[q_nonzero], diameter)
        )
        return jnp.log(E_parallel * E_perpendicular)

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the cylinder model to the log signal."""
        raise NotImplementedError(
            "Fitting for C2CylinderStejskalTannerApproximation is not implemented."
        )

    @classmethod
    def to_theta(cls, mu: ArrayLike, lambda_par: float, diameter: float) -> ArrayLike:
        """Convert parameters to the parameter space theta."""
        lambda_par = lambda_par * DIFFUSIVITY_SCALING
        diameter = diameter * DIAMETER_SCALING
        mu_normalized = (mu + jnp.pi) / (2 * jnp.pi)
        return jnp.concatenate([jnp.array([lambda_par, diameter]), mu_normalized])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the model parameters."""
        lambda_par = theta[0] / DIFFUSIVITY_SCALING
        diameter = theta[1] / DIAMETER_SCALING
        mu_normalized = theta[2:] * 2 * jnp.pi - jnp.pi
        return mu_normalized, lambda_par, diameter
