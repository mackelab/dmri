import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment, acquisition_scheme
from dmri.simulators.sphereical_distributions import SymmetricDirac
from dmri.utils.dmriutils import (
    cartesian_to_unitsphere,
    fit_diffusion_tensor_linearized,
    unitsphere_to_cartesian,
)


class Stick(SignalCompartment):
    """The Stick model represents a single fiber bundle with a fixed orientation i.e.
    a cylinder with zero radius.

    It represents fully anisotropic diffusion along the fiber orientation.
    """

    theta_dim = 3
    min_lam = 0.0
    max_lam = 0.01

    def __init__(self, mu: ArrayLike, lam_par: float) -> None:
        """Initialize the Stick model with a lambda value and eigenvector."""
        self.mu = mu
        self.lam_par = lam_par

    @classmethod
    def log_signal_fn(
        cls,
        aquisition_scheme: acquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        mu_cart = unitsphere_to_cartesian(mu)
        logS = -bvals * lam_par * (jnp.sum(bvecs * mu_cart, axis=-1)) ** 2
        return logS

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        lam_par = (lam_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([lam_par, mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value and eigenvector."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        mu1 = jnp.arccos(1 - theta[1])  # Ensures output is in upper hemisphere
        mu2 = theta[2] * 2 * jnp.pi - jnp.pi

        mu = jnp.array([mu1, mu2])

        return mu, lam_par

    def to_fod(self):
        mu_cart = unitsphere_to_cartesian(self.mu)
        return SymmetricDirac(mu_cart)

    def fit(self, logS: ArrayLike, aquisition_scheme: acquisition_scheme) -> tuple:
        """Fit the Stick model to the log signal, b-values, and b-vectors."""
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam_par = eigvals[idx]
        eigvec = eigvecs[:, idx]
        mu = cartesian_to_unitsphere(eigvec)
        return mu, lam_par
