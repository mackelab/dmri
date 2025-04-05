import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment, acquisition_scheme
from dmri.simulators.sphereical_distributions import TensorFOD
from dmri.utils.dmriutils import (
    cartesian_to_unitsphere,
    fit_diffusion_tensor_linearized,
    unitsphere_to_cartesian,
)


class Zeppelin(SignalCompartment):
    """The Zeppelin model [1]_ - an axially symmetric Tensor - typically used
    for extra-axonal diffusion.
    """

    theta_dim = 4
    min_lam = 0.0
    max_lam = 0.01

    def __init__(self, mu: ArrayLike, lambda_par: float, lambda_perp: float) -> None:
        """Initialize the Zeppelin model with orientation and diffusivity parameters."""
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    @classmethod
    def log_signal_fn(
        cls,
        acquisition_scheme: acquisition_scheme,
        mu: ArrayLike,
        lambda_par: float,
        lambda_perp: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        bvals = acquisition_scheme.bvals
        bvecs = acquisition_scheme.bvecs
        mu_cartesian = unitsphere_to_cartesian(mu)
        mu_perpendicular_plane = jnp.eye(3) - jnp.outer(mu_cartesian, mu_cartesian)
        magnitued_parallel = jnp.dot(bvecs, mu_cartesian)
        proj = jnp.einsum("...i,ij->...j", bvecs, mu_perpendicular_plane)
        magnitude_perpendicular = jnp.linalg.norm(proj, axis=-1)

        logS = -bvals * (
            lambda_par * magnitued_parallel**2
            + lambda_perp * magnitude_perpendicular**2
        )
        return logS

    @classmethod
    def to_theta(
        cls, mu: ArrayLike, lambda_par: float, lambda_perp: float
    ) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        lambda_par = (lambda_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
        lambda_perp = (lambda_perp - cls.min_lam) / (cls.max_lam - cls.min_lam)
        # Uniform sampling on the upper hemisphere
        mu1_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu2_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([lambda_par, lambda_perp, mu1_normalized, mu2_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the model parameters."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lambda_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        lambda_perp = theta[1] * (cls.max_lam - cls.min_lam) + cls.min_lam
        mu1 = jnp.arccos(1 - theta[2])  # Ensures output is in upper hemisphere
        mu2 = theta[3] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu1, mu2])
        return mu, lambda_par, lambda_perp

    def fit(self, logS: ArrayLike, aquisition_scheme) -> tuple:
        """Fit the Zeppelin model to the log signal, b-values, and b-vectors."""
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lambda_par = eigvals[idx]
        lambda_perp = jnp.mean(jnp.delete(eigvals, idx))
        mu_cartesian = eigvecs[:, idx]
        mu = cartesian_to_unitsphere(mu_cartesian)
        return mu, lambda_par, lambda_perp

    def to_fod(self):
        mu = unitsphere_to_cartesian(self.mu)
        D = self.lambda_par * jnp.outer(mu, mu) + self.lambda_perp * (
            jnp.eye(3) - jnp.outer(mu, mu)
        )
        eigvals, eigvecs = jnp.linalg.eigh(D)
        return TensorFOD(eigvecs, eigvals)
