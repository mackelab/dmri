from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment
from dmri.utils.dmriutils import (
    fit_diffusion_tensor_linearized,
    cartesian_to_unitsphere,
    unitsphere_to_cartesian,
)


class Ball(SignalCompartment):
    theta_dim: int = 1
    lam_min: float = 0.0
    lam_max: float = 0.05

    def __init__(self, lam: float) -> None:
        """Initialize the Ball model with a lambda value."""
        self.lam = lam

    @classmethod
    def log_signal_fn(
        cls, bvals: ArrayLike, bvecs: ArrayLike, lam: float, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = -bvals * lam
        return logS


    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)

    @classmethod
    def to_theta(cls, lam: ArrayLike) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the Ball model to the log signal and b-values."""
        lam = -logS / bvals
        return (jnp.mean(lam),)

    def fod_logpdf(self, mu):
        """Convert the Ball model to the Orientation Distribution Function (ODF)."""
        return 1 / (4 * jnp.pi)

    def fod_sample(self, rng: Any):
        """Sample from the Orientation Distribution Function (ODF)."""
        u = jax.random.normal(rng, (3,))
        u = u / jnp.linalg.norm(u)
        return u


class Stick(SignalCompartment):
    theta_dim = 3
    min_lam = 0.0
    max_lam = 0.05

    def __init__(self, lam: float, eigvec: ArrayLike) -> None:
        """Initialize the Stick model with a lambda value and eigenvector."""
        self.lam = lam
        self.eigvec = eigvec

    @classmethod
    def log_signal_fn(
        cls,
        bvals: ArrayLike,
        bvecs: ArrayLike,
        lam: float,
        eigvec: ArrayLike,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = -bvals * lam * (jnp.sum(bvecs * eigvec, axis=-1)) ** 2
        return logS

    @classmethod
    def to_theta(cls, lam: float, eigvec: ArrayLike) -> ArrayLike:
        """Convert the lambda value and eigenvector to the parameter space theta."""
        angle1 = jnp.arccos(eigvec[2]) / jnp.pi
        angle2 = jnp.arctan2(eigvec[1], eigvec[0])
        angle2 = (angle2 + jnp.pi) / (2 * jnp.pi)
        lam = (lam - cls.min_lam) / (cls.max_lam - cls.min_lam)

        theta = jnp.stack([lam, angle1, angle2])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value and eigenvector."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        angle1 = theta[1] * jnp.pi
        angle2 = theta[2] * 2 * jnp.pi - jnp.pi
        x = jnp.sin(angle1) * jnp.cos(angle2)
        y = jnp.sin(angle1) * jnp.sin(angle2)
        z = jnp.cos(angle1)
        eigvec = jnp.array([x, y, z])
        return lam, eigvec

    def fod_logpdf(self, u, kappa: float = 100):
        """Convert the Stick model to the Orientation Distribution Function (ODF)."""
        # Compute the ODF for a given direction u
        dot_product = jnp.dot(self.eigvec, u)

        # Compute the normalization constant for the vMF distribution in 3D.
        # Note: sinh(kappa) = (exp(kappa) - exp(-kappa))/2.
        c = kappa / (4 * jnp.pi * jnp.sinh(kappa))
        log_pdf1 = kappa * dot_product + jnp.log(c) + jnp.log(0.5)
        log_pdf2 = -kappa * dot_product + jnp.log(c) + jnp.log(0.5)
        log_prob = jax.scipy.special.logsumexp(jnp.stack([log_pdf1, log_pdf2]), axis=0)
        return log_prob

    def fod_sample(self, rng: Any, tol: float = 1e-6):
        """Sample from the Orientation Distribution Function (ODF)."""
        # Sample a direction from the ODF
        del rng, tol
        return self.eigvec

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the Stick model to the log signal, b-values, and b-vectors."""
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam = eigvals[idx]
        eigvec = eigvecs[:, idx]
        return lam, eigvec


class Zeppelin(SignalCompartment):
    theta_dim = 4
    min_lam = 0.0
    max_lam = 0.05

    def __init__(self, mu: ArrayLike, lambda_par: float, lambda_perp: float) -> None:
        """Initialize the Zeppelin model with orientation and diffusivity parameters."""
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    @classmethod
    def log_signal_fn(
        cls,
        bvals: ArrayLike,
        bvecs: ArrayLike,
        mu: ArrayLike,
        lambda_par: float,
        lambda_perp: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        mu_cartesian = unitsphere_to_cartesian(mu)
        D = lambda_par * jnp.outer(mu_cartesian, mu_cartesian) + lambda_perp * (
            jnp.eye(3) - jnp.outer(mu_cartesian, mu_cartesian)
        )
        logS = -bvals * jnp.einsum("...i,ij,...j->...", bvecs, D, bvecs)
        return logS

    @classmethod
    def to_theta(
        cls, mu: ArrayLike, lambda_par: float, lambda_perp: float
    ) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        lambda_par = (lambda_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
        lambda_perp = (lambda_perp - cls.min_lam) / (cls.max_lam - cls.min_lam)
        mu_normalized = (mu + jnp.pi) / (2 * jnp.pi)
        theta = jnp.concatenate([jnp.array([lambda_par, lambda_perp]), mu_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the model parameters."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lambda_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        lambda_perp = theta[1] * (cls.max_lam - cls.min_lam) + cls.min_lam
        mu_normalized = theta[2:] * 2 * jnp.pi - jnp.pi
        return mu_normalized, lambda_par, lambda_perp

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the Zeppelin model to the log signal, b-values, and b-vectors."""
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lambda_par = eigvals[idx]
        lambda_perp = jnp.mean(jnp.delete(eigvals, idx))
        mu_cartesian = eigvecs[:, idx]
        mu = self._cartesian_to_unitsphere(mu_cartesian)
        return mu, lambda_par, lambda_perp



class Dti(SignalCompartment):
    theta_dim: int = 6
    theta_scale = 0.05

    def __init__(self, D: ArrayLike) -> None:
        """Initialize the DTI model with a diffusion tensor D."""
        self.D = D

    @classmethod
    def log_signal_fn(
        cls, bvals: ArrayLike, bvecs: ArrayLike, D: ArrayLike, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = -bvals * jnp.einsum("bi,ij,bj->b", bvecs, D, bvecs)
        return logS


    @classmethod
    def to_theta(cls, D: ArrayLike) -> ArrayLike:
        """Convert the diffusion tensor D to the parameter space theta."""
        D = D - cls.theta_scale**2 * jnp.eye(3)
        L = jnp.linalg.cholesky(D)
        theta = L[jnp.tril_indices(3)]
        theta = theta / cls.theta_scale
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the diffusion tensor D."""
        theta = cls.theta_scale * theta
        D = jnp.zeros((3, 3))
        D = D.at[jnp.tril_indices(3)].set(theta)
        D = D @ D.T + cls.theta_scale**2 * jnp.eye(3)
        return (D,)

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the DTI model to the log signal, b-values, and b-vectors."""
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        return (D,)
