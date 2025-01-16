from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike

from dmri.dmri_models.local_models.base import Compartment
from dmri.dmri_models.local_models.utils import fit_diffusion_tensor_linearized


class Ball(Compartment):
    theta_dim = 1
    lam_min = 0.0
    lam_max = 0.05

    def __init__(self, lam: float) -> None:
        """Initialize the Ball model with a lambda value."""
        self.lam = lam

    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = -bvals * self.lam
        return logS

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the Ball model to the log signal and b-values."""
        lam = -logS / bvals
        return (jnp.mean(lam),)

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)

    @classmethod
    def to_theta(cls, lam: float) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta


class Stick(Compartment):
    theta_dim = 3
    min_lam = 0.0
    max_lam = 0.05

    def __init__(self, lam: float, eigvec: ArrayLike) -> None:
        """Initialize the Stick model with a lambda value and eigenvector."""
        self.lam = lam
        self.eigvec = eigvec

    def signal(self, bvals: ArrayLike, bvecs: ArrayLike) -> ArrayLike:
        """Compute the signal for given b-values and b-vectors."""
        S = jnp.exp(-bvals * self.lam * (jnp.sum(bvecs * self.eigvec, axis=-1)) ** 2)
        return S

    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = -bvals * self.lam * (jnp.sum(bvecs * self.eigvec, axis=-1)) ** 2
        return logS

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the Stick model to the log signal, b-values, and b-vectors."""
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam = eigvals[idx]
        eigvec = eigvecs[:, idx]
        return lam, eigvec

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


class Zeppelin(Compartment):
    def __init__(
        self, lam_parallel: float, lam_orthogonal: float, eigenvecs: ArrayLike
    ) -> None:
        """Initialize the Zeppelin model with parallel and orthogonal lambda values and eigenvectors."""
        self.lam_parallel = lam_parallel
        self.lam_orthogonal = lam_orthogonal
        self.eigenvecs = eigenvecs

    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        Lam = jnp.diag(
            jnp.concatenate(
                [self.lam_parallel, self.lam_orthogonal, self.lam_orthogonal]
            )
        )
        D = self.eigenvecs @ Lam @ jnp.linalg.inv(self.eigenvecs)
        logS = -bvals * jnp.einsum("bi,ij,bj->b", bvecs, D, bvecs)
        return logS

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the Zeppelin model to the log signal, b-values, and b-vectors."""
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam_parallel = eigvals[idx]
        lam_orthogonal = jnp.mean(jnp.delete(eigvals, idx))
        eigenvecs = eigvecs
        return lam_parallel, lam_orthogonal, eigenvecs


class Dti(Compartment):
    theta_dim: int = 6
    theta_scale = 0.05

    def __init__(self, D: ArrayLike) -> None:
        """Initialize the DTI model with a diffusion tensor D."""
        self.D = D

    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = -bvals * jnp.einsum("bi,ij,bj->b", bvecs, self.D, bvecs)
        return logS

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the DTI model to the log signal, b-values, and b-vectors."""
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        return (D,)

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
        D = D + jnp.tril(D, -1).T
        D = D @ D.T + cls.theta_scale**2 * jnp.eye(3)
        return (D,)
