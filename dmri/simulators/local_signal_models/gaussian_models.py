from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment, acquisition_scheme
from dmri.utils.dmriutils import (
    fit_diffusion_tensor_linearized,
    cartesian_to_unitsphere,
    unitsphere_to_cartesian,
)
from dmri.simulators.sphereical_distributions import (
    Uniform,
    SymmetricDirac,
    Tensor2dFOD,
    TensorFOD,
)

class Dot(SignalCompartment):
    """The Dot model is a simple model that represents a single point in space.
    It has no parameters and the signal is constant across all b-values and b-vectors.
    It represent trapped water molecules in the tissue.
    """

    theta_dim: int = 0

    def __init__(self):
        super().__init__()

    @classmethod
    def log_signal_fn(
        cls, aquisition_scheme: acquisition_scheme, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = jnp.zeros(aquisition_scheme.bvals.shape)
        return logS

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        return ()

    @classmethod
    def to_theta(cls) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        return jnp.array([])

    def to_fod(self):
        return Uniform()


class Ball(SignalCompartment):
    """The Ball model is a simple model that represents free water diffusion in
    unrestricted space. It is fully isotropic.
    It has a single parameter lambda that represents the diffusivity of water molecules.
    """

    theta_dim: int = 1
    lam_min: float = 0.0
    lam_max: float = 0.01  # 1e-3 mm^2/s is the diffusivity of free water

    def __init__(self, lam: float) -> None:
        """Initialize the Ball model with a lambda value."""
        self.lam = lam

    @classmethod
    def log_signal_fn(
        cls, aquisition_scheme: acquisition_scheme, lam: float, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        logS = -aquisition_scheme.bvals * lam
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

    def to_fod(self):
        return Uniform()


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
        mu0_normalized = mu[0] / jnp.pi
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([lam_par, mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value and eigenvector."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        mu1 = theta[1] * jnp.pi
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
        mu1_normalized = mu[0] / jnp.pi
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
        mu1 = theta[2] * jnp.pi
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
        mu = self._cartesian_to_unitsphere(mu_cartesian)
        return mu, lambda_par, lambda_perp

    def to_fod(self):
        mu = unitsphere_to_cartesian(self.mu)
        D = self.lambda_par * jnp.outer(mu, mu) + self.lambda_perp * (
            jnp.eye(3) - jnp.outer(mu, mu)
        )
        eigvals, eigvecs = jnp.linalg.eigh(D)
        return TensorFOD(eigvecs,eigvals)


class Dti(SignalCompartment):
    """The Diffusion Tensor Imaging (DTI) model represents the diffusion of water
    molecules in the brain. It is a simple model that assumes Gaussian diffusion.
    """
    theta_dim: int = 6
    D_scale = 0.001
    min_lam: float = 0.0001

    def __init__(self, D: ArrayLike) -> None:
        """Initialize the DTI model with a diffusion tensor D."""
        self.D = D

    @classmethod
    def log_signal_fn(
        cls, aquisition_scheme: acquisition_scheme, D: ArrayLike, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        logS = -bvals * jnp.einsum("bi,ij,bj->b", bvecs, D, bvecs)
        return logS


    @classmethod
    def to_theta(cls, D: ArrayLike) -> ArrayLike:
        """Convert the diffusion tensor D to the parameter space theta."""
        D = D - cls.min_lam * jnp.eye(3)
        D = D / cls.D_scale
        D_L = jnp.linalg.cholesky(D)
        theta = D_L[jnp.tril_indices(3)]
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the diffusion tensor D."""
        theta = theta
        D = jnp.zeros((3, 3))
        D = D.at[jnp.tril_indices(3)].set(theta)
        D = D @ D.T
        D = cls.D_scale * D
        D += cls.min_lam * jnp.eye(3)
        return (D,)

    def fit(self, logS: ArrayLike, aquisition_scheme: acquisition_scheme) -> tuple:
        """Fit the DTI model to the log signal, b-values, and b-vectors."""
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        return (D,)

    def to_fod(self):
        eigvals, eigvecs = jnp.linalg.eigh(self.D)
        return TensorFOD(eigvecs,eigvals)
