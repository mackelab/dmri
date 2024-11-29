import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod


class Compartment(ABC):
    theta_dim: int

    def signal(self, bvals, bvecs):
        return jnp.exp(self.log_signal(bvals, bvecs))

    @abstractmethod
    def log_signal(self, bvals, bvecs):
        pass

    @abstractmethod
    def fit(self, logS, bvals, bvecs):
        pass

    @classmethod
    @abstractmethod
    def to_theta(cls, *args):
        pass

    @classmethod
    @abstractmethod
    def to_params(cls, theta):
        pass

    @classmethod
    def from_theta(cls, theta):
        args = cls.to_params(theta)
        return cls(*args)


class Dti(Compartment):
    theta_dim = 6
    # lam_min = 0.0
    # lam_max = 0.05
    # eigvals_bias = jnp.array([-1.8459795e00, 1.2955656e-03, 1.8469909e00])
    theta_scale = 0.05

    def __init__(self, D) -> None:
        self.D = D

    def log_signal(self, bvals, bvecs):
        logS = -bvals * jnp.einsum("bi,ij,bj->b", bvecs, self.D, bvecs)
        return logS

    def fit(self, logS, bvals, bvecs):
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        return (D,)

    @classmethod
    def to_theta(cls, D):
        # eigvals, eigvecs = jnp.linalg.eigh(D)
        # Inv sigm
        # eigvals = eigvals - cls.lam_min
        # eigvals = eigvals / (cls.lam_max - cls.lam_min)
        # eigvals = jnp.log(eigvals / (1 - eigvals))
        # eigvals = eigvals + cls.eigvals_bias
        # D = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        D = D - cls.theta_scale**2 * jnp.eye(3)
        L = jnp.linalg.cholesky(D)
        theta = L[jnp.tril_indices(3)]
        theta = theta / cls.theta_scale
        return theta

    @classmethod
    def to_params(cls, theta):
        theta = cls.theta_scale * theta
        D = jnp.zeros((3, 3))
        D = D.at[jnp.tril_indices(3)].set(theta)
        D = D + jnp.tril(D, -1).T
        # eigvals, eigvecs = jnp.linalg.eigh(D)
        # eigvals = eigvals - cls.eigvals_bias
        # eigvals = jax.nn.sigmoid(eigvals) * (cls.lam_max - cls.lam_min) + cls.lam_min
        # D = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        D = D @ D.T + cls.theta_scale**2 * jnp.eye(3)
        return (D,)


class Ball(Compartment):
    theta_dim = 1
    lam_min = 0.0
    lam_max = 0.05

    def __init__(self, lam) -> None:
        self.lam = lam

    def log_signal(self, bvals, bvecs):
        logS = -bvals * self.lam
        return logS

    def fit(self, logS, bvals, bvecs):
        lam = -logS / bvals  # lam = -logS/bvals
        return (jnp.mean(lam),)  # We return the mean of the estimated lambda

    @classmethod
    def to_params(cls, theta):
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)

    @classmethod
    def to_theta(cls, lam):
        lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta


class Zeppelin(Compartment):
    def __init__(self, lam_parallel, lam_orthogonal, eigenvecs) -> None:
        self.lam_parallel = lam_parallel
        self.lam_orthogonal = lam_orthogonal
        self.eigenvecs = eigenvecs

    def log_signal(self, bvals, bvecs):
        Lam = jnp.diag(
            jnp.concatenate(
                [self.lam_parallel, self.lam_orthogonal, self.lam_orthogonal]
            )
        )
        D = self.eigenvecs @ Lam @ jnp.linalg.inv(self.eigenvecs)
        logS = -bvals * jnp.einsum("bi,ij,bj->b", bvecs, D, bvecs)
        return logS

    def fit(self, logS, bvals, bvecs):
        pass


class Stick(Compartment):
    theta_dim = 3
    min_lam = 0.0
    max_lam = 0.05

    def __init__(self, lam, eigvec) -> None:
        self.lam = lam
        self.eigvec = eigvec

    def signal(self, bvals, bvecs):
        S = jnp.exp(-bvals * self.lam * (jnp.sum(bvecs * self.eigvec, axis=-1)) ** 2)
        return S

    def log_signal(self, bvals, bvecs):
        logS = -bvals * self.lam * (jnp.sum(bvecs * self.eigvec, axis=-1)) ** 2
        return logS

    def fit(self, logS, bvals, bvecs):
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam = eigvals[idx]
        eigvec = eigvecs[:, idx]
        return lam, eigvec

    @classmethod
    def to_theta(cls, lam, eigvec):
        angle1 = jnp.arccos(eigvec[2]) / jnp.pi
        angle2 = jnp.arctan2(eigvec[1], eigvec[0])
        angle2 = (angle2 + jnp.pi) / (2 * jnp.pi)
        lam = (lam - cls.min_lam) / (cls.max_lam - cls.min_lam)

        theta = jnp.stack([lam, angle1, angle2])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta):
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        angle1 = theta[1] * jnp.pi
        angle2 = theta[2] * 2 * jnp.pi - jnp.pi
        x = jnp.sin(angle1) * jnp.cos(angle2)
        y = jnp.sin(angle1) * jnp.sin(angle2)
        z = jnp.cos(angle1)
        eigvec = jnp.array([x, y, z])
        return lam, eigvec


def fit_diffusion_tensor_linearized(logS, bvals, bvecs):
    Y = -logS
    B = bvals[:, None] * jnp.stack(
        [
            bvecs[:, 0] ** 2,
            2 * bvecs[:, 0] * bvecs[:, 1],
            bvecs[:, 1] ** 2,
            2 * bvecs[:, 0] * bvecs[:, 2],
            2 * bvecs[:, 1] * bvecs[:, 2],
            bvecs[:, 2] ** 2,
        ],
        axis=-1,
    )
    D_flat = jnp.linalg.lstsq(B, Y, rcond=None)[0]
    idx1, idx2 = jnp.tril_indices(3)
    D = jnp.zeros((3, 3))
    D = D.at[idx1, idx2].set(D_flat)
    D = D + jnp.tril(D, -1).T
    return D


def add_mri_noise(rng,signal, sigma_g):
    """
    Adds Rician noise to an MRI signal.

    Parameters:
    - signal: ndarray of the true signal magnitude (η)
    - sigma_g: standard deviation of the Gaussian noise (σ_g)

    Returns:
    - noisy_signal: ndarray of the signal with added Rician noise
    """
    # Generate Gaussian noise for real and imaginary parts
    noise_real = jax.random.normal(rng, shape=signal.shape) * sigma_g
    noise_imag = jax.random.normal(rng, shape=signal.shape) * sigma_g

    # Assume original signal is along the real axis
    signal_complex = signal + 0j

    # Add noise
    noisy_complex = signal_complex + noise_real + 1j * noise_imag

    # Compute magnitude
    noisy_signal = jnp.abs(noisy_complex)

    return noisy_signal
