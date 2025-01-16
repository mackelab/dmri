from dmri.dmri_models.local_models.utils import fit_diffusion_tensor_linearized
import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from jax.core import Array

from dmri.dmri_models.local_models.base import Compartment


class TemporalZeppelin(Compartment):
    theta_dim = 4

    def __init__(self, lam_parallel, D_inf, A, eigenvecs) -> None:
        self.lam_parallel = lam_parallel
        self.D_inf = D_inf
        self.A = A
        self.eigenvecs = eigenvecs

    def log_signal(self, bvals, bvecs, delta, Delta):
        lam_orthogonal = (
            self.D_inf + self.A * jnp.log(Delta / delta) + 3 / 2 * (Delta - delta / 3)
        )
        Lam = jnp.diag(
            jnp.concatenate([self.lam_parallel, lam_orthogonal, lam_orthogonal])
        )
        D = self.eigenvecs @ Lam @ jnp.linalg.inv(self.eigenvecs)
        logS = -bvals * jnp.einsum("bi,ij,bj->b", bvecs, D, bvecs)
        return logS

    def fit(self, logS, bvals, bvecs, delta, Delta):
        pass

    @classmethod
    def to_theta(cls, lam_parallel, D_inf, A, eigenvecs):
        theta = jnp.concatenate([lam_parallel, D_inf, A, eigenvecs.flatten()])
        return theta

    @classmethod
    def to_params(cls, theta):
        lam_parallel = theta[0]
        D_inf = theta[1]
        A = theta[2]
        eigenvecs = theta[3:].reshape(3, 3)
        return lam_parallel, D_inf, A, eigenvecs


class Cylinder(Compartment):
    theta_dim = 2
    min_lam = 0.0
    max_lam = 0.05

    def __init__(self, lam, radius) -> None:
        self.lam = lam
        self.radius = radius

    def log_signal(self, bvals, bvecs):
        q = jnp.sqrt(bvals / self.lam) / (2 * jnp.pi)
        J1 = jax.scipy.special.j1(2 * jnp.pi * q * self.radius)
        E_perp = (J1 / (jnp.pi * q * self.radius)) ** 2
        logS = -bvals * self.lam * E_perp
        return logS

    def fit(self, logS, bvals, bvecs):
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam = eigvals[idx]
        radius = jnp.sqrt(jnp.mean(eigvals) / (jnp.pi * jnp.mean(bvals)))
        return lam, radius

    @classmethod
    def to_theta(cls, lam, radius):
        lam = (lam - cls.min_lam) / (cls.max_lam - cls.min_lam)
        theta = jnp.array([lam, radius])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta):
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        radius = theta[1]
        return lam, radius
