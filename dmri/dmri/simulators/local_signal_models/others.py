from dmri.utils.dmriutils import fit_diffusion_tensor_linearized
import jax.numpy as jnp
import jax

from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment


class TemporalZeppelin(SignalCompartment):
    theta_dim = 4
    min_lam = 0.1
    max_lam = 3.0

    def __init__(self, lam_parallel, D_inf, A, eigenvecs) -> None:
        self.lam_parallel = lam_parallel
        self.D_inf = D_inf
        self.A = A
        self.eigenvecs = eigenvecs

    def log_signal(self, bvals, bvecs, delta, Delta, rng):
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
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam_parallel = eigvals[idx]
        lam_orthogonal = jnp.mean(jnp.delete(eigvals, idx))
        D_inf = (
            lam_orthogonal
            - self.A * jnp.log(Delta / delta)
            - 3 / 2 * (Delta - delta / 3)
        )
        return lam_parallel, D_inf, self.A, eigvecs

    @classmethod
    def to_theta(cls, lam_parallel, D_inf, A, eigenvecs):
        lam_parallel = (lam_parallel - cls.min_lam) / (cls.max_lam - cls.min_lam)
        D_inf = (D_inf - cls.min_lam) / (cls.max_lam - cls.min_lam)
        theta = jnp.concatenate(
            [jnp.array([lam_parallel, D_inf, A]), eigenvecs.flatten()]
        )
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta):
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_parallel = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        D_inf = theta[1] * (cls.max_lam - cls.min_lam) + cls.min_lam
        A = theta[2]
        eigenvecs = theta[3:].reshape(3, 3)
        return lam_parallel, D_inf, A, eigenvecs
