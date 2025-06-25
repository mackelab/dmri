import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment, acquisition_scheme
from dmri.simulators.sphereical_distributions import TensorFOD
from dmri.utils.dmriutils import fit_diffusion_tensor_linearized


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
        cls, acq: acquisition_scheme, D: ArrayLike, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        bvals = acq.bvals
        bvecs = acq.bvecs
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

    def fit(self, logS: ArrayLike, acq: acquisition_scheme) -> tuple:
        """Fit the DTI model to the log signal, b-values, and b-vectors."""
        bvals = acq.bvals
        bvecs = acq.bvecs
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        return (D,)

    def to_fod(self):
        eigvals, eigvecs = jnp.linalg.eigh(self.D)
        return TensorFOD(eigvecs, eigvals)
