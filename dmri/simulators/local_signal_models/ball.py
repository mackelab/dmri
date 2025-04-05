import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment, acquisition_scheme
from dmri.simulators.sphereical_distributions import Uniform


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
