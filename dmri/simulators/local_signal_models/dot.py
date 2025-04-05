import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment, acquisition_scheme
from dmri.simulators.sphereical_distributions import Uniform


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
