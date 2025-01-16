from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike


class Compartment(ABC):
    theta_dim: int

    def signal(self, bvals: ArrayLike, bvecs: ArrayLike) -> ArrayLike:
        """Simulates the signal for the compartment."""
        return jnp.exp(self.log_signal(bvals, bvecs))

    @abstractmethod
    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike) -> ArrayLike:
        """Simulates the log-signal for the compartment."""
        pass

    @abstractmethod
    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> Any:
        """Fits the compartment to the signal deterministically."""
        pass

    @classmethod
    @abstractmethod
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        pass

    @classmethod
    @abstractmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        pass

    @classmethod
    def from_theta(cls, theta: ArrayLike) -> "Compartment":
        """Creates a compartment from the optimization parameters."""
        args = cls.to_params(theta)
        return cls(*args)
