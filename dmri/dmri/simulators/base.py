from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike
import jax.tree_util as jtu


class Compartment(ABC):
    theta_dim: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_node_class(cls)

    @property
    def params(self):
        return self.__dict__

    @property
    def theta(self):
        return self.to_theta(**self.params)

    @classmethod
    @abstractmethod
    def to_theta(cls, *kwargs) -> ArrayLike:
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
    def from_theta(cls, theta: ArrayLike, **kwargs) -> "ModelCompartment":
        """Creates a compartment from the optimization parameters."""
        args = cls.to_params(theta, **kwargs)
        return cls(*args)

    def tree_flatten(self) -> tuple:
        """Flattens the compartment into a list of children and auxiliary data."""
        theta = self.theta
        return (theta,), (type(self),)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: list) -> "ModelCompartment":
        """Reconstructs the compartment from the list of children and auxiliary data."""
        return cls.from_theta(children[0])


class ModelCompartment(Compartment):
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

    def fod_logpdf(self, mu: ArrayLike) -> float:
        """Computes the log-probability of the orientation distribution function."""
        pass

    def fod_sample(self, rng: jax.random.key) -> ArrayLike:
        """Samples the fiber orientation distribution."""
        raise NotImplementedError("FOD sampling not implemented")


class NoiseCompartment(Compartment):
    @abstractmethod
    def noise(self, rng: jax.random.key) -> ArrayLike:
        """Simulates the noise for the compartment."""
        pass
