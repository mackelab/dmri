from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike
import jax.tree_util as jtu

from dmri.simulators.acquisition_scheme import acquisition_scheme


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
    def from_theta(cls, theta: ArrayLike, **kwargs) -> "SignalCompartment":
        """Creates a compartment from the optimization parameters."""
        args = cls.to_params(theta, **kwargs)
        return cls(*args)

    def tree_flatten(self) -> tuple:
        """Flattens the compartment into a list of children and auxiliary data."""
        theta = self.theta
        return (theta,), (type(self),)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: list) -> "SignalCompartment":
        """Reconstructs the compartment from the list of children and auxiliary data."""
        return cls.from_theta(children[0])


class SignalCompartment(Compartment):

    @classmethod
    def signal_fn(
        cls, acquisition_scheme: acquisition_scheme, *args, **kwargs
    ) -> ArrayLike:
        """Computes the signal for the compartment."""
        return jnp.exp(cls.log_signal_fn(acquisition_scheme, *args, **kwargs))

    @classmethod
    @abstractmethod
    def log_signal_fn(
        cls, aquisition_scheme: acquisition_scheme, *args, **kwargs
    ) -> ArrayLike:
        """Computes the log-signal for the compartment."""
        pass

    def signal(self, aquisition_scheme: acquisition_scheme, rng=None) -> ArrayLike:
        """Simulates the signal for the compartment."""
        return type(self).signal_fn(aquisition_scheme, rng=rng, **self.params)

    def log_signal(self, aquisition_scheme: acquisition_scheme, rng=None) -> ArrayLike:
        """Simulates the log-signal for the compartment."""
        return type(self).log_signal_fn(aquisition_scheme, rng=rng, **self.params)

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> Any:
        """Fits the compartment to the signal deterministically."""
        raise NotImplementedError("Fitting not implemented")

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
