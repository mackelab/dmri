"""Core abstractions for diffusion MRI compartments.

``SignalCompartment`` defines the log-signal interface :math:`\\log S(b,\\mathbf{g})`,
while ``NoiseCompartment`` adds likelihoods. ``to_theta`` / ``from_theta`` keep model
parameters Gaussian in optimization space for stable inference.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array
from jax.typing import ArrayLike

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
    def to_theta(cls, *kwargs) -> Array:
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
    def from_theta(cls, theta: ArrayLike, **kwargs) -> "Compartment":
        """Creates a compartment from the optimization parameters."""
        args = cls.to_params(theta, **kwargs)
        return cls(*args)

    def tree_flatten(self) -> tuple:
        """Flattens the compartment into a list of children and auxiliary data."""
        theta = self.theta
        return (theta,), (type(self),)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: list) -> "Compartment":
        """Reconstructs the compartment from the list of children and auxiliary data."""
        return cls.from_theta(children[0])


class SharedParameterState(Compartment):
    share_with_compartments: dict[type, list[int]]

    def __init__(self, shared_parameters: ArrayLike):
        self.shared_parameters = jnp.atleast_1d(shared_parameters)

    def set_shared_params_for_compartment(
        self, compartment_type: type
    ) -> "Compartment":
        """Returns the parameters for the compartment."""
        if (
            hasattr(compartment_type, "from_global_params")
            and compartment_type in self.share_with_compartments
        ):
            idx = self.share_with_compartments[compartment_type]
            return compartment_type.from_global_params(self.shared_parameters, idx)
        else:
            return compartment_type


class SignalCompartment(Compartment):
    @classmethod
    def signal_fn(cls, acq: acquisition_scheme, *args, **kwargs) -> ArrayLike:
        """Computes the signal for the compartment."""
        return jnp.exp(cls.log_signal_fn(acq, *args, **kwargs))

    @classmethod
    @abstractmethod
    def log_signal_fn(cls, acq: acquisition_scheme, *args, **kwargs) -> ArrayLike:
        """Computes the log-signal for the compartment."""
        pass

    def signal(self, acq: acquisition_scheme, rng=None) -> ArrayLike:
        """Simulates the signal for the compartment."""
        return type(self).signal_fn(acq, rng=rng, **self.params)

    def log_signal(self, acq: acquisition_scheme, rng=None) -> ArrayLike:
        """Simulates the log-signal for the compartment."""
        return type(self).log_signal_fn(acq, rng=rng, **self.params)

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> Any:
        """Fits the compartment to the signal deterministically."""
        raise NotImplementedError("Fitting not implemented")

    def to_fod(self):
        """Converts the signal compartment to a fiber orientation distribution compartment."""
        raise NotImplementedError("Conversion to FOD not implemented")


class NoiseCompartment(Compartment):
    @abstractmethod
    def noise(self, signal, rng: jax.random.key) -> ArrayLike:
        """Simulates the noise for the compartment."""
        pass

    @abstractmethod
    def log_likelihood(self, signal_pred, signal_obs) -> Array:
        pass
