from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike
from jax import tree_util as jtu
from dmri.simulators.base import ModelCompartment
from dmri.simulators.sphereical_distributions import Watson, SphericalDistribution
from dmri.simulators.local_signal_models.gaussian_models import Stick


class DistributionalModel(ModelCompartment):
    fod: SphericalDistribution
    signal_model: ModelCompartment

    def __init_subclass__(cls):
        assert hasattr(cls, "fod"), "fod not defined"
        assert hasattr(cls, "signal_model"), "signal_model not defined"

        cls.theta_dim = cls.fod.theta_dim + sum([m.theta_dim for m in cls.signal_model])
        jtu.register_pytree_node_class(cls)

    def __init__(self, fod, signal_model):
        self.fod = fod
        self.signal_model = signal_model

        assert issubclass(type(fod), SphericalDistribution), "Wrong fod"
        assert issubclass(type(signal_model), ModelCompartment), "Wrong signal model"

    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike, rng=None) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        return self.signal_model.log_signal(bvals, bvecs, rng=rng)
