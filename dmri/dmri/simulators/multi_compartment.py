from typing import Optional

from dmri.simulators.noise_compartments import (
    LowGaussianNoise,
    LowRicianNoise,
    MediumGaussianNoise,
    MediumRicianNoise,
    LargeGaussianNoise,
    LargeRicianNoise,
    VeryLargeGaussianNoise,
    VeryLargeRicianNoise,
)
from dmri.simulators.local_signal_models.gaussian_models import (
    Ball,
    Stick,
    Zeppelin,
    Dti,
)
from dmri.simulators.base import SignalCompartment, NoiseCompartment
from dmri.utils.transform import normal_to_dirichlet, dirichlet_to_normal


import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu

from jax.typing import ArrayLike


class MultiCompartment(SignalCompartment):
    model_types: list
    noise_types: list
    fraction_prior: ArrayLike  # Dirichelt alpha values

    def __init_subclass__(cls):
        assert hasattr(cls, "model_types"), "model_types not defined"
        assert hasattr(cls, "noise_types"), "noise_types not defined"
        assert hasattr(cls, "fraction_prior"), "fraction_prior not defined"

        assert len(cls.model_types) == len(cls.fraction_prior), (
            "Wrong number of fractions"
        )

        cls.theta_dim = (
            len(cls.model_types)
            - 1
            + sum([m.theta_dim for m in cls.model_types])
            + sum([n.theta_dim for n in cls.noise_types])
        )

        jtu.register_pytree_node_class(cls)

    def __init__(
        self,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
        model_mask: Optional[ArrayLike] = None,
    ):
        self.model_fractions = model_fractions
        self.model_compartments = model_compartments
        self.noise_compartments = noise_compartments
        self.model_mask = model_mask

        assert len(model_compartments) == len(model_fractions), (
            "Wrong number of fractions"
        )
        assert [type(m) for m in model_compartments] == self.model_types, "Wrong model"
        assert [type(m) for m in noise_compartments] == self.noise_types, "Wrong noise"

    @classmethod
    def signal_fn(
        cls,
        bvals,
        bvecs,
        model_compartments,
        noise_compartments,
        model_fractions,
        model_mask,
        rng=None,
    ):
        # Compute the signal for each compartment
        signals = jnp.stack(
            [m.signal(bvals, bvecs) for m in model_compartments], axis=0
        )
        fractions = model_fractions[:, None]
        # Combine signals with sum
        signal = jnp.sum(signals * fractions, axis=0)

        # Add noise
        if len(noise_compartments) > 0:
            assert rng is not None, "rng key  must be provided for noise"
            rngs = jax.random.split(rng, len(noise_compartments))
            for i, (noise, rng) in enumerate(zip(noise_compartments, rngs)):
                if model_mask is not None:
                    idx = len(model_compartments) + i
                    mask = model_mask[idx]
                    signal = jax.lax.cond(
                        mask,
                        lambda x, rng: noise.noise(x, rng),
                        lambda x, rng: x,
                        signal,
                        rng,
                    )
                else:
                    signal = noise.noise(signal, rng)
        return signal

    @classmethod
    def log_signal_fn(cls, bvals, bvecs, **kwargs):
        return jnp.log(cls.signal_fn(bvals, bvecs, **kwargs))

    @classmethod
    def split_idx(cls):
        theta_dims_fractions = [len(cls.model_types) - 1]
        theta_dims_model = [m.theta_dim for m in cls.model_types]
        theta_dims_noise = [m.theta_dim for m in cls.noise_types]
        total_dims = theta_dims_fractions + theta_dims_model + theta_dims_noise
        total_dims = np.array(total_dims)

        return total_dims

    @classmethod
    def split_theta(cls, theta):
        total_dims = cls.split_idx()
        # Calculate the cumulative sum of total_dims to get the split indices
        split_indices = np.cumsum(total_dims)[:-1]

        # Split theta into fractions, model, and noise
        thetas_split = jnp.split(theta, split_indices)

        return thetas_split

    @classmethod
    def to_theta(
        cls,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
        model_mask: Optional[ArrayLike] = None,
    ):
        theta_fraction = model_fractions
        if model_mask is not None:
            component_mask = model_mask[: len(model_compartments)]
        else:
            component_mask = None
        theta_fraction = dirichlet_to_normal(
            cls.fraction_prior, theta_fraction, component_mask
        )
        if len(model_compartments) > 0:
            theta_model = jnp.concatenate([m.theta for m in model_compartments])
        else:
            theta_model = jnp.array([])
        if len(noise_compartments) > 0:
            theta_noise = jnp.concatenate([m.theta for m in noise_compartments])
        else:
            theta_noise = jnp.array([])
        return jnp.concatenate([theta_fraction, theta_model, theta_noise])

    @classmethod
    def to_params(
        cls,
        theta: ArrayLike,
        model_mask: Optional[ArrayLike] = None,
    ):
        thetas_split = cls.split_theta(theta)
        fractions = thetas_split[0]
        model_thetas = thetas_split[1 : len(cls.model_types) + 1]
        noise_thetas = thetas_split[len(cls.model_types) + 1 :]

        # Model fractions should sum to 1 and follow a Dirichlet distribution
        if model_mask is not None:
            component_mask = model_mask[: len(cls.model_types)]
        else:
            component_mask = None
        fractions = normal_to_dirichlet(cls.fraction_prior, fractions, component_mask)

        # Create model compartments
        model_compartments = [
            m.from_theta(t) for m, t in zip(cls.model_types, model_thetas)
        ]
        noise_compartments = [
            m.from_theta(t) for m, t in zip(cls.noise_types, noise_thetas)
        ]
        return fractions, model_compartments, noise_compartments, model_mask

    def fod_logpdf(self, mu):
        pass

    def fod_sample(self, rng):
        rng1, rng2 = jax.random.split(rng)
        idx = jax.random.choice(
            rng1, len(self.model_compartments), p=self.model_fractions
        )
        return jax.lax.switch(
            idx, [m.fod_sample for m in self.model_compartments], rng2
        )


class BallStick(MultiCompartment):
    model_types = [Ball, Stick]
    noise_types = []
    fraction_prior = jnp.ones(2)


class Ball2Stick(MultiCompartment):
    model_types = [Ball, Stick, Stick]
    noise_types = []
    fraction_prior = jnp.ones(3)


class Ball3Stick(MultiCompartment):
    model_types = [Ball, Stick, Stick, Stick]
    noise_types = []
    fraction_prior = jnp.ones(4)


class BallStickZeppelin(MultiCompartment):
    model_types = [Ball, Stick, Zeppelin]
    noise_types = []
    fraction_prior = jnp.ones(3)


class Ball2Stick2Zeppelin2Dti(MultiCompartment):
    model_types = [
        Ball,
        Stick,
        Stick,
        Zeppelin,
        Zeppelin,
        Dti,
        Dti,
    ]
    noise_types = []
    fraction_prior = jnp.ones(7)


class AllGaussianModels(MultiCompartment):
    model_types = [Ball] + 3 * [Stick] + 3 * [Zeppelin] + 3 * [Dti]
    noise_types = [
        LowGaussianNoise,
        MediumGaussianNoise,
        LargeGaussianNoise,
        VeryLargeGaussianNoise,
    ] + [LowRicianNoise, MediumRicianNoise, LargeRicianNoise, VeryLargeRicianNoise]
    fraction_prior = jnp.ones(1 + 3 + 3 + 3)
