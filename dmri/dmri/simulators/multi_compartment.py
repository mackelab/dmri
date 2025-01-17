from dmri.dmri.simulators.local_models.gaussian_models import Ball, Stick
from dmri.dmri.simulators.noise_models import GaussianNoise
from dmri.simulators.base import ModelCompartment, NoiseCompartment

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike


class MultiCompartment(ModelCompartment):
    model_types: list
    noise_types: list

    def __init_subclass__(cls):
        assert hasattr(cls, "model_types"), "model_types not defined"
        assert hasattr(cls, "noise_types"), "noise_types not defined"

        cls.theta_dim = (
            len(cls.model_types)
            + sum([m.theta_dim for m in cls.model_types])
            + sum([n.theta_dim for n in cls.noise_types])
        )

    def __init__(
        self,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
    ):
        self.model_fractions = model_fractions
        self.model_compartments = model_compartments
        self.noise_compartments = noise_compartments

        assert len(model_compartments) == len(model_fractions), (
            "Wrong number of fractions"
        )
        assert [type(m) for m in model_compartments] == self.model_types, "Wrong model"
        assert [type(m) for m in noise_compartments] == self.noise_types, "Wrong noise"

    def signal(self, bvals, bvecs, rng):
        # Compute the signal for each compartment
        signals = jnp.stack([m.signal(bvals, bvecs) for m in self.model_compartments])
        fractions = self.model_fractions[None, ...]
        # Combine signals with sum
        signal = jnp.sum(signals * fractions, axis=0)

        # Add noise
        rngs = jax.random.split(rng, len(self.noise_compartments))
        for noise, rng in zip(self.noise_compartments, rngs):
            signal += noise.signal(bvals, bvecs, rng)
        return signal

    def log_signal(self, bvals, bvecs, rng):
        return jnp.log(self.signal(bvals, bvecs, rng))

    @classmethod
    def to_theta(
        cls,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
    ):
        theta_fraction = model_fractions
        theta_model = jnp.concatenate([m.to_theta() for m in model_compartments])
        theta_noise = jnp.concatenate([m.to_theta() for m in noise_compartments])
        return jnp.concatenate([theta_fraction, theta_model, theta_noise])

    @classmethod
    def to_params(
        cls,
        theta: ArrayLike,
    ):
        theta_dims_fractions = [len(cls.model_compartments)]
        theta_dims_model = [m.theta_dim for m in cls.model_compartments]
        theta_dims_noise = [m.theta_dim for m in cls.noise_compartments]
        total_dims = theta_dims_fractions + theta_dims_model + theta_dims_noise

        # Split theta into fractions, model, and noise
        thetas_split = jnp.split(theta, total_dims)
        fractions = thetas_split[0]
        model_thetas = thetas_split[1 : len(cls.model_compartments) + 1]
        noise_thetas = thetas_split[len(cls.model_compartments) + 1 :]

        # Create model compartments
        model_compartments = [
            m.from_theta(t) for m, t in zip(cls.model_compartments, model_thetas)
        ]
        noise_compartments = [
            m.from_theta(t) for m, t in zip(cls.noise_compartments, noise_thetas)
        ]
        return fractions, model_compartments, noise_compartments


class BallStick(MultiCompartment):
    model_types = [Ball, Stick]
    noise_types = [GaussianNoise]

    def __init__(
        self,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
    ):
        super().__init__(
            model_fractions=model_fractions,
            model_compartments=model_compartments,
            noise_compartments=noise_compartments,
        )


class Ball2Stick(MultiCompartment):
    model_types = [Ball, Stick, Stick]
    noise_types = [GaussianNoise]

    def __init__(
        self,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
    ):
        super().__init__(
            model_fractions=model_fractions,
            model_compartments=model_compartments,
            noise_compartments=noise_compartments,
        )


class Ball3Stick(MultiCompartment):
    model_types = [Ball, Stick, Stick, Stick]
    noise_types = [GaussianNoise]

    def __init__(
        self,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
    ):
        super().__init__(
            model_fractions=model_fractions,
            model_compartments=model_compartments,
            noise_compartments=noise_compartments,
        )
