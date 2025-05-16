from typing import Optional

from dmri.simulators.local_signal_models.ball import (
    MultiShellBall,
    MultiShellStaticBall,
)
from dmri.simulators.local_signal_models.stick import MultiShellStaticStick
import jax.numpy as jnp
import numpy as np
import jax
from jax import tree_util as jtu
from jax.typing import ArrayLike

from dmri.simulators import acquisition_scheme
from dmri.simulators.base import SignalCompartment, Compartment, SharedParameterState
from dmri.simulators.local_signal_models import (
    Ball,
    BinghamStick,
    BinghamZeppelin,
    Dti,
    NoddiB,
    NoddiW,
    SandiB,
    SandiW,
    Stick,
    WatsonStick,
    WatsonZeppelin,
    Zeppelin,
    StaticStick,
    StaticBall,
)
from dmri.simulators.noise_compartments import (
    GaussianNoiseSNR7080,
    GaussianNoiseSNR6070,
    GaussianNoiseSNR5060,
    GaussianNoiseSNR4050,
    GaussianNoiseSNR3040,
    GaussianNoiseSNR2030,
    GaussianNoiseSNR1020,
    GaussianNoiseSNR310,
    BoundedGaussianNoise,
    RicianNoiseSNR7080,
    RicianNoiseSNR6070,
    RicianNoiseSNR5060,
    RicianNoiseSNR4050,
    RicianNoiseSNR3040,
    RicianNoiseSNR2030,
    RicianNoiseSNR1020,
    RicianNoiseSNR310,
    BoundedRicianNoise,
)
from dmri.simulators.sphereical_distributions import MixtureOfFODs
from dmri.utils.transform import dirichlet_to_normal, normal_to_dirichlet


class MultiCompartment(SignalCompartment):
    model_types: list
    noise_types: list
    fraction_prior: ArrayLike  # Dirichelt alpha values
    shared_parameter_type: type[SharedParameterState] | None = None

    def __init_subclass__(cls):
        assert hasattr(cls, "model_types"), "model_types not defined"
        assert hasattr(cls, "noise_types"), "noise_types not defined"
        assert hasattr(cls, "fraction_prior"), "fraction_prior not defined"

        assert len(cls.model_types) == len(cls.fraction_prior), (
            "Wrong number of fractions"
        )

        model_fraction_theta_dim = len(cls.model_types) - 1
        shared_parameters_dim = (
            0
            if cls.shared_parameter_type is None
            else cls.shared_parameter_type.theta_dim
        )
        cls.theta_dim = (
            model_fraction_theta_dim
            + shared_parameters_dim
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
        shared_parameter: SharedParameterState | None = None,
    ):
        self.model_fractions = model_fractions
        self.model_compartments = model_compartments
        self.noise_compartments = noise_compartments
        self.model_mask = model_mask
        self.shared_parameter = shared_parameter
        assert len(model_compartments) == len(model_fractions), (
            "Wrong number of fractions"
        )
        # This will not work with current shared parameter state
        # assert [type(m) for m in model_compartments] == self.model_types, "Wrong model"
        assert [type(m) for m in noise_compartments] == self.noise_types, "Wrong noise"

    def get_all_params(self):
        all_params = {}
        all_params["model_fractions"] = self.model_fractions
        for m in self.model_compartments:
            all_params[m.__class__.__name__] = m.params
        for n in self.noise_compartments:
            all_params[n.__class__.__name__] = n.params
        all_params["model_mask"] = self.model_mask
        if self.shared_parameter is not None:
            all_params["shared_parameter"] = self.shared_parameter.params
        return all_params

    @classmethod
    def signal_fn(
        cls,
        aquisition_scheme: acquisition_scheme,
        model_compartments,
        noise_compartments,
        model_fractions,
        model_mask,
        shared_parameter,
        rng=None,
    ):
        del shared_parameter
        # Compute the signal for each compartment
        signals = jnp.stack(
            [m.signal(aquisition_scheme) for m in model_compartments], axis=0
        )
        fractions = model_fractions[:, None]
        # Combine signals with sum
        signal = jnp.sum(signals * fractions, axis=0)

        # Add noise
        if rng is not None and len(noise_compartments) > 0:
            if model_mask is None:
                # Select first noise compartment
                noise_model = noise_compartments[0]
                signal = noise_model.noise(signal, rng)
            else:
                # Make mask shape match noise compartments
                idx = len(model_compartments)
                noise_mask = model_mask[idx:]
                # Apply noise sequentially with where to avoid conditionals
                for i in range(len(noise_compartments)):
                    noised_signal = noise_compartments[i].noise(signal, rng)
                    signal = jnp.where(noise_mask[i], noised_signal, signal)

        return signal

    @classmethod
    def log_signal_fn(cls, aquisition_scheme, **kwargs):
        return jnp.log(cls.signal_fn(aquisition_scheme, **kwargs))

    @classmethod
    def split_idx(cls):
        theta_dims_fractions = [len(cls.model_types) - 1]
        theta_dims_shared = (
            []
            if cls.shared_parameter_type is None
            else [cls.shared_parameter_type.theta_dim]
        )
        theta_dims_model = [m.theta_dim for m in cls.model_types]
        theta_dims_noise = [m.theta_dim for m in cls.noise_types]
        total_dims = (
            theta_dims_fractions
            + theta_dims_shared
            + theta_dims_model
            + theta_dims_noise
        )
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
        shared_parameter: SharedParameterState | None = None,
    ):
        theta_fraction = model_fractions
        if model_mask is not None:
            component_mask = model_mask[: len(model_compartments)]
        else:
            component_mask = None
        theta_fraction = dirichlet_to_normal(
            cls.fraction_prior, theta_fraction, component_mask
        )
        if shared_parameter is not None:
            theta_shared = shared_parameter.theta
        else:
            theta_shared = jnp.array([])

        if len(model_compartments) > 0:
            theta_model = jnp.concatenate([m.theta for m in model_compartments])
        else:
            theta_model = jnp.array([])
        if len(noise_compartments) > 0:
            theta_noise = jnp.concatenate([m.theta for m in noise_compartments])
        else:
            theta_noise = jnp.array([])
        return jnp.concatenate([theta_fraction, theta_shared, theta_model, theta_noise])

    @classmethod
    def to_params(
        cls,
        theta: ArrayLike,
        model_mask: Optional[ArrayLike] = None,
    ):
        thetas_split = cls.split_theta(theta)
        fractions = thetas_split[0]

        # Handle shared parameters
        if cls.shared_parameter_type is not None and len(thetas_split) > 1:
            shared_parameter = cls.shared_parameter_type.from_theta(thetas_split[1])
            model_thetas = thetas_split[2 : len(cls.model_types) + 2]
            noise_thetas = thetas_split[len(cls.model_types) + 2 :]
        else:
            shared_parameter = None
            model_thetas = thetas_split[1 : len(cls.model_types) + 1]
            noise_thetas = thetas_split[len(cls.model_types) + 1 :]

        # Model fractions should sum to 1 and follow a Dirichlet distribution
        if model_mask is not None:
            component_mask = model_mask[: len(cls.model_types)]
        else:
            component_mask = None
        #print(cls.fraction_prior, fractions, component_mask)
        fractions = normal_to_dirichlet(cls.fraction_prior, fractions, component_mask)

        # Apply shared parameter
        if shared_parameter is not None:
            model_types = [
                shared_parameter.set_shared_params_for_compartment(m)
                for m in cls.model_types
            ]
        else:
            model_types = cls.model_types

        # Create model compartments
        model_compartments = [
            m.from_theta(t) for m, t in zip(model_types, model_thetas)
        ]
        noise_compartments = [
            m.from_theta(t) for m, t in zip(cls.noise_types, noise_thetas)
        ]
        return (
            fractions,
            model_compartments,
            noise_compartments,
            model_mask,
            shared_parameter,
        )

    def to_fod(self, no_isotropic=False):
        if not no_isotropic:
            fods = [m.to_fod() for m in self.model_compartments]
            fractions = self.model_fractions
            return MixtureOfFODs(fractions, fods)
        else:
            fods = [
                m.to_fod() for m in self.model_compartments if not isinstance(m, Ball)
            ]
            fractions = self.model_fractions[1:]
            fractions = fractions / jnp.sum(fractions)
            return MixtureOfFODs(fractions, fods)

    def log_likelihood(self, aquisition_scheme, signal_observed):
        # Compute the signal for each compartment
        signals = jnp.stack(
            [m.signal(aquisition_scheme) for m in self.model_compartments], axis=0
        )
        fractions = self.model_fractions[:, None]
        # Combine signals with sum
        signal = jnp.sum(signals * fractions, axis=0)

        # Compute the noise likelihood

        if len(self.noise_compartments) > 0:
            if self.model_mask is None:
                # Select first noise compartment
                noise_model = self.noise_compartments[0]
                log_likelihood = noise_model.log_likelihood(signal, signal_observed)
            else:
                # Make mask shape match noise compartments
                idx = len(self.model_compartments)
                noise_mask = self.model_mask[idx:]
                # Apply noise sequentially with where to avoid conditionals
                log_likelihood = 0.0
                for i in range(len(self.noise_compartments)):
                    ll = self.noise_compartments[i].log_likelihood(
                        signal, signal_observed
                    )
                    log_likelihood += jnp.where(noise_mask[i], ll, 0.0)
        return log_likelihood


class SharedDiffusivity(SharedParameterState):
    share_with_compartments = {
        StaticStick: [
            0,
        ],
        StaticBall: [
            0,
        ],
    }
    theta_dim = 1
    lam_min: float = 0.0001
    lam_max: float = 0.01

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        u = jax.scipy.stats.norm.cdf(theta)
        lam = cls.lam_min + u * (cls.lam_max - cls.lam_min)
        return lam

    @classmethod
    def to_theta(cls, shared_parameters: ArrayLike) -> ArrayLike:
        u = (shared_parameters - cls.lam_min) / (cls.lam_max - cls.lam_min)
        return jax.scipy.stats.norm.ppf(u)

class SharedMultiShellDiffusivity(SharedParameterState):
    share_with_compartments = {
        MultiShellStaticStick: [0, 1],
        MultiShellStaticBall: [0, 1],
    }
    theta_dim = 2
    lam_min: float = 0.0001
    lam_max: float = 0.01
    lam_std_min: float = 0.0001
    lam_std_max: float = 0.05

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        u = jax.scipy.stats.norm.cdf(theta)
        lam = cls.lam_min + u[0] * (cls.lam_max - cls.lam_min)
        lam_std = cls.lam_std_min + u[1] * (cls.lam_std_max - cls.lam_std_min)
        shared_parameters = jnp.array([lam, lam_std])
        return shared_parameters,

    @classmethod
    def to_theta(cls, shared_parameters: ArrayLike) -> ArrayLike:
        u = (shared_parameters - cls.lam_min) / (cls.lam_max - cls.lam_min)
        u_std = (shared_parameters - cls.lam_std_min) / (
            cls.lam_std_max - cls.lam_std_min
        )
        us = jnp.concatenate([u, u_std])
        return jax.scipy.stats.norm.ppf(us)

class BallStickSharedDiffusivity(MultiCompartment):
    model_types = [StaticBall, StaticStick]
    noise_types = []
    fraction_prior = jnp.ones(2)
    shared_parameter_type = SharedDiffusivity

class BallStickSharedDiffusivity2(MultiCompartment):
    model_types = [StaticBall, StaticStick]
    noise_types = []
    fraction_prior = jnp.ones(2)
    shared_parameter_type = SharedDiffusivity


class Ball3StickSharedDiffusivity(MultiCompartment):
    model_types = [StaticBall, StaticStick, StaticStick, StaticStick]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1., 0.3, 0.1])
    shared_parameter_type = SharedDiffusivity


class Ball3StickSharedDiffusivityUniformFraction(MultiCompartment):
    model_types = [StaticBall, StaticStick, StaticStick, StaticStick]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.ones(4)
    shared_parameter_type = SharedDiffusivity


class MultiShellBall3StickSharedDiffusivity(MultiCompartment):
    model_types = [
        MultiShellStaticBall,
        MultiShellStaticStick,
        MultiShellStaticStick,
        MultiShellStaticStick,
    ]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.array([3.5, 1.0, 0.3, 0.1])
    shared_parameter_type = SharedMultiShellDiffusivity


class MultiShellBall3StickSharedDiffusivityUniformFraction(MultiCompartment):
    model_types = [
        MultiShellStaticBall,
        MultiShellStaticStick,
        MultiShellStaticStick,
        MultiShellStaticStick,
    ]
    noise_types = [BoundedGaussianNoise]
    fraction_prior = jnp.ones(4)
    shared_parameter_type = SharedMultiShellDiffusivity


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

class Ball3StickNoise(MultiCompartment):
    model_types = [Ball, Stick, Stick, Stick]
    noise_types = [RicianNoiseSNR310, RicianNoiseSNR1020]
    fraction_prior = jnp.ones(4)


class BallStickZeppelinNoise(MultiCompartment):
    model_types = [Ball, Stick, Zeppelin]
    noise_types = [RicianNoiseSNR310, RicianNoiseSNR1020]
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


class Ball3Stick3ZeppelinNoise(MultiCompartment):
    model_types = [Ball] + 3 * [Stick] + 3 * [Zeppelin]
    noise_types = [
        RicianNoiseSNR310,
        RicianNoiseSNR1020,
    ]
    fraction_prior = jnp.ones(1 + 3 + 3)


class AllGaussianModels(MultiCompartment):
    model_types = [Ball] + 3 * [Stick] + 3 * [Zeppelin] + 3 * [Dti]
    noise_types = [
        RicianNoiseSNR310,
        RicianNoiseSNR1020,
        RicianNoiseSNR2030,
        RicianNoiseSNR3040,
    ]
    fraction_prior = jnp.ones(1 + 3 + 3 + 3)


class AllGaussianAndConvolvedModels(MultiCompartment):
    model_types = (
        [Ball]
        + 3 * [Stick]
        + 3 * [Zeppelin]
        + 3 * [Dti]
        + 3 * [WatsonStick]
        + 3 * [WatsonZeppelin]
        + 3 * [BinghamStick]
        + 3 * [BinghamZeppelin]
        + 3 * [NoddiB]
        + 3 * [NoddiW]
        + 3 * [SandiB]
        + 3 * [SandiW]
    )
    noise_types = [
        GaussianNoiseSNR310,
        GaussianNoiseSNR1020,
        GaussianNoiseSNR2030,
        GaussianNoiseSNR3040,
    ] + [
        RicianNoiseSNR310,
        RicianNoiseSNR1020,
        RicianNoiseSNR2030,
        RicianNoiseSNR3040,
    ]
    fraction_prior = jnp.ones(1 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3)
