from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

from dmri.utils.transform import dirichlet_to_normal, normal_to_dirichlet
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from dmri.simulators import Ball, BallStick, Stick, MultiCompartment
from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    random_hcp_acquisition,
)
from dmri.simulators.base import SharedParameterState, SignalCompartment
from dmri.utils.dmriutils import cartesian_to_unitsphere, unitsphere_to_cartesian

from .curves3d import VoxelizedCurve
from .fiber_prior import FiberPrior

def vmap3d(f, in_axes=0):
    return jax.vmap(jax.vmap(jax.vmap(f, in_axes=in_axes), in_axes=in_axes), in_axes=in_axes)

class FiberConditionedFractionPrior():
    """Deterministic Ball/Stick fractions conditioned on fiber presence."""

    def __init__(
        self,
        alpha: ArrayLike,
        beta: float = 8.0,
    ) -> None:
        self.alpha = jnp.asarray(alpha, dtype=jnp.float32)
        self.beta = float(beta)

    @classmethod
    def fiber_to_alpha(
        cls, alpha, beta, fiber_field: VoxelizedCurve
    ):
        volumes = fiber_field.volume_fraction
        voxel_size = (volumes.shape[-3], volumes.shape[-2], volumes.shape[-1])
        volumes = volumes.reshape((-1,) + voxel_size)
        ball_volume = 1-volumes.mean(axis=0)
        all_volumes = jnp.concatenate([ball_volume[None, ...], volumes], axis=0)
        alpha_updated = alpha[:, None, None, None] + beta * all_volumes
        return alpha_updated

    def sample(self, fiber_field: VoxelizedCurve, rng: jax.Array) -> jnp.ndarray:
        alpha_updated = self.fiber_to_alpha(self.alpha, self.beta, fiber_field)
        fractions = jax.random.dirichlet(rng, alpha_updated.T, shape=voxel_size).T
        return fractions


class GlobalBall(Ball):
    """Samples Ball compartment parameters conditioned on fiber representation."""

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, lam: ArrayLike, rng=None) -> ArrayLike:
        in_axes = (None, 0, None if rng is None else 0)
        return vmap3d(super().log_signal_fn, in_axes=in_axes)(acq, lam, rng)

    @classmethod
    def to_theta(cls, lam: ArrayLike, fiber_field: FiberRepresentation | None = None) -> jnp.ndarray:
        del fiber_field
        return vmap3d(super().to_theta)(lam)

    @classmethod
    def to_params(cls, theta: ArrayLike, fiber_field: FiberRepresentation | None = None) -> float:
        del fiber_field
        return vmap3d(super().to_params)(theta)


class GlobalStick(Stick):

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, mu: ArrayLike, lam_par: float, rng=None) -> ArrayLike:
        in_axes = (None, 0, 0, None if rng is None else 0)
        return vmap3d(super().log_signal_fn, in_axes=in_axes)(acq, mu, lam_par, rng)

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: ArrayLike, fiber_field: FiberRepresentation | None = None) -> jnp.ndarray:
        if fiber_field is not None:
            theta = vmap3d(super().to_theta)(mu, lam_par)
            theta_lam, theta_mu = theta[..., :1], theta[..., 1:]
            mu_cart_override = fiber_field.tangents
            no_fiber_mask = jnp.all(mu_cart_override == 0, axis=-1)
            # Project to upper hemisphere
            need_to_flip = mu_cart_override[...,2] < 0
            mu_cart_override = jnp.where(need_to_flip[..., None], -mu_cart_override, mu_cart_override)
            mu_override = vmap3d(cartesian_to_unitsphere)(mu_cart_override)
            # Convert to normalized theta
            mu0_normalized = 1 - jnp.cos(
                mu_override[...,0]
            )  # Ensures uniform distribution on upper hemisphere
            mu0_normalized = jnp.where(no_fiber_mask, theta_mu[...,0], mu0_normalized)
            mu1_normalized = (mu_override[...,1] + jnp.pi) / (2 * jnp.pi)
            mu1_normalized = jnp.where(no_fiber_mask, theta_mu[...,1], mu1_normalized)
            return jnp.concatenate([mu0_normalized[..., None], mu1_normalized[..., None], theta_lam], axis=-1)
        else:
            return vmap3d(super().to_theta)(mu, lam_par)
    @classmethod
    def to_params(cls, theta: ArrayLike, fiber_field: FiberRepresentation | None = None) -> tuple[jnp.ndarray, float]:
        if fiber_field is not None:
            # If we get a fiber representation we have to replace the theta responsible for the directions
            # with the fiber tangent
            mu_uncond, lam_par = vmap3d(super().to_params)(theta)
            mu_cart_override = fiber_field.tangents
            # Project to upper hemisphere
            need_to_flip = mu_cart_override[...,2] < 0
            mu_cart_override = jnp.where(need_to_flip[..., None], -mu_cart_override, mu_cart_override)
            mu_override = vmap3d(cartesian_to_unitsphere)(mu_cart_override)
            no_fiber_mask = jnp.all(mu_cart_override == 0, axis=-1)

            mu_override = vmap3d(cartesian_to_unitsphere)(mu_cart_override)
            mu_override = jnp.where(no_fiber_mask[..., None], mu_uncond, mu_override)
            return mu_override, lam_par
        else:
            return vmap3d(super().to_params)(theta)

class GlobalMultiCompartment(SignalCompartment):
    """MultiCompartment model with Ball and Stick compartments conditioned on fiber representation."""

    model_types: list
    noise_types: list
    fraction_prior: ArrayLike  # Dirichelt alpha values
    shared_parameter_type: type[SharedParameterState] | None = None
    normalizing_fn: Callable | None = None
    pre_normalizing_fn: Callable | None = None

    def __init_subclass__(cls):
        assert hasattr(cls, "model_types"), "model_types not defined"
        assert hasattr(cls, "noise_types"), "noise_types not defined"
        if not hasattr(cls, "fraction_prior"):
            cls.fraction_prior = jnp.ones(len(cls.model_types))

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

        jax.tree_util.register_pytree_node_class(cls)

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

    @classmethod
    def signal_fn(
        cls,
        acq: acquisition_scheme,
        model_compartments,
        noise_compartments,
        model_fractions,
        model_mask,
        shared_parameter,
        rng=None,
    ):
        del shared_parameter
        # Compute the signal for each compartment
        signals = jnp.stack([m.signal(acq) for m in model_compartments], axis=0)
        fractions = model_fractions.T[..., None]  # Add signal axis
        # Combine signals with sum
        signal = jnp.sum(signals * fractions, axis=0)

        if cls.pre_normalizing_fn is not None:
            signal = cls.pre_normalizing_fn(acq, signal)

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

        if cls.normalizing_fn is not None:
            signal = cls.normalizing_fn(acq, signal)

        return signal

    @classmethod
    def log_signal_fn(cls, acq, **kwargs):
        return jnp.log(cls.signal_fn(acq, **kwargs))


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
        thetas_split = jnp.split(theta, split_indices, axis=-1)

        return thetas_split

    @classmethod
    def fraction_conditioned_on_fiber(
        cls,
        fiber_field: Sequence[FiberRepresentation],
    ):
        # NOTE: This is specific for Ball and Stick models
        volumes = [f.volume_fraction[..., None] for f in fiber_field if f is not None]
        avg_volumes = sum(volumes)/len(volumes)
        ball_volume = 1-avg_volumes
        all_volumes = jnp.concatenate([ball_volume, avg_volumes], axis=-1)
        updated_alpha = cls.fraction_prior[None, None, None,:] + 8.0 * all_volumes
        return updated_alpha



    @classmethod
    def to_params(
        cls,
        theta: ArrayLike,
        model_mask: ArrayLike | None = None,
        fiber_field: Sequence[FiberRepresentation] | None = None,
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
        # Conditional fractions
        cond_alpha  = cls.fraction_conditioned_on_fiber(fiber_field) if fiber_field is not None else cls.fraction_prior[None, None, None,:]
        fractions = vmap3d(normal_to_dirichlet,in_axes=(0,0,None))(cond_alpha, fractions, component_mask)

        # Apply shared parameter
        if shared_parameter is not None:
            model_types = [
                shared_parameter.set_shared_params_for_compartment(m)
                for m in cls.model_types
            ]
        else:
            model_types = cls.model_types

        # Create model compartments
        if fiber_field is None:
            fiber_field = [None] * len(model_types)
        model_compartments = [
            m.from_theta(t, fiber_field=f) for m, t, f in zip(model_types, model_thetas, fiber_field)
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

    @classmethod
    def to_theta(
        cls,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
        model_mask: ArrayLike | None = None,
        fiber_field: Sequence[FiberRepresentation] | None = None,
        shared_parameter: SharedParameterState | None = None,
    ):
        theta_fraction = model_fractions
        if model_mask is not None:
            component_mask = model_mask[: len(model_compartments)]
        else:
            component_mask = None
        cond_alpha  = cls.fraction_conditioned_on_fiber(fiber_field) if fiber_field is not None else cls.fraction_prior[None, None, None,:]
        theta_fraction = vmap3d(dirichlet_to_normal, in_axes=(0, 0, None))(
            cond_alpha, theta_fraction, component_mask
        )
        theta_parts = [theta_fraction]
        if shared_parameter is not None:
            theta_shared = shared_parameter.theta
            theta_parts.append(theta_shared)

        if len(model_compartments) > 0:
            thetas = [m.to_theta(**m.params, fiber_field=f) for m, f in zip(model_compartments, fiber_field or [None]*len(model_compartments))]
            theta_model = jnp.concatenate(thetas, axis=-1)
            theta_parts.append(theta_model)

        if len(noise_compartments) > 0:
            theta_noise = jnp.concatenate([m.theta for m in noise_compartments])
            theta_parts.append(theta_noise)

        return jnp.concatenate(theta_parts, axis=-1)

class GlobalBallStick(GlobalMultiCompartment):
    """MultiCompartment model with Ball and Stick compartments conditioned on fiber representation."""

    model_types = [GlobalBall, GlobalStick]
    noise_types = []
    fraction_prior = jnp.array([1.0, 1.0])
    shared_parameter_type = None
