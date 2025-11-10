from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from dmri.simulators import Ball, Stick
from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
)
from dmri.simulators.base import SharedParameterState, SignalCompartment
from dmri.simulators.semi_global_signal_models.fiber_prior import (
    FiberField,
)
from dmri.utils.dmriutils import cart2sph
from dmri.utils.transform import dirichlet_to_normal, normal_to_dirichlet


def vmap3d(f, in_axes=0):
    return jax.vmap(
        jax.vmap(jax.vmap(f, in_axes=in_axes), in_axes=in_axes), in_axes=in_axes
    )


class FiberConditionedFractionPrior:
    """Deterministic Ball/Stick fractions conditioned on fiber presence."""

    def __init__(
        self,
        alpha: ArrayLike,
        beta: float = 8.0,
    ) -> None:
        self.alpha = jnp.asarray(alpha, dtype=jnp.float32)
        self.beta = float(beta)

    @classmethod
    def _normalize_fiber_sequence(
        cls, fiber_field: FiberField | Sequence[FiberField | None], num_components: int
    ) -> tuple[FiberField | None, ...]:
        if isinstance(fiber_field, Sequence):
            fiber_seq = list(fiber_field)
        else:
            fiber_seq = [fiber_field]

        if num_components > 1 and len(fiber_seq) == num_components - 1:
            fiber_seq = [None] + fiber_seq

        if len(fiber_seq) != num_components:
            raise ValueError(
                f"Expected {num_components} fiber fields (including None for isotropic components), "
                f"got {len(fiber_seq)}"
            )

        return tuple(fiber_seq)

    @staticmethod
    def _resolve_rng(fiber_fields: Sequence[FiberField | None]) -> jax.random.KeyArray:
        for field in fiber_fields:
            if isinstance(field, FiberField):
                return field.rng
        raise ValueError("At least one FiberField with an RNG key must be provided")

    @classmethod
    def fiber_to_alpha(
        cls,
        alpha: ArrayLike,
        beta: float,
        fiber_fields: Sequence[FiberField | None],
        component_mask: ArrayLike | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[int, int, int]]:
        alpha_arr = jnp.asarray(alpha, dtype=jnp.float32)
        mask_bool = None
        if component_mask is not None:
            mask_arr = jnp.asarray(component_mask, dtype=jnp.bool_).reshape(-1)
            if mask_arr.shape[0] != alpha_arr.shape[0]:
                raise ValueError("component_mask must have the same length as alpha")
            mask_bool = np.asarray(mask_arr, dtype=bool)
        else:
            mask_bool = np.ones(alpha_arr.shape[0], dtype=bool)
            mask_arr = jnp.asarray(mask_bool, dtype=jnp.bool_)

        voxel_shape: tuple[int, int, int] | None = None
        volume_fields: list[jnp.ndarray | None] = []
        for field in fiber_fields:
            if isinstance(field, FiberField):
                vf = jnp.asarray(field.volume_fraction, dtype=jnp.float32)
                if voxel_shape is None:
                    voxel_shape = vf.shape
                elif vf.shape != voxel_shape:
                    raise ValueError("All fiber fields must share the same voxel shape")
                volume_fields.append(vf)
            else:
                volume_fields.append(None)

        if voxel_shape is None:
            raise ValueError(
                "At least one FiberField is required to infer the voxel grid"
            )

        zero_volume = jnp.zeros(voxel_shape, dtype=jnp.float32)
        resolved_volumes = [zero_volume if vf is None else vf for vf in volume_fields]

        active_fiber_volumes = [
            resolved_volumes[idx]
            for idx in range(1, len(resolved_volumes))
            if mask_bool[idx]
        ]
        if active_fiber_volumes:
            stacked = jnp.stack(active_fiber_volumes, axis=0)
            fiber_total = jnp.clip(stacked.sum(axis=0), 0.0, 1.0)
        else:
            fiber_total = jnp.zeros(voxel_shape, dtype=jnp.float32)

        resolved_volumes[0] = jnp.clip(1.0 - fiber_total, 0.0, 1.0)

        for idx, active in enumerate(mask_bool):
            if not active:
                resolved_volumes[idx] = jnp.zeros_like(resolved_volumes[idx])

        all_volumes = jnp.stack(resolved_volumes, axis=0)
        alpha_updated = alpha_arr[:, None, None, None] + float(beta) * all_volumes
        return alpha_updated, mask_arr, voxel_shape

    def sample(
        self,
        fiber_field: FiberField | Sequence[FiberField | None],
        component_mask: ArrayLike | None = None,
    ) -> jnp.ndarray:
        fiber_fields = self._normalize_fiber_sequence(
            fiber_field, int(self.alpha.shape[0])
        )
        alpha_updated, mask_arr, _voxel_shape = self.fiber_to_alpha(
            self.alpha, self.beta, fiber_fields, component_mask
        )
        rng = self._resolve_rng(fiber_fields)

        concentrations = jnp.moveaxis(alpha_updated, 0, -1)
        fractions = jax.random.dirichlet(rng, concentrations)

        mask_broadcast = mask_arr.reshape(
            (1,) * (fractions.ndim - 1) + (mask_arr.shape[0],)
        )
        fractions = jnp.where(mask_broadcast, fractions, 0.0)
        denom = fractions.sum(axis=-1, keepdims=True)
        denom_safe = jnp.where(denom > 0, denom, jnp.ones_like(denom))
        fractions = fractions / denom_safe
        fractions = jnp.where(denom > 0, fractions, 0.0)
        return fractions


class GlobalBall(Ball):
    """Samples Ball compartment parameters conditioned on fiber representation."""

    @classmethod
    def log_signal_fn(
        cls, acq: acquisition_scheme, lam: ArrayLike, rng=None
    ) -> ArrayLike:
        in_axes = (None, 0, None if rng is None else 0)
        return vmap3d(super().log_signal_fn, in_axes=in_axes)(acq, lam, rng)

    @classmethod
    def to_theta(
        cls, lam: ArrayLike, fiber_field: FiberField | None = None
    ) -> jnp.ndarray:
        del fiber_field
        return vmap3d(super().to_theta)(lam)

    @classmethod
    def to_params(
        cls, theta: ArrayLike, fiber_field: FiberField | None = None
    ) -> float:
        del fiber_field
        return vmap3d(super().to_params)(theta)


class GlobalStick(Stick):
    @classmethod
    def log_signal_fn(
        cls, acq: acquisition_scheme, mu: ArrayLike, lam_par: float, rng=None
    ) -> ArrayLike:
        in_axes = (None, 0, 0, None if rng is None else 0)
        return vmap3d(super().log_signal_fn, in_axes=in_axes)(acq, mu, lam_par, rng)

    @classmethod
    def to_theta(
        cls, mu: ArrayLike, lam_par: ArrayLike, fiber_field: FiberField | None = None
    ) -> jnp.ndarray:
        if fiber_field is not None:
            theta = vmap3d(super().to_theta)(mu, lam_par)
            theta_lam, theta_mu = theta[..., :1], theta[..., 1:]
            mu_cart_override = fiber_field.tangents
            no_fiber_mask = jnp.all(mu_cart_override == 0, axis=-1)
            # Project to upper hemisphere
            need_to_flip = mu_cart_override[..., 2] < 0
            mu_cart_override = jnp.where(
                need_to_flip[..., None], -mu_cart_override, mu_cart_override
            )
            mu_override = vmap3d(cart2sph)(mu_cart_override)
            # Convert to normalized theta
            mu0_normalized = 1 - jnp.cos(
                mu_override[..., 0]
            )  # Ensures uniform distribution on upper hemisphere
            mu0_normalized = jnp.where(no_fiber_mask, theta_mu[..., 0], mu0_normalized)
            mu1_normalized = (mu_override[..., 1] + jnp.pi) / (2 * jnp.pi)
            mu1_normalized = jnp.where(no_fiber_mask, theta_mu[..., 1], mu1_normalized)
            return jnp.concatenate(
                [mu0_normalized[..., None], mu1_normalized[..., None], theta_lam],
                axis=-1,
            )
        else:
            return vmap3d(super().to_theta)(mu, lam_par)

    @classmethod
    def to_params(
        cls, theta: ArrayLike, fiber_field: FiberField | None = None
    ) -> tuple[jnp.ndarray, float]:
        if fiber_field is not None:
            # If we get a fiber representation we have to replace the theta responsible for the directions
            # with the fiber tangent
            mu_uncond, lam_par = vmap3d(super().to_params)(theta)
            mu_cart_override = fiber_field.tangents
            # Project to upper hemisphere
            need_to_flip = mu_cart_override[..., 2] < 0
            mu_cart_override = jnp.where(
                need_to_flip[..., None], -mu_cart_override, mu_cart_override
            )
            mu_override = vmap3d(cart2sph)(mu_cart_override)
            no_fiber_mask = jnp.all(mu_cart_override == 0, axis=-1)

            mu_override = vmap3d(cart2sph)(mu_cart_override)
            mu_override = jnp.where(no_fiber_mask[..., None], mu_uncond, mu_override)
            return mu_override, lam_par
        else:
            return vmap3d(super().to_params)(theta)


class GlobalMultiCompartment(SignalCompartment):
    """MultiCompartment model with Ball and Stick compartments conditioned on fiber representation."""

    model_types: list
    noise_types: list
    fraction_prior: ArrayLike  # Dirichelt alpha values
    fiber_conditioned_fraction: type[FiberConditionedFractionPrior] = (
        FiberConditionedFractionPrior
    )
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
    def to_params(
        cls,
        theta: ArrayLike,
        model_mask: ArrayLike | None = None,
        fiber_field: Sequence[FiberField] | None = None,
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
        fiber_cond_prior = cls.fiber_conditioned_fraction(cls.fraction_prior)
        if fiber_field is not None:
            fractions = fiber_cond_prior.sample(fiber_field, component_mask)
        else:
            fractions = vmap3d(normal_to_dirichlet, in_axes=(None, 0, None))(
                cls.fraction_prior, fractions, component_mask
            )

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
            m.from_theta(t, fiber_field=f)
            for m, t, f in zip(model_types, model_thetas, fiber_field)
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
        fiber_field: Sequence[FiberField] | None = None,
        shared_parameter: SharedParameterState | None = None,
    ):
        if model_mask is not None:
            component_mask = model_mask[: len(model_compartments)]
        else:
            component_mask = None

        theta_fraction = vmap3d(dirichlet_to_normal, in_axes=(None, 0, None))(
            cls.fraction_prior, model_fractions, component_mask
        )
        theta_parts = [theta_fraction]
        if shared_parameter is not None:
            theta_shared = shared_parameter.theta
            theta_parts.append(theta_shared)

        if len(model_compartments) > 0:
            thetas = [
                m.to_theta(**m.params, fiber_field=f)
                for m, f in zip(
                    model_compartments, fiber_field or [None] * len(model_compartments)
                )
            ]
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
