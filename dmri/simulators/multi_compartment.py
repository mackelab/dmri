from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu
from jax.typing import ArrayLike
from probjax.utils.special import gammaincinv

from dmri.simulators import acquisition_scheme
from dmri.simulators.base import SharedParameterState, SignalCompartment
from dmri.simulators.local_signal_models import (
    Ball,
    SSFPStaticBall,
    SSFPStaticStick,
    StaticBall,
    StaticStick,
)
from dmri.simulators.local_signal_models.ball import MultiShellStaticBall
from dmri.simulators.local_signal_models.stick import MultiShellStaticStick
from dmri.simulators.mask_prior import BetaBernoulliMaskPrior, MaskPrior
from dmri.simulators.sphereical_distributions import MixtureOfFODs
from dmri.utils.transform import dirichlet_to_normal, eps_mask, normal_to_dirichlet


class MultiCompartment(SignalCompartment):
    """Composable mixture of signal compartments with optional shared parameters.

    A ``MultiCompartment`` turns a set of signal compartments :math:`S_k(b, \\mathbf{g})`
    and fractions :math:`f_k` into a single signal

    .. math::

        S(b, \\mathbf{g}) = \\sum_k f_k \\; S_k(b, \\mathbf{g})

    Noise compartments can be added on top through ``noise_types``. The class exposes
    ``to_theta`` / ``from_theta`` so priors over fractions, shared diffusivities and
    per-compartment parameters stay Gaussian in optimization space.

    References
    ----------
    * Stejskal & Tanner, 1965. Spin diffusion measurements: spin echoes in the presence of a time‐dependent field gradient.
    * Behrens et al., 2003. Characterization and propagation of uncertainty in diffusion-weighted MR imaging.
    """

    model_types: list
    noise_types: list
    fraction_prior: ArrayLike  # Dirichelt alpha values
    shared_parameter_type: type[SharedParameterState] | None = None
    normalizing_fn: Callable | None = None
    pre_normalizing_fn: Callable | None = None
    mask_prior_cls: type[MaskPrior] = BetaBernoulliMaskPrior
    mask_prior_kwargs: dict[str, Any] | None = None
    _split_dims: tuple[int, ...] = ()
    _split_indices: tuple[int, ...] = ()
    _num_models: int = 0
    _num_noises: int = 0
    _has_shared: bool = False

    def __init_subclass__(cls):
        assert hasattr(cls, "model_types"), "model_types not defined"
        assert hasattr(cls, "noise_types"), "noise_types not defined"
        if not hasattr(cls, "fraction_prior"):
            cls.fraction_prior = jnp.ones(len(cls.model_types))

        assert len(cls.model_types) == len(cls.fraction_prior), (
            "Wrong number of fractions"
        )

        cls._num_models = len(cls.model_types)
        cls._num_noises = len(cls.noise_types)
        cls._has_shared = cls.shared_parameter_type is not None

        model_fraction_theta_dim = max(cls._num_models - 1, 0)
        shared_parameters_dim = (
            0 if not cls._has_shared else cls.shared_parameter_type.theta_dim
        )
        cls.theta_dim = (
            model_fraction_theta_dim
            + shared_parameters_dim
            + sum([m.theta_dim for m in cls.model_types])
            + sum([n.theta_dim for n in cls.noise_types])
        )
        split_dims: list[int] = [model_fraction_theta_dim]
        if cls._has_shared:
            split_dims.append(shared_parameters_dim)
        split_dims.extend(m.theta_dim for m in cls.model_types)
        split_dims.extend(n.theta_dim for n in cls.noise_types)
        cls._split_dims = tuple(split_dims)
        if len(split_dims) > 1:
            cls._split_indices = tuple(np.cumsum(split_dims, dtype=int)[:-1].tolist())
        else:
            cls._split_indices = ()

        jtu.register_pytree_node_class(cls)
        if not hasattr(cls, "mask_prior_cls") or cls.mask_prior_cls is None:
            cls.mask_prior_cls = BetaBernoulliMaskPrior

    def __init__(
        self,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
        model_mask: ArrayLike | None = None,
        shared_parameter: SharedParameterState | None = None,
    ):
        self.model_fractions = jnp.asarray(model_fractions)
        self.model_compartments = tuple(model_compartments)
        self.noise_compartments = tuple(noise_compartments)
        self.model_mask = model_mask
        self.shared_parameter = shared_parameter
        assert len(model_compartments) == len(model_fractions), (
            "Wrong number of fractions"
        )
        # This will not work with current shared parameter state
        # assert [type(m) for m in model_compartments] == self.model_types, "Wrong model"
        assert [type(m) for m in noise_compartments] == self.noise_types, "Wrong noise"

    @classmethod
    def num_compartments(cls):
        return len(cls.model_types) + len(cls.noise_types)

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

    @staticmethod
    def _mix_model_signals(acq, model_compartments, model_fractions):
        num_models = len(model_compartments)
        if num_models == 0:
            return jnp.zeros_like(acq.bvals)

        signals = [m.signal(acq) for m in model_compartments]
        stacked = jnp.stack(signals, axis=0)
        return jnp.sum(stacked * model_fractions[:, None], axis=0)

    @staticmethod
    def _noise_mask(model_mask, num_models: int, num_noise: int):
        if num_noise == 0:
            return None
        if model_mask is None:
            mask = jnp.zeros((num_noise,), dtype=bool)
            return mask.at[0].set(True)
        mask_slice = model_mask[num_models : num_models + num_noise]
        return jnp.asarray(mask_slice, dtype=bool)

    @staticmethod
    def _mask_weights(mask: ArrayLike, target_ndim: int, dtype):
        if target_ndim < 1:
            raise ValueError("target_ndim must be at least 1")
        expand_shape = mask.shape + (1,) * (target_ndim - 1)
        return mask.astype(dtype).reshape(expand_shape)

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
        signal = cls._mix_model_signals(acq, model_compartments, model_fractions)
        base_signal = signal

        if cls.pre_normalizing_fn is not None:
            signal = cls.pre_normalizing_fn(acq, signal)

        # Add noise
        num_noise = len(noise_compartments)
        if rng is not None and num_noise > 0:
            noise_mask = cls._noise_mask(model_mask, len(model_compartments), num_noise)
            noise_outputs = [
                noise_compartments[i].noise(base_signal, rng) for i in range(num_noise)
            ]
            stacked_noise = jnp.stack(noise_outputs, axis=0)
            mask_weights = cls._mask_weights(
                noise_mask, stacked_noise.ndim, stacked_noise.dtype
            )
            weighted_noise = jnp.sum(stacked_noise * mask_weights, axis=0)
            signal = jnp.where(jnp.any(noise_mask), weighted_noise, base_signal)

        if cls.normalizing_fn is not None:
            signal = cls.normalizing_fn(acq, signal)

        return signal

    @classmethod
    def create_mask_prior(cls, **overrides: Any) -> MaskPrior:
        """Instantiate the configured mask-prior distribution."""
        kwargs = dict(cls.mask_prior_kwargs or {})
        kwargs.update(overrides)
        return cls.mask_prior_cls(
            len(cls.model_types),
            len(cls.noise_types),
            **kwargs,
        )

    @classmethod
    def log_signal_fn(cls, acq, **kwargs):
        return jnp.log(cls.signal_fn(acq, **kwargs))

    def _reconstruct_params(self):
        """Rebuild all parameters from theta so eager and JIT paths stay in sync."""
        return type(self).to_params(self.theta, model_mask=self.model_mask)

    def signal(self, acq: acquisition_scheme, rng=None):
        fractions, model_compartments, noise_compartments, model_mask, shared_parameter = (
            self._reconstruct_params()
        )
        return type(self).signal_fn(
            acq,
            model_compartments,
            noise_compartments,
            fractions,
            model_mask,
            shared_parameter,
            rng=rng,
        )

    def log_signal(self, acq: acquisition_scheme, rng=None):
        """Compute log-signal using reconstructed parameters to mirror the JIT path."""
        return jnp.log(self.signal(acq, rng=rng))

    @classmethod
    def split_idx(cls):
        return np.asarray(cls._split_dims)

    @classmethod
    def split_theta(cls, theta):
        if cls._split_indices:
            thetas_split = jnp.split(theta, cls._split_indices)
        else:
            thetas_split = (theta,)

        return thetas_split

    @classmethod
    def theta_mask(
        cls,
        model_mask,
        model_idx: Sequence[int] | None = None,
        noise_idx: Sequence[int] | None = None,
    ):
        if model_idx is not None or noise_idx is not None:
            if model_idx is None:
                model_idx_tuple = tuple(range(len(cls.model_types)))
            else:
                model_idx_tuple = tuple(int(i) for i in model_idx)
            if noise_idx is None:
                noise_idx_tuple = tuple(range(len(cls.noise_types)))
            else:
                noise_idx_tuple = tuple(int(i) for i in noise_idx)
            cls = cls.sub_model(model_idx=model_idx_tuple, noise_idx=noise_idx_tuple)
        with jax.ensure_compile_time_eval():
            if model_mask is None:
                return jnp.ones((cls.theta_dim,), dtype=jnp.bool_)

            mask_arr = jnp.ravel(jnp.asarray(model_mask, dtype=jnp.bool_))
            num_models = len(cls.model_types)
            num_noise = len(cls.noise_types)
            expected = num_models + num_noise
            if mask_arr.shape[0] < expected:
                raise ValueError(
                    f"model_mask has length {mask_arr.shape[0]}, but "
                    f"{cls.__name__} expects at least {expected} entries."
                )

            mask_list = []
            # Model fractions: only sticks that participate in active components
            if num_models > 1:
                fraction_mask = eps_mask(mask_arr[:num_models])
            else:
                fraction_mask = jnp.zeros((0,), dtype=jnp.bool_)
            mask_list.append(fraction_mask)

            # Shared parameters
            if cls._has_shared:
                shared_param_mask = jnp.ones(
                    (cls.shared_parameter_type.theta_dim,), dtype=jnp.bool_
                )
                mask_list.append(shared_param_mask)

            # Model compartments (reuse the same mask slice as the fractions)
            model_component_mask = mask_arr[:num_models]
            for is_active, compartment in zip(model_component_mask, cls.model_types):
                mask_list.append(
                    jnp.full((compartment.theta_dim,), is_active, dtype=jnp.bool_)
                )

            # Noise compartments (occupy the tail of the mask)
            noise_component_mask = mask_arr[num_models : num_models + num_noise]
            for is_active, compartment in zip(noise_component_mask, cls.noise_types):
                mask_list.append(
                    jnp.full((compartment.theta_dim,), is_active, dtype=jnp.bool_)
                )

            return (
                jnp.concatenate(mask_list)
                if mask_list
                else jnp.ones((0,), dtype=jnp.bool_)
            )

    @classmethod
    def to_theta(
        cls,
        model_fractions: ArrayLike,
        model_compartments: list,
        noise_compartments: list,
        model_mask: ArrayLike | None = None,
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
        model_mask: ArrayLike | None = None,
    ):
        thetas_split = cls.split_theta(theta)
        fractions = thetas_split[0]

        # Handle shared parameters
        idx = 1
        shared_parameter = None
        if cls._has_shared and len(thetas_split) > 1:
            shared_parameter = cls.shared_parameter_type.from_theta(thetas_split[idx])
            idx += 1
        model_end = idx + cls._num_models
        model_thetas = thetas_split[idx:model_end]
        noise_thetas = thetas_split[model_end:]

        # Model fractions should sum to 1 and follow a Dirichlet distribution
        if model_mask is not None:
            component_mask = model_mask[: len(cls.model_types)]
        else:
            component_mask = None
        # print(cls.fraction_prior, fractions, component_mask)
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

    @classmethod
    def sub_model(
        cls,
        model_idx: Sequence[int],
        noise_idx: Sequence[int],
    ) -> type[MultiCompartment]:
        """Return a reduced ``MultiCompartment`` subclass with selected components."""

        model_idx_tuple = tuple(int(i) for i in model_idx)
        noise_idx_tuple = tuple(int(i) for i in noise_idx)

        if not model_idx_tuple:
            raise ValueError("model_idx must contain at least one entry")

        if any(i < 0 or i >= len(cls.model_types) for i in model_idx_tuple):
            raise IndexError("model_idx contains entries outside available model types")
        if any(i < 0 or i >= len(cls.noise_types) for i in noise_idx_tuple):
            raise IndexError("noise_idx contains entries outside available noise types")

        return _build_submodel(cls, model_idx_tuple, noise_idx_tuple)

    def to_fod(self, no_isotropic=False):
        if not no_isotropic:
            fods = [m.to_fod() for m in self.model_compartments]
            fractions = self.model_fractions
            return MixtureOfFODs(fractions, fods)
        else:
            fods = [
                m.to_fod()
                for m in self.model_compartments
                if not isinstance(m, Ball) or not isinstance(m, MultiShellStaticBall)
            ]
            fractions = self.model_fractions[1:]
            fractions = fractions / jnp.sum(fractions)
            return MixtureOfFODs(fractions, fods)

    def log_likelihood(self, acq, signal_observed):
        # Compute the signal for each compartment
        # signals = jnp.stack([m.signal(acq) for m in self.model_compartments], axis=0)
        # fractions = self.model_fractions[:, None]
        # # Combine signals with sum
        # signal = jnp.sum(signals * fractions, axis=0)
        signal = self.signal_fn(
            acq,
            self.model_compartments,
            self.noise_compartments,
            self.model_fractions,
            self.model_mask,
            self.shared_parameter,
        )

        # Compute the noise likelihood

        num_noise = len(self.noise_compartments)
        if num_noise == 0:
            return jnp.array(0.0)

        noise_mask = self._noise_mask(
            self.model_mask, len(self.model_compartments), num_noise
        )
        ll_values = [
            self.noise_compartments[i].log_likelihood(signal, signal_observed)
            for i in range(num_noise)
        ]
        stacked_ll = jnp.stack(ll_values, axis=0)
        stacked_ll = jnp.nan_to_num(stacked_ll)
        mask_weights = self._mask_weights(noise_mask, stacked_ll.ndim, stacked_ll.dtype)
        return jnp.sum(stacked_ll * mask_weights, axis=0)


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
    lam_min: float = 0.0
    lam_max: float = 0.01

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        u = jax.scipy.stats.norm.cdf(theta)
        lam = cls.lam_min + u * (cls.lam_max - cls.lam_min)
        return (lam,)

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
    lam_min: float = 0.0
    lam_max: float = 0.01
    lam_std_min: float = 0.0
    lam_std_max: float = 0.005

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        u = jax.scipy.stats.norm.cdf(theta)
        lam = cls.lam_min + u[0] * (cls.lam_max - cls.lam_min)
        lam_std = cls.lam_std_min + u[1] * (cls.lam_std_max - cls.lam_std_min)
        shared_parameters = jnp.array([lam, lam_std])
        return (shared_parameters,)

    @classmethod
    def to_theta(cls, shared_parameters: ArrayLike) -> ArrayLike:
        u = (shared_parameters[0] - cls.lam_min) / (cls.lam_max - cls.lam_min)
        u_std = (shared_parameters[1] - cls.lam_std_min) / (
            cls.lam_std_max - cls.lam_std_min
        )
        us = jnp.array([u, u_std])
        return jax.scipy.stats.norm.ppf(us)


@cache
def _build_submodel(
    base_cls: type[MultiCompartment],
    model_idx_tuple: tuple[int, ...],
    noise_idx_tuple: tuple[int, ...],
) -> type[MultiCompartment]:
    model_types = [base_cls.model_types[i] for i in model_idx_tuple]
    noise_types = [base_cls.noise_types[i] for i in noise_idx_tuple]
    fraction_prior = jnp.asarray(base_cls.fraction_prior)[list(model_idx_tuple)]

    attrs = {
        "model_types": model_types,
        "noise_types": noise_types,
        "fraction_prior": fraction_prior,
        "shared_parameter_type": base_cls.shared_parameter_type,
        "normalizing_fn": base_cls.normalizing_fn,
        "pre_normalizing_fn": base_cls.pre_normalizing_fn,
        "__module__": base_cls.__module__,
    }

    subclass_name = (
        f"{base_cls.__name__}Sub_"
        f"{'_'.join(map(str, model_idx_tuple))}__"
        f"{'_'.join(map(str, noise_idx_tuple)) if noise_idx_tuple else 'none'}"
    )

    return type(subclass_name, (base_cls,), attrs)


class SharedMultiShellDiffusivityGammaPrior(SharedParameterState):
    share_with_compartments = {
        MultiShellStaticStick: [0, 1],
        MultiShellStaticBall: [0, 1],
    }
    theta_dim = 2
    lam_min: float = 0.0
    lam_max: float = 0.01
    lam_std_alpha: float = 0.4
    lam_std_beta: float = 400

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        u = jax.scipy.stats.norm.cdf(theta)
        lam = cls.lam_min + u[0] * (cls.lam_max - cls.lam_min)
        lam_std = gammaincinv(cls.lam_std_alpha, u[1]) * 1 / cls.lam_std_beta
        shared_parameters = jnp.array([lam, lam_std])
        return (shared_parameters,)

    @classmethod
    def to_theta(cls, shared_parameters: ArrayLike) -> ArrayLike:
        u = (shared_parameters[0] - cls.lam_min) / (cls.lam_max - cls.lam_min)
        u_std = jax.scipy.stats.gamma.cdf(
            shared_parameters[1], a=cls.lam_std_alpha, scale=1 / cls.lam_std_beta
        )
        us = jnp.array([u, u_std])
        return jax.scipy.stats.norm.ppf(us)


class SharedSSFPDiffusivity(SharedDiffusivity):
    share_with_compartments = {
        SSFPStaticBall: [0],
        SSFPStaticStick: [0],
    }
    theta_dim = 1
    lam_min: float = 0.0
    lam_max: float = 0.01


_MODEL_CLASS_NAMES = {
    "AllGaussianAndConvolvedModels",
    "AllGaussianModels",
    "AllGaussianModelsParamCountPrior",
    "Ball2Stick",
    "Ball2Stick2Zeppelin2Dti",
    "Ball3Stick",
    "Ball3Stick3ZeppelinNoise",
    "Ball3StickNoise",
    "Ball3StickSharedDiffusivity",
    "Ball3StickSharedDiffusivityTotalParamPenalizedPrior",
    "Ball3StickSharedDiffusivityUniformFraction",
    "BallStick",
    "BallStickSharedDiffusivity",
    "BallStickSharedDiffusivity2",
    "BallStickZeppelinNoise",
    "MultiShellBall3StickSharedDiffusivity",
    "MultiShellBall3StickSharedDiffusivityGammaPrior",
    "MultiShellBall3StickSharedDiffusivityUniformFraction",
    "SSFPBall3StickSharedDiffusivity",
    "SSFPBall3StickSharedDiffusivityBetterNorm",
}


def __getattr__(name: str):
    if name in _MODEL_CLASS_NAMES:
        from importlib import import_module

        models = import_module("dmri.simulators.models")
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
