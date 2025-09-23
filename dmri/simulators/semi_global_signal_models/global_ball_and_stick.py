from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from dmri.simulators import Ball, BallStick, Stick
from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    random_hcp_acquisition,
)
from dmri.simulators.base import SignalCompartment
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

    def sample(self, fiber_field: VoxelizedCurve, rng: jax.Array) -> jnp.ndarray:
        volumes = fiber_field.volume_fraction
        voxel_size = (volumes.shape[-3], volumes.shape[-2], volumes.shape[-1])
        volumes = volumes.reshape((-1,) + voxel_size)
        ball_volume = 1-volumes.mean(axis=0)
        all_volumes = jnp.concatenate([ball_volume[None, ...], volumes], axis=0)
        alpha_updated = self.alpha[:, None, None, None] + self.beta * all_volumes
        fractions = jax.random.dirichlet(rng, alpha_updated.T, shape=voxel_size).T
        return fractions


class GlobalBall(Ball):
    """Samples Ball compartment parameters conditioned on fiber representation."""

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
        return vmap3d(partial(super().log_signal_fn, acq))(mu, lam_par, rng)

    @classmethod
    def signal_fn(cls, acq: acquisition_scheme, *args, **kwargs) -> ArrayLike:
        return vmap3d(partial(super().signal_fn, acq))(*args, **kwargs)

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: ArrayLike, fiber_field: FiberRepresentation | None = None) -> jnp.ndarray:
        if fiber_field is not None:
            lam_par = (lam_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
            theta_lam = jax.scipy.stats.norm.ppf(lam_par)
            mu_cart_override = fiber_field
            # Project to upper hemisphere
            need_to_flip = mu_cart_override[...,2] < 0
            mu_cart_override = jnp.where(need_to_flip[..., None], -mu_cart_override, mu_cart_override)
            mu_override = vmap3d(cartesian_to_unitsphere)(mu_cart_override)
            mu0_normalized = 1 - jnp.cos(
                mu_override[0]
            )  # Ensures uniform distribution on upper hemisphere
            mu1_normalized = (mu_override[1] + jnp.pi) / (2 * jnp.pi)
            return jnp.array([theta_lam, mu0_normalized, mu1_normalized], dtype=jnp.float32)
        else:
            return super().to_theta(mu, lam_par)
    @classmethod
    def to_params(cls, theta: ArrayLike, fiber_field: FiberRepresentation | None = None) -> tuple[jnp.ndarray, float]:
        if fiber_field is not None:
            # If we get a fiber representation we have to replace the theta responsible for the directions
            # with the fiber tangent
            theta = jax.scipy.stats.norm.cdf(theta)
            lam_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
            mu_cart_override = fiber_field
            mu_override = vmap3d(cartesian_to_unitsphere)(mu_cart_override)
            return mu_override, lam_par
        else:
            return super().from_theta(theta)
