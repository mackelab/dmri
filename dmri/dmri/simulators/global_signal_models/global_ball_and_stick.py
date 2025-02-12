from functools import partial
from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike

from dmri.simulators.base import ModelCompartment
from dmri.simulators import BallStick, Ball, Stick

from .fiber_prior import FiberPrior


class GlobalBallStick(ModelCompartment):
    def __init__(self, n_voxels: int = 4):
        self.fiber_prior = FiberPrior(n_voxels, 2)
        # Not any fiber vs. 1 fiber
        self.num_fiber_probs = jnp.array([0.2, 0.8])

    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike, rng=None) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        volumes_fibers, tangent_fibers = self._sample_n_fibers(rng1)
        fiber_fraction = self._fiber_conditional_fraction_prior(rng2, volumes_fibers)
        fiber_direction = self._fiber_conditional_direction_prior(tangent_fibers, rng3)

        fiber_direction = fiber_direction[0]
        fiber_fraction = fiber_fraction.T

        @partial(jax.vmap, in_axes=(0, 0, None, None, None))
        @partial(jax.vmap, in_axes=(0, 0, None, None, None))
        @partial(jax.vmap, in_axes=(0, 0, None, None, None))
        def _single_voxel_log_signal(
            fiber_fraction, fiber_direction, bvals, bvecs, rng
        ):
            # print(fiber_fraction.shape, fiber_direction.shape)
            # jax.debug.print("{fiber_fraction}", fiber_fraction=fiber_fraction)
            ball = Ball(0.01)
            stick = Stick(0.001, fiber_direction)
            ball_stick = BallStick(fiber_fraction, [ball, stick], [])
            return ball_stick.log_signal(bvals, bvecs)

        # Compute the signal for the ball
        return _single_voxel_log_signal(
            fiber_fraction, fiber_direction, bvals, bvecs, rng4
        )

    def fit(self, logS, bvals, bvecs):
        raise NotImplementedError("Fitting not implemented for GlobalBallStick")

    @classmethod
    def to_params(cls, theta):
        return ()

    @classmethod
    def to_theta(cls, *kwargs, **params):
        return jnp.array([])

    def _sample_n_fibers(self, rng: Any) -> int:
        """Sample the number of fibers."""

        max_fibers = len(self.num_fiber_probs) - 1
        key_num, *key_fibers = jax.random.split(rng, 1 + max_fibers)
        key_fibers = jnp.array(key_fibers)
        # key_fibers = key_fibers.reshape(-1, key_fibers.shape[-1])
        volumes_fibers, tangent_fibers = jax.vmap(self.fiber_prior.sample)(key_fibers)

        num_active_fibers = jax.random.choice(
            key_num, jnp.arange(max_fibers + 1), p=self.num_fiber_probs
        )
        # Zero out the inactive fibers
        volumes_fibers = jnp.where(
            jnp.arange(max_fibers) < num_active_fibers, volumes_fibers, 0
        )
        tangent_fibers = jnp.where(
            jnp.arange(max_fibers) < num_active_fibers, tangent_fibers, 0
        )
        return volumes_fibers, tangent_fibers

    def _fiber_conditional_fraction_prior(self, rng, volumes_fibers):
        """Compute the conditional prior of the fiber."""
        # Per fiber volumes fractions (Fs, *voxels)
        is_fiber = jnp.where(volumes_fibers > 0, True, False)

        # If there is no fiber -> is a ball alphas[0] = 1, alphas[1] = 0.1
        # If there is a fiber -> alphas[0] = 0.1, alphas[1] = 1
        alpha1 = is_fiber.astype(jnp.float32) + 0.01
        alpha2 = (~is_fiber) + 0.01
        alpha1 = 10 * alpha1
        alpha2 = 10 * alpha2

        alphas = jnp.concatenate([alpha2, alpha1], axis=0)

        fractions = jax.random.dirichlet(rng, alpha=alphas.T).T

        return fractions

    def _fiber_conditional_direction_prior(self, tangent_fibers, rng):
        zero_tangent = jnp.array([0.0, 0.0, 0.0])
        random_tangents = jax.random.normal(rng, tangent_fibers.shape)
        random_tangents = random_tangents / jnp.linalg.norm(
            random_tangents, axis=-1, keepdims=True
        )
        tangent_fibers_normed = tangent_fibers / jnp.linalg.norm(
            tangent_fibers, axis=-1, keepdims=True
        )
        tangent_fibers_normed = jnp.nan_to_num(tangent_fibers_normed)
        return jnp.where(
            jnp.all(tangent_fibers == zero_tangent, axis=-1, keepdims=True),
            random_tangents,
            tangent_fibers_normed,
        )
