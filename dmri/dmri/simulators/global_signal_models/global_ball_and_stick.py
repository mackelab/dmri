from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike

from dmri.simulators.base import ModelCompartment
from dmri.simulators import BallStick, Ball, Stick

from .fiber_prior import FiberPrior


class GlobalBallStick(ModelCompartment):
    def __init__(self, fiber_prior: FiberPrior):
        self.fiber_prior = fiber_prior
        # Not any fiber vs. 1 fiber
        self.num_fiber_probs = jnp.array([0.2, 0.8])

    def log_signal(self, bvals: ArrayLike, bvecs: ArrayLike, rng) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        volumes_fibers, tangent_fibers = self._sample_n_fibers(rng)
        fiber_fraction = self._fiber_conditional_fraction_prior(rng, volumes_fibers)
        fiber_direction = self._fiber_conditional_direction_prior(tangent_fibers, rng)

        # Compute the signal for the ball
        ball = Ball(1.7e-3)
        stick = Stick(1.7e-3, fiber_direction)
        ball_stick = BallStick(fiber_fraction, [ball, stick], [])
        return ball_stick.log_signal(bvals, bvecs)

    def _sample_n_fibers(self, rng: Any) -> int:
        """Sample the number of fibers."""

        max_fibers = len(self.num_fiber_probs) - 1
        key_num, *fibers = jax.random.split(rng, 1 + max_fibers - 1)
        volumes_fibers, tangent_fibers = jax.vmap(self.fiber_prior.sample)(fibers)
        num_active_fibers = jax.random.choice(
            key_num, jnp.arange(max_fibers), p=self.num_fiber_probs
        )
        # Zero out the inactive fibers
        volumes_fibers = jnp.where(
            jnp.arange(max_fibers) < num_active_fibers[:, None], volumes_fibers, 0
        )
        tangent_fibers = jnp.where(
            jnp.arange(max_fibers) < num_active_fibers[:, None], tangent_fibers, 0
        )
        return volumes_fibers, tangent_fibers

    def _fiber_conditional_fraction_prior(self, rng, volumes_fibers):
        """Compute the conditional prior of the fiber."""
        # Per fiber volumes fractions (Fs, *voxels)
        is_fiber = jnp.where(volumes_fibers > 0, 1, 0)

        # If there is no fiber -> is a ball alphas[0] = 1, alphas[1] = 0.1
        # If there is a fiber -> alphas[0] = 0.1, alphas[1] = 1
        alpha1 = jnp.sum(is_fiber, axis=0) + 0.1
        alpha2 = jnp.sum(~is_fiber, axis=0) + 0.1

        alphas = jnp.concatenate([alpha1, alpha2])
        return jax.random.dirichlet(rng, alphas)

    def _fiber_conditional_direction_prior(self, tangent_fibers, rng):
        """Compute the conditional prior of the fiber."""
        # Highly centered von Mises distribution for the fiber direction
        kappa = 30
        return jax.random.von_mises(rng, kappa, tangent_fibers)
