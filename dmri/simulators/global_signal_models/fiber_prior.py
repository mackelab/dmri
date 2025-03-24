from typing import Any
import jax.numpy as jnp
import jax


from .curves3d import (
    sample_splines,
    VoxelGrid,
    get_average_tangent_in_voxels,
    get_spline_volume_fraction_in_voxels,
)


class FiberPrior:
    r"""
    A class for generating fiber prior distributions in 3D space.
    """

    def __init__(self, n_voxels: int, max_degree: int):
        self.max_degree = max_degree
        self.grid = VoxelGrid(n_voxels, n_voxels, n_voxels)

    def sample(self, rng: Any) -> jnp.ndarray:
        """Sample from the fiber prior distribution."""
        rng1, rng2 = jax.random.split(rng)
        degree = jax.random.choice(
            rng1, jnp.arange(1, self.max_degree + 1), p=jnp.array([0.7, 0.3])
        )
        spline = sample_splines(rng2, degree)
        # This for now assumes a constant radius
        volume_fractions = get_spline_volume_fraction_in_voxels(
            spline, self.grid, 1.0 / self.grid.n_voxels_x
        )
        tangents = get_average_tangent_in_voxels(spline, self.grid)
        return volume_fractions, tangents
