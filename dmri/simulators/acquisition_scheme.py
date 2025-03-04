from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from jax.typing import ArrayLike
import numpy as np
import jax.numpy as jnp
import jax

from jax.tree_util import register_dataclass

WATER_DIFFUSION_CONSTANT = 2.299e-3  # mm^2/s
WATER_IN_AXON_DIFFUSION_CONSTANT = 1.7e-3  # mm^2/s
NAA_IN_AXONS = 0.00015e-3  # mm^2/s
WATER_GYROMAGNETIC_RATIO = 267.513e6  # 1/(sT)


@dataclass
class acquisition_scheme:
    bvals: ArrayLike  # s/mm^2
    bvecs: ArrayLike  # unit vectors
    delta: ArrayLike = field(default_factory=lambda: jnp.array(0.03))  # In seconds
    Delta: ArrayLike = field(default_factory=lambda: jnp.array(0.05))  # In seconds

    @property
    def q_values(self):
        return jnp.sqrt(self.bvals / (4 * jnp.pi**2 * self.Delta))  # In 1/mm

    @property
    def tau(self):
        return self.Delta - self.delta / 3

    @property
    def gradient_strengths(self):
        return (
            (self.bvals * 1e6)
            / (self.tau * WATER_GYROMAGNETIC_RATIO**2 * self.delta**2)
        ) ** 0.5  # Units in T/m

    @classmethod
    def from_gradient_strengths(
        cls,
        gradient_strengths: ArrayLike,
        gradient_directions: ArrayLike,
        delta: ArrayLike,
        Delta: ArrayLike,
    ):
        """
        Create an aquisition scheme from gradient strengths and directions and delta and Delta values

        Args:
            gradient_strengths: ArrayLike
                Gradient strengths in T/m usually 0.003-0.008 T/m but can also go upto 0.03 or 0.1 T/m in animal scanners.
            gradient_directions: ArrayLike
                Gradient directions as 3D unit vectors.
            delta: ArrayLike
                Time between the start of the diffusion encoding and the first gradient pulse
            Delta: ArrayLike
                Time between the start of the diffusion encoding and the second gradient pulse
        """
        assert np.all(Delta > delta) and np.all(delta > 0), (
            "Delta must be greater than delta and both must be greater than 0"
        )
        assert gradient_strengths.shape[0] == gradient_directions.shape[0], (
            "Gradient strengths and directions must have the same length"
        )
        assert gradient_directions.shape[1] == 3 and np.allclose(
            np.linalg.norm(gradient_directions, axis=1), 1.0
        ), "Gradient directions must be 3D unit vectors"

        tau = Delta - delta / 3
        bvals = (
            (gradient_strengths * WATER_GYROMAGNETIC_RATIO * delta) ** 2 * tau
        ) / 1e6  # s/m^2

        return acquisition_scheme(bvals, gradient_directions, delta, Delta)


register_dataclass(
    acquisition_scheme, data_fields=("bvals", "bvecs"), meta_fields=("delta", "Delta")
)


def random_clinical_acquisition(rng, num_acquisitions=35) -> acquisition_scheme:
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)

    # Bvalues are usually 1000 here with 1-3 b0 images
    bvals_typical = jnp.array([1000.0] * num_acquisitions)
    bvals_rand_int = jax.random.choice(
        rng1, jnp.linspace(0, 2000, 200), shape=(num_acquisitions,), replace=False
    )
    bvals_float = jax.random.uniform(rng2, shape=(num_acquisitions,)) * 2000

    # Choose one of the above randomly
    bvals = jax.random.choice(
        rng3,
        jnp.stack([bvals_typical, bvals_rand_int, bvals_float])
    )
    bvals = jnp.sort(bvals)
    # Create a mask for b0 images
    mask = jax.random.choice(
        rng4, jnp.array([0, 1]), shape=(num_acquisitions,), p=jnp.array([0.1, 0.9])
    )
    bvals = bvals * mask

    # Create random directions
    bvecs = jax.random.normal(rng5, shape=(num_acquisitions, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1)[:, None]

    return acquisition_scheme(bvals, bvecs)


def random_hardi_acquisition(rng, num_acquisitions=200):
    rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(rng, 6)
    bvals_typical = jax.random.choice(rng1, jnp.array([1000.0, 2000.0, 3000.0]))
    bvals_typical = jnp.array([bvals_typical] * num_acquisitions)
    bvals_rand_int = jax.random.choice(
        rng2, jnp.linspace(0, 4000, 400), shape=(num_acquisitions,), replace=False
    )
    bvals_float = jax.random.uniform(rng3, shape=(num_acquisitions,)) * 4000

    bvals = jax.random.choice(
        rng4,
        jnp.stack([bvals_typical, bvals_rand_int, bvals_float]),
    )
    bvals = jnp.sort(bvals)

    # Create a mask for b0 images
    mask = jax.random.choice(
        rng5, jnp.array([0, 1]), shape=(num_acquisitions,), p=jnp.array([0.05, 0.95])
    )
    bvals = bvals * mask

    # Create random directions
    bvecs = jax.random.normal(rng6, shape=(num_acquisitions, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1)[:, None]

    return acquisition_scheme(bvals, bvecs)


def random_advanced_reasearch_acquisition_scheme(rng, num_acquisitions=500):
    rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(rng, 6)
    bvals_typical = jax.random.choice(rng1, jnp.array([1000.0, 2000.0, 3000.0]))
    bvals_typical = jnp.array([bvals_typical] * num_acquisitions)
    bvals_rand_int = jax.random.choice(
        rng2, jnp.linspace(0, 4000, 400), shape=(num_acquisitions,), replace=False
    )
    bvals_float = jax.random.uniform(rng3, shape=(num_acquisitions,)) * 4000

    bvals = jax.random.choice(
        rng4,
        jnp.stack([bvals_typical, bvals_rand_int, bvals_float]),
    )

    bvals = jnp.sort(bvals)

    # Create a mask for b0 images
    mask = jax.random.choice(
        rng5, jnp.array([0, 1]), shape=(num_acquisitions,), p=jnp.array([0.05, 0.95])
    )
    bvals = bvals * mask

    # Create random directions
    bvecs = jax.random.normal(rng6, shape=(num_acquisitions, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1)[:, None]

    return acquisition_scheme(bvals, bvecs)
