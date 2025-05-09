from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike

WATER_DIFFUSION_CONSTANT = 2.299e-3  # mm^2/s
WATER_IN_AXON_DIFFUSION_CONSTANT = 1.7e-3  # mm^2/s
NAA_IN_AXONS = 0.00015e-3  # mm^2/s
WATER_GYROMAGNETIC_RATIO = 267.513e6  # 1/(sT)

ACQ_CONSTANTS = {
    "HCP": {"delta": 0.0106, "Delta": 0.0431},
    "Biobank": {"delta": 0.0214, "Delta": 0.0455},
}


@dataclass
class acquisition_scheme:  # noqa: N801
    """A class representing a diffusion MRI acquisition scheme.

    This class encapsulates the parameters needed to define a diffusion MRI acquisition,
    including b-values, gradient directions, and timing parameters.

    Attributes:
        bvals (ArrayLike): Array of b-values in s/mm^2, representing the diffusion weighting.
        bvecs (ArrayLike): Array of 3D unit vectors representing gradient directions.
        delta (ArrayLike): Time between the start of the diffusion encoding and the first gradient pulse in seconds.
        Delta (ArrayLike): Time between the start of the diffusion encoding and the second gradient pulse in seconds.
    """

    bvals: ArrayLike  # s/mm^2
    bvecs: ArrayLike  # unit vectors
    delta: ArrayLike = field(default_factory=lambda: 0.0106)  # In seconds
    Delta: ArrayLike = field(default_factory=lambda: 0.0431)  # In seconds

    @property
    def q_values(self):
        """Calculate the q-values for the acquisition scheme.

        Returns:
            ArrayLike: Array of q-values in 1/mm, calculated from b-values and Delta.
        """
        return jnp.sqrt(self.bvals / (4 * jnp.pi**2 * self.Delta))  # In 1/mm

    @property
    def tau(self):
        """Calculate the effective diffusion time tau.

        Returns:
            ArrayLike: Array of effective diffusion times in seconds.
        """
        return self.Delta - self.delta / 3

    @property
    def gradient_strengths(self):
        """Calculate the gradient strengths required for the acquisition scheme.

        Returns:
            ArrayLike: Array of gradient strengths in T/m.
        """
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
        Delta: ArrayLike,  # noqa: N803
    ):
        """Create an acquisition scheme from gradient parameters.

        Args:
            gradient_strengths (ArrayLike): Gradient strengths in T/m (typically 0.003-0.008 T/m,
                can reach 0.03-0.1 T/m in animal scanners).
            gradient_directions (ArrayLike): Gradient directions as 3D unit vectors.
            delta (ArrayLike): Time between the start of the diffusion encoding and the first gradient pulse.
            Delta (ArrayLike): Time between the start of the diffusion encoding and the second gradient pulse.

        Returns:
            acquisition_scheme: A new acquisition scheme instance.

        Raises:
            AssertionError: If Delta is not greater than delta, or if gradient strengths and directions
                have mismatched shapes, or if gradient directions are not 3D unit vectors.
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

from dipy.io.gradients import read_bvals_bvecs

import os
_bvals_typ, bvecs_typ = read_bvals_bvecs(os.path.join(os.path.dirname(__file__), "data/bvals"), os.path.join(os.path.dirname(__file__), "data/bvecs"))
# Round bvals to nearest (0, 1000, 2000)
bvals_typ = np.round(_bvals_typ / 1000) * 1000
bvals_typ = np.clip(bvals_typ, 0, 2000)
idx = np.argsort(bvals_typ)
bvals_typ = bvals_typ[idx]
bvecs_typ = bvecs_typ[idx]


def random_hcp_acquisition(rng, num_acquisitions=105) -> acquisition_scheme:
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
    bvals_typical = jnp.array(bvals_typ)
    bvecs_typical = jnp.array(bvecs_typ)
    bvals_float = jax.random.uniform(rng2, shape=(num_acquisitions,)) * 4000
    # TODO remove restrictrictions
    bvals = jax.random.choice(
        rng3, jnp.stack([bvals_typical, bvals_float]), p=jnp.array([1., 0.0])
    )

    mask = jax.random.choice(
        rng4, jnp.array([0, 1]), shape=(num_acquisitions,), p=jnp.array([0.0, 1.])
    )
    bvals = bvals * mask

    bvals = jnp.sort(bvals)

    # Create random directions
    rng5_1, rng5_2 = jax.random.split(rng5, 2)
    bvecs = jax.random.normal(rng5_1, shape=(num_acquisitions, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1)[:, None]

    bvecs = jax.random.choice(
        rng5_2, jnp.stack([bvecs_typical, bvecs], axis=0), axis=0, p=jnp.array([1., 0.0])
    )

    return acquisition_scheme(bvals, bvecs)


def random_clinical_acquisition(rng, num_acquisitions=35) -> acquisition_scheme:
    """Generate a random clinical diffusion MRI acquisition scheme.

    Creates an acquisition scheme typical for clinical diffusion MRI scans, with b-values
    around 1000 s/mm² and 1-3 b0 images.

    Args:
        rng: JAX random number generator key.
        num_acquisitions (int, optional): Number of acquisitions to generate. Defaults to 35.

    Returns:
        acquisition_scheme: A randomly generated acquisition scheme suitable for clinical use.
    """
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)

    # Bvalues are usually 1000 here with 1-3 b0 images
    bvals_typical = jnp.array([1000.0] * num_acquisitions)
    bvals_rand_int = jax.random.choice(
        rng1, jnp.linspace(0, 2000, 200), shape=(num_acquisitions,), replace=False
    )
    bvals_float = jax.random.uniform(rng2, shape=(num_acquisitions,)) * 2000

    # Choose one of the above randomly
    bvals = jax.random.choice(
        rng3, jnp.stack([bvals_typical, bvals_rand_int, bvals_float])
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
    """Generate a random HARDI (High Angular Resolution Diffusion Imaging) acquisition scheme.

    Creates an acquisition scheme typical for HARDI scans, with higher b-values and more
    gradient directions than clinical scans.

    Args:
        rng: JAX random number generator key.
        num_acquisitions (int, optional): Number of acquisitions to generate. Defaults to 200.

    Returns:
        acquisition_scheme: A randomly generated acquisition scheme suitable for HARDI.
    """
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
    """Generate a random advanced research diffusion MRI acquisition scheme.

    Creates an acquisition scheme suitable for advanced research purposes, with a large number
    of acquisitions and high b-values.

    Args:
        rng: JAX random number generator key.
        num_acquisitions (int, optional): Number of acquisitions to generate. Defaults to 500.

    Returns:
        acquisition_scheme: A randomly generated acquisition scheme suitable for advanced research.
    """
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


def get_biobank_bvals():
    bvals = np.array(
        [
            5,
            1000,
            2005,
            1000,
            1990,
            995,
            2000,
            990,
            2000,
            995,
            1995,
            1005,
            2010,
            1005,
            2000,
            995,
            2000,
            1005,
            1995,
            995,
            2005,
            5,
            990,
            1990,
            1000,
            2000,
            1005,
            1990,
            1000,
            1985,
            995,
            1995,
            1000,
            1995,
            1000,
            2000,
            1000,
            2005,
            995,
            1990,
            1000,
            1990,
            5,
            990,
            2000,
            990,
            1995,
            1000,
            1990,
            990,
            1995,
            1000,
            1995,
            995,
            1995,
            995,
            1995,
            995,
            2005,
            1000,
            1995,
            1005,
            1990,
            5,
            995,
            2000,
            1000,
            2005,
            1000,
            1990,
            995,
            1985,
            1000,
            1990,
            1000,
            2000,
            1005,
            2010,
            1005,
            1995,
            1005,
            2005,
            1000,
            2000,
            5,
            1005,
            2000,
            1000,
            2000,
            995,
            2005,
            1000,
            2000,
            1005,
            2005,
            1000,
            2010,
            995,
            2005,
            1005,
            1990,
            1005,
            1995,
            1000,
            1990,
        ]
    )
    return bvals


def get_hcp_bvals():
    bvals = np.array(
        [
            5,
            1000,
            1995,
            3005,
            995,
            2995,
            2005,
            990,
            1990,
            3000,
            1000,
            1985,
            2990,
            1005,
            1995,
            2995,
            5,
            995,
            2000,
            3010,
            3005,
            995,
            2005,
            995,
            1990,
            2985,
            1005,
            2000,
            3005,
            995,
            1995,
            2990,
            5,
            3005,
            995,
            2000,
            990,
            1985,
            2990,
            995,
            2985,
            1995,
            3005,
            995,
            2005,
            1000,
            1990,
            2995,
            5,
            1005,
            2005,
            2995,
            3000,
            1005,
            1985,
            1005,
            2985,
            2005,
            995,
            1990,
            3005,
            1000,
            2990,
            2000,
            5,
            990,
            2005,
            2995,
            1000,
            2000,
            3000,
            995,
            2005,
            3005,
            1005,
            2000,
            3010,
            990,
            2005,
            1995,
            5,
            1000,
            3010,
            2985,
            1005,
            1995,
            3000,
            1000,
            2010,
            3000,
            990,
            2005,
            2990,
            3000,
            1000,
            5,
            2990,
            1000,
            1995,
            1995,
            1005,
            2000,
            2994,
            1005,
            2010,
            2995,
            1005,
            2000,
            3010,
            1000,
            2005,
            2990,
            5,
            990,
            2985,
            2005,
            1000,
            2005,
            2985,
            995,
            2005,
            2990,
            995,
            2005,
            2990,
            1005,
            2000,
            3005,
            5,
            995,
            2995,
            1990,
            1005,
            995,
            2005,
            2990,
            2000,
            2995,
            995,
            2005,
            2985,
            990,
            1990,
            2985,
            5,
            990,
            2000,
            3010,
            995,
            1990,
            2995,
            1005,
            2000,
            3005,
            1000,
            1990,
            2995,
            2000,
            995,
            2990,
            5,
            995,
            3005,
            1990,
            3005,
            1000,
            2000,
            1000,
            1990,
            2995,
            2990,
            990,
            2010,
            1000,
            2005,
            3000,
            5,
            1005,
            2980,
            2005,
            1000,
            1985,
            2005,
            1000,
            2985,
            2995,
            990,
            1995,
            3005,
            2995,
            995,
            5,
            1995,
            1005,
            3000,
            1990,
            995,
            2010,
            2990,
            995,
            2005,
            2990,
            995,
            2000,
            2985,
            1005,
            2990,
            1995,
            5,
            3010,
            1005,
            1990,
            1000,
            2005,
            2990,
            995,
            2000,
            2995,
            1000,
            2005,
            2990,
            3000,
            1005,
            1990,
            5,
            995,
            3010,
            2005,
            995,
            2000,
            3000,
            2985,
            995,
            1990,
            2995,
            995,
            2000,
            995,
            2005,
            3010,
            5,
            2995,
            1000,
            1990,
            995,
            1990,
            3000,
            1000,
            2000,
            2985,
            990,
            1990,
            2990,
            1000,
            3005,
            1990,
            5,
            1005,
            1995,
            2990,
            1000,
            1990,
            3000,
            1000,
            2000,
            2985,
            995,
            1990,
            3000,
            1005,
            3005,
            2005,
            5,
            995,
            1995,
            3005,
            1000,
            995,
            2990,
            2985,
            2005,
            1990,
            1000,
            2990,
            2000,
            1005,
            2000,
            3000,
        ]
    )
    return bvals
