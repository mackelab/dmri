"""Acquisition helpers for diffusion and SSFP experiments.

This module wraps Stejskal–Tanner timing (:math:`b, \\delta, \\Delta`) into a
typed dataclass, exposes q-values :math:`q = \\sqrt{b/(4\\pi^2\\Delta)}` and gradient
strengths, and provides a second dataclass for SSFP sequences with common unit
conversions.
"""

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from jax.tree_util import register_dataclass
from probjax.utils.typing import Array, ArrayLike

WATER_DIFFUSION_CONSTANT = 2.299e-3  # mm^2/s
WATER_IN_AXON_DIFFUSION_CONSTANT = 1.7e-3  # mm^2/s
NAA_IN_AXONS = 0.00015e-3  # mm^2/s
WATER_GYROMAGNETIC_RATIO = 267.513e6  # 1/(sT)
WATER_GYROMAGNETIC_RATIO_MS_MT = 267.513  # rad/ms/mT

ACQ_CONSTANTS = {
    "HCP": {"delta": 0.0106, "Delta": 0.0431},
    "Biobank": {"delta": 0.0214, "Delta": 0.0455},
}


@runtime_checkable
class AcquisitionScheme(Protocol):
    """Abstract acquisition scheme interface for typing."""

    bvals: Array
    bvecs: Array
    delta: ArrayLike
    Delta: ArrayLike

    @property
    def qvals(self) -> ArrayLike: ...


@dataclass
class acquisition_scheme(AcquisitionScheme):  # noqa: N801
    """A class representing a diffusion MRI acquisition scheme.

    This class encapsulates the parameters needed to define a diffusion MRI acquisition,
    including b-values, gradient directions, and timing parameters.

    Attributes:
        bvals (ArrayLike): Array of b-values in s/mm^2, representing the diffusion weighting.
        bvecs (ArrayLike): Array of 3D unit vectors representing gradient directions.
        delta (ArrayLike): Time between the start of the diffusion encoding and the first gradient pulse in seconds.
        Delta (ArrayLike): Time between the start of the diffusion encoding and the second gradient pulse in seconds.
    """

    bvals: Array  # s/mm^2
    bvecs: Array  # unit vectors
    delta: ArrayLike = field(default_factory=lambda: 0.0106)  # In seconds
    Delta: ArrayLike = field(default_factory=lambda: 0.0431)  # In seconds

    @property
    def qvals(self) -> ArrayLike:
        """Calculate the q-values for the acquisition scheme.

        Returns:
            ArrayLike: Array of q-values in 1/mm, calculated from b-values and Delta.
        """
        return jnp.sqrt(self.bvals / (4 * jnp.pi**2 * self.Delta))  # In 1/mm

    @property
    def tau(self) -> ArrayLike:
        """Calculate the effective diffusion time tau.

        Returns:
            ArrayLike: Array of effective diffusion times in seconds.
        """
        return self.Delta - self.delta / 3

    @property
    def gradient_strengths(self) -> ArrayLike:
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
    ) -> "acquisition_scheme":
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


@dataclass
class ssfp_acquisition_scheme(AcquisitionScheme):  # noqa: N801
    r"""Steady-State Free Precession (SSFP) acquisition description.

    Parameters are stored in common MR units and converted as properties:

    - ``bvecs``: diffusion gradient unit vectors.
    - ``TRs``: repetition times [s].
    - ``flipAngles_raw``: flip angles [deg] (property ``flipAngles`` yields radians).
    - ``diffGradAmps_raw``: diffusion gradient amplitudes [T/m] (property ``diffGradAmps`` yields G/mm).
    - ``diffGradDur``: diffusion gradient durations [s].
    - ``B1``: transmit field scaling, unitless.
    - ``T1_raw`` / ``T2_raw``: relaxation times [ms] (properties ``T1``/``T2`` yield seconds).
    - ``delta`` / ``Delta``: diffusion pulse duration and separation [s].

    Derived quantities:

    - ``sa``, ``ca``: \(\sin(\alpha B_1)\), \(\cos(\alpha B_1)\) with \(\alpha\) in radians.
    - ``E1``, ``E2``: relaxation terms \(\exp(-\mathrm{TR}/T_1)\), \(\exp(-\mathrm{TR}/T_2)\).
    - ``qvals``: \(q = \gamma \, G \, \delta\) [1/mm] using ``gyromag_ratio`` in rad/ms/mT.
    - ``bvals``: \(b = q^2 \, 4\pi^2 \, \Delta\).
    """

    bvecs: Array  # unit vectors
    T1_raw: ArrayLike  # ms (will be converted to s in properties)
    T2_raw: ArrayLike  # ms (will be converted to s in properties)
    B1: ArrayLike  # unitless
    diffGradAmps_raw: ArrayLike  # noqa: N815  # T/m (will be converted to G/mm in properties)
    flipAngles_raw: ArrayLike = field(  # noqa: N815
        default_factory=lambda: 14.0
    )  # degrees (will be converted to radians in properties)
    TRs: ArrayLike = field(default_factory=lambda: 0.0210)  # seconds
    diffGradDur: ArrayLike = field(default_factory=lambda: 0.01016)  # noqa: N815  # seconds
    delta: ArrayLike = field(default_factory=lambda: 0.0106)
    Delta: ArrayLike = field(default_factory=lambda: 0.008)
    gyromag_ratio: ArrayLike = field(default_factory=lambda: 4258 * 2 * jnp.pi)  # Hz/G

    @property
    def T1(self) -> ArrayLike:
        """T1 converted to seconds."""
        return self.T1_raw * 1e-3  # ms -> s

    @property
    def T2(self) -> ArrayLike:
        """T2 converted to seconds."""
        return self.T2_raw * 1e-3  # ms -> s

    @property
    def diffGradAmps(self) -> ArrayLike:
        """Diffusion gradient amplitudes converted to G/mm."""
        return self.diffGradAmps_raw * 1e-1  # Convert to G/mm

    @property
    def flipAngles(self) -> ArrayLike:
        """Flip angles converted to radians."""
        return self.flipAngles_raw * jnp.pi / 180.0  # Convert to radians

    @property
    def sa(self) -> ArrayLike:
        """sin(flip angle * B1) [unitless]."""
        return jnp.sin(self.flipAngles * self.B1)

    @property
    def ca(self) -> ArrayLike:
        """cos(flip angle * B1) [unitless]."""
        return jnp.cos(self.flipAngles * self.B1)

    @property
    def E1(self) -> ArrayLike:
        """Longitudinal relaxation term E1 = exp(-TR/T1) [unitless]."""
        return jnp.exp(-self.TRs / self.T1)

    @property
    def E2(self) -> ArrayLike:
        """Transverse relaxation term E2 = exp(-TR/T2) [unitless]."""
        return jnp.exp(-self.TRs / self.T2)

    @property
    def logE1(self) -> ArrayLike:
        """Logarithm of longitudinal relaxation: logE1 = -TR/T1 [unitless]."""
        return -self.TRs / self.T1

    @property
    def logE2(self) -> ArrayLike:
        """Logarithm of transverse relaxation: logE2 = -TR/T2 [unitless]."""
        return -self.TRs / self.T2

    @property
    def qvals(self) -> ArrayLike:
        """Diffusion encoding q-values, q = γ·G·δ [1/mm].

        γ in rad/(ms·mT), G in T/m (≡ mT/mm), δ in ms.
        """
        return self.gyromag_ratio * self.diffGradAmps * self.diffGradDur

    @property
    def bvals(self) -> ArrayLike:
        """Effective b-values, b = - 1/D log(S_ssfp/S_ssfp)"""
        return self.qvals**2 * 4 * jnp.pi**2 * self.Delta
        # diffusivity = 0.0001# mm^2/ms
        # ssfp_signal_up = ssfp_signal_fn(diffusivity, self.qvals, self.E1, self.E2, self.sa, self.ca, self.TRs, self.diffGradDur)
        # ssfp_signal_down = ssfp_signal_fn(diffusivity, self.qvals, self.E1, self.E2, self.sa, self.ca, self.TRs, 0.)
        # return -1/diffusivity * jnp.log(ssfp_signal_up / ssfp_signal_down)


register_dataclass(
    acquisition_scheme, data_fields=("bvals", "bvecs"), meta_fields=("delta", "Delta")
)

register_dataclass(
    ssfp_acquisition_scheme,
    data_fields=(
        "bvecs",
        "TRs",
        "flipAngles_raw",
        "diffGradAmps_raw",
        "diffGradDur",
        "B1",
        "T1_raw",
        "T2_raw",
    ),
    meta_fields=("delta", "Delta", "gyromag_ratio"),
)


_bvals_typ, bvecs_typ = read_bvals_bvecs(
    os.path.join(os.path.dirname(__file__), "data/bvals"),
    os.path.join(os.path.dirname(__file__), "data/bvecs"),
)
# Round bvals to nearest (0, 1000, 2000)
bvals_typ = np.round(_bvals_typ / 1000) * 1000
bvals_typ = np.clip(bvals_typ, 0, 2000)
idx = np.argsort(bvals_typ)
bvals_typ = bvals_typ[idx]
bvecs_typ = bvecs_typ[idx]

_bvals_large, bvecs_large = read_bvals_bvecs(
    os.path.join(os.path.dirname(__file__), "data/bvals_large"),
    os.path.join(os.path.dirname(__file__), "data/bvecs_large"),
)
# Round bvals to nearest (0, 1000, 2000)
bvals_large = np.round(_bvals_large / 1000) * 1000
bvals_large = np.clip(bvals_large, 0, 4000)
idx = np.argsort(bvals_large)
bvals_typ_large = bvals_large[idx]
bvecs_typ_large = bvecs_large[idx]

diff_grad_amps_typical = np.loadtxt(
    os.path.join(os.path.dirname(__file__), "data/diffgrad_amps_ssfp")
)


bvecs_ssfp = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/bvecs_ssfp")).T


def random_ssfp_acquisition(
    rng, num_acquisitions=120, typical_prob=0.5, random_prob=0.5
):
    rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8, rng9, rng10 = jax.random.split(
        rng, 10
    )

    # For typical SSFP acquisition, we use fixed values from data files
    if num_acquisitions == 120:
        bvecs_typical = jnp.array(bvecs_ssfp)
        diffGradAmps_typical = jnp.array(diff_grad_amps_typical)
    else:
        bvecs_typical = jax.random.choice(
            rng1,
            jnp.array(bvecs_ssfp),
            shape=(num_acquisitions,),
            replace=True,
            axis=0,
        )
        diffGradAmps_typical = jax.random.choice(
            rng2,
            jnp.array(diff_grad_amps_typical),
            shape=(num_acquisitions,),
            replace=True,
            axis=0,
        )

    # Generate random values for the random component
    bvecs_random = jax.random.normal(rng1, (num_acquisitions, 3))
    bvecs_random = bvecs_random / jnp.linalg.norm(bvecs_random, axis=-1, keepdims=True)

    diffGradAmps_random = jax.random.uniform(rng2, (num_acquisitions,)) * 100
    flipAngles_random = (jax.random.uniform(rng3, ()) * 49 + 1) * jnp.ones((
        num_acquisitions,
    ))
    T1_random = jax.random.uniform(rng4, (num_acquisitions,)) * 1800 + 600
    T2_random = jax.random.uniform(rng5, (num_acquisitions,)) * 115 + 10
    diffGradDur_random = ((jax.random.uniform(rng6, ()) * 15 + 5) * 1e-3) * jnp.ones((
        num_acquisitions,
    ))
    TRs_random = ((jax.random.uniform(rng7, ()) * 45 + 5) * 1e-3) * jnp.ones((
        num_acquisitions,
    ))
    B1_random = jax.random.uniform(rng8, (num_acquisitions,)) * 2

    # For typical SSFP acquisition, use fixed values
    T1_typical = jax.random.gamma(rng9, 20, (num_acquisitions,)) * 50
    T2_typical = jax.random.gamma(rng10, 8, (num_acquisitions,)) * 8
    B1_typical = -jax.random.gamma(rng9, 2, (num_acquisitions,)) * 0.2 + 1.3
    diffGradDur_typical = jnp.ones((num_acquisitions,)) * 0.01016
    flipAngles_typical = jnp.ones((num_acquisitions,)) * 14.0
    TRs_typical = jnp.ones((num_acquisitions,)) * 0.0210

    # Choose between typical and random values based on probabilities
    bvecs = jax.random.choice(
        rng1,
        jnp.stack([bvecs_typical, bvecs_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    diffGradAmps = jax.random.choice(
        rng2,
        jnp.stack([diffGradAmps_typical, diffGradAmps_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    flipAngles = jax.random.choice(
        rng3,
        jnp.stack([flipAngles_typical, flipAngles_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    T1 = jax.random.choice(
        rng4,
        jnp.stack([T1_typical, T1_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    T2 = jax.random.choice(
        rng5,
        jnp.stack([T2_typical, T2_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    diffGradDur = jax.random.choice(
        rng6,
        jnp.stack([diffGradDur_typical, diffGradDur_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    TRs = jax.random.choice(
        rng7,
        jnp.stack([TRs_typical, TRs_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    B1 = jax.random.choice(
        rng8,
        jnp.stack([B1_typical, B1_random]),
        p=jnp.array([typical_prob, random_prob]),
    )

    return ssfp_acquisition_scheme(
        bvecs=bvecs,
        T1_raw=T1,
        T2_raw=T2,
        B1=B1,
        diffGradAmps_raw=diffGradAmps,
        flipAngles_raw=flipAngles,
        TRs=TRs,
        diffGradDur=diffGradDur,
    )


def random_hcp_large_acquisition(
    rng, num_acquisitions=297, typical_prob=0.8, random_prob=0.2
) -> acquisition_scheme:
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
    bvals_typical_large = jnp.array(bvals_typ_large)
    bvecs_typical_large = jnp.array(bvecs_typ_large)
    bvals_float = jax.random.uniform(rng2, shape=(num_acquisitions,)) * 4000
    # TODO remove restrictrictions
    bvals = jax.random.choice(
        rng3,
        jnp.stack([bvals_typical_large, bvals_float]),
        p=jnp.array([typical_prob, random_prob]),
    )
    bvals = jnp.sort(bvals)
    mask = jax.random.choice(
        rng4, jnp.array([0, 1]), shape=(num_acquisitions,), p=jnp.array([0.01, 0.99])
    )

    bvals = bvals * mask

    bvals = jnp.sort(bvals)

    # Create random directions
    rng5_1, rng5_2 = jax.random.split(rng5, 2)
    bvecs = jax.random.normal(rng5_1, shape=(num_acquisitions, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1)[:, None]

    bvecs = jax.random.choice(
        rng5_2,
        jnp.stack([bvecs_typical_large, bvecs], axis=0),
        axis=0,
        p=jnp.array([typical_prob, random_prob]),
    )

    return acquisition_scheme(bvals, bvecs)


def random_hcp_acquisition(
    rng, num_acquisitions=105, typical_prob=0.5, random_prob=0.5
) -> acquisition_scheme:
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
    bvals_typical = jnp.array(bvals_typ)
    bvecs_typical = jnp.array(bvecs_typ)
    bvals_float = jax.random.uniform(rng2, shape=(num_acquisitions,)) * 4000
    # TODO remove restrictrictions
    bvals = jax.random.choice(
        rng3,
        jnp.stack([bvals_typical, bvals_float]),
        p=jnp.array([typical_prob, random_prob]),
    )

    mask = jax.random.choice(
        rng4, jnp.array([0, 1]), shape=(num_acquisitions,), p=jnp.array([0.01, 0.99])
    )
    bvals = bvals * mask

    bvals = jnp.sort(bvals)

    # Create random directions
    rng5_1, rng5_2 = jax.random.split(rng5, 2)
    bvecs = jax.random.normal(rng5_1, shape=(num_acquisitions, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1)[:, None]

    bvecs = jax.random.choice(
        rng5_2,
        jnp.stack([bvecs_typical, bvecs], axis=0),
        axis=0,
        p=jnp.array([typical_prob, random_prob]),
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
    bvals = np.array([
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
    ])
    return bvals


def get_hcp_bvals():
    bvals = np.array([
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
    ])
    return bvals
