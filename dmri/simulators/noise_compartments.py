"""Noise models for diffusion MRI signals.

Implements bounded Rician and Gaussian noise commonly observed in magnitude
MR images. The likelihoods follow:

.. math::
    p(z \\mid \\nu, \\sigma) = \\frac{z}{\\sigma^2} \\exp\\Big(-\\tfrac{z^2+\\nu^2}{2\\sigma^2}\\Big) I_0\\Big(\\tfrac{z\\nu}{\\sigma^2}\\Big)

with :math:`I_0` the modified Bessel function (Gudbjartsson & Patz, 1995). SNR
priors are kept in bounded ranges for numerical stability.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import NoiseCompartment

# NOTE: Review on noise models in MRI:
# https://www.lpi.tel.uva.es/~santi/personal/docus/noise_survey_tec_report.pdf

# Note: We now follow: https://www.biorxiv.org/content/10.1101/2024.11.19.624267v1.full.pdf
# Uses a inverse uniform distribution to sample the std or a uniform distribution on the snr
# and then compute the std by 1/snr


class RicianNoise(NoiseCompartment):
    theta_dim = 1

    def __init__(self, snr):
        self.snr = snr

    def noise(self, signal, rng):
        return add_rician_noise(rng, signal, 1 / self.snr)

    def log_likelihood(self, signal_pred, signal_true):
        sigma = 1.0 / self.snr
        z = signal_true
        nu = signal_pred

        # Avoid log(0); clamp if needed:
        z = jnp.clip(z, 1e-12, None)

        x = z * nu / sigma**2  # argument for I0

        log_i0 = jnp.log(jax.scipy.special.i0e(x)) + jnp.abs(x)

        log_p = (
            jnp.log(z) - 2.0 * jnp.log(sigma) - 0.5 * (z**2 + nu**2) / sigma**2 + log_i0
        )

        return log_p.sum(-1)

    @classmethod
    def to_theta(cls, snr: ArrayLike) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        return jnp.log(snr)

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        return (jnp.exp(theta),)


class GaussianNoise(NoiseCompartment):
    theta_dim = 1

    def __init__(self, snr):
        self.snr = snr

    def noise(self, signal, rng):
        return add_gaussian_noise(rng, signal, 1 / self.snr)

    def log_likelihood(self, signal_pred, signal_true):
        log_likelihood = jax.scipy.stats.norm.logpdf(
            signal_true, loc=signal_pred, scale=1 / self.snr
        )
        return log_likelihood.sum(-1)

    @classmethod
    def to_theta(cls, snr: ArrayLike) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        return jnp.log(snr)

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        return (jnp.exp(theta),)


class BoundedRicianNoise(RicianNoise):
    theta_dim = 1
    min_snr = 3
    max_snr = 80

    def __init__(self, snr):
        self.snr = snr

    @classmethod
    def to_theta(cls, snr: ArrayLike) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        # Invert scaling
        theta_u = (snr - cls.min_snr) / (cls.max_snr - cls.min_snr)
        theta = jax.scipy.stats.norm.ppf(theta_u)
        # Apply inverse sigmoid
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        u = jax.scipy.stats.norm.cdf(theta)
        return (u * (cls.max_snr - cls.min_snr) + cls.min_snr,)


class BoundedGaussianNoise(GaussianNoise):
    theta_dim = 1
    min_snr = 3
    max_snr = 80

    def __init__(self, snr):
        self.snr = snr

    @classmethod
    def to_theta(cls, snr: ArrayLike) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        # Invert scaling
        theta_u = (snr - cls.min_snr) / (cls.max_snr - cls.min_snr)
        theta = jax.scipy.stats.norm.ppf(theta_u)
        # Apply inverse sigmoid
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        u = jax.scipy.stats.norm.cdf(theta)
        return (u * (cls.max_snr - cls.min_snr) + cls.min_snr,)


class RicianNoiseSNR7080(BoundedRicianNoise):
    min_snr = 70
    max_snr = 80


class RicianNoiseSNR6070(BoundedRicianNoise):
    min_snr = 60
    max_snr = 70


class RicianNoiseSNR5060(BoundedRicianNoise):
    min_snr = 50
    max_snr = 60


class RicianNoiseSNR4050(BoundedRicianNoise):
    min_snr = 40
    max_snr = 50


class RicianNoiseSNR3040(BoundedRicianNoise):
    min_snr = 30
    max_snr = 40


class RicianNoiseSNR2030(BoundedRicianNoise):
    min_snr = 20
    max_snr = 30


class RicianNoiseSNR1020(BoundedRicianNoise):
    min_snr = 10
    max_snr = 20


class RicianNoiseSNR310(BoundedRicianNoise):
    min_snr = 3
    max_snr = 10


class GaussianNoiseSNR7080(BoundedGaussianNoise):
    min_snr = 70
    max_snr = 80


class GaussianNoiseSNR6070(BoundedGaussianNoise):
    min_snr = 60
    max_snr = 70


class GaussianNoiseSNR5060(BoundedGaussianNoise):
    min_snr = 50
    max_snr = 60


class GaussianNoiseSNR4050(BoundedGaussianNoise):
    min_snr = 40
    max_snr = 50


class GaussianNoiseSNR3040(BoundedGaussianNoise):
    min_snr = 30
    max_snr = 40


class GaussianNoiseSNR2030(BoundedGaussianNoise):
    min_snr = 20
    max_snr = 30


class GaussianNoiseSNR1020(BoundedGaussianNoise):
    min_snr = 10
    max_snr = 20


class GaussianNoiseSNR310(BoundedGaussianNoise):
    min_snr = 3
    max_snr = 10


def add_rician_noise(rng, signal, sigma_g):
    """
    Adds Rician noise to an MRI signal.

    NOTE: The noise follows a Rician distribution when the signal-to-noise ratio (SNR)
    is moderate to high

    Parameters:
    - signal: ndarray of the true signal magnitude (η)
    - sigma_g: standard deviation of the Gaussian noise (σ_g)


    Returns:
    - noisy_signal: ndarray of the signal with added Rician noise
    """
    # Independent keys: drawing both parts from `rng` would make them identical
    # and the magnitude would not be Rician.
    rng_real, rng_imag = jax.random.split(rng)
    noise_real = jax.random.normal(rng_real, shape=signal.shape) * sigma_g
    noise_imag = jax.random.normal(rng_imag, shape=signal.shape) * sigma_g

    # Assume original signal is along the real axis
    signal_complex = signal + 0j

    # Add noise
    noisy_complex = signal_complex + noise_real + 1j * noise_imag

    # Compute magnitude
    noisy_signal = jnp.abs(noisy_complex)

    return noisy_signal


def add_gaussian_noise(rng, signal, sigma_g):
    """
    Adds Gaussian noise to an MRI signal.

    NOTE: Often used for simplicity, especially when modeling raw complex data or when
    SNR is high.

    Parameters:
    - signal: ndarray of the true signal magnitude (η)
    - sigma_g: standard deviation of the Gaussian noise (σ_g)

    Returns:
    - noisy_signal: ndarray of the signal with added Gaussian noise
    """
    # Generate Gaussian noise
    noise = jax.random.normal(rng, shape=signal.shape) * sigma_g

    # Add noise
    noisy_signal = signal + noise

    return noisy_signal


def add_noncentral_chi_noise(rng, signal, sigma_g, num_coils):
    """
    Adds Non-Central Chi noise to an MRI signal.

    Nature: Arises in multi-coil acquisition systems due to the combination of multiple
    complex signals.Characteristics: The noise distribution in the magnitude image
    follows a non-central Chi distribution. Degree of freedom depends on the number of
    coils used in parallel imaging. Implications: More complex than the Rician model and
    can complicate bias correction.

    Parameters:
    - signal: ndarray of the true signal magnitude (η)
    - sigma_g: standard deviation of the Gaussian noise (σ_g)
    - num_coils: number of coils used in parallel imaging

    Returns:
    - noisy_signal: ndarray of the signal with added Non-Central Chi noise
    """
    # Independent keys per part, as in add_rician_noise.
    rng_real, rng_imag = jax.random.split(rng)
    noise_real = (
        jax.random.normal(rng_real, shape=(num_coils,) + signal.shape) * sigma_g
    )
    noise_imag = (
        jax.random.normal(rng_imag, shape=(num_coils,) + signal.shape) * sigma_g
    )

    # Assume original signal is along the real axis and replicate for each coil
    signal_complex = jnp.tile(signal, (num_coils, 1, 1)) + 0j

    # Add noise
    noisy_complex = signal_complex + noise_real + 1j * noise_imag

    # Compute magnitude and sum over coils
    noisy_signal = jnp.sqrt(jnp.sum(jnp.abs(noisy_complex) ** 2, axis=0))

    return noisy_signal
