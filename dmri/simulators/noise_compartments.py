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

    def log_likelihood(cls, signal_pred, signal_true):
        log_likelihood = jnp.log(signal_true) - 2 * jnp.log(cls.snr)
        std = 1 / cls.snr
        log_likelihood -= 0.5 * (signal_true**2 + signal_pred**2) / std**2
        x = signal_true * signal_pred / std**2
        log_i0_value = x + jnp.log(jax.scipy.special.i0e(x))
        log_likelihood += log_i0_value
        return log_likelihood.sum(-1)

    @classmethod
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        return jnp.log(args[0])

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

    def log_likelihood(cls, signal_pred, signal_true):
        log_likelihood = jax.scipy.stats.norm.logpdf(
            signal_true, loc=signal_pred, scale=1 / cls.snr
        )
        return log_likelihood.sum(-1)

    @classmethod
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        return jnp.log(args[0])

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
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        # Invert scaling
        theta_sig = (args[0] - cls.min_snr) / (cls.max_snr - cls.min_snr)
        # Apply inverse sigmoid
        return jnp.log(theta_sig / (1 - theta_sig))

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        return (jax.nn.sigmoid(theta) * (cls.max_snr - cls.min_snr) + cls.min_snr,)


class BoundedGaussianNoise(GaussianNoise):
    theta_dim = 1
    min_snr = 10
    max_snr = 60

    def __init__(self, snr):
        self.snr = snr

    @classmethod
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        # Invert scaling
        theta_sig = (args[0] - cls.min_snr) / (cls.max_snr - cls.min_snr)
        # Apply inverse sigmoid
        return jnp.log(theta_sig / (1 - theta_sig))

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        return (jax.nn.sigmoid(theta) * (cls.max_snr - cls.min_snr) + cls.min_snr,)

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
    # Generate Gaussian noise for real and imaginary parts
    noise_real = jax.random.normal(rng, shape=signal.shape) * sigma_g
    noise_imag = jax.random.normal(rng, shape=signal.shape) * sigma_g

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
    # Generate Gaussian noise for real and imaginary parts for each coil
    noise_real = jax.random.normal(rng, shape=(num_coils,) + signal.shape) * sigma_g
    noise_imag = jax.random.normal(rng, shape=(num_coils,) + signal.shape) * sigma_g

    # Assume original signal is along the real axis and replicate for each coil
    signal_complex = jnp.tile(signal, (num_coils, 1, 1)) + 0j

    # Add noise
    noisy_complex = signal_complex + noise_real + 1j * noise_imag

    # Compute magnitude and sum over coils
    noisy_signal = jnp.sqrt(jnp.sum(jnp.abs(noisy_complex) ** 2, axis=0))

    return noisy_signal
