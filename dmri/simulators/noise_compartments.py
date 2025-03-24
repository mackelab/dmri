from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import NoiseCompartment

# NOTE: Review on noise models in MRI:
# https://www.lpi.tel.uva.es/~santi/personal/docus/noise_survey_tec_report.pdf


class RicianNoise(NoiseCompartment):
    theta_dim = 1

    def __init__(self, sigma_g):
        self.sigma_g = sigma_g

    def noise(self, signal, rng):
        return add_rician_noise(rng, signal, self.sigma_g)

    def log_likelihood(cls, signal_pred, signal_true):
        log_likelihood = jnp.log(signal_true) - 2 * jnp.log(cls.sigma_g)
        log_likelihood -= 0.5 * (signal_true**2 + signal_pred**2) / cls.sigma_g**2
        x = signal_true * signal_pred / cls.sigma_g**2
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

    def __init__(self, sigma_g):
        self.sigma_g = sigma_g

    def noise(self, signal, rng):
        return add_gaussian_noise(rng, signal, self.sigma_g)

    def log_likelihood(cls, signal_pred, signal_true):
        log_likelihood = jax.scipy.stats.norm.logpdf(
            signal_true, loc=signal_pred, scale=cls.sigma_g
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
    min_sigma_g = 0.0
    max_sigma_g = 0.5

    def __init__(self, sigma_g):
        self.sigma_g = sigma_g

    def noise(self, signal, rng):
        return add_rician_noise(rng, signal, self.sigma_g)

    @classmethod
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        # Invert scaling
        theta_sig = (args[0] - cls.min_sigma_g) / (cls.max_sigma_g - cls.min_sigma_g)
        # Apply inverse sigmoid
        return jnp.log(theta_sig / (1 - theta_sig))

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        return (
            jax.nn.sigmoid(theta) * (cls.max_sigma_g - cls.min_sigma_g)
            + cls.min_sigma_g,
        )


class BoundedGaussianNoise(GaussianNoise):
    theta_dim = 1
    min_sigma_g = 0.0
    max_sigma_g = 0.5

    def __init__(self, sigma_g):
        self.sigma_g = sigma_g

    def noise(self, signal, rng):
        return add_gaussian_noise(rng, signal, self.sigma_g)

    @classmethod
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        # Invert scaling
        theta_sig = (args[0] - cls.min_sigma_g) / (cls.max_sigma_g - cls.min_sigma_g)
        # Apply inverse sigmoid
        return jnp.log(theta_sig / (1 - theta_sig))

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        return (
            jax.nn.sigmoid(theta) * (cls.max_sigma_g - cls.min_sigma_g)
            + cls.min_sigma_g,
        )


class LowRicianNoise(BoundedRicianNoise):
    min_sigma_g = 0.0
    max_sigma_g = 0.05


class MediumRicianNoise(BoundedRicianNoise):
    min_sigma_g = 0.05
    max_sigma_g = 0.1


class LargeRicianNoise(BoundedRicianNoise):
    min_sigma_g = 0.1
    max_sigma_g = 0.2


class VeryLargeRicianNoise(BoundedRicianNoise):
    min_sigma_g = 0.2
    max_sigma_g = 0.5


class LowGaussianNoise(BoundedGaussianNoise):
    min_sigma_g = 0.0
    max_sigma_g = 0.05


class MediumGaussianNoise(BoundedGaussianNoise):
    min_sigma_g = 0.05
    max_sigma_g = 0.1


class LargeGaussianNoise(BoundedGaussianNoise):
    min_sigma_g = 0.1
    max_sigma_g = 0.2


class VeryLargeGaussianNoise(BoundedGaussianNoise):
    min_sigma_g = 0.2
    max_sigma_g = 0.5


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
