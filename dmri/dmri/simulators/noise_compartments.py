import jax
import jax.numpy as jnp

from simulators.base import NoiseCompartment


class RicianNoise(NoiseCompartment):
    def __init__(self, sigma_g):
        self.sigma_g = sigma_g

    def noise(self, rng, signal):
        return add_rician_noise(rng, signal, self.sigma_g)


class GaussianNoise(NoiseCompartment):
    def __init__(self, sigma_g):
        self.sigma_g = sigma_g

    def noise(self, rng, signal):
        return add_gaussian_noise(rng, signal, self.sigma_g)


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
