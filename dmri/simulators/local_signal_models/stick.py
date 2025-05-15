import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment, acquisition_scheme
from dmri.simulators.sphereical_distributions import SymmetricDirac
from dmri.utils.dmriutils import (
    cartesian_to_unitsphere,
    fit_diffusion_tensor_linearized,
    unitsphere_to_cartesian,
)


class Stick(SignalCompartment):
    """The Stick model represents a single fiber bundle with a fixed orientation i.e.
    a cylinder with zero radius.

    It represents fully anisotropic diffusion along the fiber orientation.
    """

    theta_dim = 3
    min_lam = 0.00001
    max_lam = 0.01

    def __init__(self, mu: ArrayLike, lam_par: float) -> None:
        """Initialize the Stick model with a lambda value and eigenvector."""
        self.mu = mu
        self.lam_par = lam_par

    @classmethod
    def log_signal_fn(
        cls,
        aquisition_scheme: acquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        mu_cart = unitsphere_to_cartesian(mu)
        logS = -bvals * lam_par * (jnp.sum(bvecs * mu_cart, axis=-1)) ** 2
        return logS

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        lam_par = (lam_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([lam_par, mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value and eigenvector."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        mu1 = jnp.arccos(1 - theta[1])  # Ensures output is in upper hemisphere
        mu2 = theta[2] * 2 * jnp.pi - jnp.pi

        mu = jnp.array([mu1, mu2])

        return mu, lam_par

    def to_fod(self):
        mu_cart = unitsphere_to_cartesian(self.mu)
        return SymmetricDirac(mu_cart)

    def fit(self, logS: ArrayLike, aquisition_scheme: acquisition_scheme) -> tuple:
        """Fit the Stick model to the log signal, b-values, and b-vectors."""
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam_par = eigvals[idx]
        eigvec = eigvecs[:, idx]
        mu = cartesian_to_unitsphere(eigvec)
        return mu, lam_par


class StaticStick(Stick):
    """
    The StaticStick model is a Stick with a fixed lambda value.
    The lam_par parameter is shared from a global parameter state as a class attribute,
    while the mu parameter remains learnable.
    """

    theta_dim: int = 2  # Only mu parameters are learnable
    lam_par: float = None  # Class attribute for the fixed lambda value

    def __init__(self, mu: ArrayLike) -> None:
        """Initialize the StaticStick model with a mu value.
        The lam_par parameter is a class attribute and not passed to the constructor.
        """
        if type(self).lam_par is None:
            raise ValueError("lam_par must be set before initializing StaticStick")
        self.mu = mu

    @classmethod
    def to_theta(cls, mu: ArrayLike) -> ArrayLike:
        """Convert only the mu parameter to the parameter space theta.
        The lam_par parameter is a class attribute and not included in theta.
        """
        # Only convert mu parameters to theta space
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the mu value.
        The lam_par parameter is a class attribute and not included in theta.
        """
        theta = jax.scipy.stats.norm.cdf(theta)
        mu1 = jnp.arccos(1 - theta[0])  # Ensures output is in upper hemisphere
        mu2 = theta[1] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu1, mu2])
        return (mu,)

    @classmethod
    def from_global_params(cls, params: ArrayLike, idx: list[int]) -> type:
        """Create a StaticStick from a global theta value.

        Parameters
        ----------
        theta : ArrayLike
            The global parameter array
        idx : list[int]
            Indices in the global parameter array that correspond to this model's parameters

        Returns
        -------
        StaticStick
            A StaticStick instance with parameters extracted from the global theta
        """
        # Set the fixed lam_par value as a class attribute
        cls.lam_par = params[idx[0]]
        assert len(idx) == 1, "StaticStick only has one fixed parameter, lam_par"
        return cls

    @classmethod
    def log_signal_fn(
        cls,
        aquisition_scheme: acquisition_scheme,
        mu: ArrayLike,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors.

        Parameters
        ----------
        aquisition_scheme : acquisition_scheme
            The acquisition scheme containing b-values and b-vectors
        mu : ArrayLike
            The orientation of the stick in spherical coordinates
        rng : jax.random.key, optional
            Random number generator key, by default None

        Returns
        -------
        ArrayLike
            The log signal
        """
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        mu_cart = unitsphere_to_cartesian(mu)
        logS = -bvals * cls.lam_par * (jnp.sum(bvecs * mu_cart, axis=-1)) ** 2
        return logS


class MultiShellStick(Stick):
    """
    The MultiShellStick model is a Stick with a fixed lambda value.
    The lam_par parameter is shared from a global parameter state as a class attribute,
    while the mu parameter remains learnable.
    """

    theta_dim: int = 4
    lam_par: float
    lam_par_std: float
    lam_par_std_min: float = 0
    lam_par_std_max: float = 0.005

    def __init__(self, mu: ArrayLike, lam_par: float, lam_par_std: float) -> None:
        """Initialize the MultiShellStick model with a mu value, lam_par, and lam_par_std."""
        self.mu = mu
        self.lam_par = lam_par
        self.lam_par_std = lam_par_std

    @classmethod
    def log_signal_fn(
        cls,
        aquisition_scheme: acquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        lam_par_std: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors.

        Parameters
        ----------
        aquisition_scheme : acquisition_scheme
            The acquisition scheme containing b-values and b-vectors
        """
        bvals = aquisition_scheme.bvals
        bvecs = aquisition_scheme.bvecs
        mu_cart = unitsphere_to_cartesian(mu)

        scaling = lam_par**2 / lam_par_std**2
        dot_product = (jnp.sum(bvecs * mu_cart, axis=-1)) ** 2
        logS = scaling * jnp.log(
            lam_par / (lam_par + bvals * dot_product * lam_par_std**2)
        )
        return logS

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float, lam_par_std: float) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        theta = jnp.array([mu, lam_par, lam_par_std])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float, lam_par_std: float) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        lam_par = (lam_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
        lam_par_std = (lam_par_std - cls.lam_par_std_min) / (
            cls.lam_par_std_max - cls.lam_par_std_min
        )
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([lam_par, lam_par_std, mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value and eigenvector."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        lam_par_std = (
            theta[1] * (cls.lam_par_std_max - cls.lam_par_std_min) + cls.lam_par_std_min
        )
        mu1 = jnp.arccos(1 - theta[2])  # Ensures output is in upper hemisphere
        mu2 = theta[3] * 2 * jnp.pi - jnp.pi

        mu = jnp.array([mu1, mu2])

        return mu, lam_par, lam_par_std


class MultiShellStaticStick(MultiShellStick):
    """
    The MultiShellStaticStick model is a Stick with a fixed lambda value.
    The lam_par parameter is shared from a global parameter state as a class attribute,
    while the mu parameter remains learnable.
    """

    theta_dim: int = 2
    lam_par: float = None
    lam_par_std: float = None

    def __init__(self, mu: ArrayLike) -> None:
        """Initialize the MultiShellStaticStick model with a mu value."""
        self.mu = mu

    @classmethod
    def to_theta(cls, mu: ArrayLike) -> ArrayLike:
        """Convert only the mu parameter to the parameter space theta.
        The lam_par parameter is a class attribute and not included in theta.
        """
        # Only convert mu parameters to theta space
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the mu value.
        The lam_par parameter is a class attribute and not included in theta.
        """
        theta = jax.scipy.stats.norm.cdf(theta)
        mu1 = jnp.arccos(1 - theta[0])  # Ensures output is in upper hemisphere
        mu2 = theta[1] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu1, mu2])
        return (mu,)

    @classmethod
    def from_global_params(cls, params: ArrayLike, idx: list[int]) -> type:
        """Create a MultiShellStaticStick from a global theta value.

        Parameters
        ----------
        theta : ArrayLike
            The global parameter array
        idx : list[int]
            Indices in the global parameter array that correspond to this model's parameters

        Returns
        -------
        MultiShellStaticStick
            A MultiShellStaticStick instance with parameters extracted from the global theta
        """
        cls.lam_par = params[idx[0]]
        cls.lam_par_std = params[idx[1]]
        return cls

    @classmethod
    def log_signal_fn(
        cls,
        aquisition_scheme: acquisition_scheme,
        mu: ArrayLike,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors.

        Parameters
        ----------
        aquisition_scheme : acquisition_scheme
            The acquisition scheme containing b-values and b-vectors
        mu : ArrayLike
            The orientation of the stick in spherical coordinates
        rng : jax.random.key, optional
            Random number generator key, by default None

        Returns
        -------
        ArrayLike
            The log signal
        """
        return MultiShellStick.log_signal_fn(
            aquisition_scheme, mu, cls.lam_par, cls.lam_par_std
        )
