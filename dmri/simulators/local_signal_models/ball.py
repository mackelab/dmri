import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    ssfp_acquisition_scheme,
)
from dmri.simulators.base import SignalCompartment
from dmri.simulators.sphereical_distributions import Uniform
from dmri.utils.dmriutils import ssfp_signal_fn


class Ball(SignalCompartment):
    """The Ball model is a simple model that represents free water diffusion in
    unrestricted space. It is fully isotropic.
    It has a single parameter lambda that represents the diffusivity of water molecules.
    """

    theta_dim: int = 1
    lam_min: float = 0.0
    lam_max: float = 0.01  # 1e-3 mm^2/s is the diffusivity of free water

    def __init__(self, lam: float) -> None:
        """Initialize the Ball model with a lambda value."""
        self.lam = lam

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, lam: float, rng=None) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        logS = -acq.bvals * lam
        return logS

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[...,0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)

    @classmethod
    def to_theta(cls, lam: ArrayLike) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> tuple:
        """Fit the Ball model to the log signal and b-values."""
        lam = -logS / bvals
        return (jnp.mean(lam),)

    def to_fod(self):
        return Uniform()


class StaticBall(Ball):
    """
    The StaticBall model is a Ball with a fixed lambda value.
    The lam parameter is shared from a global parameter state as a class attribute.
    """

    theta_dim: int = 0  # No learnable parameters, lam is fixed
    lam: float = None  # Class attribute for the fixed lambda value

    def __init__(self) -> None:
        """Initialize the StaticBall model.
        The lam parameter is a class attribute and not passed to the constructor.
        """
        if type(self).lam is None:
            raise ValueError("lam must be set before initializing StaticBall")

    @classmethod
    def to_theta(cls) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        return jnp.array([])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to parameters.
        Since there are no learnable parameters, return an empty tuple.
        """
        return ()

    @classmethod
    def from_global_params(cls, params: ArrayLike, idx: list[int]) -> "StaticBall":
        """Create a StaticBall from a global theta value.

        Parameters
        ----------
        theta : ArrayLike
            The global parameter array
        idx : list[int]
            Indices in the global parameter array that correspond to this model's parameters

        Returns
        -------
        StaticBall
            A StaticBall instance with parameters extracted from the global theta
        """
        # Set the fixed lambda value as a class attribute
        cls.lam = params[idx[0]]
        assert len(idx) == 1, "StaticBall only has one fixed parameter, lam"
        return cls

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, rng=None) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        acq: acquisition_scheme
        logS = -acq.bvals * cls.lam
        return logS


class MultiShellBall(Ball):
    """
    The MultiShellBall model is a Ball with multiple lambda values.
    The lam parameter is shared from a global parameter state as a class attribute.
    """

    theta_dim: int = 2  # No learnable parameters, lam is fixed
    lam: float
    lam_std: float
    lam_std_min: float = 0.0000001
    lam_std_max: float = 0.01

    def __init__(self, lam: float, lam_std: float) -> None:
        """Initialize the MultiShellBall model.
        The lam parameter is a class attribute and not passed to the constructor.
        """
        self.lam = lam
        self.lam_std = lam_std

    @classmethod
    def log_signal_fn(
        cls, acq: acquisition_scheme, lam: float, lam_std: float, rng=None
    ) -> ArrayLike:
        """Computes the log-signal for the ball compartment with uncertainty."""
        acq: acquisition_scheme
        return multi_shell_ball_log_signal_fn(acq.bvals, lam, lam_std)

    @classmethod
    def to_theta(cls, lam: float, lam_std: float) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        u_lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        u_lam_std = (lam_std - cls.lam_std_min) / (cls.lam_std_max - cls.lam_std_min)
        theta_u = jnp.array([u_lam, u_lam_std])
        theta = jax.scipy.stats.norm.ppf(theta_u)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to parameters.
        Since there are no learnable parameters, return an empty tuple.
        """
        u = jax.scipy.stats.norm.cdf(theta)
        lam = u[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        lam_std = u[1] * (cls.lam_std_max - cls.lam_std_min) + cls.lam_std_min
        return (lam, lam_std)


class MultiShellStaticBall(MultiShellBall):
    """
    The MultiShellBall model is a Ball with multiple lambda values.
    The lam parameter is shared from a global parameter state as a class attribute.
    """

    theta_dim: int = 0  # No learnable parameters, lam is fixed
    lam: float = None
    lam_std: float = None

    def __init__(self) -> None:
        pass

    @classmethod
    def from_global_params(cls, params: ArrayLike, idx: list[int]) -> "StaticBall":
        """Create a StaticBall from a global theta value.

        Parameters
        ----------
        theta : ArrayLike
            The global parameter array
        idx : list[int]
            Indices in the global parameter array that correspond to this model's parameters

        Returns
        -------
        StaticBall
            A StaticBall instance with parameters extracted from the global theta
        """
        # Set the fixed lambda value as a class attribute
        cls.lam = params[idx[0]]
        cls.lam_std = params[idx[1]]
        assert len(idx) == 2, "MultiShellBall has two fixed parameters, lam and lam_std"
        return cls

    @classmethod
    def to_theta(cls) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        return jnp.array([])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to parameters.
        Since there are no learnable parameters, return an empty tuple.
        """
        return ()

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, rng=None) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        acq: acquisition_scheme
        return multi_shell_ball_log_signal_fn(acq.bvals, cls.lam, cls.lam_std)


class SSFPBall(SignalCompartment):
    """
    The SSFPBall model is a Ball with a fixed lambda value.
    The lam parameter is shared from a global parameter state as a class attribute.
    """

    theta_dim: int = 1
    lam_min: float = 0.0
    lam_max: float = 0.01

    def __init__(self, lam: float) -> None:
        self.lam = lam

    @classmethod
    def signal_fn(cls, acq: ssfp_acquisition_scheme, lam: float, rng=None) -> ArrayLike:
        """Computes the signal for the ball compartment."""
        # Relaxation terms
        # E1 = jnp.exp(-acq.TRs / (1e-3 * acq.T1))  # T1
        # E2 = jnp.exp(-acq.TRs / (1e-3 * acq.T2))  # T2

        # # Flipping terms
        # sa = jnp.sin(acq.flipAngles * acq.B1 * jnp.pi / 180.0)  # sin(flip * B1)
        # ca = jnp.cos(acq.flipAngles * acq.B1 * jnp.pi / 180.0)  # cos(flip * B1)
        qvals = acq.qvals
        E1 = acq.E1
        E2 = acq.E2
        sa = acq.sa
        ca = acq.ca
        TR = acq.TRs
        grad_diff_dur = acq.diffGradDur
        lam = lam

        signal = ssfp_signal_fn(lam, qvals, E1, E2, sa, ca, TR, grad_diff_dur)
        return signal

    @classmethod
    def log_signal_fn(
        cls, acq: ssfp_acquisition_scheme, lam: float, rng=None
    ) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        signal = cls.signal_fn(acq, lam, rng)
        return jnp.log(signal)

    @classmethod
    def to_theta(cls, lam: float) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        u_lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jax.scipy.stats.norm.ppf(u_lam)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to parameters.
        Since there are no learnable parameters, return an empty tuple.
        """
        u_lam = jax.scipy.stats.norm.cdf(theta)
        lam = u_lam * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)


class SSFPStaticBall(SSFPBall):
    """
    The SSFPStaticBall model is a SSFPBall with a fixed lambda value.
    The lam parameter is shared from a global parameter state as a class attribute.
    """

    theta_dim: int = 0  # No learnable parameters, lam is fixed
    lam: float = None

    def __init__(self) -> None:
        if type(self).lam is None:
            raise ValueError("lam must be set before initializing SSFPStaticBall")

    @classmethod
    def from_global_params(cls, params: ArrayLike, idx: list[int]) -> "StaticBall":
        """Create a StaticBall from a global theta value."""
        cls.lam = params[idx[0]]
        assert len(idx) == 1, "SSFPStaticBall only has one fixed parameter, lam"
        return cls

    @classmethod
    def to_theta(cls) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        return jnp.array([])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to parameters.
        Since there are no learnable parameters, return an empty tuple.
        """
        return ()

    @classmethod
    def signal_fn(cls, acq: ssfp_acquisition_scheme, rng=None) -> ArrayLike:
        """Computes the signal for the ball compartment."""
        return SSFPBall.signal_fn(acq, cls.lam, rng)

    @classmethod
    def log_signal_fn(cls, acq: ssfp_acquisition_scheme, rng=None) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        return SSFPBall.log_signal_fn(acq, cls.lam, rng)


def multi_shell_ball_log_signal_fn(
    bvals: ArrayLike,
    lam: float,
    lam_std: float,
) -> ArrayLike:
    """Implementation with float32 numerical stability for MultiShellBall.

    Parameters
    ----------
    bvals : ArrayLike
        The b-values of the acquisition scheme
    lam : float
        The mean diffusivity
    lam_std : float
        The standard deviation of the diffusivity

    Returns
    -------
    ArrayLike
        The log signal
    """
    nugget = jnp.finfo(bvals.dtype).eps

    # Add small constant to prevent division by zero
    scaling = (lam / (lam_std + nugget)) ** 2

    # Use stable log computation
    logS = jnp.log(lam + nugget) - jnp.log(lam + bvals * lam_std**2 + nugget)

    return scaling * logS
