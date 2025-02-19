from functools import partial
from typing import Any
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

from jax.typing import ArrayLike
from jax import tree_util as jtu
from dmri.simulators.base import Compartment, SignalCompartment
from dmri.simulators.sphereical_distributions import (
    Bingham,
    Watson,
    SphericalDistribution,
    inverse_sh_matrix,
    hemisphere_default,
)
from dmri.utils.dmriutils import cartesian_to_unitsphere
from dmri.simulators.local_signal_models.gaussian_models import Stick, Zeppelin


class SignalKernel(Compartment):
    @classmethod
    @abstractmethod
    def kernel_fn(cls, mu: ArrayLike, bvals, bvecs) -> ArrayLike:
        pass

    def sh_coeff(self, bvals, bvecs, sh_order):
        inverse_real_sh = inverse_sh_matrix(sh_order, sphere=hemisphere_default)
        kernel = partial(self.kernel_fn, **self.params)
        signal = jax.vmap(kernel, in_axes=(0, None, None))(
            hemisphere_default.vertices, bvals, bvecs
        )
        sh_coeff = inverse_real_sh @ signal.squeeze()
        return sh_coeff


class StickKernel(SignalKernel):
    theta_dim: int = 1
    lam_min: float = Stick.max_lam
    lam_max: float = Stick.min_lam

    def __init__(self, lam: float):
        self.lam = lam

    @classmethod
    def kernel_fn(cls, mu: ArrayLike, bvals, bvecs, lam: float, rng=None) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        signal_fn = jax.vmap(Stick.signal_fn, in_axes=(0, 0, None, None))
        return signal_fn(bvals, bvecs, lam, mu)

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)

    @classmethod
    def to_theta(cls, lam: ArrayLike) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta


class ZeppelinKernel(SignalKernel):
    theta_dim: int = 2
    lam_min: float = Zeppelin.min_lam
    lam_max: float = Zeppelin.max_lam

    def __init__(self, lam_perp: float, lam_par: float):
        self.lam_perp = lam_perp
        self.lam_par = lam_par

    @classmethod
    def kernel_fn(
        cls, mu: ArrayLike, bvals, bvecs, lam_perp: float, lam_par: float, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        signal_fn = jax.vmap(Zeppelin.signal_fn, in_axes=(0, 0, None, None, None))
        mu = cartesian_to_unitsphere(mu)
        return signal_fn(bvals, bvecs, mu, lam_perp, lam_par)

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_perp = theta[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        lam_par = theta[1] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam_perp, lam_par)

    @classmethod
    def to_theta(cls, lam_perp: float, lam_par: float) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        lam_perp = (lam_perp - cls.lam_min) / (cls.lam_max - cls.lam_min)
        lam_par = (lam_par - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam_perp, lam_par])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta


class StickZeppelinKernel(SignalKernel):
    theta_dim: int = 3

    def __init__(self, fraction, lam_perp, lam_par):
        self.fraction = fraction
        self.lam_perp = lam_perp
        self.lam_par = lam_par

    @classmethod
    def kernel_fn(
        cls, mu: ArrayLike, bvals, bvecs, fraction, lam_perp, lam_par, rng=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        stick_signal = StickKernel.kernel_fn(mu, bvals, bvecs, lam=lam_par)
        zeppelin_signal = ZeppelinKernel.kernel_fn(
            mu, bvals, bvecs, lam_par=lam_par, lam_perp=lam_perp
        )
        return fraction * stick_signal + (1 - fraction) * zeppelin_signal

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        fraction = theta[0]
        lam_perp = theta[1]
        lam_par = theta[2]

        fraction = jax.scipy.stats.norm.cdf(fraction)
        lam_perp_u = jax.scipy.stats.norm.cdf(lam_perp)
        lam_par_u = jax.scipy.stats.norm.cdf(lam_par)

        lam_perp_u = (
            lam_perp_u * (Zeppelin.max_lam - Zeppelin.min_lam) + Zeppelin.min_lam
        )
        lam_par_u = lam_par_u * (Zeppelin.max_lam - Zeppelin.min_lam) + Zeppelin.min_lam

        return fraction, lam_perp_u, lam_par_u

    @classmethod
    def to_theta(cls, fraction, lam_perp, lam_par) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        fraction = fraction
        lam_perp = (lam_perp - Zeppelin.min_lam) / (Zeppelin.max_lam - Zeppelin.min_lam)
        lam_par = (lam_par - Zeppelin.min_lam) / (Zeppelin.max_lam - Zeppelin.min_lam)
        theta = jnp.array([fraction, lam_perp, lam_par])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta


class DistributionalModel(SignalCompartment):
    fod_type: type
    signal_kernel_type: type

    def __init_subclass__(cls):
        assert hasattr(cls, "fod_type"), "fod_type not defined"
        assert hasattr(cls, "signal_kernel_type"), "signal_kernel_type not defined"

        cls.theta_dim = (
            cls.fod_type.theta_dim + cls.signal_kernel_type.theta_dim
        )  # sum([m.theta_dim for m in cls.signal_kernel])
        jtu.register_pytree_node_class(cls)

    def __init__(self, fod, signal_kernel):
        self.fod = fod
        self.signal_kernel = signal_kernel
        assert type(fod) is self.fod_type, "Wrong fod"
        assert type(signal_kernel) is self.signal_kernel_type, "Wrong signal model"

    @classmethod
    def signal_fn(
        cls,
        bvals: ArrayLike,
        bvecs: ArrayLike,
        fod,
        signal_kernel,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        sh_coeff_fod = fod.sh_coeff(sh_order=22)
        sh_coeff_signal = signal_kernel.sh_coeff(bvals, bvecs, sh_order=22)
        return jnp.dot(sh_coeff_fod, sh_coeff_signal)

    @classmethod
    def log_signal_fn(cls, bvals, bvecs, **kwargs):
        return jnp.log(cls.signal_fn(bvals, bvecs, **kwargs))

    @classmethod
    def to_theta(cls, fod, signal_kernel):
        fod_theta = fod.theta
        signal_theta = signal_kernel.theta
        return jnp.concatenate([fod_theta, signal_theta])

    @classmethod
    def to_params(cls, theta):
        dim1 = cls.fod_type.theta_dim
        fod_theta, signal_theta = jnp.split(theta, [dim1])
        fod = cls.fod_type.from_theta(fod_theta)
        signal_kernel = cls.signal_kernel_type.from_theta(signal_theta)
        return fod, signal_kernel


class WatsonStick(DistributionalModel):
    fod_type = Watson
    signal_kernel_type = StickKernel


class WatsonZeppelin(DistributionalModel):
    fod_type = Watson
    signal_kernel_type = ZeppelinKernel


class BinghamStick(DistributionalModel):
    fod_type = Bingham
    signal_kernel_type = StickKernel


class BinghamZeppelin(DistributionalModel):
    fod_type = Bingham
    signal_kernel_type = ZeppelinKernel


class NoddiW(DistributionalModel):
    fod_type = Watson
    signal_kernel_type = StickZeppelinKernel

class NoddiB(DistributionalModel):
    fod_type = Bingham
    signal_kernel_type = StickZeppelinKernel
