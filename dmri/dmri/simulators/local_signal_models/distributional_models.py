from functools import partial
from typing import Any
from dmri.simulators.acquisition_scheme import acquisition_scheme
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
    vmap_on_sphere: bool = False

    @classmethod
    @abstractmethod
    def kernel_fn(
        cls, mu: ArrayLike, aquisition_scheme: acquisition_scheme, **kwargs
    ) -> ArrayLike:
        pass

    def sh_coeff(self, aquisition_scheme, sh_order):
        with jax.ensure_compile_time_eval():
            inverse_real_sh = inverse_sh_matrix(sh_order, sphere=hemisphere_default)

        inverse_real_sh = jnp.array(inverse_real_sh)
        kernel = partial(self.kernel_fn, **self.params)
        # This her can be quite memory intensive so might be better to use a for loop
        if type(self).vmap_on_sphere:
            signal = jax.vmap(kernel, in_axes=(0, None))(
                hemisphere_default.vertices, aquisition_scheme
            )
            sh_coeff = inverse_real_sh @ signal.squeeze()
        else:
            signal = jax.lax.map(
                partial(kernel, aquisition_scheme=aquisition_scheme),
                hemisphere_default.vertices,
            )
            sh_coeff = inverse_real_sh @ signal.squeeze()

        return sh_coeff


class StickKernel(SignalKernel):
    theta_dim: int = 1
    lam_min: float = Stick.max_lam
    lam_max: float = Stick.min_lam

    def __init__(self, lam_par: float):
        self.lam_par = lam_par

    @classmethod
    def kernel_fn(
        cls,
        mu: ArrayLike,
        aquisition_scheme: acquisition_scheme,
        lam_par: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        signal_fn = jax.vmap(Stick.signal_fn, in_axes=(0, None, None))
        mu = cartesian_to_unitsphere(mu)
        return signal_fn(aquisition_scheme, mu, lam_par)

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)

    @classmethod
    def to_theta(cls, lam_par: ArrayLike) -> ArrayLike:
        """Convert the lambda value to the parameter space theta."""
        lam_par = (lam_par - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam_par])
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
        cls,
        mu: ArrayLike,
        aquisition_scheme: acquisition_scheme,
        lam_perp: float,
        lam_par: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        signal_fn = jax.vmap(Zeppelin.signal_fn, in_axes=(0, None, None, None))
        mu = cartesian_to_unitsphere(mu)
        return signal_fn(aquisition_scheme, mu, lam_perp, lam_par)

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


class NODDIKernel(SignalKernel):
    theta_dim: int = 3

    def __init__(self, fraction, lam_perp, lam_par):
        self.fraction = fraction
        self.lam_perp = lam_perp
        self.lam_par = lam_par

    @classmethod
    def kernel_fn(
        cls,
        mu: ArrayLike,
        aquisition_scheme: acquisition_scheme,
        fraction,
        lam_perp,
        lam_par,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        stick_signal = StickKernel.kernel_fn(mu, aquisition_scheme, lam_par=lam_par)
        zeppelin_signal = ZeppelinKernel.kernel_fn(
            mu, aquisition_scheme, lam_par=lam_par, lam_perp=lam_perp
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

class SimpleSANDIKernel(SignalKernel):
    theta_dim: int = 5

    def __init__(self, fraction_in, fraction_ec, lam_par_in, lam_perp_ex, lam_par_ex):
        self.fraction_in = fraction_in
        self.fraction_ec = fraction_ec
        self.lam_par_in = lam_par_in
        self.lam_perp_ex = lam_perp_ex
        self.lam_par_ex = lam_par_ex

    @classmethod
    def kernel_fn(
        cls,
        mu: ArrayLike,
        aquisition_scheme: acquisition_scheme,
        fraction_in,
        fraction_ec,
        lam_par_in,
        lam_perp_ex,
        lam_par_ex,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        axon_signal_in = StickKernel.kernel_fn(
            mu, aquisition_scheme, lam_par=lam_par_in
        )
        soma_signal = 1.0  # Dot

        zeppelin_signal_ex = ZeppelinKernel.kernel_fn(
            mu, aquisition_scheme, lam_par=lam_par_ex, lam_perp=lam_perp_ex
        )
        signal_in = fraction_in * axon_signal_in + (1 - fraction_in) * soma_signal
        signal_ex = zeppelin_signal_ex
        return (1 - fraction_ec) * signal_in + fraction_ec * signal_ex

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        u_theta = jax.scipy.stats.norm.cdf(theta)
        fraction_in = u_theta[0]
        fraction_ec = u_theta[1]
        lam_par_in = u_theta[2] * (Stick.max_lam - Stick.min_lam) + Stick.min_lam
        lam_perp_ex = (
            u_theta[3] * (Zeppelin.max_lam - Zeppelin.min_lam) + Zeppelin.min_lam
        )
        lam_par_ex = (
            u_theta[4] * (Zeppelin.max_lam - Zeppelin.min_lam) + Zeppelin.min_lam
        )
        return fraction_in, fraction_ec, lam_par_in, lam_perp_ex, lam_par_ex

    @classmethod
    def to_theta(
        cls, fraction_in, fraction_ec, lam_par_in, lam_perp_ex, lam_par_ex
    ) -> ArrayLike:
        fraction_in = (fraction_in - 0.0) / 1.0
        fraction_ec = (fraction_ec - 0.0) / 1.0
        lam_par_in = (lam_par_in - Stick.min_lam) / (Stick.max_lam - Stick.min_lam)
        lam_perp_ex = (lam_perp_ex - Zeppelin.min_lam) / (
            Zeppelin.max_lam - Zeppelin.min_lam
        )
        lam_par_ex = (lam_par_ex - Zeppelin.min_lam) / (
            Zeppelin.max_lam - Zeppelin.min_lam
        )
        theta = jnp.array(
            [fraction_in, fraction_ec, lam_par_in, lam_perp_ex, lam_par_ex]
        )
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
        aquisition_scheme: acquisition_scheme,
        fod,
        signal_kernel,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        sh_coeff_fod = fod.sh_coeff(sh_order=14)
        sh_coeff_signal = signal_kernel.sh_coeff(aquisition_scheme, sh_order=14)
        return jnp.dot(sh_coeff_fod, sh_coeff_signal)

    @classmethod
    def log_signal_fn(cls, aquisition_scheme: acquisition_scheme, **kwargs):
        return jnp.log(cls.signal_fn(aquisition_scheme, **kwargs))

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
    signal_kernel_type = NODDIKernel

class NoddiB(DistributionalModel):
    fod_type = Bingham
    signal_kernel_type = NODDIKernel


class SandiW(DistributionalModel):
    fod_type = Watson
    signal_kernel_type = SimpleSANDIKernel


class SandiB(DistributionalModel):
    fod_type = Bingham
    signal_kernel_type = SimpleSANDIKernel
