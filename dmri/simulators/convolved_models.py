from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.typing import ArrayLike
from probjax.utils.typing import Array

from dmri.simulators.acquisition_scheme import acquisition_scheme
from dmri.simulators.base import Compartment, SignalCompartment
from dmri.simulators.sphereical_distributions import (
    hemisphere_default,
    inverse_sh_matrix,
)

HEMISPHERE_VERTICES = jnp.array(hemisphere_default.vertices)


class SignalKernel(Compartment):
    vmap_on_sphere: bool = False

    @classmethod
    @abstractmethod
    def kernel_fn(cls, mu: ArrayLike, acq: acquisition_scheme, **kwargs) -> Array:
        pass

    def sh_coeff(self, acq: acquisition_scheme, sh_order: int) -> ArrayLike:
        with jax.ensure_compile_time_eval():
            inverse_real_sh = inverse_sh_matrix(sh_order, sphere=hemisphere_default)
        inverse_real_sh = jnp.array(inverse_real_sh)
        kernel = partial(self.kernel_fn, **self.params)
        eval_kernel = partial(kernel, acq)

        # Vectorizing over the hemisphere can explode memory when the grid is large,
        # so allow kernels to opt-out via vmap_on_sphere.
        if type(self).vmap_on_sphere:
            signal = jax.vmap(eval_kernel)(HEMISPHERE_VERTICES)
        else:
            signal = jax.lax.map(eval_kernel, HEMISPHERE_VERTICES, batch_size=64)

        sh_coeff = inverse_real_sh @ jnp.squeeze(signal)
        return sh_coeff


class ConvolvedSignalCompartment(SignalCompartment):
    fod_type: type
    signal_kernel_type: type
    default_eval_sh_order: int = 14

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
        acq: acquisition_scheme,
        fod,
        signal_kernel,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        sh_coeff_fod = fod.sh_coeff(sh_order=cls.default_eval_sh_order)
        sh_coeff_signal = signal_kernel.sh_coeff(
            acq, sh_order=cls.default_eval_sh_order
        )
        return jnp.dot(sh_coeff_fod, sh_coeff_signal)

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, **kwargs):
        return jnp.log(cls.signal_fn(acq, **kwargs))

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

    def to_fod(self):
        return self.fod
