from dmri.simulators.base import SignalCompartment
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import lax  # Add this import for special functions

DIAMETER_SCALING = 1e-6


class S2Sphere(SignalCompartment):
    r"""
    The Stejskal Tanner signal approximation of a sphere model. It assumes
    that pulse length is infinitesimally small and diffusion time large enough
    so that the diffusion is completely restricted. Only depends on q-value.
    """

    theta_dim = 1

    def __init__(self, diameter=None):
        self.diameter = diameter

    def sphere_attenuation(self, q, diameter):
        """The signal attenuation for the sphere model."""
        radius = diameter / 2.0
        q_argument = 2 * jnp.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = jnp.zeros_like(q)

        # J = special.spherical_jn(q_argument)
        Jder = lax.spherical_bessel_j1(q_argument)  # Using JAX's spherical Bessel function derivative
        for k in range(0, self.alpha.shape[0]):
            for n in range(0, self.alpha.shape[1]):
                a_nk2 = self.alpha[k, n] ** 2
                update = jnp.exp(-a_nk2 * self.Dintra * tau / radius ** 2)
                update *= ((2 * n + 1) * a_nk2) / (a_nk2 - (n - 0.5) ** 2 + 0.25)
                update *= q_argument * Jder
                update /= (q_argument_2 - a_nk2) ** 2
                res += update
        return res

    def log_signal(self, bvals, bvecs, **kwargs):
        """Calculates the log signal attenuation."""
        q = jnp.sqrt(bvals)  # Assuming q-values are derived from b-values
        diameter = kwargs.get("diameter", self.diameter)
        E_sphere = jnp.ones_like(q)
        q_nonzero = q > 0  # only q>0 attenuate
        E_sphere = E_sphere.at[q_nonzero].set(
            self.sphere_attenuation(q[q_nonzero], diameter)
        )
        return jnp.log(E_sphere)

    def fit(self, logS: ArrayLike, q: ArrayLike) -> float:
        """Fit the sphere model to the log signal and q-values."""
        # This requires a more complex fitting mechanism (not implemented here).
        raise NotImplementedError(
            "Fitting for S2SphereStejskalTannerApproximation is not implemented."
        )

    @classmethod
    def to_theta(cls, diameter: ArrayLike) -> ArrayLike:
        """Convert the diameter to the parameter space theta."""
        scaled_diameter = diameter
        return jnp.log(scaled_diameter)

    @classmethod
    def to_params(cls, theta: ArrayLike) -> float:
        """Convert the parameter space theta to the diameter."""
        diameter = jnp.exp(theta)
        return (diameter,)
