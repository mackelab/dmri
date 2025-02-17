from abc import abstractmethod
from typing import Any
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from .utils import unitsphere_to_cartesian

from dmri.simulators.base import Compartment
from dmri.utils.shm import real_sh
from dipy.data import get_sphere, HemiSphere
import functools
import numpy as np

sphere = get_sphere("symmetric724")
hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)


def odi2kappa(odi):
    "Calculates concentration (kappa) from orientation dispersion index (odi)."
    return 1.0 / jnp.tan(odi * (jnp.pi / 2.0))


def get_sh_order_from_odi(odi):
    "Returns minimum sh_order to estimate spherical harmonics for given odi."
    odis = np.array(
        [0.80606061, 0.46666667, 0.25333333, 0.15636364, 0.09818182, 0.06909091, 0.0]
    )
    sh_orders = np.arange(2, 15, 2)
    return sh_orders[np.argmax(odis < odi)]


@functools.lru_cache(None)
def inverse_sh_matrix(sh_order, sphere=hemisphere):
    full_basis = not isinstance(sphere, HemiSphere)

    real_sh_basis, _, _ = real_sh(
        sh_order, sphere.theta, sphere.phi, full_basis=full_basis
    )
    inverse_real_sh = jnp.linalg.pinv(real_sh_basis)
    return inverse_real_sh


class SphericalDistribution(Compartment):
    @abstractmethod
    def pdf(self, n):
        pass

    @abstractmethod
    def sample(self, key, shape):
        pass


class SD1Watson(SphericalDistribution):
    r"""The Watson spherical distribution model [1]_ [2]_.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    kappa : float,
        concentration parameter of the Watson distribution.

    References
    ----------
    .. [1] Kaden et al.
           "Parametric spherical deconvolution: inferring anatomical
            connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Zhang et al.
           "NODDI: practical in vivo neurite orientation dispersion and density
            imaging of the human brain". NeuroImage (2012)
    """

    theta_dim = 3

    odi_min = 0.02
    odi_max = 0.99

    def __init__(self, mu, odi):
        self.mu = mu
        self.odi = odi

    def pdf(self, n):
        r"""The Watson spherical distribution model [1, 2].

        Parameters
        ----------
        n : array of shape(3) or array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        Wn: float or array of shape(N),
            Probability density at orientations n, given mu and kappa.
        """

        kappa = odi2kappa(self.odi)
        mu_cart = unitsphere_to_cartesian(self.mu)
        numerator = jnp.exp(kappa * jnp.dot(n, mu_cart) ** 2)
        denominator = 4 * jnp.pi * jax.scipy.special.hyp1f1(0.5, 1.5, kappa)
        Wn = numerator / denominator
        return Wn

    def sample(self, key, shape):
        """
        Now defers to _sample_watson_distribution.
        """
        kappa = odi2kappa(self.odi)
        mu_cart = unitsphere_to_cartesian(self.mu)
        _sample_fn = functools.partial(sample_watson_ar_1, mu=mu_cart, kappa=kappa)
        keys = jax.random.split(key, shape)
        for _ in range(keys.ndim):
            _sample_fn = jax.vmap(_sample_fn)
        return _sample_fn(keys)

    def sh_coeff(self, sh_order=None, **kwargs):
        r"""The Watson spherical distribution model in spherical harmonics.
        The minimum order is automatically derived from numerical experiments
        to ensure fast function executation and accurate results.

        Parameters
        ----------
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        watson_sh : array,
            spherical harmonics of Watson probability density.
        """
        if sh_order is None:
            sh_order = get_sh_order_from_odi(self.odi)

        watson_sf = self.pdf(hemisphere.vertices)
        sh_mat_inv = inverse_sh_matrix(sh_order)
        watson_sh = jnp.dot(sh_mat_inv, watson_sf)
        return watson_sh

    @classmethod
    def to_theta(cls, *args) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        mu, odi = args

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        mu_theta, odi_theta = theta[:2], theta[2]

        # Map mu_theta -> uniform on the hemisphere via spherical coordinates
        u0 = jax.scipy.stats.norm.cdf(mu_theta[0])
        u1 = jax.scipy.stats.norm.cdf(mu_theta[1])
        phi = 2.0 * jnp.pi * u0
        # Restrict to the upper hemisphere by keeping cos_theta in [0, 1]
        cos_theta = jnp.clip(u1, 0.0, 1.0)
        mu = jnp.array([jnp.arccos(cos_theta), phi])

        # Map odi_theta -> uniform in [odi_min, odi_max]
        odi_unif = jax.scipy.stats.norm.cdf(odi_theta)
        odi = cls.odi_min + (cls.odi_max - cls.odi_min) * odi_unif

        return mu, odi


def sample_watson_ar_1(key, mu, kappa):
    """
    Draw a single sample from Watson(mu, kappa) on the unit sphere using
    acceptance-rejection from the uniform distribution on S^{d-1}.

    Arguments:
      key:    a jax.random.PRNGKey
      mu:     a jnp.ndarray of shape (d,) — will be normalized internally
      kappa:  a nonnegative float (concentration parameter)
    Returns:
      A jnp.ndarray of shape (d,) lying on the unit sphere, distributed ~ Watson(mu,kappa).
    """
    mu = mu / jnp.linalg.norm(mu)  # ensure mu is a unit vector

    def cond_fn(state):
        # state = (key, accepted, candidate)
        _, accepted, _ = state
        return jnp.logical_not(accepted)

    def body_fn(state):
        key, _, _ = state
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # 1) Propose x ~ Uniform(S^{d-1})
        z = jax.random.normal(subkey1, shape=mu.shape)
        x_proposal = z / jnp.linalg.norm(z)

        # 2) Acceptance probability
        log_accept_ratio = kappa * ((mu @ x_proposal) ** 2 - 1.0)
        u = jax.random.uniform(subkey2)

        accepted = jnp.log(u) <= log_accept_ratio
        return (key, accepted, x_proposal)

    # Initialize and run the while loop
    init_state = (key, False, jnp.zeros_like(mu))
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    _, _, x_accepted = final_state

    return x_accepted
