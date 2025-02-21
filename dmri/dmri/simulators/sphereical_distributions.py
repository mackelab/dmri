from abc import abstractmethod
import math
from typing import Any
from dmri.simulators.local_signal_models.gaussian_models import Stick
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from dmri.utils.dmriutils import (
    rotation_matrix_100_to_theta_phi_psi,
    unitsphere_to_cartesian,
)

from dmri.simulators.base import Compartment
from dmri.utils.shm import real_sh
from dmri.utils.sample_fns import sample_watson_ar_1
from dipy.data import get_sphere, HemiSphere
import functools
import numpy as np

sphere_default = get_sphere("symmetric724")
hemisphere_default = HemiSphere(phi=sphere_default.phi, theta=sphere_default.theta)

big_sphere = get_sphere("repulsion724")
bigger_hemisphere = HemiSphere(phi=big_sphere.phi, theta=big_sphere.theta)


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
def inverse_sh_matrix(sh_order, sphere=None, full_basis=False):
    sphere = hemisphere if sphere is None else sphere
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

    def sh_coeff(self, sh_order=None, sphere=None, full_basis=False, **kwargs):
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

        if sphere is not None:
            hemisphere = HemiSphere.from_sphere(sphere)
        else:
            hemisphere = hemisphere_default

        pdf_sf = self.pdf(hemisphere.vertices)
        sh_mat_inv = inverse_sh_matrix(
            sh_order, sphere=hemisphere, full_basis=full_basis
        )
        sh_coef = jnp.dot(sh_mat_inv, pdf_sf)
        return sh_coef


class Watson(SphericalDistribution):
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



    @classmethod
    def to_theta(cls, mu, odi) -> ArrayLike:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        phi = mu[1]
        theta = mu[0]

        # Map phi -> [0,1]
        u0 = (phi + jnp.pi) / (2.0 * jnp.pi)

        # Map theta -> [0,1]
        u1 = theta / (jnp.pi / 2.0)

        mu_theta = jnp.array(
            [jax.scipy.stats.norm.ppf(u0), jax.scipy.stats.norm.ppf(u1)]
        )

        # Map odi -> uniform in [odi_min, odi_max]
        odi_unif = (odi - cls.odi_min) / (cls.odi_max - cls.odi_min)
        odi_theta = jax.scipy.stats.norm.ppf(odi_unif)

        return jnp.concatenate([mu_theta, odi_theta[None]])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        mu_theta, odi_theta = theta[:2], theta[2]

        # Map mu_theta -> uniform on the hemisphere via spherical coordinates
        u0 = jax.scipy.stats.norm.cdf(mu_theta[0])
        u1 = jax.scipy.stats.norm.cdf(mu_theta[1])
        phi = 2.0 * jnp.pi * u0 - jnp.pi
        theta = 0.5 * jnp.pi * u1
        mu = jnp.array([theta, phi])

        # Map odi_theta -> uniform in [odi_min, odi_max]
        odi_unif = jax.scipy.stats.norm.cdf(odi_theta)
        odi = cls.odi_min + (cls.odi_max - cls.odi_min) * odi_unif

        return mu, odi


class Bingham(SphericalDistribution):
    # NOTE: This is not a canonical Bingham distribution.
    theta_dim = 5
    odi_min = 0.02
    odi_max = 0.99
    psi_min = 0.0
    psi_max = np.pi
    beta_fraction_min = 0.0
    beta_fraction_max = 1.0

    def __init__(self, mu, odi, psi, beta_fraction):
        self.mu = mu
        self.odi = odi
        self.psi = psi
        self.beta_fraction = beta_fraction

    def pdf(self, n):
        kappa = odi2kappa(self.odi)
        beta = self.beta_fraction * kappa
        mu = self.mu
        psi = self.psi

        mu_cart = unitsphere_to_cartesian(mu)

        R = rotation_matrix_100_to_theta_phi_psi(mu[0], mu[1], psi)
        mu_beta = jnp.dot(R, jnp.array([0.0, 1.0, 0.0]))


        numerator = jnp.exp(
            kappa * jnp.dot(n, mu_cart) ** 2 + beta * jnp.dot(n, mu_beta) ** 2
        )

        denominator = deterministic_sphere_integration(
            kappa=kappa, beta=beta, mu=mu_cart, mu_beta=mu_beta
        )
        Bn = numerator / denominator
        return Bn

    def sample(self, key, shape):
        num_samples = math.prod(shape)
        mu_cart = unitsphere_to_cartesian(self.mu)
        R = rotation_matrix_100_to_theta_phi_psi(self.mu[0], self.mu[1], self.psi)
        mu_beta = jnp.dot(R, jnp.array([0.0, 1.0, 0.0]))
        kappa = odi2kappa(self.odi)
        beta = self.beta_fraction * kappa

        sample_fn = functools.partial(
            sample_quad_exp_distribution,
            kappa=kappa,
            beta=beta,
            mu=mu_cart,
            mu_beta=mu_beta,
        )
        keys = jax.random.split(key, num_samples)
        for _ in range(keys.ndim):
            sample_fn = jax.vmap(sample_fn)
        return sample_fn(keys)

    @classmethod
    def to_params(cls, theta):
        mu = theta[:2]
        odi = theta[2]
        psi = theta[3]
        beta_fraction = theta[4]

        # Odi to range
        odi_u = jax.scipy.stats.norm.cdf(odi)
        odi = cls.odi_min + (cls.odi_max - cls.odi_min) * odi_u

        # Psi to range
        psi_u = jax.scipy.stats.norm.cdf(psi)
        psi = cls.psi_min + (cls.psi_max - cls.psi_min) * psi_u

        # Beta fraction to range
        beta_fraction_u = jax.scipy.stats.norm.cdf(beta_fraction)
        beta_fraction = (
            cls.beta_fraction_min
            + (cls.beta_fraction_max - cls.beta_fraction_min) * beta_fraction_u
        )

        # Mu to uniform on the hemisphere via spherical coordinates
        u0 = jax.scipy.stats.norm.cdf(mu[0])
        u1 = jax.scipy.stats.norm.cdf(mu[1])
        phi = 2.0 * jnp.pi * u0 - jnp.pi
        theta = 0.5 * jnp.pi * u1
        mu = jnp.array([theta, phi])

        return mu, odi, psi, beta_fraction

    @classmethod
    def to_theta(cls, mu, odi, psi, beta_fraction):
        # Mu to normal
        u0 = (mu[1] + jnp.pi) / (2.0 * jnp.pi)
        u1 = mu[0] / (jnp.pi / 2.0)
        mu0 = jax.scipy.stats.norm.ppf(u0)
        mu1 = jax.scipy.stats.norm.ppf(u1)

        # Odi to uniform
        odi_unif = (odi - cls.odi_min) / (cls.odi_max - cls.odi_min)
        odi_theta = jax.scipy.stats.norm.ppf(odi_unif)

        # Psi to uniform
        psi_unif = (psi - cls.psi_min) / (cls.psi_max - cls.psi_min)
        psi_theta = jax.scipy.stats.norm.ppf(psi_unif)

        # Beta fraction to uniform
        beta_fraction_unif = (beta_fraction - cls.beta_fraction_min) / (
            cls.beta_fraction_max - cls.beta_fraction_min
        )
        beta_fraction_theta = jax.scipy.stats.norm.ppf(beta_fraction_unif)

        return jnp.array([mu0, mu1, odi_theta, psi_theta, beta_fraction_theta])


def deterministic_sphere_integration(
    kappa, beta, mu, mu_beta, n_theta=400, n_phi=400
):
    """
    Use a 2D trapezoidal rule in spherical coords to approximate
        ∫ exp(kappa (n·mu)^2 + beta (n·mu_beta)^2 ) dΩ(n).
    """

    # 1) Discretize angles
    thetas = jnp.linspace(0, jnp.pi, n_theta)
    phis = jnp.linspace(0, 2 * jnp.pi, n_phi)

    dtheta = jnp.pi / (n_theta - 1)
    dphi = 2 * jnp.pi / (n_phi - 1)

    # 2) Build a mesh
    Theta, Phi = jnp.meshgrid(thetas, phis, indexing="ij")  # shape (n_phi, n_theta)

    # 3) Cartesian coords for each (theta, phi)
    sinT = jnp.sin(Theta)
    cosT = jnp.cos(Theta)
    cosP = jnp.cos(Phi)
    sinP = jnp.sin(Phi)

    # n = (sinθ cosφ, sinθ sinφ, cosθ) at each grid point
    nx = sinT * cosP
    ny = sinT * sinP
    nz = cosT

    # 4) Dot with mu, mu_beta
    dot_mu = nx * mu[0] + ny * mu[1] + nz * mu[2]  # shape (n_phi, n_theta)
    dot_mubeta = nx * mu_beta[0] + ny * mu_beta[1] + nz * mu_beta[2]
    exps = jnp.exp(kappa * dot_mu**2 + beta * dot_mubeta**2)

    # 5) Multiply by sinθ and sum
    sinT_2d = sinT  # shape (n_phi, n_theta)
    integrand = exps * sinT_2d  # shape (n_phi, n_theta)

    # 6) Trapezoidal approximate the double integral
    integral_approx = jnp.trapezoid(
        jnp.trapezoid(integrand, dx=dtheta, axis=1), dx=dphi, axis=0
    )
    return integral_approx


def sample_sphere(key, n_samples):
    """
    Generate n_samples random unit vectors on S^2 by sampling Normal(0,1) in 3D
    and normalizing. Returns shape (n_samples, 3).
    """
    xyz = jax.random.normal(key, shape=(n_samples, 3))
    r = jnp.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / r


def sample_quad_exp_distribution(key, kappa, beta, mu, mu_beta):
    max_val = jnp.maximum(kappa, beta)
    max_iter = 100

    def cond_fun(state):
        i, done, _, _ = state
        return (i < max_iter) & ~done

    def body_fun(state):
        i, done, rng, sample_val = state
        rng, sk_samp, sk_acc = jax.random.split(rng, 3)
        candidate = sample_sphere(sk_samp, 1)[0]
        dot_mu = jnp.sum(candidate * mu)
        dot_mubeta = jnp.sum(candidate * mu_beta)
        exponent = kappa * dot_mu**2 + beta * dot_mubeta**2
        accept_prob = jnp.exp(exponent - max_val)
        uniform = jax.random.uniform(sk_acc)
        accepted = uniform < accept_prob
        new_sample = jnp.where(accepted, candidate, sample_val)
        return i + 1, done | accepted, rng, new_sample

    init = (0, False, key, jnp.zeros(3))
    _, _, _, final_sample = jax.lax.while_loop(cond_fun, body_fun, init)
    return final_sample
