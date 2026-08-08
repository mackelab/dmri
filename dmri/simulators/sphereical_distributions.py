import functools
import math
from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from dipy.data import HemiSphere, get_sphere
from jax.typing import ArrayLike

from dmri.simulators.base import Compartment
from dmri.utils.dmriutils import rotation_matrix_100_to_theta_phi_psi, sph2cart
from dmri.utils.shm import real_sh


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


from dmri.utils.viz import (
    plot_spherical_distribution_cartesian,
    plot_spherical_distribution_fod,
    plot_spherical_distribution_polar,
)

sphere_default = get_sphere(name="symmetric724")
hemisphere_default = HemiSphere(phi=sphere_default.phi, theta=sphere_default.theta)

big_sphere = get_sphere(name="repulsion724")
bigger_hemisphere = HemiSphere(phi=big_sphere.phi, theta=big_sphere.theta)


def diffusion_tensor2d_odf(dirs, evals, evecs):
    """
    Compute the ODF for a single diffusion tensor at directions `dirs`.
    Handles the degenerate case, where the last eigenvalue is 0, by
    restricting the evaluation to the plane spanned by the first two eigenvectors.
    For directions falling outside the plane, returns 0.
    """
    assert evals.shape == (2,)
    assert evecs.shape == (3, 2)

    evec1, evec2 = evecs.T
    evec3 = jnp.cross(evec1, evec2)

    # Build the inverse tensor in the plane.
    inv_vals = jnp.array([1.0 / evals[0], 1.0 / evals[1]])
    mat_perp = jnp.column_stack([evec1, evec2])
    D_perp_inv = mat_perp @ jnp.diag(inv_vals) @ mat_perp.T

    # Compute projection on the degenerate (normal) vector.
    proj = jnp.abs(dirs @ evec3)
    tol = 1e-6

    # Quadratic form for each direction.
    quad = jnp.sum(dirs @ D_perp_inv * dirs, axis=1)

    # In-plane ODF (ignoring the vanished normalization factor).
    odf_inplane = 1.0 / (4.0 * jnp.pi * (quad**1.5))

    # Set ODF to 0 for directions not in the plane.
    odf_vals = jnp.where(proj < tol, odf_inplane, 0.0)
    return odf_vals


def sample_single_from_odf_jax(evals, evecs, rng):
    """
    Sample a single unit direction from the ODF defined by the diffusion tensor
    using naive rejection sampling in JAX, implemented with jax.lax.while_loop.
    Parameters
    ----------
    evals : array-like, shape (3,)
        Eigenvalues of the diffusion tensor (assumed positive).
    evecs : array-like, shape (3, 3)
        Eigenvectors of the diffusion tensor (columns = eigenvectors).
    rng : jax.random.PRNGKey
        Random key for JAX.
    max_iter : int, optional
        Maximum proposals for rejection sampling.
    Returns
    -------
    direction : jnp.ndarray, shape (3,)
        A single sampled unit direction, or None (a Python object) if rejected
        in all attempts.
    """

    R = jnp.asarray(evecs)
    D = R @ jnp.diag(evals) @ R.T
    mv_norm = jax.random.multivariate_normal(rng, jnp.zeros(3), D)
    sample = mv_norm / jnp.linalg.norm(mv_norm)
    return sample


def sample_single_from_odf_jax_degenerate(evals, evecs, rng, max_iter=10000):
    """
    Sample a direction from the ODF of a 'degenerate' diffusion tensor
    with eigenvalues [lambda1, lambda2, 0] using naive rejection sampling
    in the plane of nonzero diffusion.
    Parameters
    ----------
    evals : array-like of shape (3,)
        Eigenvalues of the diffusion tensor. We assume evals[2] == 0 and
        evals[0], evals[1] > 0 (ordered or not).
    evecs : array-like of shape (3,3)
        Eigenvectors (columns) of the diffusion tensor.
        evecs[:,2] is the direction corresponding to the zero eigenvalue.
    rng : jax.random.PRNGKey
        Random key for JAX.
    max_iter : int, optional
        Maximum proposals for rejection sampling in the plane.
    Returns
    -------
    direction : jnp.ndarray of shape (3,)
        A sampled unit direction in the plane spanned by the two nonzero
        eigenvalues, or None (Python object) if rejected in all attempts.
    """
    # --- 1. Identify the plane vectors & eigenvalues ---
    # Sort the eigenvalues just to be sure we know which is zero.
    # Alternatively, you can skip sorting if you already know the order.
    idx_sorted = jnp.argsort(evals)  # ascending order
    evals_sorted = evals[idx_sorted]
    evecs_sorted = evecs[:, idx_sorted]

    # rename them for clarity
    lam1, lam2, lam3 = evals_sorted
    # evec1, evec2, evec3
    evec1 = evecs_sorted[:, 0]
    evec2 = evecs_sorted[:, 1]
    _evec3 = evecs_sorted[:, 2]

    # We assume lam3 == 0, lam1>0, lam2>0
    # Build the 2D inverse sub-tensor in that plane:
    #    D_perp_inv = [evec1 evec2] diag(1/lam1, 1/lam2) [evec1 evec2]^T
    mat_perp = jnp.column_stack([evec1, evec2])  # shape (3,2)
    inv_vals = jnp.array([1.0 / lam1, 1.0 / lam2])  # (2,)
    D_perp_inv = mat_perp @ jnp.diag(inv_vals) @ mat_perp.T  # shape (3,3)

    # --- 2. Define the ODF function restricted to the plane ---
    # ignoring normalization constants for rejection sampling
    def in_plane_odf(theta):
        """
        Return ODF(theta) ~ 1 / ( u^T D_perp_inv u )^(3/2 ),
        where u(theta) = cos(theta)*evec1 + sin(theta)*evec2.
        """
        # direction in-plane
        u = jnp.cos(theta) * evec1 + jnp.sin(theta) * evec2
        quad = u @ D_perp_inv @ u
        # We only need it up to a scale factor for acceptance
        return 1.0 / (quad**1.5)

    # --- 3. Find a crude upper bound by sampling angles ---
    angles_grid = jnp.linspace(0.0, 2 * jnp.pi, num=1000, endpoint=False)
    odf_grid = jax.vmap(in_plane_odf)(angles_grid)
    max_odf_est = jnp.max(odf_grid) * 1.2  # a small safety margin

    # --- 4. Naive rejection sampling in [0, 2*pi) ---
    # We'll do a while_loop that tries up to `max_iter` times

    def cond_fun(state):
        i, done, theta_accepted, key = state
        return (i < max_iter) & (~done)

    def body_fun(state):
        i, done, theta_acc, key = state
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Propose an angle uniformly in [0, 2*pi)
        theta_prop = jax.random.uniform(subkey1, minval=0.0, maxval=2 * jnp.pi)
        # Evaluate acceptance probability
        accept_prob = in_plane_odf(theta_prop) / max_odf_est
        # Accept if uniform(0,1) < accept_prob
        accept = (jax.random.uniform(subkey2) < accept_prob) & (~done)

        new_done = done | accept
        new_theta = jnp.where(accept, theta_prop, theta_acc)
        return (i + 1, new_done, new_theta, key)

    init_state = (0, False, 0.0, rng)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    i_final, done_final, theta_final, _ = final_state

    # --- 5. Return the result in Python space ---
    if bool(done_final):
        # Convert the accepted angle to a 3D unit vector in-plane
        u = jnp.cos(theta_final) * evec1 + jnp.sin(theta_final) * evec2
        # (Should already be unit in principle if evec1, evec2 are orthonormal.)
        return u / jnp.linalg.norm(u)
    else:
        # If no acceptance, return None
        return None


def diffusion_tensor_odf(dirs, evals, evecs):
    """
    Compute the ODF for a single diffusion tensor at directions `dirs`.
    """
    R = jnp.asarray(evecs)
    eigvals_inv = 1.0 / evals
    D_inv = R @ jnp.diag(eigvals_inv) @ R.T
    det_factor = jnp.sqrt(jnp.prod(evals))

    # Quadratic form u^T D_inv u
    quad = jnp.sum(dirs @ D_inv * dirs, axis=1)
    # ODF(u) = 1 / (4*pi * sqrt(det(D)) * (quad)^(3/2))
    odf_vals = 1.0 / (4.0 * jnp.pi * det_factor * (quad**1.5))
    return odf_vals


def odi2kappa(odi):
    "Calculates concentration (kappa) from orientation dispersion index (odi)."
    return 1.0 / jnp.tan(odi * (jnp.pi / 2.0))


def get_sh_order_from_odi(odi):
    "Returns minimum sh_order to estimate spherical harmonics for given odi."
    odis = np.array([
        0.80606061,
        0.46666667,
        0.25333333,
        0.15636364,
        0.09818182,
        0.06909091,
        0.0,
    ])
    sh_orders = np.arange(2, 15, 2)
    return sh_orders[np.argmax(odis < odi)]


@functools.lru_cache(None)
def inverse_sh_matrix(sh_order, sphere=None, full_basis=False):
    r"""Returns the inverse of the spherical harmonics basis matrix."""

    sphere = hemisphere_default if sphere is None else sphere
    full_basis = not isinstance(sphere, HemiSphere)

    real_sh_basis, _, _ = real_sh(
        sh_order, sphere.theta, sphere.phi, full_basis=full_basis
    )
    inverse_real_sh = np.linalg.pinv(real_sh_basis)
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
        with jax.ensure_compile_time_eval():
            if sh_order is None:
                sh_order = get_sh_order_from_odi(self.odi)

            if sphere is not None:
                hemisphere = HemiSphere.from_sphere(sphere)
            else:
                hemisphere = hemisphere_default

            sh_mat_inv = inverse_sh_matrix(
                sh_order, sphere=hemisphere, full_basis=full_basis
            )

        vertices = jnp.array(hemisphere.vertices)
        sh_mat_inv = jnp.array(sh_mat_inv)
        pdf_sf = self.pdf(vertices)

        sh_coef = jnp.dot(sh_mat_inv, pdf_sf)
        return sh_coef

    def viz(
        self,
        plot_type="fod",
        sphere=None,
        n_samples=1000,
        ax=None,
        color=None,
        cmap=None,
        levels=3,
        alpha=None,
    ):
        r"""Visualize the spherical distribution model on the sphere.

        Parameters
        ----------
        plot_type : str, optional
            Type of plot to generate. Options are:
            - "polar": Plot in polar coordinates (theta, phi)
            - "cartesian": Plot in Cartesian coordinates (x, y, z)
            - "fod": Plot as a fiber orientation distribution
        sphere : object, optional
            Sphere object for vertices (used in cartesian plot)
        n_samples : int, optional
            Number of samples to generate
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to plot on
        color : str, optional
            Colormap to use
        levels : int, optional
            Number of contour levels for polar plot
        alpha : float, optional
            Transparency of the surface for FOD plot

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis containing the plot
        """
        if plot_type == "polar":
            return plot_spherical_distribution_polar(
                self, n_samples=n_samples, ax=ax, color=color, levels=levels
            )
        elif plot_type == "cartesian":
            return plot_spherical_distribution_cartesian(
                self, n_samples=n_samples, sphere=sphere, ax=ax
            )
        elif plot_type == "fod":
            return plot_spherical_distribution_fod(self, ax=ax, alpha=alpha, cmap=cmap)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def to_pmf(self, sphere=None, n_samples=1_000):
        """Convert the continuous distribution to a discrete probability mass function.

        Parameters
        ----------
        sphere : object, optional
            Sphere object for vertices
        n_samples : int, optional
            Number of samples to generate

        Returns
        -------
        tuple
            (vertices, probabilities) where vertices are the sphere points and
            probabilities are the corresponding PMF values
        """
        sphere = sphere_default if sphere is None else sphere
        samples = self.sample(jax.random.key(0), (n_samples,))
        pdfs = self.pdf(samples)
        vertices = sphere.vertices

        # Compute cosine similarity between each vertex and each sample.
        # Resulting shape is (num_vertices, num_samples)
        cos_sim = jnp.dot(vertices, samples.T)
        # For each sample, find the vertex index with maximum cosine similarity.
        idx_closest = jnp.argmax(cos_sim, axis=0)

        # Initialize vertex weights and accumulate pdfs using segment_sum.
        vertex_weights = jnp.zeros(vertices.shape[0])
        vertex_weights = jax.ops.segment_sum(pdfs, idx_closest, vertices.shape[0])

        fpmf = vertex_weights / jnp.sum(vertex_weights)
        return fpmf


class Uniform(SphericalDistribution):
    theta_dim: int = 0

    def pdf(self, n):
        return 1.0 / (4 * jnp.pi) * jnp.ones(n.shape[0])

    def sample(self, key, shape):
        normal = jax.random.normal(key, shape + (3,))
        return normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)

    @classmethod
    def to_theta(cls, mu, odi) -> ArrayLike:
        return jnp.array([])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        return ()


class SymmetricDirac(SphericalDistribution):
    theta_dim = 3

    def __init__(self, mu):
        self.mu = mu

    def pdf(self, n):
        return jnp.where(jnp.all((n == self.mu) | (n == -self.mu), axis=-1), 1.0, 0.0)

    def sample(self, key, shape):
        # Point symmetric on origin
        sign = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=shape + (1,))
        return sign * self.mu

    @classmethod
    def to_theta(cls, mu, odi) -> ArrayLike:
        return mu

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        return theta


class TensorFOD(SphericalDistribution):
    def __init__(self, evecs, evals):
        assert evecs.shape == (3, 3), "evecs must have shape (3, 3)"
        assert evals.shape == (3,), "evals must have shape (3,)"
        self.evecs = evecs
        self.evals = evals

    def pdf(self, n):
        return diffusion_tensor_odf(evals=self.evals, evecs=self.evecs, dirs=n)

    def sample(self, key, shape):
        num_samples = math.prod(shape)
        keys = jax.random.split(key, num_samples)
        sample_fn = functools.partial(
            sample_single_from_odf_jax, self.evals, self.evecs
        )
        sample_fn = jax.vmap(sample_fn)
        return sample_fn(keys).reshape(shape + (3,))

    @classmethod
    def to_theta(cls, evecs, evals) -> ArrayLike:
        raise NotImplementedError()

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        raise NotImplementedError()


class Tensor2dFOD(SphericalDistribution):
    def __init__(self, evecs, evals):
        assert evecs.shape == (2, 3), "evecs must have shape (2, 3)"
        assert evals.shape == (2,), "evals must have shape (2,)"
        self.evecs = evecs
        self.evals = evals

    def pdf(self, n):
        return diffusion_tensor2d_odf(self.evals, self.evecs, n)

    def sample(self, key, shape):
        num_samples = math.prod(shape)
        keys = jax.random.split(key, num_samples)
        sample_fn = functools.partial(
            sample_single_from_odf_jax, self.evals, self.evecs
        )
        sample_fn = jax.vmap(sample_fn)
        return sample_fn(keys).reshape(shape + (3,))

    @classmethod
    def to_theta(cls, evecs, evals) -> ArrayLike:
        raise NotImplementedError()

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        raise NotImplementedError()


class MixtureOfFODs(SphericalDistribution):
    def __init__(self, fractions, components):
        self.components = components
        self.fractions = fractions

    def pdf(self, n):
        return jnp.sum(
            jnp.array([
                f * m.pdf(n)
                for f, m in zip(self.fractions, self.components, strict=True)
            ]),
            axis=0,
        )

    def sample(self, key, shape):
        num_samples = math.prod(shape)
        keys = jax.random.split(key, num_samples)
        sample_fn = jax.vmap(self._sample_one)
        return sample_fn(keys).reshape(shape + (3,))

    def _sample_one(self, key):
        rng1, rng2 = jax.random.split(key)
        idx = jax.random.choice(rng1, len(self.components), p=self.fractions)
        sample_fns = [functools.partial(m.sample, shape=(1,)) for m in self.components]
        return jax.lax.switch(idx, sample_fns, rng2)

    @classmethod
    def to_theta(cls, fractions, components) -> ArrayLike:
        raise NotImplementedError()

    @classmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        raise NotImplementedError()


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
        mu_cart = sph2cart(self.mu)
        numerator = jnp.exp(kappa * jnp.dot(n, mu_cart) ** 2)
        denominator = 4 * jnp.pi * jax.scipy.special.hyp1f1(0.5, 1.5, kappa)
        Wn = numerator / denominator
        return Wn

    def sample(self, key, shape):
        """
        Now defers to _sample_watson_distribution.
        """
        kappa = odi2kappa(self.odi)
        mu_cart = sph2cart(self.mu)
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

        mu_theta = jnp.array([
            jax.scipy.stats.norm.ppf(u0),
            jax.scipy.stats.norm.ppf(u1),
        ])

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

        mu_cart = sph2cart(mu)

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
        mu_cart = sph2cart(self.mu)
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


def deterministic_sphere_integration(kappa, beta, mu, mu_beta, n_theta=200, n_phi=200):
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
