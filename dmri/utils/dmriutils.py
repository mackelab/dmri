import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def fit_diffusion_tensor_linearized(
    logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike
) -> ArrayLike:
    """Linearized fit of the diffusion tensor.

    Args:
        logS (ArrayLike): Signal in log-space.
        bvals (ArrayLike): Bvalues.
        bvecs (ArrayLike): Bvectors.

    Returns:
        Array: _description_
    """
    Y = -logS
    B = bvals[:, None] * jnp.stack(
        [
            bvecs[:, 0] ** 2,
            2 * bvecs[:, 0] * bvecs[:, 1],
            bvecs[:, 1] ** 2,
            2 * bvecs[:, 0] * bvecs[:, 2],
            2 * bvecs[:, 1] * bvecs[:, 2],
            bvecs[:, 2] ** 2,
        ],
        axis=-1,
    )
    D_flat = jnp.linalg.lstsq(B, Y, rcond=None)[0]
    idx1, idx2 = jnp.tril_indices(3)
    D = jnp.zeros((3, 3))
    D = D.at[idx1, idx2].set(D_flat)
    D = D + jnp.tril(D, -1).T
    return D


def cartesian_to_unitsphere(cartesian: ArrayLike) -> ArrayLike:
    """Convert Cartesian coordinates to spherical coordinates."""
    x, y, z = cartesian
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return jnp.array([theta, phi])


def unitsphere_to_cartesian(mu: ArrayLike) -> ArrayLike:
    """Convert spherical coordinates to Cartesian coordinates."""
    theta, phi = mu
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.array([x, y, z])


def normalize_bvecs(bvecs: ArrayLike) -> ArrayLike:
    """Normalize b-vectors to unit length."""
    norms = jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    return bvecs / norms


def compute_FA(D: ArrayLike) -> float:
    """Compute fractional anisotropy (FA) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    mean_diffusivity = jnp.mean(evals)
    fa = jnp.sqrt(1.5 * jnp.sum((evals - mean_diffusivity) ** 2) / jnp.sum(evals**2))
    return fa


def compute_MD(D: ArrayLike) -> float:
    """Compute mean diffusivity (MD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return jnp.mean(evals)


def compute_RD(D: ArrayLike) -> float:
    """Compute radial diffusivity (RD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return jnp.mean(evals[:2])


def compute_AD(D: ArrayLike) -> float:
    """Compute axial diffusivity (AD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return evals[2]


def rotation_matrix_100_to_theta_phi(theta, phi):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position on the unit sphere.

    Parameters
    ----------
    theta : float,
        inclination of polar angle of main angle mu [0, pi].
    phi : float,
        polar angle of main angle mu [-pi, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    x, y, z = unitsphere_to_cartesian([theta, phi])
    return rotation_matrix_100_to_xyz(x, y, z)


def rotation_matrix_100_to_xyz(x, y, z):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position in Cartesian space.

    Parameters
    ----------
    x, y, z : floats,
        position in Cartesian space.

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    is_identity = (x == 1.0) & (y == 0.0) & (z == 0.0)
    R_identity = jnp.eye(3)

    # Avoid division by zero
    denom = jnp.maximum((y * y + z * z), 1e-12)
    R_general = jnp.array(
        [
            [x, -y, -z],
            [
                y,
                (x * (y * y) + (z * z)) / denom,
                ((x - 1) * (y * z)) / denom,
            ],
            [
                z,
                ((x - 1) * (y * z)) / denom,
                ((y * y) + x * (z * z)) / denom,
            ],
        ]
    )
    # Use a data-dependent condition instead of Python ifs
    R = jnp.where(is_identity, R_identity, R_general)
    return R


def rotation_matrix_100_to_theta_phi_psi(theta, phi, psi):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position in Cartesian space, and rotates about its axis.

    Parameters
    ----------
    theta : float,
        inclination of polar angle of main angle mu [0, pi].
    phi : float,
        polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    R_100_to_theta_phi = rotation_matrix_100_to_theta_phi(theta, phi)
    R_around_100 = rotation_matrix_around_100(psi)
    return jnp.dot(R_100_to_theta_phi, R_around_100)


def rotation_matrix_around_100(psi):
    """Generates a rotation matrix that rotates around the x-axis (1, 0, 0).

    Parameters
    ----------
    psi : float,
        euler angle [0, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    cos_psi = jnp.cos(psi)
    sin_psi = jnp.sin(psi)

    R = jnp.array([[1.0, 0.0, 0.0], [0.0, cos_psi, -sin_psi], [0.0, sin_psi, cos_psi]])
    return R


def canonical_bingham_normalization_series(kappa, beta, max_terms=30):
    """
    Series expansion for Z(kappa, beta) using JAX.
    Sums up to 'max_terms' to approximate.
    """
    A = kappa - beta
    n = jnp.arange(max_terms)
    terms = A**n / (jax.scipy.special.gamma(n + 1) * (n + 0.5))
    summation = jnp.sum(terms)
    return 2.0 * jnp.pi * jnp.exp(beta) * summation
