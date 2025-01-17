from jax.typing import ArrayLike


import jax.numpy as jnp


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
