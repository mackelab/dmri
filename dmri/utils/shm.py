from functools import partial

import jax
import jax.numpy as jnp
import numpy as np  # optional, for generating test data, etc.
from dipy.reconst.shm import sph_harm_ind_list
from jax.scipy.special import sph_harm  # Key function
from jax.typing import ArrayLike


def real_sh_descoteaux_from_index_jax(
    m_values: ArrayLike,
    l_values: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    sh_order_max: int = 4,
    legacy: bool = True,
) -> ArrayLike:
    """Compute real spherical harmonics using the Descoteaux basis.

    This function implements the real spherical harmonics basis as described in
    Descoteaux et al. (2007). The basis is commonly used in diffusion MRI for
    representing orientation distribution functions (ODFs).

    Args:
        m_values: Array of order values (m) for spherical harmonics.
        l_values: Array of degree values (l) for spherical harmonics.
        theta: Array of polar angles (colatitude) in radians.
        phi: Array of azimuthal angles in radians.
        sh_order_max: Maximum order of spherical harmonics (default: 4).
        legacy: If True, uses absolute value of m_values as in legacy implementations.
               If False, uses m_values directly (default: True).

    Returns:
        Array of real spherical harmonics values.

    Notes:
        - The basis is normalized according to the Descoteaux convention.
        - For m > 0, the imaginary part is used; for m ≤ 0, the real part is used.
        - The basis is scaled by sqrt(2) for m ≠ 0 to ensure orthonormality.
    """
    # In the cited paper, the basis is defined without the absolute value
    if legacy:
        sh = sph_harm(jnp.abs(m_values), l_values, phi, theta, n_max=sh_order_max)
    else:
        sh = sph_harm(m_values, l_values, phi, theta, n_max=sh_order_max)

    real_sh = jnp.where(m_values > 0, sh.imag, sh.real)
    real_sh *= jnp.where(m_values == 0, 1.0, np.sqrt(2))

    return real_sh


def real_sh(
    sh_order_max: int,
    theta: ArrayLike,
    phi: ArrayLike,
    *,
    full_basis: bool = False,
    legacy: bool = True,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute real spherical harmonics for a given maximum order.

    This function generates real spherical harmonics up to a specified maximum order
    for given angular coordinates. It uses the Descoteaux basis and supports both
    full and symmetric basis sets.

    Args:
        sh_order_max: Maximum order of spherical harmonics.
        theta: Array of polar angles (colatitude) in radians.
        phi: Array of azimuthal angles in radians.
        full_basis: If True, returns the full basis including negative m values.
                   If False, returns only the symmetric part (default: False).
        legacy: If True, uses legacy implementation with absolute m values.
               If False, uses direct m values (default: True).

    Returns:
        Tuple containing:
        - Array of real spherical harmonics values
        - Array of m values (order)
        - Array of l values (degree)

    Notes:
        - The function uses JAX's vmap for efficient vectorization.
        - The basis is normalized according to the Descoteaux convention.
        - The returned arrays are squeezed to remove unnecessary dimensions.
    """
    with jax.ensure_compile_time_eval():
        m_value, l_value = sph_harm_ind_list(sh_order_max, full_basis=full_basis)
    m_value = jnp.asarray(m_value).reshape(-1, 1)
    l_value = jnp.asarray(l_value).reshape(-1, 1)
    phi = jnp.asarray(phi).reshape(-1, 1)
    theta = jnp.asarray(theta).reshape(-1, 1)

    _real_sph_harm = partial(
        real_sh_descoteaux_from_index_jax, sh_order_max=sh_order_max, legacy=legacy
    )
    real_sh = jax.vmap(
        jax.vmap(_real_sph_harm, in_axes=(0, 0, None, None)), in_axes=(None, None, 0, 0)
    )(m_value, l_value, theta, phi)

    return real_sh.squeeze(-1), m_value.squeeze(-1), l_value.squeeze(-1)
