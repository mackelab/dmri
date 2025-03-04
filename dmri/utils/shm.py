from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import sph_harm  # Key function
import numpy as np  # optional, for generating test data, etc.

from dipy.reconst.shm import sph_harm_ind_list, real_sh_descoteaux, spherical_harmonics


def real_sh_descoteaux_from_index_jax(
    m_values, l_values, theta, phi, *, sh_order_max=4, legacy=True
):
    # In the cited paper, the basis is defined without the absolute value
    if legacy:
        sh = sph_harm(jnp.abs(m_values), l_values, phi, theta, n_max=sh_order_max)
    else:
        sh = sph_harm(m_values, l_values, phi, theta, n_max=sh_order_max)

    real_sh = jnp.where(m_values > 0, sh.imag, sh.real)
    real_sh *= jnp.where(m_values == 0, 1.0, np.sqrt(2))

    return real_sh


def real_sh(sh_order_max, theta, phi, *, full_basis=False, legacy=True):
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


