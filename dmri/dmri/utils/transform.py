from probjax.utils.stats import betainc, betaincinv

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def normal_to_dirichlet(alpha, eps, mask=None):
    assert len(eps) == len(alpha) - 1, "eps should be of size len(alpha) - 1"
    u = jax.scipy.stats.norm.cdf(eps)

    def scan_fn(carry, i):
        pi_sum = carry

        def new_phi(alpha, u, i):
            a = alpha[i]
            larger_i = jnp.arange(len(alpha)) > i
            b = jnp.sum(alpha * mask * larger_i)
            phi_i = betaincinv(a, b, u[i])
            return phi_i

        def masked_phi(alpha, u, i):
            return jnp.array(0.0)

        if mask is None:
            phi_i = new_phi(alpha, u, i)
        else:
            phi_i = jax.lax.cond(mask[i], new_phi, masked_phi, alpha, u, i)

        pi_i = phi_i * (1 - pi_sum)
        pi_sum += pi_i
        return pi_sum, pi_i

    pi_sum = 0.0
    pi_sum, pis = lax.scan(scan_fn, pi_sum, jnp.arange(len(alpha) - 1))
    pi_K = 1 - pi_sum
    pis = jnp.append(pis, pi_K)
    if mask is not None:
        last_entry = jnp.argmax(jnp.flip(mask)) - 1
        pis = pis.at[-1].set(0.0)
        pis = pis.at[last_entry].set(pi_K)
        pis = jnp.where(~mask, 0.0, pis)
        pis /= jnp.sum(pis)
        pis = jnp.nan_to_num(pis)
    return pis


@jax.jit
def dirichlet_to_normal(alpha, pi, mask=None):
    """
    Inverse of the normal_to_dirichlet stick-breaking transform.

    Args:
        alpha (array_like): Dirichlet concentration parameters, shape (K,)
        pi    (array_like): A Dirichlet sample, shape (K,), sums to 1.

    Returns:
        eps   (array): Shape (K - 1,), the Normal samples that would map
                       back to `pi` under `normal_to_dirichlet(alpha, eps)`.
    """
    if mask is not None:
        alpha = jnp.where(mask, alpha, 0.0)

    def scan_fn(carry, i):
        pi_sum = carry

        a = alpha[i]
        mask = jnp.arange(len(alpha)) > i
        b = jnp.sum(alpha * mask)
        phi_i = pi[i] / (1.0 - pi_sum)
        u_i = betainc(a, b, phi_i)
        eps_i = jax.scipy.stats.norm.ppf(u_i)
        return pi_sum + pi[i], eps_i

    init_pi_sum = 0.0
    _, eps = lax.scan(scan_fn, init_pi_sum, jnp.arange(len(alpha) - 1))
    eps = jnp.nan_to_num(eps)

    return eps
