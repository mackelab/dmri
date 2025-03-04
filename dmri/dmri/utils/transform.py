from functools import partial
from probjax.utils.stats import betainc, betaincinv

import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def _normal_to_dirichlet(alpha, eps, mask=None):
    assert len(eps) == len(alpha) - 1, "eps should be of size len(alpha) - 1"
    u = jax.scipy.stats.norm.cdf(eps)

    def scan_fn(carry, i):
        pi_sum = carry

        def new_phi(alpha, u, i):
            a = alpha[i]
            larger_i = jnp.arange(len(alpha)) > i
            _alpha = alpha if mask is None else jnp.where(mask, alpha, 0.0)
            # jax.debug.print("{a}", a=a)
            b = jnp.sum(_alpha * larger_i)
            phi_i = betaincinv(a, b, u[i])
            phi_i = jnp.nan_to_num(phi_i)  # If b is 0
            return phi_i

        def masked_phi(alpha, u, i):
            return jnp.array(0.0)

        if mask is None:
            phi_i = new_phi(alpha, u, i)
        else:
            phi_i = jax.lax.cond(mask[i], new_phi, masked_phi, alpha, u, i)
            # jax.debug.print("{phi_i}", phi_i=phi_i)

        pi_i = phi_i * (1 - pi_sum)
        pi_sum += pi_i
        return pi_sum, pi_i

    pi_sum = 0.0
    pi_sum, pis = lax.scan(scan_fn, pi_sum, jnp.arange(len(alpha) - 1))
    pi_K = 1.0 - pi_sum
    pis = jnp.append(pis, pi_K)
    if mask is not None:
        last_entry = -jnp.argmax(mask[::-1]) - 1
        # jax.debug.print("{last_entry}", last_entry=last_entry)
        zero = jnp.array(0.0)
        pis = pis.at[-1].set(zero)
        pis = pis.at[last_entry].set(pi_K)
        pis = jnp.where(~mask, zero, pis)
        # jax.debug.print("{pis}", pis=pis)
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
    zero = jnp.array(0.0)
    if mask is not None:
        alpha = jnp.where(mask, alpha, zero)

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

# Forward function (returns output and residuals for backward pass)
@jax.custom_vjp
def normal_to_dirichlet(alpha, eps, mask=None):
    """Forward function that we want a custom backward (VJP) for."""
    return _normal_to_dirichlet(alpha, eps, mask)


# Forward pass for custom VJP. Return (output, residuals_for_bwd).
def normal_to_dirichlet_fwd(alpha, eps, mask):
    out = _normal_to_dirichlet(alpha, eps, mask)
    # We save (alpha, eps, mask, out) so we can use them in backward pass.
    return out, (alpha, eps, mask, out)


# Backward pass. Given residuals and the gradient w.r.t. output (g),
# return the gradient w.r.t. each input: (grad_alpha, grad_eps, grad_mask).
def normal_to_dirichlet_bwd(res, g):
    alpha, eps, mask, out = res

    # Suppose we define an "inverse" function that recovers normal coords
    # from the final dirichlet coords:
    def f_inv(dirichlet_out):
        return dirichlet_to_normal(alpha, dirichlet_out, mask=mask)

    # J is the Jacobian of f_inv at "out"
    J = jax.jacfwd(f_inv)(out)
    J_inv = jnp.linalg.pinv(J)
    print(J_inv.shape, g.shape)

    # For demonstration, we treat everything as if we only care about eps.
    # So, example chain rule: grad w.r.t. eps = J_inv * g
    grad_eps = jnp.dot(g, J_inv)

    # Return (grad_alpha, grad_eps, grad_mask)
    # If alpha/mask are not trainable or not used, return None for them:
    return (None, grad_eps, None)


# Register the forward/backward with JAX
normal_to_dirichlet.defvjp(normal_to_dirichlet_fwd, normal_to_dirichlet_bwd)
