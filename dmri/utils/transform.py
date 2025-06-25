import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from probjax.utils.special import betaincinv
from jax.scipy.special import betainc
import math


@jax.jit
def dirichlet_to_normal(
    alpha: ArrayLike, pi: ArrayLike, mask: ArrayLike | None = None
) -> ArrayLike:
    """
    Inverse of the normal_to_dirichlet stick-breaking transform.

    Args:
        alpha: Dirichlet concentration parameters, shape (K,)
        pi: A Dirichlet sample, shape (K,), sums to 1
        mask: Optional boolean mask of shape (K,) to zero out certain components

    Returns:
        Array of shape (K-1,) containing the Normal samples that would map
        back to `pi` under `normal_to_dirichlet(alpha, eps)`
    """
    zero = jnp.array(0.0)

    # Apply mask to both alpha and pi
    if mask is not None:
        alpha = jnp.where(mask, alpha, zero)
        pi = jnp.where(mask, pi, zero)

    # We need to reverse the stick-breaking process
    # For masked components, we'll just return zeros
    def scan_fn(carry, i):
        pi_sum = carry

        # Handle alpha masking
        a = alpha[i]
        mask = jnp.arange(len(alpha)) > i
        b = jnp.sum(alpha * mask)
        phi_i = pi[i] / (1.0 - pi_sum)
        u_i = betainc(a, b, phi_i)
        eps_i = jax.scipy.stats.norm.ppf(u_i)
        return pi_sum + pi[i], eps_i

    init_pi_sum = 0.0
    _, eps = lax.scan(scan_fn, init_pi_sum, jnp.arange(len(alpha) - 1))

    # Handle NaN values
    eps = jnp.nan_to_num(eps)

    return eps


# ------------------------------------------------------------------------------
# 1) A pure forward stick-breaking that also returns the intermediate phi's.
# ------------------------------------------------------------------------------
def _normal_to_dirichlet_and_phi(alpha, eps, mask=None):
    """
    Returns (pi, phi) where pi is the final Dirichlet sample and
    phi[i] = betaincinv(alpha[i], sum_{k>i} alpha[k], u_i).
    """
    assert len(eps) == len(alpha) - 1, "eps should be of size len(alpha) - 1"
    u = jax.scipy.stats.norm.cdf(eps)

    def scan_fn(carry, i):
        pi_sum = carry

        def new_phi(alpha, u, i):
            a = alpha[i]
            larger_i = jnp.arange(len(alpha)) > i
            _alpha = alpha if mask is None else jnp.where(mask, alpha, 0.0)
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

        pi_i = phi_i * (1 - pi_sum)
        pi_sum += pi_i
        return pi_sum, (pi_i, phi_i)

    pi_sum = 0.0
    pi_sum, (pis, phis) = lax.scan(scan_fn, pi_sum, jnp.arange(len(alpha) - 1))
    pi_K = 1.0 - pi_sum
    pis = jnp.append(pis, pi_K)
    phis = jnp.append(phis, 1.0 - phis.sum())
    if mask is not None:
        last_entry = -jnp.argmax(mask[::-1]) - 1
        zero = jnp.array(0.0)
        pis = pis.at[-1].set(zero)
        pis = pis.at[last_entry].set(pi_K)
        pis = jnp.where(~mask, zero, pis)
        pis /= jnp.sum(pis)
        pis = jnp.nan_to_num(pis)
    return pis, phis


# ------------------------------------------------------------------------------
# 2) A convenience forward function that also returns (pi) only,
#    but we store (alpha, eps, mask, pi, phi) as residuals for the backward pass.
# ------------------------------------------------------------------------------


def eps_mask(mask):
    # For each eps_i, check if current and any future component are active
    num_elements = len(mask)
    mask_eps = jnp.zeros((num_elements - 1,), dtype=bool)
    for i in range(num_elements - 1):
        current_active = mask[i]
        future_active = jnp.any(
            jnp.array([mask[j] for j in range(i + 1, num_elements)])
        )
        mask_eps = mask_eps.at[i].set(current_active & future_active)
    return mask_eps


@jax.custom_vjp
@jax.jit
def normal_to_dirichlet(alpha, eps, mask=None):
    pi, _phi = _normal_to_dirichlet_and_phi(alpha, eps, mask)
    return pi


def normal_to_dirichlet_fwd(alpha, eps, mask):
    pi, phi = _normal_to_dirichlet_and_phi(alpha, eps, mask)
    # Save everything needed for backward pass
    return pi, (alpha, eps, mask, pi, phi)


def normal_to_dirichlet_bwd(res: tuple, g: ArrayLike) -> tuple[None, ArrayLike, None]:
    """
    Given gradients w.r.t. pi (the Dirichlet sample), compute gradients w.r.t.:
      - alpha (we can return None or jnp.zeros_like(alpha) if you want no gradient there)
      - eps
      - mask

    For masked components:
    1. Their probabilities are set to zero
    2. Remaining probabilities are renormalized to sum to 1
    3. Gradients must account for both direct effects and renormalization
    """
    alpha, eps, mask, pi, phi = res

    # Define a function that uses _normal_to_dirichlet_and_phi directly to avoid recursion
    def compute_output(alpha_in, eps_in):
        pi_out, _ = _normal_to_dirichlet_and_phi(alpha_in, eps_in, mask)
        return jnp.sum(pi_out * g)

    # Use JAX's autodiff on the non-custom function
    _, grad_eps = jax.grad(compute_output, (0, 1))(alpha, eps)

    # Handle NaN values
    grad_eps = jnp.nan_to_num(grad_eps)

    # Apply masking if needed
    if mask is not None:
        # For each eps_i, check if current and any future component are active
        mask_eps = jnp.zeros_like(eps, dtype=jnp.bool_)
        for i in range(len(eps)):
            current_active = mask[i]
            future_active = jnp.any(
                jnp.array([mask[j] for j in range(i + 1, len(alpha))])
            )
            mask_eps = mask_eps.at[i].set(current_active & future_active)

        grad_eps = jnp.where(mask_eps, grad_eps, 0.0)

    return None, grad_eps, None


# # Register the forward/backward
normal_to_dirichlet.defvjp(normal_to_dirichlet_fwd, normal_to_dirichlet_bwd)
