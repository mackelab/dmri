import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import betainc
from jax.typing import ArrayLike
from probjax.utils.special import betaincinv


# ------------------------------------------------------------------------------
# 1. eps_mask (kept, JIT-safe)
#    eps_mask(mask)[i] == True  iff:
#      - mask[i] is True AND
#      - there exists at least one True in mask[j] for some j > i
#
#    Interpretation: eps[i] participates in stick-breaking for an "active"
#    category that is NOT the last active category. Shape: (K-1,)
# ------------------------------------------------------------------------------
def eps_mask(mask: jnp.ndarray) -> jnp.ndarray:
    mask_bool = jnp.asarray(mask, dtype=jnp.bool_)

    # suffix_any[i] = any(mask[i:])
    suffix_any = jnp.flip(jnp.cumsum(jnp.flip(mask_bool.astype(jnp.int32))) > 0)

    # future_any[i] = any(mask[i+1:])
    future_any = suffix_any[1:]  # shape (K-1,)
    current_active = mask_bool[:-1]  # shape (K-1,)

    return current_active & future_any  # shape (K-1,), bool


# ------------------------------------------------------------------------------
# 2. Helper: make sure we always have a boolean mask
# ------------------------------------------------------------------------------
def _bool_mask(mask, K: int):
    if mask is None:
        return jnp.ones((K,), dtype=jnp.bool_)
    return jnp.asarray(mask, dtype=jnp.bool_)


# ------------------------------------------------------------------------------
# 3. Helper: precompute Beta "b" parameters for each stick index i
#
#    For the active-only stick-breaking, at position i we want:
#      a_i = alpha[i]
#      b_i = sum_{j > i and mask[j]} alpha[j]
#
#    We'll build an array b_all of shape (K,) where:
#      b_all[i] = sum_{j > i} alpha[j] * mask[j]
#
#    This uses only static-shape ops (flip + cumsum).
# ------------------------------------------------------------------------------
def _precompute_b(alpha: jnp.ndarray, mask_bool: jnp.ndarray) -> jnp.ndarray:
    alpha = jnp.asarray(alpha)
    mask_bool = jnp.asarray(mask_bool, dtype=jnp.bool_)

    alpha_active = alpha * mask_bool.astype(alpha.dtype)  # (K,)
    # suffix_sum[k] = sum_{j >= k} alpha_active[j]
    suffix_sum = jnp.flip(jnp.cumsum(jnp.flip(alpha_active)))
    # b_i = sum_{j > i} alpha_active[j] = suffix_sum[i+1]
    # For convenience make shape (K,), last entry unused for eps/scan
    b_all = jnp.concatenate([suffix_sum[1:], jnp.array([0.0], dtype=alpha.dtype)])
    return b_all  # (K,)


# ------------------------------------------------------------------------------
# 4. Helper: last active index
#
#    last_active_idx = index of last True in mask.
#    We do this with argmax on reversed mask, which is JIT-friendly and
#    returns a scalar integer index. Using that index in .at[...] is OK.
# ------------------------------------------------------------------------------
def _last_active_index(mask_bool: jnp.ndarray) -> jnp.ndarray:
    K = mask_bool.shape[0]
    rev_mask_int = jnp.flip(mask_bool.astype(jnp.int32))
    # argmax gives first max in reversed array = distance from end
    last_active_from_end = jnp.argmax(rev_mask_int)
    last_active_idx = (K - 1) - last_active_from_end
    return last_active_idx  # scalar int index


# ------------------------------------------------------------------------------
# 5. Core forward: _forward_pure
#
# This produces the masked Dirichlet sample pi_full of shape (K,) from:
#   - alpha: (K,)
#   - eps:   (K-1,)
#   - mask_bool: (K,) bool
#
# Logic:
#   * Only eps[i] where eps_mask(mask_bool)[i] == True actually matter.
#   * We do a fixed-length lax.scan over i=0..K-2 to build all "non-final"
#     active sticks.
#   * We accumulate the active mass.
#   * After the scan we drop the leftover mass into the *last active* index,
#     just like standard stick-breaking assigns the remainder to the last stick.
#
# Result:
#   - pi_full[mask_bool] sums to 1
#   - pi_full[~mask_bool] = 0
#   - No renormalization tricks, no boolean indexing, static shapes only.
# ------------------------------------------------------------------------------
def _forward_pure(
    alpha: jnp.ndarray, eps: jnp.ndarray, mask_bool: jnp.ndarray
) -> jnp.ndarray:
    alpha = jnp.asarray(alpha)
    eps = jnp.asarray(eps)
    mask_bool = jnp.asarray(mask_bool, dtype=jnp.bool_)

    K = alpha.shape[0]
    assert eps.shape[0] == K - 1, "eps must have shape (K-1,)"

    # Which eps entries are actually meaningful?
    em = eps_mask(mask_bool)  # (K-1,) bool

    # Beta b parameters for each position
    b_all = _precompute_b(alpha, mask_bool)  # (K,)

    def scan_fn(pi_sum_active, i):
        a_i = alpha[i]
        b_i = b_all[i]

        # u_i = Phi(eps_i)
        u_i = jax.scipy.stats.norm.cdf(eps[i])
        u_i = jnp.clip(u_i, 1e-12, 1.0 - 1e-12)

        # phi_raw = betaincinv(a_i, b_i, u_i)
        phi_raw = betaincinv(
            a_i,
            b_i,
            u_i,
            max_halley_steps=8,
            max_bisection_steps=16,
        )
        phi_raw = jnp.nan_to_num(phi_raw, nan=0.5, posinf=0.5, neginf=0.5)

        # Only keep it if this index participates in stick-breaking
        phi_i = jnp.where(em[i], phi_raw, 0.0)

        # pi_i = phi_i * (1 - pi_sum_active)
        pi_i = phi_i * (1.0 - pi_sum_active)

        # If em[i] is False, this should not contribute any mass
        pi_i = jnp.where(em[i], pi_i, 0.0)

        new_sum = pi_sum_active + pi_i
        return new_sum, pi_i

    # Scan over the first K-1 sticks
    init_pi_sum = jnp.array(0.0, dtype=alpha.dtype)
    pi_sum_after, pis_partial = lax.scan(scan_fn, init_pi_sum, jnp.arange(K - 1))
    # pis_partial: (K-1,)

    # Remainder mass for the final active category
    remainder_mass = 1.0 - pi_sum_after

    # Build full pi vector
    pi_full = jnp.concatenate(
        [pis_partial, jnp.array([0.0], dtype=alpha.dtype)],
        axis=0,
    )  # shape (K,)

    # Put that remainder mass into the last active component
    last_idx = _last_active_index(mask_bool)
    pi_full = pi_full.at[last_idx].set(remainder_mass)

    # Zero out inactive entries explicitly (numerical safety)
    pi_full = jnp.where(mask_bool, pi_full, 0.0)

    # Final tiny renorm (should already sum to 1 on mask, but guard fp drift)
    active_mass = jnp.sum(pi_full * mask_bool.astype(pi_full.dtype))
    pi_full = jnp.where(
        mask_bool,
        pi_full / (active_mass + 1e-16),
        0.0,
    )

    # Clean NaNs/Infs
    pi_full = jnp.nan_to_num(pi_full, nan=0.0, posinf=0.0, neginf=0.0)
    return pi_full  # (K,)


# ------------------------------------------------------------------------------
# 6. Public forward with custom_vjp
#
# normal_to_dirichlet(alpha, eps, mask) -> pi_full
#
# This is now JIT-safe, static-shape, and respects the bijection on the
# active subset. eps MUST be shape (K-1,).
# ------------------------------------------------------------------------------
@jax.custom_vjp
def normal_to_dirichlet(
    alpha: ArrayLike, eps: ArrayLike, mask: ArrayLike | None = None
) -> ArrayLike:
    alpha = jnp.asarray(alpha)
    eps = jnp.asarray(eps)
    K = alpha.shape[0]
    mask_bool = _bool_mask(mask, K)
    return _forward_pure(alpha, eps, mask_bool)


def normal_to_dirichlet_fwd(alpha, eps, mask):
    alpha = jnp.asarray(alpha)
    eps = jnp.asarray(eps)
    K = alpha.shape[0]
    mask_bool = _bool_mask(mask, K)

    pi_full = _forward_pure(alpha, eps, mask_bool)
    # residuals we need for backward
    return pi_full, (alpha, eps, mask_bool)


def normal_to_dirichlet_bwd(res, g: ArrayLike):
    """
    We return gradients only where eps actually matters
    (eps_mask(mask)==True). Everywhere else grad_eps is forced to 0.
    No gradient for alpha or mask (consistent with your earlier design).
    """
    alpha, eps, mask_bool = res
    alpha = jnp.asarray(alpha)
    eps = jnp.asarray(eps)

    def scalarized(e_in):
        out = _forward_pure(alpha, e_in, mask_bool)
        return jnp.sum(out * g)

    grad_eps = jax.grad(scalarized)(eps)  # (K-1,)
    grad_eps = jnp.nan_to_num(grad_eps, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero out grads where eps does not participate
    em = eps_mask(mask_bool).astype(grad_eps.dtype)  # (K-1,)
    grad_eps = grad_eps * em

    return (None, grad_eps, None)


normal_to_dirichlet.defvjp(normal_to_dirichlet_fwd, normal_to_dirichlet_bwd)
normal_to_dirichlet = jax.jit(normal_to_dirichlet)


# ------------------------------------------------------------------------------
# 7. Inverse: dirichlet_to_normal
#
# Given:
#   alpha: (K,)
#   pi:    (K,) with pi[mask] summing to 1, inactive entries ~0
#   mask:  (K,) bool or None
#
# We reconstruct eps_full of shape (K-1,). Only entries where
# eps_mask(mask) == True are meaningful; others are 0.
#
# This is the exact inverse of _forward_pure at those coordinates,
# and it's all static-shape/JIT-safe.
# ------------------------------------------------------------------------------
@jax.jit
def dirichlet_to_normal(
    alpha: ArrayLike, pi: ArrayLike, mask: ArrayLike | None = None
) -> ArrayLike:
    alpha = jnp.asarray(alpha)
    pi = jnp.asarray(pi)
    K = alpha.shape[0]
    assert pi.shape[0] == K

    mask_bool = _bool_mask(mask, K)

    # Normalize only over active components for safety
    active_mass = jnp.sum(pi * mask_bool.astype(pi.dtype))
    pi_norm = jnp.where(
        mask_bool,
        pi / (active_mass + 1e-16),
        0.0,
    )

    # Precompute helpers (same static logic as forward)
    em = eps_mask(mask_bool)  # (K-1,) bool
    b_all = _precompute_b(alpha, mask_bool)  # (K,)
    # We'll reconstruct eps[i] for i in 0..K-2 where em[i] is True.

    def scan_fn(pi_sum_active, i):
        a_i = alpha[i]
        b_i = b_all[i]

        denom = 1.0 - pi_sum_active
        # phi_i = pi_i / (1 - sum_previous_active)
        pi_i = pi_norm[i]

        phi_raw = jnp.where(
            (denom > 0.0),
            pi_i / denom,
            0.0,
        )
        phi_raw = jnp.clip(phi_raw, 1e-12, 1.0 - 1e-12)

        u_i = betainc(a_i, b_i, phi_raw)
        u_i = jnp.clip(u_i, 1e-12, 1.0 - 1e-12)

        eps_raw = jax.scipy.stats.norm.ppf(u_i)

        # Only meaningful if em[i] is True. Otherwise set 0.
        eps_i = jnp.where(em[i], eps_raw, 0.0)

        # Update running sum ONLY for active/non-final sticks
        pi_sum_new = pi_sum_active + jnp.where(em[i], pi_i, 0.0)

        return pi_sum_new, eps_i

    init_sum = jnp.array(0.0, dtype=alpha.dtype)
    _, eps_full = lax.scan(scan_fn, init_sum, jnp.arange(K - 1))
    # eps_full: (K-1,)

    eps_full = jnp.nan_to_num(eps_full, nan=0.0, posinf=0.0, neginf=0.0)
    return eps_full  # shape (K-1,)


# ------------------------------------------------------------------------------
