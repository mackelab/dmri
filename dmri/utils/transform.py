import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from probjax.utils.stats import betainc, betaincinv


@jax.jit
def _normal_to_dirichlet(
    alpha: ArrayLike, eps: ArrayLike, mask: ArrayLike | None = None
) -> ArrayLike:
    """
    Internal implementation of the normal to Dirichlet transformation.

    Args:
        alpha: Dirichlet concentration parameters, shape (K,)
        eps: Normal samples, shape (K-1,)
        mask: Optional boolean mask of shape (K,) to zero out certain components

    Returns:
        Array of shape (K,) representing a Dirichlet sample
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
        return pi_sum, pi_i

    pi_sum = 0.0
    pi_sum, pis = lax.scan(scan_fn, pi_sum, jnp.arange(len(alpha) - 1))
    pi_K = 1.0 - pi_sum
    pis = jnp.append(pis, pi_K)
    if mask is not None:
        last_entry = -jnp.argmax(mask[::-1]) - 1
        zero = jnp.array(0.0)
        pis = pis.at[-1].set(zero)
        pis = pis.at[last_entry].set(pi_K)
        pis = jnp.where(~mask, zero, pis)
        pis /= jnp.sum(pis)
        pis = jnp.nan_to_num(pis)
    return pis


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
def normal_to_dirichlet(
    alpha: ArrayLike, eps: ArrayLike, mask: ArrayLike | None = None
) -> ArrayLike:
    """
    Custom VJP implementation of normal to Dirichlet transformation.

    Args:
        alpha: Dirichlet concentration parameters, shape (K,)
        eps: Normal samples, shape (K-1,)
        mask: Optional boolean mask of shape (K,) to zero out certain components

    Returns:
        Array of shape (K,) representing a Dirichlet sample
    """
    return _normal_to_dirichlet(alpha, eps, mask)


# Forward pass for custom VJP. Return (output, residuals_for_bwd).
def normal_to_dirichlet_fwd(
    alpha: ArrayLike, eps: ArrayLike, mask: ArrayLike | None
) -> tuple[ArrayLike, tuple]:
    out = _normal_to_dirichlet(alpha, eps, mask)
    return out, (alpha, eps, mask, out)


# Backward pass. Given residuals and the gradient w.r.t. output (g),
# return the gradient w.r.t. each input: (grad_alpha, grad_eps, grad_mask).
def normal_to_dirichlet_bwd(res: tuple, g: ArrayLike) -> tuple[None, ArrayLike, None]:
    alpha, eps, mask, out = res

    def f_inv(dirichlet_out: ArrayLike) -> ArrayLike:
        return dirichlet_to_normal(alpha, dirichlet_out, mask=mask)

    J = jax.jacfwd(f_inv)(out)
    J_inv = jnp.linalg.pinv(J)

    grad_eps = jnp.dot(g, J_inv)
    return (None, grad_eps, None)


# Register the forward/backward with JAX
normal_to_dirichlet.defvjp(normal_to_dirichlet_fwd, normal_to_dirichlet_bwd)


def uniform_rotation(rng_key: jax.random.PRNGKey) -> ArrayLike:
    """
    Generate a uniform random rotation matrix using quaternions.

    Args:
        rng_key: JAX PRNG key for random number generation

    Returns:
        Array of shape (3, 3) representing a random rotation matrix in SO(3)
    """
    # Generate a random quaternion
    q = jax.random.uniform(rng_key, (4,))
    q /= jnp.linalg.norm(q)

    # Convert quaternion to rotation matrix
    R = transform_quaternion_to_S03(q)
    return R


def r3_to_s3(v: ArrayLike) -> ArrayLike:
    """
    Stereographic projection from R^3 to S^3 (minus one point).

    Args:
        v: Array of shape (3,) representing a point in R^3

    Returns:
        Array of shape (4,) representing a quaternion (w,x,y,z) of norm 1
        The origin maps to (1,0,0,0).
        The 'north pole' is (-1,0,0,0) and is not reached by any finite v.
    """
    x, y, z = v
    r_sq = x * x + y * y + z * z
    denom = 1.0 + r_sq

    w = (1.0 - r_sq) / denom
    i = (2.0 * x) / denom
    j = (2.0 * y) / denom
    k = (2.0 * z) / denom
    q = jnp.array([w, i, j, k])

    return q


def s3_to_r3(q: ArrayLike) -> ArrayLike:
    """
    Inverse stereographic projection from a quaternion in S^3 back to R^3.

    Args:
        q: Array of shape (4,) representing a quaternion (w,x,y,z) in S^3
           The excluded point on S^3 is (-1,0,0,0)

    Returns:
        Array of shape (3,) representing a point in R^3
    """
    w, x, y, z = q
    r_sq = (1.0 - w) / (1.0 + w)

    factor = 0.5 * (1.0 + r_sq)
    x_3D = x * factor
    y_3D = y * factor
    z_3D = z * factor
    return jnp.array([x_3D, y_3D, z_3D])


def transform_quaternion_to_S03(q: ArrayLike) -> ArrayLike:
    """
    Convert a unit quaternion to a 3x3 rotation matrix in SO(3).

    Args:
        q: Array of shape (4,) representing a unit quaternion (w, x, y, z)

    Returns:
        Array of shape (3, 3) representing a rotation matrix in SO(3)
    """
    w, x, y, z = q
    R = jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R


def transform_S03_to_quaternion(R: ArrayLike) -> ArrayLike:
    """
    Convert rotation matrix in SO(3) to a unit quaternion, enforcing w >= 0.

    Args:
        R: Array of shape (3, 3) representing a rotation matrix in SO(3)

    Returns:
        Array of shape (4,) representing a unit quaternion (w,x,y,z) with w >= 0
    """
    w = jnp.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    w4 = 4.0 * w
    x = (R[2, 1] - R[1, 2]) / w4
    y = (R[0, 2] - R[2, 0]) / w4
    z = (R[1, 0] - R[0, 1]) / w4
    q = jnp.array([w, x, y, z])

    # Enforce w >= 0 for a canonical representative in double-cover
    q = jnp.where(q[0] < 0, -q, q)
    return q
