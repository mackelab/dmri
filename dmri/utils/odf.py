import jax
import jax.numpy as jnp
from jax import lax

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
    evec3 = evecs_sorted[:, 2]

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
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
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
