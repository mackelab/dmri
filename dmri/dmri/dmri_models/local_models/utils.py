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
