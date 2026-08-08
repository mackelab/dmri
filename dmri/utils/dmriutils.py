import os
from typing import Any

import jax.numpy as jnp
import nibabel as nb
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def ssfp_signal_fn(
    adc: ArrayLike,
    qval: ArrayLike,
    E1: ArrayLike,
    E2: ArrayLike,
    sa: ArrayLike,
    ca: ArrayLike,
    TR: ArrayLike,
    diff_grad_dur: ArrayLike,
    S0: ArrayLike = 1.0,
):
    """
    Numerically-stable SSFP diffusion-attenuation signal.

    Identical output to `ssfp_signal_fn`, but robust for very large ADCs.
    Works with scalars or arbitrary-shaped arrays and remains JIT/grad-safe.
    """
    # ---------- cast once ----------
    adc = jnp.asarray(adc)
    qval = jnp.asarray(qval)
    dtype = jnp.result_type(adc, qval)
    adc, qval = adc.astype(dtype), qval.astype(dtype)

    # ---------- logs of all exponentials ----------
    g_TR = qval**2 * TR * adc  # γ_TR
    g_dur = qval**2 * diff_grad_dur * adc  # γ_dur

    logA1 = -g_TR  # log(A1)
    logA2 = -g_dur  # log(A2)
    logA2_03 = -g_dur / 3.0  # log(A2_03)

    logE2 = jnp.log(E2 + 1e-30)  # avoid log(0)
    log1mE1 = jnp.log1p(-E1)  # stable for E1≈1

    exp_ = jnp.exp

    # ---------- helper ----------
    def e(logx):  # shorthand exp(log(x))
        return exp_(logx)

    # ---------- s ----------
    s = e(logE2 + logA1 - 4 * logA2_03) * (1 - E1 * ca) + e(logE2 - logA2_03) * (
        ca - E1
    )

    # ---------- r ----------
    r = (1 - E1 * ca) + e(2 * logE2 + logA1 + logA2_03) * (ca - E1)

    # ---------- K ----------
    num = (
        1
        - E1 * e(logA1) * ca
        - e(2 * logE2 + 2 * logA1 - 2 * logA2_03) * (E1 * e(logA1) - ca)
    )

    den = e(logE2 + logA1 - 4 * logA2_03) * (1 + ca) * (1 - E1 * e(logA1))
    K = num / (den + 1e-30)

    # ---------- F1 (stable form) ----------
    A2_sq = e(2 * logA2)
    sqrt_disc = jnp.sqrt(jnp.maximum(K * K - A2_sq, 0))
    F1 = A2_sq / (K + sqrt_disc + 1e-30)

    # ---------- M− ----------
    top_factor = -e(log1mE1 + logE2 - 2 * logA2_03) * sa
    prod2 = e(logE2 + logA1 + 2 * logA2_03)  # E2*A1*A2_03²
    Mminus_top = top_factor * (F1 - prod2)
    Mminus_bottom = r - F1 * s

    signal = jnp.abs(S0 * Mminus_top / (Mminus_bottom + 1e-30))
    return jnp.nan_to_num(signal)


def fit_diffusion_tensor_linearized(
    logS: Array, bvals: Array, bvecs: Array
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


def _split_vector(vec: ArrayLike, size: int, *, name: str) -> tuple[Array, ...]:
    arr = jnp.asarray(vec)
    if arr.ndim == 0:
        raise ValueError(f"{name} vector must have at least one dimension.")
    if arr.shape[-1] == size:
        components = tuple(arr[..., idx] for idx in range(size))
    elif arr.shape[0] == size:
        components = tuple(arr[idx] for idx in range(size))
    else:
        raise ValueError(
            f"{name} vector must have length {size} along its leading or trailing axis."
        )
    return components  # type: ignore[return-value]


def cart2sph(
    x: ArrayLike,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
) -> Array | tuple[Array, Array]:
    """Convert Cartesian coordinates to spherical coordinates.

    Accepts either a single stacked array (..., 3) / (3, ...) or three separate
    components. Returns a stacked output when the input is stacked, otherwise a
    tuple ``(theta, phi)``.
    """
    stacked_input = y is None or z is None
    if stacked_input:
        x, y, z = _split_vector(x, 3, name="Cartesian")
    else:
        x, y, z = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)

    r = jnp.sqrt(x**2 + y**2 + z**2)
    safe_ratio = jnp.where(r == 0, z, z / r)
    theta = jnp.arccos(jnp.clip(safe_ratio, -1.0, 1.0))
    phi = jnp.arctan2(y, x)
    if stacked_input:
        return jnp.stack([theta, phi], axis=-1)
    return theta, phi


def sph2cart(
    theta: ArrayLike,
    phi: ArrayLike | None = None,
) -> Array | tuple[Array, Array, Array]:
    """Convert spherical coordinates to Cartesian coordinates.

    Accepts stacked ``[..., 2]`` / ``(2, ...)`` inputs or two separate arrays.
    Returns a stacked output when the input is stacked, otherwise a tuple
    ``(x, y, z)``.
    """
    stacked_input = phi is None
    if stacked_input:
        theta, phi = _split_vector(theta, 2, name="Spherical")
    else:
        theta, phi = jnp.asarray(theta), jnp.asarray(phi)

    sin_theta = jnp.sin(theta)
    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = jnp.cos(theta)
    if stacked_input:
        return jnp.stack([x, y, z], axis=-1)
    return x, y, z


def normalize_bvecs(bvecs: ArrayLike) -> ArrayLike:
    """Normalize b-vectors to unit length."""
    norms = jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    return bvecs / norms


def compute_fa(D: ArrayLike) -> ArrayLike:
    """Compute fractional anisotropy (FA) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    mean_diffusivity = jnp.mean(evals)
    fa = jnp.sqrt(1.5 * jnp.sum((evals - mean_diffusivity) ** 2) / jnp.sum(evals**2))
    return fa


def compute_md(D: ArrayLike) -> ArrayLike:
    """Compute mean diffusivity (MD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return jnp.mean(evals)


def compute_rd(D: ArrayLike) -> ArrayLike:
    """Compute radial diffusivity (RD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return jnp.mean(evals[:2])


def compute_ad(D: ArrayLike) -> ArrayLike:
    """Compute axial diffusivity (AD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return evals[2]


def rotation_matrix_100_to_theta_phi(theta, phi):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position on the unit sphere.

    Parameters
    ----------
    theta : float,
        inclination of polar angle of main angle mu [0, pi].
    phi : float,
        polar angle of main angle mu [-pi, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    x, y, z = sph2cart(theta, phi)
    return rotation_matrix_100_to_xyz(x, y, z)


def rotation_matrix_100_to_xyz(x, y, z):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position in Cartesian space.

    Parameters
    ----------
    x, y, z : floats,
        position in Cartesian space.

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    is_identity = (x == 1.0) & (y == 0.0) & (z == 0.0)
    R_identity = jnp.eye(3)

    # Avoid division by zero
    denom = jnp.maximum((y * y + z * z), 1e-12)
    R_general = jnp.array([
        [x, -y, -z],
        [
            y,
            (x * (y * y) + (z * z)) / denom,
            ((x - 1) * (y * z)) / denom,
        ],
        [
            z,
            ((x - 1) * (y * z)) / denom,
            ((y * y) + x * (z * z)) / denom,
        ],
    ])
    # Use a data-dependent condition instead of Python ifs
    R = jnp.where(is_identity, R_identity, R_general)
    return R


def rotation_matrix_100_to_theta_phi_psi(theta, phi, psi):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position in Cartesian space, and rotates about its axis.

    Parameters
    ----------
    theta : float,
        inclination of polar angle of main angle mu [0, pi].
    phi : float,
        polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    R_100_to_theta_phi = rotation_matrix_100_to_theta_phi(theta, phi)
    R_around_100 = rotation_matrix_around_100(psi)
    return jnp.dot(R_100_to_theta_phi, R_around_100)


def rotation_matrix_around_100(psi):
    """Generates a rotation matrix that rotates around the x-axis (1, 0, 0).

    Parameters
    ----------
    psi : float,
        euler angle [0, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    cos_psi = jnp.cos(psi)
    sin_psi = jnp.sin(psi)

    R = jnp.array([[1.0, 0.0, 0.0], [0.0, cos_psi, -sin_psi], [0.0, sin_psi, cos_psi]])
    return R


def make_dyads(
    theta_samples: Array, phi_samples: Array, percentile: float = None
) -> tuple[Array, Array]:
    """
    Uses fibre orientation samples (in spherical coordinates) from the posterior to estimate the mean fibre orientation
    (in cartesian coordinates [x,y,z]) and the uncertainty (dispersion) around it.

    Args:
        theta_samples: Array of inclination angles
        phi_samples: Array of azimuthal angles
        percentile: Optional percentile for dispersion calculation

    Returns:
        tuple: (v, disp) where v is the mean orientation and disp is the dispersion
    """
    v = jnp.stack(
        [
            jnp.sin(theta_samples) * jnp.cos(phi_samples),
            jnp.sin(theta_samples) * jnp.sin(phi_samples),
            jnp.cos(theta_samples),
        ],
        axis=0,
    )

    dyadic_tensor = jnp.matmul(v, v.T) / len(theta_samples)
    L, E = jnp.linalg.eigh(dyadic_tensor)

    ind = jnp.argsort(-L)
    v1 = E[:, ind[0]]
    disp = 1 - jnp.max(jnp.abs(L))

    if percentile is not None:
        # Calculate angular deviations from the principal eigenvector
        dot_prod = jnp.matmul(v.T, v1)
        angles = jnp.arccos(jnp.clip(dot_prod, -1, 1)) * (
            180 / jnp.pi
        )  # Convert to degrees
        angles = jnp.where(angles > 90, 180 - angles, angles)
        # Determine the cone angle at the specified percentile
        disp = jnp.percentile(angles, percentile)

    return v1, disp


def _normalize_volume_slice(volume_slice, spatial_shape):
    if volume_slice is None:
        return ()
    normalized = []
    axes = min(len(volume_slice), len(spatial_shape))
    for axis in range(axes):
        axis_slice = volume_slice[axis]
        axis_len = spatial_shape[axis]
        if axis_slice is None:
            normalized.append(slice(0, axis_len, 1))
            continue
        start, stop, step = axis_slice.indices(axis_len)
        normalized.append(slice(start, stop, step))
    return tuple(normalized)


def _apply_volume_slice_to_data(data, normalized_slice):
    if not normalized_slice:
        return data
    spatial_dims = min(len(normalized_slice), data.ndim)
    slicing = normalized_slice[:spatial_dims] + (slice(None),) * (
        data.ndim - spatial_dims
    )
    return data[slicing]


def _adjust_affine_for_slice(affine, normalized_slice):
    if not normalized_slice:
        return affine
    new_affine = np.array(affine, copy=True)
    spatial_axes = min(len(normalized_slice), 3)
    for axis in range(spatial_axes):
        axis_slice = normalized_slice[axis]
        basis = new_affine[:3, axis].copy()
        new_affine[:3, 3] += basis * axis_slice.start
        new_affine[:3, axis] = basis * axis_slice.step
    return new_affine


def export_nifti(
    data: ArrayLike,
    orig_data: Any,
    output_path: str,
    name: str,
    volume_slice: tuple[slice, ...] | None = None,
) -> None:
    """Save an array to NIfTI, preserving affine/slices from the source image."""
    # Copy the header of the original image
    vol_slice = (
        volume_slice
        if volume_slice is not None
        else getattr(orig_data, "volume_slice", None)
    )
    spatial_shape = getattr(orig_data, "shape", data.shape)
    normalized_slice = _normalize_volume_slice(vol_slice, spatial_shape)
    aff_mat = (
        np.array(orig_data.affine, copy=True)
        if hasattr(orig_data, "affine")
        else np.eye(4, dtype=float)
    )
    if normalized_slice:
        data = _apply_volume_slice_to_data(data, normalized_slice)
        aff_mat = _adjust_affine_for_slice(aff_mat, normalized_slice)
    nb.save(nb.Nifti2Image(data, affine=aff_mat), os.path.join(output_path, name))


def reorder_angles_3fib(mu1, mu2, mu3, f1, f2, f3):
    """Reorder angles to maintain consistency across samples by comparing to reference vectors.

    Sample 0 provides the reference directions and is left untouched; every other
    sample is permuted so its first direction is the one closest to ``v1_ref``
    and its second is whichever of the remaining two is closer to ``v2_ref``.
    Each sample is compared only against the references, never against its
    predecessor, so all samples are handled in parallel.

    Args:
        mu1, mu2, mu3: Arrays of shape (n_samples, 2) containing spherical angles (theta, phi)
        f1, f2, f3: Arrays of shape (n_samples,) containing the fractions

    Returns:
        new_mu1, new_mu2, new_mu3: Reordered angles arrays of same shape as inputs
        new_f1, new_f2, new_f3: Reordered fraction arrays of same shape as inputs
    """
    mus = jnp.stack([mu1, mu2, mu3], axis=1)  # (S, 3, 2)
    fracs = jnp.stack([f1, f2, f3], axis=1)  # (S, 3)

    vecs = jnp.stack(sph2cart(mus[..., 0], mus[..., 1]), axis=-1)  # (S, 3, 3)
    unit = vecs / jnp.linalg.norm(vecs, axis=-1, keepdims=True)
    v1_ref, v2_ref = unit[0, 0], unit[0, 1]

    # Best match for the first reference direction.
    first = jnp.argmax(unit @ v1_ref, axis=-1)  # (S,)

    # Of the two directions that are left, take the one closer to v2_ref.
    remaining = jnp.stack(
        [(first + 1) % 3, (first + 2) % 3], axis=-1
    )  # (S, 2), in ascending order of original index modulo the rotation
    remaining = jnp.sort(remaining, axis=-1)
    dots_v2 = jnp.take_along_axis(unit @ v2_ref, remaining, axis=-1)  # (S, 2)
    take_first = dots_v2[:, 0] > dots_v2[:, 1]
    second = jnp.where(take_first, remaining[:, 0], remaining[:, 1])
    third = jnp.where(take_first, remaining[:, 1], remaining[:, 0])

    order = jnp.stack([first, second, third], axis=-1)  # (S, 3)
    # Sample 0 defines the references and is kept as-is.
    order = order.at[0].set(jnp.arange(3))

    new_mus = jnp.take_along_axis(mus, order[..., None], axis=1)
    new_fracs = jnp.take_along_axis(fracs, order, axis=1)

    return (
        new_mus[:, 0],
        new_mus[:, 1],
        new_mus[:, 2],
        new_fracs[:, 0],
        new_fracs[:, 1],
        new_fracs[:, 2],
    )
