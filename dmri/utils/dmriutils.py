import os

import jax
import jax.numpy as jnp
import nibabel as nb
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


def freed_ssfp_signal_fn(
    adc: ArrayLike,
    qval: ArrayLike,
    TR: ArrayLike,
    T1: ArrayLike,
    T2: ArrayLike,
    sa: ArrayLike,
    ca: ArrayLike,
    S0: ArrayLike = 1.0,
    num_terms: int = 10,
):
    """Numerically-stable, shape-safe freed-diffusion SSFP signal."""

    # ---------- helper exponentials ----------
    def E1p(p):
        return jnp.exp(-TR / T1 - adc * qval**2 * TR * p**2)

    def E2p(p):
        return jnp.exp(-TR / T2 - adc * qval**2 * ((p**2 + p + 1 / 3) * TR))

    # ---------- short-hand symbols ----------
    def Ap(p):
        return 0.5 * (E1p(p) - 1) * (1 + ca)

    def Bp(p):
        return 0.5 * (E1p(p) + 1) * (1 - ca)

    def Cp(p):
        return E1p(p) - ca

    def np_(p):
        return -E2p(-p) * E2p(p - 1) * Ap(p) ** 2 * Bp(p - 1) / Bp(p)

    def dp(p):
        return (Ap(p) - Bp(p)) + E2p(-p - 1) * E2p(p) * Bp(p) * Cp(p + 1) / Bp(p + 1)

    def ep(p):
        return -E2p(p) * E2p(-p - 1) * Bp(p) * Cp(p + 1) / Bp(p + 1)

    # ---------- scan body ----------
    def scan_body(carry, k):
        # choose the correct denominator while keeping shapes identical
        denom = jax.lax.cond(
            k == 1,
            lambda _: dp(k) + ep(k),  # last recursion: base case
            lambda _: dp(k) + carry,  # recursive case
            operand=None,
        )
        new_carry = np_(k) / denom
        return new_carry, None

    # ---------- run backward recurrence ----------
    k_values = jnp.arange(num_terms, 0, -1, dtype=jnp.int32)
    init_carry = jnp.zeros_like(dp(1))  # <<< shape-correct seed
    x1, _ = jax.lax.scan(scan_body, init_carry, k_values)

    # ---------- finish analytical part ----------
    r1 = x1 / (E2p(-1) * Bp(0)) + (E2p(0) * Cp(1)) / Bp(1)
    S = r1 * sa * (1 - E1p(0)) * E2p(-1) / (Ap(0) - Bp(0) + E2p(-1) * Cp(0) * r1)

    return jnp.nan_to_num(S0 * jnp.abs(S))


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


def cartesian_to_unitsphere(cartesian: ArrayLike) -> ArrayLike:
    """Convert Cartesian coordinates to spherical coordinates."""
    x, y, z = cartesian
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return jnp.array([theta, phi])


def unitsphere_to_cartesian(mu: ArrayLike) -> ArrayLike:
    """Convert spherical coordinates to Cartesian coordinates."""
    theta, phi = mu
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.array([x, y, z])


def normalize_bvecs(bvecs: ArrayLike) -> ArrayLike:
    """Normalize b-vectors to unit length."""
    norms = jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    return bvecs / norms


def compute_FA(D: ArrayLike) -> ArrayLike:
    """Compute fractional anisotropy (FA) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    mean_diffusivity = jnp.mean(evals)
    fa = jnp.sqrt(1.5 * jnp.sum((evals - mean_diffusivity) ** 2) / jnp.sum(evals**2))
    return fa


def compute_MD(D: ArrayLike) -> ArrayLike:
    """Compute mean diffusivity (MD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return jnp.mean(evals)


def compute_RD(D: ArrayLike) -> ArrayLike:
    """Compute radial diffusivity (RD) from a diffusion tensor."""
    evals = jnp.linalg.eigvalsh(D)
    return jnp.mean(evals[:2])


def compute_AD(D: ArrayLike) -> ArrayLike:
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
    x, y, z = unitsphere_to_cartesian([theta, phi])
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
    R_general = jnp.array(
        [
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
        ]
    )
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


def canonical_bingham_normalization_series(kappa, beta, max_terms=30):
    """
    Series expansion for Z(kappa, beta) using JAX.
    Sums up to 'max_terms' to approximate.
    """
    A = kappa - beta
    n = jnp.arange(max_terms)
    terms = A**n / (jax.scipy.special.gamma(n + 1) * (n + 0.5))
    summation = jnp.sum(terms)
    return 2.0 * jnp.pi * jnp.exp(beta) * summation


def make_dyads(
    theta_samples: ArrayLike, phi_samples: ArrayLike, percentile: float = None
) -> tuple[ArrayLike, float]:
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


def cart2sph(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        tuple: (theta, phi) spherical coordinates
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.where(
        r == 0,
        jnp.arccos(z),  # To avoid NaN when r==0
        jnp.arccos(z / r),
    )
    phi = jnp.arctan2(y, x)
    return theta, phi


def sph2cart(theta: ArrayLike, phi: ArrayLike) -> ArrayLike:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        theta: Inclination angle
        phi: Azimuthal angle

    Returns:
        Array: Cartesian coordinates [x, y, z]
    """
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)


def export_nifti(data, orig_data, output_path, name):
    """
    Args:
        data:
        orig_data:
        output_path:
        name:
    """
    # Copy the header of the original image
    aff_mat = orig_data.affine
    nb.save(nb.Nifti2Image(data, affine=aff_mat), os.path.join(output_path, name))


def reorder_angles_3fib(mu1, mu2, mu3, f1, f2, f3):
    """Reorder angles to maintain consistency across samples by comparing to reference vectors.

    Args:
        mu1, mu2, mu3: Arrays of shape (n_samples, 2) containing spherical angles (theta, phi)
        f1, f2, f3: Arrays of shape (n_samples,) containing the fractions

    Returns:
        new_mu1, new_mu2, new_mu3: Reordered angles arrays of same shape as inputs
        new_f1, new_f2, new_f3: Reordered fraction arrays of same shape as inputs
    """

    print(mu1.shape, mu2.shape, mu3.shape, f1.shape, f2.shape, f3.shape)

    # Initialize output arrays
    new_mu1 = jnp.zeros_like(mu1)
    new_mu2 = jnp.zeros_like(mu2)
    new_mu3 = jnp.zeros_like(mu3)
    new_f1 = jnp.zeros_like(f1)
    new_f2 = jnp.zeros_like(f2)
    new_f3 = jnp.zeros_like(f3)

    # Use first sample as reference vectors
    v1_ref = sph2cart(mu1[0, 0], mu1[0, 1])
    v2_ref = sph2cart(mu2[0, 0], mu2[0, 1])
    v3_ref = sph2cart(mu3[0, 0], mu3[0, 1])

    # Copy first sample directly
    new_mu1 = new_mu1.at[0].set(mu1[0])
    new_mu2 = new_mu2.at[0].set(mu2[0])
    new_mu3 = new_mu3.at[0].set(mu3[0])
    new_f1 = new_f1.at[0].set(f1[0])
    new_f2 = new_f2.at[0].set(f2[0])
    new_f3 = new_f3.at[0].set(f3[0])

    # Process remaining samples
    for j in range(1, mu1.shape[0]):
        # Convert current sample to cartesian
        v1 = sph2cart(mu1[j, 0], mu1[j, 1])
        v2 = sph2cart(mu2[j, 0], mu2[j, 1])
        v3 = sph2cart(mu3[j, 0], mu3[j, 1])

        # Calculate dot products with v1_ref
        dots = jnp.array(
            [
                jnp.dot(v1_ref, v1) / (jnp.linalg.norm(v1_ref) * jnp.linalg.norm(v1)),
                jnp.dot(v1_ref, v2) / (jnp.linalg.norm(v1_ref) * jnp.linalg.norm(v2)),
                jnp.dot(v1_ref, v3) / (jnp.linalg.norm(v1_ref) * jnp.linalg.norm(v3)),
            ]
        )

        # Find best match for v1_ref
        best_match = jnp.argmax(dots)

        # Reorder based on best match with v1_ref using where
        angles1 = jnp.where(
            best_match == 0, mu1[j], jnp.where(best_match == 1, mu2[j], mu3[j])
        )
        frac1 = jnp.where(
            best_match == 0, f1[j], jnp.where(best_match == 1, f2[j], f3[j])
        )

        # Get remaining angles and fractions
        remaining_angles = jnp.where(
            best_match == 0,
            jnp.array([mu2[j], mu3[j]]),
            jnp.where(
                best_match == 1,
                jnp.array([mu1[j], mu3[j]]),
                jnp.array([mu1[j], mu2[j]]),
            ),
        )
        remaining_fracs = jnp.where(
            best_match == 0,
            jnp.array([f2[j], f3[j]]),
            jnp.where(
                best_match == 1, jnp.array([f1[j], f3[j]]), jnp.array([f1[j], f2[j]])
            ),
        )

        v_remaining = jnp.where(
            best_match == 0,
            jnp.array([v2, v3]),
            jnp.where(best_match == 1, jnp.array([v1, v3]), jnp.array([v1, v2])),
        )

        # Find best match for v2_ref among remaining vectors
        dots_v2 = jnp.array(
            [
                jnp.dot(v2_ref, v_remaining[0])
                / (jnp.linalg.norm(v2_ref) * jnp.linalg.norm(v_remaining[0])),
                jnp.dot(v2_ref, v_remaining[1])
                / (jnp.linalg.norm(v2_ref) * jnp.linalg.norm(v_remaining[1])),
            ]
        )

        # Order remaining two vectors based on similarity to v2_ref using where
        angles2 = jnp.where(
            dots_v2[0] > dots_v2[1], remaining_angles[0], remaining_angles[1]
        )
        angles3 = jnp.where(
            dots_v2[0] > dots_v2[1], remaining_angles[1], remaining_angles[0]
        )
        frac2 = jnp.where(
            dots_v2[0] > dots_v2[1], remaining_fracs[0], remaining_fracs[1]
        )
        frac3 = jnp.where(
            dots_v2[0] > dots_v2[1], remaining_fracs[1], remaining_fracs[0]
        )

        # Store reordered angles and fractions
        new_mu1 = new_mu1.at[j].set(angles1)
        new_mu2 = new_mu2.at[j].set(angles2)
        new_mu3 = new_mu3.at[j].set(angles3)
        new_f1 = new_f1.at[j].set(frac1)
        new_f2 = new_f2.at[j].set(frac2)
        new_f3 = new_f3.at[j].set(frac3)

    return new_mu1, new_mu2, new_mu3, new_f1, new_f2, new_f3


def export_SBI_estimates(
    samples: ArrayLike,
    mask: ArrayLike,
    data_brain_orig: ArrayLike,
    outPath: str,
    nfib: int = 3,
    modelnum: int = 1,
) -> None:
    """
    Export SBI estimates to NIfTI files.

    Args:
        samples: Array of samples
        mask: Brain mask
        data_brain_orig: Original brain data for header information
        outPath: Output directory path
        nfib: Number of fiber components
        modelnum: Model number
    """
    # d
    export_nifti(samples[..., 0], data_brain_orig, outPath, "merged_dsamples.nii.gz")
    export_nifti(
        jnp.median(samples[..., 0], axis=3),
        data_brain_orig,
        outPath,
        "mean_dsamples.nii.gz",
    )
    export_nifti(
        jnp.std(samples[..., 0], axis=3),
        data_brain_orig,
        outPath,
        "std_dsamples.nii.gz",
    )

    # fibre components
    mean_fsumsamples = jnp.zeros((samples.shape[0], samples.shape[1], samples.shape[2]))
    for i in range(nfib):
        # f
        export_nifti(
            samples[..., 1 + 3 * i],
            data_brain_orig,
            outPath,
            f"merged_f{i + 1}samples.nii.gz",
        )
        export_nifti(
            jnp.median(samples[..., 1 + 3 * i], axis=3),
            data_brain_orig,
            outPath,
            f"mean_f{i + 1}samples.nii.gz",
        )
        export_nifti(
            jnp.std(samples[..., 1 + 3 * i], axis=3),
            data_brain_orig,
            outPath,
            f"std_f{i + 1}samples.nii.gz",
        )
        mean_fsumsamples += jnp.median(samples[..., 1 + 3 * i], axis=3)

        # v
        export_nifti(
            samples[..., 2 + 3 * i],
            data_brain_orig,
            outPath,
            f"merged_th{i + 1}samples.nii.gz",
        )
        export_nifti(
            samples[..., 3 + 3 * i],
            data_brain_orig,
            outPath,
            f"merged_ph{i + 1}samples.nii.gz",
        )

        v, disp = make_dyads(samples[..., 2 + 3 * i], samples[..., 3 + 3 * i])
        export_nifti(v, data_brain_orig, outPath, f"dyads{i + 1}.nii.gz")
        export_nifti(disp, data_brain_orig, outPath, f"dyads{i + 1}_dispersion.nii.gz")

    # SNR
    export_nifti(samples[..., -1], data_brain_orig, outPath, "merged_SNRsamples.nii.gz")
    export_nifti(
        jnp.median(samples[..., -1], axis=3),
        data_brain_orig,
        outPath,
        "mean_SNRsamples.nii.gz",
    )
    export_nifti(
        jnp.std(samples[..., -1], axis=3),
        data_brain_orig,
        outPath,
        "std_SNRsamples.nii.gz",
    )

    if modelnum == 2:
        # d_std
        export_nifti(
            samples[..., -2], data_brain_orig, outPath, "merged_d_stdsamples.nii.gz"
        )
        export_nifti(
            jnp.median(samples[..., -2], axis=3),
            data_brain_orig,
            outPath,
            "mean_d_stdsamples.nii.gz",
        )
        export_nifti(
            jnp.std(samples[..., -2], axis=3),
            data_brain_orig,
            outPath,
            "std_d_stdsamples.nii.gz",
        )
