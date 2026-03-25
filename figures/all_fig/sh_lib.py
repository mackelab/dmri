import numpy as np
from dipy.reconst.shm import sf_to_sh

def kappa_from_sigma_deg(sigma_deg: float) -> float:
    sigma = np.deg2rad(float(sigma_deg))
    return 1.0 / (sigma * sigma + 1e-12)

def _effective_sample_size(weights):
    # Kish effective sample size: (sum w)^2 / sum(w^2)
    w = np.asarray(weights, dtype=np.float64)
    sw = w.sum()
    sw2 = np.sum(w * w)
    if sw <= 0 or sw2 <= 0:
        return 0.0
    return (sw * sw) / sw2

def choose_sigma_deg(samples_xyz, weights=None, antipodal=True,
                     sigma_min=3.0, sigma_max=18.0):
    """
    Heuristic bandwidth (degrees) for spherical KDE from direction samples.

    Rationale:
    - Baseline spacing from Neff: area per point ≈ 4π/Neff, spacing ≈ sqrt(area).
    - Concentration from orientation tensor: more peaked => smaller sigma.

    Returns
    -------
    sigma_deg : float
    """
    default_sigma = float(np.clip(12.0, sigma_min, sigma_max))
    X = np.asarray(samples_xyz, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("samples_xyz must have shape (M, 3).")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if not np.all(np.isfinite(X)):
        return default_sigma

    if weights is None:
        w = np.ones((X.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (X.shape[0],):
            raise ValueError("weights must have shape (M,).")
        if (w < 0).any():
            raise ValueError("weights must be nonnegative.")
    if not np.all(np.isfinite(w)):
        return default_sigma

    Neff = _effective_sample_size(w)
    if Neff < 5:
        # Too few points to infer structure reliably; use a conservative default.
        return default_sigma

    # Baseline spacing (radians) from equal-area argument.
    # spacing_rad ~ sqrt(4π / Neff)
    spacing_rad = np.sqrt(4.0 * np.pi / Neff)
    spacing_deg = np.rad2deg(spacing_rad)

    # Concentration estimate from second-moment tensor.
    # For antipodal data, second moment is appropriate (sign cancels).
    # T = Σ w x x^T / Σ w, trace(T)=1.
    sw = w.sum()
    if not np.isfinite(sw) or sw <= 0:
        return default_sigma
    T = (X * w[:, None]).T @ X / (sw + 1e-12)
    if not np.all(np.isfinite(T)):
        return default_sigma
    T = 0.5 * (T + T.T)
    # normalize trace to 1 for stability (should already be ~1)
    tr = np.trace(T)
    if not np.isfinite(tr) or tr <= 1e-12:
        return default_sigma
    T = T / tr
    T = 0.5 * (T + T.T) + np.eye(3) * 1e-8

    try:
        evals = np.linalg.eigvalsh(T)
    except np.linalg.LinAlgError:
        # Rare non-convergence: add more diagonal jitter, then fall back.
        try:
            evals = np.linalg.eigvalsh(T + np.eye(3) * 1e-6)
        except np.linalg.LinAlgError:
            return default_sigma
    evals.sort()
    l1, l2, l3 = evals[2], evals[1], evals[0]  # l1 >= l2 >= l3

    # Concentration proxy in [0, 1] (approx):
    # isotropic: l1≈l2≈l3≈1/3 => c≈0
    # strongly peaked: l1→1, l2,l3→0 => c→1
    c = (l1 - l2) / (l1 + 1e-12)
    c = float(np.clip(c, 0.0, 1.0))

    # Map (spacing, concentration) -> sigma.
    # - diffuse (c~0): sigma ~ 1.1 * spacing (more smoothing)
    # - peaked (c~1):  sigma ~ 0.5 * spacing (preserve sharpness)
    factor = 1.1 - 0.6 * c
    sigma = factor * spacing_deg

    # Additional guardrails: if antipodal=False and data is oriented with strong mean,
    # you may want a slightly smaller sigma; keep it mild.
    if not antipodal:
        sigma *= 0.9

    return float(np.clip(sigma, sigma_min, sigma_max))

def spherical_kde_vmf_from_samples(
    samples_xyz,
    eval_sphere,
    sigma_deg=7.5,
    weights=None,
    antipodal=True,
    normalize="sum",
    eps=1e-12,
):
    X = np.asarray(samples_xyz, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("samples_xyz must have shape (M, 3).")
    X /= np.linalg.norm(X, axis=1, keepdims=True) + eps

    U = np.asarray(eval_sphere.vertices, dtype=np.float64)
    U /= np.linalg.norm(U, axis=1, keepdims=True) + eps

    if weights is None:
        w = np.ones((X.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (X.shape[0],):
            raise ValueError("weights must have shape (M,).")
        if (w < 0).any():
            raise ValueError("weights must be nonnegative.")

    kappa = kappa_from_sigma_deg(float(sigma_deg))

    dots = U @ X.T  # (N_eval, M)
    if antipodal:
        K = np.cosh(kappa * dots)
    else:
        K = np.exp(kappa * dots)

    sf = K @ w  # (N_eval,)

    if normalize == "sum":
        s = sf.sum()
        if s > 0:
            sf = sf / s

    return sf

def samples_to_sh(
    samples_xyz,
    eval_sphere,
    sigma_deg="auto",
    sh_order_max=8,
    sh_basis="descoteaux07",
    weights=None,
    antipodal=True,
    full_basis=False,
):
    if sigma_deg == "auto":
        sigma_deg = choose_sigma_deg(samples_xyz, weights=weights, antipodal=antipodal)

    sf_kde = spherical_kde_vmf_from_samples(
        samples_xyz,
        eval_sphere,
        sigma_deg=sigma_deg,
        weights=weights,
        antipodal=antipodal,
        normalize="sum",
    )

    sh = sf_to_sh(
        sf_kde,
        eval_sphere,
        sh_order_max=sh_order_max,
        basis_type=sh_basis,
        full_basis=full_basis,
    )
    return sf_kde, sh, float(sigma_deg)
