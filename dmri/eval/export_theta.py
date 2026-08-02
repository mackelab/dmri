import logging
import os

import jax
import numpy as np

from dmri import console
from dmri.simulators.local_signal_models.ball import MultiShellStaticBall
from dmri.utils.dmriutils import export_nifti, make_dyads, reorder_angles_3fib

#: Voxels per chunk for the export's own device passes.
DEFAULT_EXPORT_BATCH_SIZE = 20_000

#: Minimum mean fraction for a stick to count as a resolved fiber.
FIBER_FRACTION_THRESHOLD = 0.05


def spherical_to_cartesian(theta, phi):
    """Stack ``(theta, phi)`` angles into unit vectors of shape ``(..., 3)``."""
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)],
        axis=-1,
    ).astype(np.float32)


def _conditional_mean(values, keep, fallback):
    """Mean of ``values`` over the samples selected by ``keep``, per voxel.

    Voxels where ``keep`` selects nothing fall back to ``fallback``.
    """
    counts = keep.sum(axis=1)
    totals = np.where(keep, values, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), fallback).astype(
        np.float32
    )


def _conditional_std(values, keep, fallback):
    """Standard deviation of ``values`` over the samples selected by ``keep``."""
    counts = keep.sum(axis=1)
    means = _conditional_mean(values, keep, fallback * 0.0)
    sq = np.where(keep, (values - means[:, None]) ** 2, 0.0).sum(axis=1)
    return np.where(counts > 0, np.sqrt(sq / np.maximum(counts, 1)), fallback).astype(
        np.float32
    )


def _divide_safe(numerator, denominator, eps=1e-8):
    """Elementwise division that yields 0 where the denominator vanishes."""
    out = np.zeros_like(numerator, dtype=np.float32)
    return np.divide(numerator, denominator, out=out, where=np.abs(denominator) > eps)


def _valid_fraction_samples(fractions):
    """Return validity per simplex-valued posterior draw."""
    fractions = np.asarray(fractions)
    return (
        np.all(np.isfinite(fractions), axis=-1)
        & np.all(fractions >= 0.0, axis=-1)
        & np.isclose(np.sum(fractions, axis=-1), 1.0, rtol=1e-5, atol=1e-6)
    )


def _cfg_flag(cfg, name, default):
    """Read an optional config flag that older config files may not define."""
    try:
        value = cfg.get(name, default)
    except (AttributeError, TypeError):
        value = getattr(cfg, name, default)
    return default if value is None else bool(value)


def _cfg_batch_size(cfg):
    """Voxels per export chunk, honouring an explicit `batch_size` override.

    One cheap forward pass per sample, so a fixed chunk is enough -- no probing.
    """
    override = None
    try:
        override = cfg.get("batch_size")
    except (AttributeError, TypeError):
        override = None
    return int(override) if override else DEFAULT_EXPORT_BATCH_SIZE


def map_over_voxels(fn, *arrays, batch_size=DEFAULT_EXPORT_BATCH_SIZE):
    """Apply ``fn(theta, mask)`` over the (voxel, sample) axes in voxel chunks.

    ``fn`` may return a pytree; the results are concatenated on the host as
    float32 numpy arrays so nothing larger than one chunk stays on the device.
    """
    mapped = jax.jit(jax.vmap(jax.vmap(fn)))
    num_voxels = arrays[0].shape[0]
    if num_voxels == 0:
        return jax.tree_util.tree_map(
            lambda x: np.asarray(x, dtype=np.float32), mapped(*arrays)
        )

    chunks = []
    for start in range(0, num_voxels, batch_size):
        stop = min(start + batch_size, num_voxels)
        result = mapped(*(np.asarray(a[start:stop]) for a in arrays))
        chunks.append(
            jax.tree_util.tree_map(
                lambda x: np.asarray(jax.device_get(x), dtype=np.float32), result
            )
        )

    if len(chunks) == 1:
        return chunks[0]
    return jax.tree_util.tree_map(lambda *parts: np.concatenate(parts, axis=0), *chunks)


def embed_in_full_brain_array(to_embed, brain_mask_flat, brain_shape):
    """Embed a tensor in the full brain array"""
    event_shape = to_embed.shape[1:]
    full_brain = np.zeros(
        (brain_mask_flat.shape[0],) + event_shape, dtype=to_embed.dtype
    )
    full_brain[brain_mask_flat, ...] = to_embed
    full_brain = full_brain.reshape(brain_shape + event_shape)
    return full_brain


def export_thetas_raw(
    cfg,
    thetas,
    true_model_mask,
    true_thetas,
    brain_mask_flat,
    brain_shape,
    out_path,
    orig_data,
):
    """Export raw theta samples without model-specific post-processing."""
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.savez_compressed(os.path.join(out_path, "theta_samples.npz"), thetas=thetas)
    np.savez_compressed(os.path.join(out_path, "true_theta.npz"), thetas=true_thetas)
    np.savez_compressed(
        os.path.join(out_path, "true_model_mask.npz"), model_mask=true_model_mask
    )

    full_thetas = embed_in_full_brain_array(
        thetas, brain_mask_flat, brain_shape
    ).astype(np.float32)
    export_nifti(full_thetas, orig_data, out_path, "raw_thetas.nii.gz")


def export_thetas_to_files_ball3stick(
    cfg,
    thetas,
    sim_type,
    model_mask,
    brain_mask_flat,
    brain_shape,
    out_path,
    orig_data,
    batch_size=None,
):
    """Export thetas to files for ball3stick

    Supported models:
    - Ball3StickSharedDiffusivity
    - MultiShellBall3StickSharedDiffusivity
    - MultiShellBall3StickSharedDiffusivityGammaPrior

    If the model is not supported, the output might be wrong.
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    thetas = np.asarray(thetas)
    theta_valid_samples = np.all(np.isfinite(thetas), axis=-1)
    safe_thetas = np.where(theta_valid_samples[..., None], thetas, 0.0)
    logging.info(f"Exporting raw inferred parameters to {out_path}")
    # Export raw samples
    full_thetas = embed_in_full_brain_array(
        np.where(theta_valid_samples[..., None], thetas, np.nan),
        brain_mask_flat,
        brain_shape,
    ).astype(np.float32)
    export_nifti(full_thetas, orig_data, out_path, "raw_thetas.nii.gz")

    batch_size = batch_size or _cfg_batch_size(cfg)
    condition_fsum = _cfg_flag(cfg.export, "condition_fsum_on_ball", True)

    num_components = len(sim_type.model_types) + len(sim_type.noise_types)
    if model_mask is None:
        model_mask = np.ones((*thetas.shape[:2], num_components), dtype=np.bool_)
    else:
        model_mask = np.asarray(model_mask, dtype=np.bool_)
        if model_mask.ndim == 1:
            model_mask = np.broadcast_to(
                model_mask, (*thetas.shape[:2], model_mask.shape[-1])
            )
        elif model_mask.ndim == 2:
            model_mask = np.broadcast_to(
                model_mask[:, None, :],
                (thetas.shape[0], thetas.shape[1], model_mask.shape[-1]),
            )
        if model_mask.shape[:2] != thetas.shape[:2]:
            raise ValueError(
                "model_mask must align with the voxel and sample dimensions of thetas"
            )

    # To save multishell stds
    is_multi_shell = sim_type.model_types[0] is MultiShellStaticBall

    def extract(theta, mask):
        """Pull every exported quantity out of one theta in a single pass."""
        simulator = sim_type.from_theta(theta, model_mask=mask)
        ball = simulator.model_compartments[0]
        out = {
            "fractions": simulator.model_fractions,
            "diffusitivity": ball.lam,
            "mu1": simulator.model_compartments[1].mu,
            "mu2": simulator.model_compartments[2].mu,
            "mu3": simulator.model_compartments[3].mu,
            "snr": simulator.noise_compartments[0].snr,
        }
        if is_multi_shell:
            out["diffusitivity_std"] = ball.lam_std
        return out

    # A single vmapped call over every brain voxel exhausts a small GPU.
    extracted = map_over_voxels(extract, safe_thetas, model_mask, batch_size=batch_size)
    extracted = jax.tree_util.tree_map(
        lambda values: np.where(
            theta_valid_samples.reshape(
                theta_valid_samples.shape + (1,) * (values.ndim - 2)
            ),
            values,
            np.nan,
        ).astype(np.float32),
        extracted,
    )

    fractions = extracted["fractions"]
    signal_active = np.any(model_mask[..., : len(sim_type.model_types)], axis=-1)
    fraction_math_valid = _valid_fraction_samples(fractions)
    fraction_valid_samples = theta_valid_samples & signal_active & fraction_math_valid
    noise_only_samples = theta_valid_samples & ~signal_active
    malformed_fraction_samples = (
        theta_valid_samples & signal_active & ~fraction_math_valid
    )

    def warn_samples(samples, description):
        if not np.any(samples):
            return
        sample_count = int(np.count_nonzero(samples))
        affected_voxels = int(np.count_nonzero(np.any(samples, axis=1)))
        message = (
            f"{sample_count} posterior draws across {affected_voxels} in-brain "
            f"voxels {description}; affected fraction outputs will be NaN."
        )
        if console.enabled():
            console.warning(message)
        else:
            logging.warning(message)

    warn_samples(noise_only_samples, "are noise-only and have no signal fractions")
    warn_samples(malformed_fraction_samples, "have malformed signal fractions")

    # Noise-only masks are valid, but signal fractions are not defined for them.
    # Embedding later still leaves every outside-brain voxel at zero.
    fractions = np.where(fraction_valid_samples[..., None], fractions, np.nan)
    diffusitivity = extracted["diffusitivity"]
    mu1 = extracted["mu1"]
    mu2 = extracted["mu2"]
    mu3 = extracted["mu3"]
    snr = extracted["snr"]

    if is_multi_shell:
        diffusitivity_std = extracted["diffusitivity_std"]
        diffusitivity_std_mean = np.mean(diffusitivity_std, axis=1)

        full_diffusitivity_std_mean = embed_in_full_brain_array(
            diffusitivity_std_mean, brain_mask_flat, brain_shape
        )
        export_nifti(
            full_diffusitivity_std_mean,
            orig_data,
            out_path,
            "mean_dstdsamples_std.nii.gz",
        )

    if cfg.export.sort_by_fractions:
        logging.info("Sorting fiber fractions and corresponding directions.")
        # Stable sort so equal fractions keep a deterministic order.
        order = np.argsort(-fractions[..., 1:], axis=-1, kind="stable")  # (V, S, 3)

        fractions = fractions.copy()
        fractions[..., 1:] = np.take_along_axis(fractions[..., 1:], order, axis=-1)

        mus = np.stack([mu1, mu2, mu3], axis=-2)  # (V, S, 3, 2)
        mus = np.take_along_axis(mus, order[..., None], axis=-2)
        mu1, mu2, mu3 = mus[..., 0, :], mus[..., 1, :], mus[..., 2, :]

    logging.info("Exporting moments and processed inferred parameters.")

    # Moments
    fractions_mean = np.mean(fractions, axis=1)
    diffusitivity_mean = np.mean(diffusitivity, axis=1)

    f0_mean = fractions_mean[..., 0]
    f1_mean = fractions_mean[..., 1]
    f2_mean = fractions_mean[..., 2]
    f3_mean = fractions_mean[..., 3]

    # A sample that drops the ball renormalises to f0 = 0, pinning its f_sum at
    # 1; conditioning on an active ball keeps the map comparable to a fit that
    # always has one. See conf/evaluation/export/theta/ball3stick.yaml.
    ball_active = np.asarray(model_mask[..., 0], dtype=np.bool_)  # (V, S)
    fsum_samples = fractions[..., 1:].sum(axis=-1)  # (V, S)
    fsum_mean_all = fsum_samples.mean(axis=1)
    ball_active_fraction = ball_active.mean(axis=1).astype(np.float32)

    if condition_fsum:
        fsum_mean = _conditional_mean(fsum_samples, ball_active, fsum_mean_all)
    else:
        fsum_mean = fsum_mean_all
    fraction_voxels_valid = np.all(fraction_valid_samples, axis=1)
    fsum_mean = np.where(fraction_voxels_valid, fsum_mean, np.nan)

    if cfg.export.export_stds:
        fractions_std = np.std(fractions, axis=1)
        diffusitivity_std = np.std(diffusitivity, axis=1)
        f0_std = fractions_std[..., 0]
        f1_std = fractions_std[..., 1]
        f2_std = fractions_std[..., 2]
        f3_std = fractions_std[..., 3]
        # std of the sum, not the sum of stds: the fractions are constrained to
        # sum to 1 and so are strongly negatively correlated.
        fsum_std_all = np.std(fsum_samples, axis=1)
        if condition_fsum:
            fsum_std = _conditional_std(fsum_samples, ball_active, fsum_std_all)
        else:
            fsum_std = fsum_std_all
        fsum_std = np.where(fraction_voxels_valid, fsum_std, np.nan)

    # Export diffusitivity
    full_diffusitivity_mean = embed_in_full_brain_array(
        diffusitivity_mean, brain_mask_flat, brain_shape
    )
    full_diffusitivity_samples = embed_in_full_brain_array(
        diffusitivity, brain_mask_flat, brain_shape
    )

    export_nifti(full_diffusitivity_mean, orig_data, out_path, "mean_dsamples.nii.gz")
    export_nifti(
        full_diffusitivity_samples, orig_data, out_path, "merged_dsamples.nii.gz"
    )

    if cfg.export.export_stds:
        full_diffusitivity_std = embed_in_full_brain_array(
            diffusitivity_std, brain_mask_flat, brain_shape
        )
        export_nifti(full_diffusitivity_std, orig_data, out_path, "std_dsamples.nii.gz")

    # Export SNR
    snr_mean = np.mean(snr, axis=1)

    full_snr_mean = embed_in_full_brain_array(snr_mean, brain_mask_flat, brain_shape)
    full_snr_samples = embed_in_full_brain_array(snr, brain_mask_flat, brain_shape)

    if cfg.export.export_stds:
        snr_std = np.std(snr, axis=1)
        full_snr_std = embed_in_full_brain_array(snr_std, brain_mask_flat, brain_shape)
        export_nifti(full_snr_std, orig_data, out_path, "std_snrsamples.nii.gz")

    export_nifti(full_snr_mean, orig_data, out_path, "mean_snrsamples.nii.gz")
    export_nifti(full_snr_samples, orig_data, out_path, "merged_snrsamples.nii.gz")

    # Export mean fractions
    full_f0_mean = embed_in_full_brain_array(f0_mean, brain_mask_flat, brain_shape)
    full_f1_mean = embed_in_full_brain_array(f1_mean, brain_mask_flat, brain_shape)
    full_f2_mean = embed_in_full_brain_array(f2_mean, brain_mask_flat, brain_shape)
    full_f3_mean = embed_in_full_brain_array(f3_mean, brain_mask_flat, brain_shape)
    full_fsum_mean = embed_in_full_brain_array(fsum_mean, brain_mask_flat, brain_shape)

    # `f*_mean` is already reduced over samples, so this counts per voxel.
    num_fib_pred = (
        (f1_mean > FIBER_FRACTION_THRESHOLD).astype(np.float32)
        + (f2_mean > FIBER_FRACTION_THRESHOLD).astype(np.float32)
        + (f3_mean > FIBER_FRACTION_THRESHOLD).astype(np.float32)
    )

    # Some useful auxiliary. f1_mean is exactly 0 wherever stick 1 is masked off
    # in every sample, so guard the division.
    f2_f1_ratio = _divide_safe(f2_mean, f1_mean)
    f3_f1_ratio = _divide_safe(f3_mean, f1_mean)
    num_fib_pred = np.where(fraction_voxels_valid, num_fib_pred, np.nan)
    f2_f1_ratio = np.where(fraction_voxels_valid, f2_f1_ratio, np.nan)
    f3_f1_ratio = np.where(fraction_voxels_valid, f3_f1_ratio, np.nan)
    full_f2_f1_ratio = embed_in_full_brain_array(
        f2_f1_ratio, brain_mask_flat, brain_shape
    )
    full_f3_f1_ratio = embed_in_full_brain_array(
        f3_f1_ratio, brain_mask_flat, brain_shape
    )
    full_num_fib_pred = embed_in_full_brain_array(
        num_fib_pred, brain_mask_flat, brain_shape
    )

    # Export as nifti
    export_nifti(full_f0_mean, orig_data, out_path, "mean_f0samples.nii.gz")
    export_nifti(full_f1_mean, orig_data, out_path, "mean_f1samples.nii.gz")
    export_nifti(full_f2_mean, orig_data, out_path, "mean_f2samples.nii.gz")
    export_nifti(full_f3_mean, orig_data, out_path, "mean_f3samples.nii.gz")
    export_nifti(full_fsum_mean, orig_data, out_path, "mean_fsumsamples.nii.gz")
    export_nifti(
        embed_in_full_brain_array(
            fsum_mean_all.astype(np.float32), brain_mask_flat, brain_shape
        ),
        orig_data,
        out_path,
        "mean_fsumsamples_all.nii.gz",
    )
    export_nifti(
        embed_in_full_brain_array(ball_active_fraction, brain_mask_flat, brain_shape),
        orig_data,
        out_path,
        "frac_ball_active.nii.gz",
    )
    export_nifti(
        full_f2_f1_ratio, orig_data, out_path, "mean_f2_f1_ratiosamples.nii.gz"
    )
    export_nifti(
        full_f3_f1_ratio, orig_data, out_path, "mean_f3_f1_ratiosamples.nii.gz"
    )
    export_nifti(
        full_num_fib_pred, orig_data, out_path, "mean_num_fib_predsamples.nii.gz"
    )

    # Export std fractions
    if cfg.export.export_stds:
        full_f0_std = embed_in_full_brain_array(f0_std, brain_mask_flat, brain_shape)
        full_f1_std = embed_in_full_brain_array(f1_std, brain_mask_flat, brain_shape)
        full_f2_std = embed_in_full_brain_array(f2_std, brain_mask_flat, brain_shape)
        full_f3_std = embed_in_full_brain_array(f3_std, brain_mask_flat, brain_shape)
        full_fsum_std = embed_in_full_brain_array(
            fsum_std, brain_mask_flat, brain_shape
        )

        export_nifti(full_f0_std, orig_data, out_path, "std_f0samples.nii.gz")
        export_nifti(full_f1_std, orig_data, out_path, "std_f1samples.nii.gz")
        export_nifti(full_f2_std, orig_data, out_path, "std_f2samples.nii.gz")
        export_nifti(full_f3_std, orig_data, out_path, "std_f3samples.nii.gz")
        export_nifti(full_fsum_std, orig_data, out_path, "std_fsumsamples.nii.gz")

    # Export samples fractions
    full_fractions_samples = embed_in_full_brain_array(
        fractions, brain_mask_flat, brain_shape
    )
    export_nifti(
        full_fractions_samples[..., 0], orig_data, out_path, "merged_f0samples.nii.gz"
    )
    export_nifti(
        full_fractions_samples[..., 1], orig_data, out_path, "merged_f1samples.nii.gz"
    )
    export_nifti(
        full_fractions_samples[..., 2], orig_data, out_path, "merged_f2samples.nii.gz"
    )
    export_nifti(
        full_fractions_samples[..., 3], orig_data, out_path, "merged_f3samples.nii.gz"
    )

    # Angles
    mu1_theta = mu1[..., 0]
    mu2_theta = mu2[..., 0]
    mu3_theta = mu3[..., 0]

    mu1_phi = mu1[..., 1]
    mu2_phi = mu2[..., 1]
    mu3_phi = mu3[..., 1]

    # Export theta angles samples
    full_mu1_theta = embed_in_full_brain_array(mu1_theta, brain_mask_flat, brain_shape)
    full_mu2_theta = embed_in_full_brain_array(mu2_theta, brain_mask_flat, brain_shape)
    full_mu3_theta = embed_in_full_brain_array(mu3_theta, brain_mask_flat, brain_shape)
    export_nifti(full_mu1_theta, orig_data, out_path, "merged_th1samples.nii.gz")
    export_nifti(full_mu2_theta, orig_data, out_path, "merged_th2samples.nii.gz")
    export_nifti(full_mu3_theta, orig_data, out_path, "merged_th3samples.nii.gz")

    # Export phi angles samples
    full_mu1_phi = embed_in_full_brain_array(mu1_phi, brain_mask_flat, brain_shape)
    full_mu2_phi = embed_in_full_brain_array(mu2_phi, brain_mask_flat, brain_shape)
    full_mu3_phi = embed_in_full_brain_array(mu3_phi, brain_mask_flat, brain_shape)
    export_nifti(full_mu1_phi, orig_data, out_path, "merged_ph1samples.nii.gz")
    export_nifti(full_mu2_phi, orig_data, out_path, "merged_ph2samples.nii.gz")
    export_nifti(full_mu3_phi, orig_data, out_path, "merged_ph3samples.nii.gz")

    # Cartesian samples
    mu1_cart = spherical_to_cartesian(mu1_theta, mu1_phi)
    mu2_cart = spherical_to_cartesian(mu2_theta, mu2_phi)
    mu3_cart = spherical_to_cartesian(mu3_theta, mu3_phi)

    # Export cartesian samples
    full_mu1_cart = embed_in_full_brain_array(mu1_cart, brain_mask_flat, brain_shape)
    full_mu2_cart = embed_in_full_brain_array(mu2_cart, brain_mask_flat, brain_shape)
    full_mu3_cart = embed_in_full_brain_array(mu3_cart, brain_mask_flat, brain_shape)
    export_nifti(full_mu1_cart, orig_data, out_path, "merged_cart1samples.nii.gz")
    export_nifti(full_mu2_cart, orig_data, out_path, "merged_cart2samples.nii.gz")
    export_nifti(full_mu3_cart, orig_data, out_path, "merged_cart3samples.nii.gz")

    # Dyads unarranged
    dyads1_in_brain_unordered, dyads1_disp_unordered = jax.vmap(make_dyads)(
        mu1_theta, mu1_phi
    )
    dyads2_in_brain_unordered, dyads2_disp_unordered = jax.vmap(make_dyads)(
        mu2_theta, mu2_phi
    )
    dyads3_in_brain_unordered, dyads3_disp_unordered = jax.vmap(make_dyads)(
        mu3_theta, mu3_phi
    )

    # Export dyads unordered
    full_dyads1_in_brain_unordered = embed_in_full_brain_array(
        dyads1_in_brain_unordered, brain_mask_flat, brain_shape
    )
    full_dyads2_in_brain_unordered = embed_in_full_brain_array(
        dyads2_in_brain_unordered, brain_mask_flat, brain_shape
    )
    full_dyads3_in_brain_unordered = embed_in_full_brain_array(
        dyads3_in_brain_unordered, brain_mask_flat, brain_shape
    )
    export_nifti(full_dyads1_in_brain_unordered, orig_data, out_path, "dyads1.nii.gz")
    export_nifti(full_dyads2_in_brain_unordered, orig_data, out_path, "dyads2.nii.gz")
    export_nifti(full_dyads3_in_brain_unordered, orig_data, out_path, "dyads3.nii.gz")

    # Export dyads dispersion
    full_dyads1_disp_unordered = embed_in_full_brain_array(
        dyads1_disp_unordered, brain_mask_flat, brain_shape
    )
    full_dyads2_disp_unordered = embed_in_full_brain_array(
        dyads2_disp_unordered, brain_mask_flat, brain_shape
    )
    full_dyads3_disp_unordered = embed_in_full_brain_array(
        dyads3_disp_unordered, brain_mask_flat, brain_shape
    )
    export_nifti(
        full_dyads1_disp_unordered, orig_data, out_path, "dyads1_dispersion.nii.gz"
    )
    export_nifti(
        full_dyads2_disp_unordered, orig_data, out_path, "dyads2_dispersion.nii.gz"
    )
    export_nifti(
        full_dyads3_disp_unordered, orig_data, out_path, "dyads3_dispersion.nii.gz"
    )

    # Reorder
    # Dyads reordered
    if cfg.export.reorder_dyads:
        reordered_path = os.path.join(out_path, "reordered")
        if not os.path.exists(reordered_path):
            os.makedirs(reordered_path)

        f1 = fractions[..., 1]
        f2 = fractions[..., 2]
        f3 = fractions[..., 3]
        (
            mu1_reordered,
            mu2_reordered,
            mu3_reordered,
            f1_reordered,
            f2_reordered,
            f3_reordered,
        ) = jax.vmap(reorder_angles_3fib)(mu1, mu2, mu3, f1, f2, f3)

        dyads1_in_brain_reordered, dyads1_disp_reordered = jax.vmap(make_dyads)(
            mu1_reordered[..., 0], mu1_reordered[..., 1]
        )
        dyads2_in_brain_reordered, dyads2_disp_reordered = jax.vmap(make_dyads)(
            mu2_reordered[..., 0], mu2_reordered[..., 1]
        )
        dyads3_in_brain_reordered, dyads3_disp_reordered = jax.vmap(make_dyads)(
            mu3_reordered[..., 0], mu3_reordered[..., 1]
        )

        fractions_reordered = np.concatenate(
            [
                fractions[..., 0][..., None],
                f1_reordered[..., None],
                f2_reordered[..., None],
                f3_reordered[..., None],
            ],
            axis=-1,
        )
        # Export reordered
        full_fractions_reordered = embed_in_full_brain_array(
            fractions_reordered, brain_mask_flat, brain_shape
        )

        export_nifti(
            full_fractions_reordered,
            orig_data,
            reordered_path,
            "merged_fsamples.nii.gz",
        )

        f0_reordered = fractions_reordered[..., 0]
        f1_reordered = fractions_reordered[..., 1]
        f2_reordered = fractions_reordered[..., 2]
        f3_reordered = fractions_reordered[..., 3]
        fsum_reordered = f1_reordered + f2_reordered + f3_reordered

        full_f0_reordered = embed_in_full_brain_array(
            f0_reordered, brain_mask_flat, brain_shape
        )
        full_f1_reordered = embed_in_full_brain_array(
            f1_reordered, brain_mask_flat, brain_shape
        )
        full_f2_reordered = embed_in_full_brain_array(
            f2_reordered, brain_mask_flat, brain_shape
        )
        full_f3_reordered = embed_in_full_brain_array(
            f3_reordered, brain_mask_flat, brain_shape
        )
        full_fsum_reordered = embed_in_full_brain_array(
            fsum_reordered, brain_mask_flat, brain_shape
        )
        export_nifti(
            full_f0_reordered, orig_data, reordered_path, "merged_f0samples.nii.gz"
        )
        export_nifti(
            full_f1_reordered, orig_data, reordered_path, "merged_f1samples.nii.gz"
        )
        export_nifti(
            full_f2_reordered, orig_data, reordered_path, "merged_f2samples.nii.gz"
        )
        export_nifti(
            full_f3_reordered, orig_data, reordered_path, "merged_f3samples.nii.gz"
        )
        export_nifti(
            full_fsum_reordered, orig_data, reordered_path, "merged_fsumsamples.nii.gz"
        )

        if cfg.export.export_stds:
            fractions_std_reordered = np.std(fractions_reordered, axis=1)
            f0_std_reordered = fractions_std_reordered[..., 0]
            f1_std_reordered = fractions_std_reordered[..., 1]
            f2_std_reordered = fractions_std_reordered[..., 2]
            f3_std_reordered = fractions_std_reordered[..., 3]
            fsum_samples_reordered = fractions_reordered[..., 1:].sum(axis=-1)
            fsum_std_reordered_all = np.std(fsum_samples_reordered, axis=1)
            if condition_fsum:
                fsum_std_reordered = _conditional_std(
                    fsum_samples_reordered, ball_active, fsum_std_reordered_all
                )
            else:
                fsum_std_reordered = fsum_std_reordered_all

            full_f0_std_reordered = embed_in_full_brain_array(
                f0_std_reordered, brain_mask_flat, brain_shape
            )
            full_f1_std_reordered = embed_in_full_brain_array(
                f1_std_reordered, brain_mask_flat, brain_shape
            )
            full_f2_std_reordered = embed_in_full_brain_array(
                f2_std_reordered, brain_mask_flat, brain_shape
            )
            full_f3_std_reordered = embed_in_full_brain_array(
                f3_std_reordered, brain_mask_flat, brain_shape
            )
            full_fsum_std_reordered = embed_in_full_brain_array(
                fsum_std_reordered, brain_mask_flat, brain_shape
            )
            export_nifti(
                full_f0_std_reordered, orig_data, reordered_path, "std_f0samples.nii.gz"
            )
            export_nifti(
                full_f1_std_reordered, orig_data, reordered_path, "std_f1samples.nii.gz"
            )
            export_nifti(
                full_f2_std_reordered, orig_data, reordered_path, "std_f2samples.nii.gz"
            )
            export_nifti(
                full_f3_std_reordered, orig_data, reordered_path, "std_f3samples.nii.gz"
            )
            export_nifti(
                full_fsum_std_reordered,
                orig_data,
                reordered_path,
                "std_fsumsamples.nii.gz",
            )

        full_dyads1_in_brain_reordered = embed_in_full_brain_array(
            dyads1_in_brain_reordered, brain_mask_flat, brain_shape
        )
        full_dyads2_in_brain_reordered = embed_in_full_brain_array(
            dyads2_in_brain_reordered, brain_mask_flat, brain_shape
        )
        full_dyads3_in_brain_reordered = embed_in_full_brain_array(
            dyads3_in_brain_reordered, brain_mask_flat, brain_shape
        )
        export_nifti(
            full_dyads1_in_brain_reordered, orig_data, reordered_path, "dyads1.nii.gz"
        )
        export_nifti(
            full_dyads2_in_brain_reordered, orig_data, reordered_path, "dyads2.nii.gz"
        )
        export_nifti(
            full_dyads3_in_brain_reordered, orig_data, reordered_path, "dyads3.nii.gz"
        )

        full_dyads1_disp_reordered = embed_in_full_brain_array(
            dyads1_disp_reordered, brain_mask_flat, brain_shape
        )
        full_dyads2_disp_reordered = embed_in_full_brain_array(
            dyads2_disp_reordered, brain_mask_flat, brain_shape
        )
        full_dyads3_disp_reordered = embed_in_full_brain_array(
            dyads3_disp_reordered, brain_mask_flat, brain_shape
        )
        export_nifti(
            full_dyads1_disp_reordered,
            orig_data,
            reordered_path,
            "dyads1_dispersion.nii.gz",
        )
        export_nifti(
            full_dyads2_disp_reordered,
            orig_data,
            reordered_path,
            "dyads2_dispersion.nii.gz",
        )
        export_nifti(
            full_dyads3_disp_reordered,
            orig_data,
            reordered_path,
            "dyads3_dispersion.nii.gz",
        )
