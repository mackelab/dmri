import logging
import os

import jax
import numpy as np

from dmri.simulators.local_signal_models.ball import MultiShellStaticBall
from dmri.utils.dmriutils import export_nifti, make_dyads, reorder_angles_3fib, sph2cart


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
    cfg, thetas, sim_type, model_mask, brain_mask_flat, brain_shape, out_path, orig_data
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
    logging.info(f"Exporting raw inferred parameters to {out_path}")
    # Export raw samples
    full_thetas = embed_in_full_brain_array(
        thetas, brain_mask_flat, brain_shape
    ).astype(np.float32)
    export_nifti(full_thetas, orig_data, out_path, "raw_thetas.nii.gz")

    def to_fractions(theta):
        return sim_type.from_theta(theta, model_mask=model_mask).model_fractions

    def to_diffusivities(theta):
        return (
            sim_type.from_theta(theta, model_mask=model_mask).model_compartments[0].lam
        )

    def direction_s1(theta):
        return (
            sim_type.from_theta(theta, model_mask=model_mask).model_compartments[1].mu
        )

    def direction_s2(theta):
        return (
            sim_type.from_theta(theta, model_mask=model_mask).model_compartments[2].mu
        )

    def direction_s3(theta):
        return (
            sim_type.from_theta(theta, model_mask=model_mask).model_compartments[3].mu
        )

    def snr(theta):
        return (
            sim_type.from_theta(theta, model_mask=model_mask).noise_compartments[0].snr
        )

    fractions = np.array(jax.vmap(jax.vmap(to_fractions))(thetas), dtype=np.float32)
    diffusitivity = np.array(
        jax.vmap(jax.vmap(to_diffusivities))(thetas), dtype=np.float32
    )
    mu1 = np.array(jax.vmap(jax.vmap(direction_s1))(thetas), dtype=np.float32)
    mu2 = np.array(jax.vmap(jax.vmap(direction_s2))(thetas), dtype=np.float32)
    mu3 = np.array(jax.vmap(jax.vmap(direction_s3))(thetas), dtype=np.float32)
    snr = np.array(jax.vmap(jax.vmap(snr))(thetas), dtype=np.float32)

    # To save multishell stds
    is_multi_shell = sim_type.model_types[0] is MultiShellStaticBall
    if is_multi_shell:

        def to_diffusivities_std(theta):
            return (
                sim_type
                .from_theta(theta, model_mask=model_mask)
                .model_compartments[0]
                .lam_std
            )

        diffusitivity_std = np.array(jax.vmap(jax.vmap(to_diffusivities_std))(thetas))
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

    if model_mask is not None:
        # Mask out voxels that are not in the model
        if model_mask.ndim == 2:
            model_mask = model_mask[..., None, :]
            model_mask = np.repeat(model_mask, fractions.shape[1], axis=-1)
        fractions = np.where(model_mask, fractions, 0)
    else:
        fractions = fractions

    if cfg.export.sort_by_fractions:
        logging.info("Sorting fiber fractions and corresponding directions.")
        # Keep ball fraction unchanged
        fractions_new = np.zeros_like(fractions)
        fractions_new[..., 0] = fractions[..., 0]

        # Initialize arrays for sorted parameters
        mu1_new = np.zeros_like(mu1)
        mu2_new = np.zeros_like(mu2)
        mu3_new = np.zeros_like(mu3)

        for i in range(fractions.shape[0]):
            for j in range(fractions.shape[1]):
                # Get indices that would sort stick fractions in descending order
                idx = np.argsort(-fractions[i, j, 1:])  # Negative to sort descending

                # Sort stick fractions
                fractions_new[i, j, 1:] = fractions[i, j, 1:][idx]
                # Stack and sort corresponding mu parameters
                mus = np.stack([mu1[i, j], mu2[i, j], mu3[i, j]], axis=0)
                mus_sorted = mus[idx]

                # Unpack sorted mus
                mu1_new[i, j] = mus_sorted[0]
                mu2_new[i, j] = mus_sorted[1]
                mu3_new[i, j] = mus_sorted[2]

        # Replace original arrays with sorted versions
        fractions = fractions_new
        mu1 = mu1_new
        mu2 = mu2_new
        mu3 = mu3_new

        # Check that fraction still sums to 1
        assert np.allclose(np.sum(fractions, axis=-1), 1.0), (
            "Fraction does not sum to 1"
        )

        # Check that sorting worked
        assert np.all(fractions[..., 1] >= fractions[..., 2]), (
            "f1 should be greater than f2"
        )
        assert np.all(fractions[..., 2] >= fractions[..., 3]), (
            "f2 should be greater than f3"
        )
    logging.info("Exporting moments and processed inferred parameters.")

    # Moments
    fractions_mean = np.mean(fractions, axis=1)
    diffusitivity_mean = np.mean(diffusitivity, axis=1)

    f0_mean = fractions_mean[..., 0]
    f1_mean = fractions_mean[..., 1]
    f2_mean = fractions_mean[..., 2]
    f3_mean = fractions_mean[..., 3]
    fsum_mean = f1_mean + f2_mean + f3_mean

    if cfg.export.export_stds:
        fractions_std = np.std(fractions, axis=1)
        diffusitivity_std = np.std(diffusitivity, axis=1)
        f0_std = fractions_std[..., 0]
        f1_std = fractions_std[..., 1]
        f2_std = fractions_std[..., 2]
        f3_std = fractions_std[..., 3]
        fsum_std = f1_std + f2_std + f3_std

    diffusitivity_mean = diffusitivity_mean

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

    nfib1_pred = np.sum(f1_mean > 0.05, axis=-1).astype(np.float32)
    nfib2_pred = np.sum(f2_mean > 0.05, axis=-1).astype(np.float32)
    nfib3_pred = np.sum(f3_mean > 0.05, axis=-1).astype(np.float32)
    num_fib_pred = nfib1_pred + nfib2_pred + nfib3_pred

    # Some useful auxiliary
    f2_f1_ratio = f2_mean / f1_mean
    f3_f1_ratio = f3_mean / f1_mean
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
    mu1_cart = jax.vmap(jax.vmap(sph2cart, in_axes=(0, 0)), in_axes=(0, 0))(
        mu1_theta, mu1_phi
    )
    mu2_cart = jax.vmap(jax.vmap(sph2cart, in_axes=(0, 0)), in_axes=(0, 0))(
        mu2_theta, mu2_phi
    )
    mu3_cart = jax.vmap(jax.vmap(sph2cart, in_axes=(0, 0)), in_axes=(0, 0))(
        mu3_theta, mu3_phi
    )

    mu1_cart = np.stack(mu1_cart, dtype=np.float32, axis=-1)
    mu2_cart = np.stack(mu2_cart, dtype=np.float32, axis=-1)
    mu3_cart = np.stack(mu3_cart, dtype=np.float32, axis=-1)
    print("mu1_cart shape:", mu1_cart.shape)

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
        assert np.allclose(np.sum(fractions_reordered, axis=-1), 1.0), (
            "Fraction does not sum to 1"
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

        f0_std_reordered = fractions_reordered[..., 0]
        f1_std_reordered = fractions_reordered[..., 1]
        f2_std_reordered = fractions_reordered[..., 2]
        f3_std_reordered = fractions_reordered[..., 3]
        fsum_std_reordered = f1_std_reordered + f2_std_reordered + f3_std_reordered

        if cfg.export.export_stds:
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
