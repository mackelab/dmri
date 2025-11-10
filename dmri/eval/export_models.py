import os

import jax
import jax.numpy as jnp
import numpy as np

from dmri.utils.dmriutils import export_nifti


def embed_in_full_brain_array(to_embed, brain_mask_flat, brain_shape):
    """Embed a tensor in the full brain array"""
    event_shape = to_embed.shape[1:]
    full_brain = np.zeros(
        (brain_mask_flat.shape[0],) + event_shape, dtype=to_embed.dtype
    )
    full_brain[brain_mask_flat, ...] = to_embed
    full_brain = full_brain.reshape(brain_shape + event_shape)
    return full_brain


def export_model_selection_to_files(
    cfg,
    model_mask,
    out_path,
    orig_data,
    brain_mask_flat,
    brain_shape,
    model,
    acq,
    data,
):
    """Export model selection results to files"""
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    def embed(values):
        return embed_in_full_brain_array(values, brain_mask_flat, brain_shape)

    def save(image, filename):
        export_nifti(
            image,
            orig_data,
            out_path,
            filename,
        )

    # Export the samples
    full_model_mask = embed(
        model_mask
    ).astype(np.float32)
    save(full_model_mask, "merged_model_mask.nii.gz")

    marginal_probabilities = jnp.mean(model_mask, axis=1)
    full_marginal_probabilities = embed(marginal_probabilities)
    save(full_marginal_probabilities, "mean_marginal_probabilities.nii.gz")

    # Get feasible models from config
    feasible_models = jnp.array(
        cfg.export_model_selection.feasible_models, dtype=jnp.bool
    )
    p_mask = cfg.mask_sample.p_mask

    # Calculate log probabilities for each feasible model
    def eval_feasible_log_probs(x):
        model_logpmf = jax.vmap(
            model.log_prob_mask, in_axes=(0, None, None, None, None)
        )(feasible_models, acq.bvals, acq.bvecs, x, jnp.array([p_mask]))
        return model_logpmf

    # Process in batches to avoid memory issues
    batch_size = 10_000
    probabilities = []
    for i in range(0, data.shape[0], batch_size):
        batch_data = data[i : i + batch_size]
        batch_logpmf = jax.vmap(eval_feasible_log_probs, in_axes=(0,))(batch_data)
        # Convert log probabilities to probabilities
        batch_probs = jax.nn.softmax(batch_logpmf, axis=-1)
        probabilities.append(batch_probs)
    probabilities = np.concatenate(probabilities, axis=0)

    # Export probabilities for each feasible model
    for i in range(len(feasible_models)):
        p_model_i = probabilities[..., i]
        full_p_model_i = embed(p_model_i)
        save(full_p_model_i, f"p_feasible_model_{i}.nii.gz")
