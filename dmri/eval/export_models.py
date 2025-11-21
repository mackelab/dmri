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


def export_model_selection_ball3stick(
    cfg,
    export_cfg,
    model_mask,
    out_path,
    orig_data,
    brain_mask_flat,
    brain_shape,
    model,
    acq,
    data,
):
    """Export model selection results for the ball-and-stick simulator."""
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    def embed(values):
        return embed_in_full_brain_array(values, brain_mask_flat, brain_shape)

    def save(image, filename):
        export_nifti(image, orig_data, out_path, filename)

    model_mask = np.asarray(model_mask)
    full_model_mask = embed(model_mask).astype(np.float32)
    save(full_model_mask, "merged_model_mask.nii.gz")

    if model_mask.ndim >= 3:
        marginal_probabilities = jnp.mean(model_mask, axis=1)
    else:
        marginal_probabilities = model_mask
    full_marginal_probabilities = embed(marginal_probabilities)
    save(full_marginal_probabilities, "mean_marginal_probabilities.nii.gz")

    feasible_models = jnp.array(export_cfg.feasible_models, dtype=jnp.bool)
    p_mask = cfg.mask_sample.p_mask

    def eval_feasible_log_probs(x):
        model_logpmf = jax.vmap(
            model.log_prob_mask, in_axes=(0, None, None, None, None)
        )(feasible_models, acq.bvals, acq.bvecs, x, jnp.array([p_mask]))
        return model_logpmf

    batch_size = 10_000
    probabilities = []
    for i in range(0, data.shape[0], batch_size):
        batch_data = data[i : i + batch_size]
        batch_logpmf = jax.vmap(eval_feasible_log_probs, in_axes=(0,))(batch_data)
        batch_probs = jax.nn.softmax(batch_logpmf, axis=-1)
        probabilities.append(batch_probs)
    probabilities = np.concatenate(probabilities, axis=0)

    for i in range(len(feasible_models)):
        p_model_i = probabilities[..., i]
        full_p_model_i = embed(p_model_i)
        save(full_p_model_i, f"p_feasible_model_{i}.nii.gz")


def export_model_selection_raw(
    cfg,
    export_cfg,
    model_mask,
    out_path,
    orig_data,
    brain_mask_flat,
    brain_shape,
    model,
    acq,
    data,
):
    """Export raw model-selection samples without model-specific processing."""
    del cfg, export_cfg, model, acq, data  # Unused for raw export

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.savez_compressed(
        os.path.join(out_path, "model_selection_samples.npz"), model_mask=model_mask
    )

    full_model_mask = embed_in_full_brain_array(
        model_mask, brain_mask_flat, brain_shape
    ).astype(np.float32)
    export_nifti(full_model_mask, orig_data, out_path, "raw_model_selection.nii.gz")


MODEL_SELECTION_EXPORTERS = {
    "ball3stick": export_model_selection_ball3stick,
    "raw": export_model_selection_raw,
}


def get_model_selection_exporter(export_type):
    exporter = MODEL_SELECTION_EXPORTERS.get(export_type)
    if exporter is None:
        raise ValueError(f"Unknown model selection export type '{export_type}'.")
    return exporter
