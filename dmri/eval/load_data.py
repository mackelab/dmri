"""Loading and normalisation of FSL/HCP-style diffusion data.

The brain mask is applied while reading, so only in-brain voxels are ever
materialised and everything downstream works on a flat ``(voxels, gradients)``
array.
"""

import os

import nibabel as nb
import numpy as np
from dipy.io import read_bvals_bvecs

# Gradient volumes per read. NIfTI stores the gradient axis slowest, so reading
# in order streams forwards through the (possibly gzipped) file. Measured on a
# 145x174x145x90 scan: 32 matches a whole-volume read at ~2.5x less memory.
DEFAULT_VOLUME_CHUNK = 32


def load_and_process_data(
    path,
    brain_mask="nodif_brain_mask.nii.gz",
    mri_data="data.nii.gz",
    bvals_data="bvals",
    bvecs_data="bvecs",
    round_bvals=True,
    mask_filter=None,
    clip_quantile=None,
):
    """Load a data folder and return the normalised in-brain signal.

    Args:
        mask_filter: optional callable applied to the brain mask before the
            signal is read, used to restrict processing to a sub-volume. Any
            voxel it drops is never loaded.
        clip_quantile: optional quantile in (0, 1]; signals above it are clipped.

    Returns:
        ``(img, data_in_brain, brain_mask, bvals, bvecs)`` where ``data_in_brain``
        has shape ``(num_in_brain_voxels, num_gradients)`` and ``img`` is the
        unread source image, kept for its affine and shape.
    """
    img = nb.load(os.path.join(path, mri_data))
    brain_mask = load_brain_mask(path, brain_mask)
    bvals, bvecs = read_bvals_bvecs(
        os.path.join(path, bvals_data), os.path.join(path, bvecs_data)
    )

    if mask_filter is not None:
        brain_mask = mask_filter(brain_mask)

    if brain_mask.shape != img.shape[:3]:
        raise ValueError(
            f"Brain mask shape {brain_mask.shape} does not match the volume shape "
            f"{img.shape[:3]}."
        )

    bvals = process_bvals(bvals, round_bvals)
    order = np.argsort(bvals, kind="stable")
    bvals = bvals[order]
    bvecs = bvecs[order]

    data_in_brain = read_in_brain(img, brain_mask, order)
    data_in_brain = normalize_in_brain(data_in_brain, bvals)

    np.nan_to_num(data_in_brain, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if clip_quantile is not None and data_in_brain.size:
        upper = np.quantile(data_in_brain, clip_quantile)
        np.clip(data_in_brain, 0, upper, out=data_in_brain)

    return img, data_in_brain, brain_mask, bvals, bvecs


def load_brain_mask(path, brain_mask="nodif_brain_mask.nii.gz"):
    """Load a brain mask as a boolean array."""
    mask_img = nb.load(os.path.join(path, brain_mask))
    return np.asanyarray(mask_img.dataobj) > 0


def read_in_brain(img, brain_mask, order=None, volume_chunk=DEFAULT_VOLUME_CHUNK):
    """Read only the in-brain voxels of a 4-D image as ``(voxels, gradients)``.

    ``order`` reorders the gradient axis (e.g. by b-value) without a second copy.
    """
    num_gradients = img.shape[3]
    if order is None:
        order = np.arange(num_gradients)
    mask_flat = np.asarray(brain_mask, dtype=bool).reshape(-1)
    num_voxels = int(np.count_nonzero(mask_flat))

    out = np.empty((num_voxels, num_gradients), dtype=np.float32)
    # Invert `order` so the file can still be read front to back.
    destination = np.empty(num_gradients, dtype=np.intp)
    destination[np.asarray(order, dtype=np.intp)] = np.arange(num_gradients)

    for start in range(0, num_gradients, volume_chunk):
        stop = min(start + volume_chunk, num_gradients)
        chunk = np.asanyarray(img.dataobj[..., start:stop], dtype=np.float32)
        chunk = chunk.reshape(-1, stop - start)[mask_flat]
        out[:, destination[start:stop]] = chunk

    return out


def process_bvals(bvals, round_bvals):
    # Round bvals to nearest (0, 1000, 2000, ...)
    if round_bvals:
        bvals_rounded = np.round(bvals / 1000) * 1000
        bvals = np.clip(bvals_rounded, 0, None)
    else:
        bvals = bvals
    return bvals


def sort_by_bvals(bvals, bvecs, data):
    """Sort b-values, b-vectors and the gradient axis of ``data`` together."""
    idx = np.argsort(bvals, kind="stable")
    bvals = bvals[idx]
    bvecs = bvecs[idx]
    data = data[..., idx]
    return bvals, bvecs, data


def normalize_in_brain(data, bvals, out=None):
    """Divide the in-brain signal ``(voxels, gradients)`` by its mean b0 signal.

    Voxels whose b0 signal is non-positive are set to zero rather than producing
    NaNs, and a missing b0 shell is an error instead of a silently empty volume.
    """
    b0_mask = bvals == 0
    if not np.any(b0_mask):
        raise ValueError(
            "No b=0 volumes found; the signal cannot be normalised. Check the "
            "bvals file, or enable data.round_bvals to fold low b-values into b0."
        )

    s0 = data[:, b0_mask].mean(axis=-1, keepdims=True)
    valid = s0 > 0
    out = data if out is None else out
    np.divide(data, s0, out=out, where=valid)
    out[~valid[:, 0], :] = 0.0
    return out
