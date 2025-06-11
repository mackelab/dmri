import os

import nibabel as nb
import numpy as np
from dipy.io import read_bvals_bvecs


def load_and_process_data(
    path, brain_mask, mri_data, bvals_data, bvecs_data, round_bvals
):
    data, data_norm, brain_mask, bvals, bvecs = load_data(
        path, brain_mask, mri_data, bvals_data, bvecs_data
    )
    bvals = process_bvals(bvals, round_bvals)
    bvals, bvecs, data_norm = sort_by_bvals(bvals, bvecs, data_norm)
    data_norm, _ = normalize_data(data_norm, brain_mask, bvals)
    return data, data_norm, brain_mask, bvals, bvecs


def load_data(path, brain_mask, mri_data, bvals_data, bvecs_data):
    data = nb.load(os.path.join(path, mri_data))
    data_norm = data.get_fdata()
    data_norm = data_norm.astype(np.float32)

    brain_mask = nb.load(os.path.join(path, brain_mask))
    brain_mask = brain_mask.get_fdata().astype(np.bool)

    bvals, bvecs = read_bvals_bvecs(
        os.path.join(path, bvals_data), os.path.join(path, bvecs_data)
    )

    return data, data_norm, brain_mask, bvals, bvecs


def process_bvals(bvals, round_bvals):
    # Round bvals to nearest (0, 1000, 2000, ...)
    if round_bvals:
        bvals_rounded = np.round(bvals / 1000) * 1000
        bvals = np.clip(bvals_rounded, 0, None)
    else:
        bvals = bvals
    return bvals


def sort_by_bvals(bvals, bvecs, data):
    idx = np.argsort(bvals)
    bvals = bvals[idx]
    bvecs = bvecs[idx]
    data = data[..., idx]
    return bvals, bvecs, data


def normalize_data(data_norm, brain_mask, bvals):
    # Normalized data
    b0_mask = bvals == 0
    S0 = np.mean(data_norm[..., b0_mask], axis=-1, keepdims=True)
    data_norm = data_norm / S0
    data_norm = np.where(brain_mask[..., None], data_norm, 0.0)
    data_norm = np.where(np.isnan(data_norm), 0.0, data_norm)
    S0 = np.where(brain_mask, S0[..., 0], 0.0)

    return data_norm, S0
