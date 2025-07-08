import os

from dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    ssfp_acquisition_scheme,
)
import nibabel as nb
import numpy as np
from dipy.io import read_bvals_bvecs


def load_and_process_data(path, brain_mask, mri_data, data_type, **kwargs):
    if data_type == "dmri":
        bvals_data = kwargs["bvals_data"]
        bvecs_data = kwargs["bvecs_data"]
        round_bvals = kwargs["round_bvals"]
        data, data_norm, brain_mask, acq = load_data(
            path,
            brain_mask,
            mri_data,
            bvals_data,
            bvecs_data,
            round_bvals,
        )
        data_norm, _ = normalize_data(data_norm, brain_mask, acq.bvals)
    elif data_type == "ssfp":
        bvals_data = kwargs["bvals_data"]
        bvecs_data = kwargs["bvecs_data"]
        diffGradAmps_data = kwargs["diffGradAmps_data"]
        diffGradDurs_data = kwargs["diffGradDurs_data"]
        flipAngles_data = kwargs["flipAngles_data"]
        TRs_data = kwargs["TRs_data"]
        T1_map_data = kwargs["T1_map_data"]
        T2_map_data = kwargs["T2_map_data"]
        B1_map_data = kwargs["B1_map_data"]
        (
            data,
            data_norm,
            brain_mask,
            acq,
        ) = load_data_ssfp(
            path,
            brain_mask,
            mri_data,
            bvals_data,
            bvecs_data,
            diffGradAmps_data,
            diffGradDurs_data,
            flipAngles_data,
            TRs_data,
            T1_map_data,
            T2_map_data,
            B1_map_data,
        )
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    return data, data_norm, brain_mask, acq


def load_data_ssfp(
    path,
    brain_mask,
    mri_data,
    bvals_data,
    bvecs_data,
    diffGradAmps_data,
    diffGradDurs_data,
    flipAngles_data,
    TRs_data,
    T1_map_data,
    T2_map_data,
    B1_map_data,
):
    data = nb.load(os.path.join(path, mri_data))
    data_norm = data.get_fdata()
    data_norm = data_norm.astype(np.float32)

    brain_mask = nb.load(os.path.join(path, brain_mask))
    brain_mask = brain_mask.get_fdata().astype(np.bool)

    bvals, bvecs = read_bvals_bvecs(
        os.path.join(path, bvals_data), os.path.join(path, bvecs_data)
    )

    diffGradAmp = np.loadtxt(os.path.join(path, diffGradAmps_data)).astype(np.float32)
    diffGradDurs = np.loadtxt(os.path.join(path, diffGradDurs_data)).astype(np.float32)
    flipAngles = np.loadtxt(os.path.join(path, flipAngles_data)).astype(np.float32)
    T1map = nb.load(os.path.join(path, T1_map_data)).get_fdata().astype(np.float32)
    T2map = nb.load(os.path.join(path, T2_map_data)).get_fdata().astype(np.float32)
    B1map = nb.load(os.path.join(path, B1_map_data)).get_fdata().astype(np.float32)
    TRs = np.loadtxt(os.path.join(path, TRs_data)).astype(np.float32)

    acq = ssfp_acquisition_scheme(
        bvecs, T1map, T2map, B1map, diffGradAmp, flipAngles, TRs, diffGradDurs
    )

    data_norm, _ = normalize_data(data_norm, brain_mask, bvals)

    return data, data_norm, brain_mask, acq


def load_data(path, brain_mask, mri_data, bvals_data, bvecs_data, round_bvals):
    data = nb.load(os.path.join(path, mri_data))
    data_norm = data.get_fdata()
    data_norm = data_norm.astype(np.float32)

    brain_mask = nb.load(os.path.join(path, brain_mask))
    brain_mask = brain_mask.get_fdata().astype(np.bool)

    bvals, bvecs = read_bvals_bvecs(
        os.path.join(path, bvals_data), os.path.join(path, bvecs_data)
    )

    bvals = process_bvals(bvals, round_bvals)
    bvals, bvecs, data_norm = sort_by_bvals(bvals, bvecs, data_norm)

    acq = acquisition_scheme(bvals, bvecs)
    return data, data_norm, brain_mask, acq


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
