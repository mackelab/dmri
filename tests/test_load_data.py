"""Tests for the diffusion data loading and normalisation path."""

import nibabel as nb
import numpy as np
import pytest

from dmri.eval.load_data import (
    load_and_process_data,
    normalize_in_brain,
    process_bvals,
    read_in_brain,
    sort_by_bvals,
)

SHAPE = (5, 6, 7)
NUM_GRADIENTS = 12


@pytest.fixture
def dataset(tmp_path):
    """Write a small FSL-style folder with two b0s and two shells."""
    rng = np.random.default_rng(0)
    bvals = np.array([2000, 0, 1000, 0] * 3, dtype=float)
    bvecs = rng.normal(size=(NUM_GRADIENTS, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)

    data = rng.uniform(100, 1000, size=SHAPE + (NUM_GRADIENTS,)).astype(np.float32)
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[1:4, 2:5, 1:6] = 1

    nb.save(nb.Nifti1Image(data, np.eye(4)), tmp_path / "data.nii.gz")
    nb.save(nb.Nifti1Image(mask, np.eye(4)), tmp_path / "nodif_brain_mask.nii.gz")
    np.savetxt(tmp_path / "bvals", bvals[None, :], fmt="%g")
    np.savetxt(tmp_path / "bvecs", bvecs.T, fmt="%.8f")

    return tmp_path, data, mask.astype(bool), bvals, bvecs


def test_load_and_process_returns_normalised_in_brain_signal(dataset):
    path, data, mask, bvals, _ = dataset
    img, signal, brain_mask, out_bvals, out_bvecs = load_and_process_data(path)

    assert signal.shape == (int(mask.sum()), NUM_GRADIENTS)
    assert signal.dtype == np.float32
    np.testing.assert_array_equal(brain_mask, mask)
    # b-values come back sorted, and the gradient axis follows them.
    assert np.all(np.diff(out_bvals) >= 0)
    assert out_bvecs.shape == (NUM_GRADIENTS, 3)

    # Compare against the straightforward whole-volume computation.
    order = np.argsort(process_bvals(bvals, True), kind="stable")
    expected = data.reshape(-1, NUM_GRADIENTS)[mask.reshape(-1)][:, order]
    expected = expected / expected[:, out_bvals == 0].mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(signal, expected, rtol=1e-5)


def test_mask_filter_restricts_what_is_loaded(dataset):
    path, _, mask, _, _ = dataset

    def keep_one_slice(brain_mask):
        restricted = np.zeros_like(brain_mask)
        restricted[2] = brain_mask[2]
        return restricted

    _, signal, brain_mask, _, _ = load_and_process_data(
        path, mask_filter=keep_one_slice
    )
    assert signal.shape[0] == int(mask[2].sum())
    assert brain_mask.sum() == mask[2].sum()


def test_clip_quantile_bounds_the_signal(dataset):
    path, _, _, _, _ = dataset
    _, unclipped, _, _, _ = load_and_process_data(path)
    _, clipped, _, _, _ = load_and_process_data(path, clip_quantile=0.5)

    assert clipped.max() < unclipped.max()
    assert clipped.min() >= 0.0


def test_read_in_brain_matches_a_whole_volume_read(dataset):
    path, data, mask, _, _ = dataset
    img = nb.load(path / "data.nii.gz")
    expected = data.reshape(-1, NUM_GRADIENTS)[mask.reshape(-1)]

    for chunk in (1, 5, NUM_GRADIENTS, NUM_GRADIENTS + 4):
        actual = read_in_brain(img, mask, volume_chunk=chunk)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_read_in_brain_applies_the_gradient_order(dataset):
    path, data, mask, bvals, _ = dataset
    img = nb.load(path / "data.nii.gz")
    order = np.argsort(bvals, kind="stable")

    actual = read_in_brain(img, mask, order=order, volume_chunk=5)
    expected = data.reshape(-1, NUM_GRADIENTS)[mask.reshape(-1)][:, order]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_missing_b0_shell_raises_instead_of_zeroing_everything():
    """Previously an empty b0 axis produced NaNs and a silently blank volume."""
    signal = np.ones((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="No b=0 volumes"):
        normalize_in_brain(signal, np.array([1000.0, 2000.0, 3000.0]))


def test_non_positive_s0_yields_zero_not_nan():
    signal = np.array([[0.0, 4.0], [2.0, 6.0]], dtype=np.float32)
    bvals = np.array([0.0, 1000.0])
    out = normalize_in_brain(signal.copy(), bvals)

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[0], [0.0, 0.0])
    np.testing.assert_allclose(out[1], [1.0, 3.0])


def test_sort_by_bvals_keeps_arrays_aligned():
    bvals = np.array([2000.0, 0.0, 1000.0])
    bvecs = np.eye(3)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)

    out_bvals, out_bvecs, out_data = sort_by_bvals(bvals, bvecs, data)
    np.testing.assert_allclose(out_bvals, [0.0, 1000.0, 2000.0])
    np.testing.assert_array_equal(out_bvecs, bvecs[[1, 2, 0]])
    np.testing.assert_array_equal(out_data, data[:, [1, 2, 0]])


def test_process_bvals_rounds_to_shells():
    bvals = np.array([-5.0, 4.0, 995.0, 1490.0, 2010.0])
    np.testing.assert_allclose(
        process_bvals(bvals, True), [0.0, 0.0, 1000.0, 1000.0, 2000.0]
    )
    np.testing.assert_allclose(process_bvals(bvals, False), bvals)


def test_mask_shape_mismatch_is_reported(dataset, tmp_path):
    path, _, _, _, _ = dataset
    nb.save(
        nb.Nifti1Image(np.ones((2, 2, 2), np.uint8), np.eye(4)),
        path / "wrong_mask.nii.gz",
    )
    with pytest.raises(ValueError, match="does not match the volume shape"):
        load_and_process_data(path, brain_mask="wrong_mask.nii.gz")
