"""Tests for the scalar maps written by the ball-and-3-stick exporter."""

import os
from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pytest
from omegaconf import OmegaConf

from dmri.eval.export_theta import (
    FIBER_FRACTION_THRESHOLD,
    _conditional_mean,
    _conditional_std,
    _divide_safe,
    _valid_fraction_samples,
    export_thetas_to_files_ball3stick,
    map_over_voxels,
    spherical_to_cartesian,
)
from dmri.simulators.models import Ball3StickSharedDiffusivity

NUM_VOXELS = 24
NUM_SAMPLES = 16
BRAIN_SHAPE = (3, 4, 5)


@pytest.fixture
def exported(tmp_path):
    """Run the exporter on synthetic thetas with a partially inactive ball."""
    rng = np.random.default_rng(0)
    sim_type = Ball3StickSharedDiffusivity

    brain_mask_flat = np.zeros(int(np.prod(BRAIN_SHAPE)), dtype=bool)
    brain_mask_flat[:NUM_VOXELS] = True
    thetas = rng.normal(size=(NUM_VOXELS, NUM_SAMPLES, sim_type.theta_dim)).astype(
        np.float32
    )

    model_mask = np.ones((NUM_VOXELS, NUM_SAMPLES, 5), dtype=bool)
    model_mask[..., 0] = rng.random((NUM_VOXELS, NUM_SAMPLES)) > 0.3
    model_mask[..., 1:4] = rng.random((NUM_VOXELS, NUM_SAMPLES, 3)) > 0.3
    # The prior keeps at least one compartment active.
    model_mask[np.all(~model_mask[..., :4], axis=-1), 0] = True

    cfg = OmegaConf.create({
        "export": {
            "sort_by_fractions": True,
            "reorder_dyads": True,
            "export_stds": True,
            "condition_fsum_on_ball": True,
        }
    })
    orig_data = SimpleNamespace(
        affine=np.eye(4), shape=BRAIN_SHAPE + (10,), volume_slice=None
    )
    out_path = str(tmp_path / "export")
    export_thetas_to_files_ball3stick(
        cfg,
        thetas,
        sim_type,
        model_mask,
        brain_mask_flat,
        BRAIN_SHAPE,
        out_path,
        orig_data,
    )

    def load(name, sub=""):
        path = os.path.join(out_path, sub, name)
        return np.asarray(nb.load(path).dataobj).reshape(int(np.prod(BRAIN_SHAPE)), -1)[
            :NUM_VOXELS
        ]

    return SimpleNamespace(load=load, out_path=out_path, model_mask=model_mask)


def test_fsum_is_conditioned_on_an_active_ball(exported):
    """Samples that drop the ball pin f_sum at 1 and must not inflate the map."""
    fsum = exported.load("mean_fsumsamples.nii.gz")[:, 0]
    fsum_all = exported.load("mean_fsumsamples_all.nii.gz")[:, 0]
    ball_active = exported.load("frac_ball_active.nii.gz")[:, 0]

    expected_active = exported.model_mask[..., 0].mean(axis=1)
    np.testing.assert_allclose(ball_active, expected_active, atol=1e-6)

    # Conditioning can only remove f_sum == 1 samples, so it never increases.
    assert np.all(fsum <= fsum_all + 1e-6)
    # With a ball that is off ~30% of the time the two maps must actually differ.
    assert np.any(ball_active < 1.0)
    assert not np.allclose(fsum, fsum_all)


def test_unconditioned_fsum_matches_one_minus_f0(exported):
    """The four fractions sum to 1 per sample, so f_sum is identically 1 - f0."""
    fsum_all = exported.load("mean_fsumsamples_all.nii.gz")[:, 0]
    f0 = exported.load("mean_f0samples.nii.gz")[:, 0]
    np.testing.assert_allclose(fsum_all, 1.0 - f0, atol=1e-5)


def test_fsum_std_is_std_of_the_sum_not_sum_of_stds(exported):
    """The fractions are negatively correlated, so the two differ substantially."""
    fsum_std = exported.load("std_fsumsamples.nii.gz")[:, 0]
    sum_of_stds = sum(exported.load(f"std_f{i}samples.nii.gz")[:, 0] for i in (1, 2, 3))
    assert not np.allclose(fsum_std, sum_of_stds)
    assert np.all(fsum_std <= sum_of_stds + 1e-6)


def test_num_fib_pred_varies_per_voxel(exported):
    """The map counts sticks per voxel, not voxels above a threshold."""
    num_fib = exported.load("mean_num_fib_predsamples.nii.gz")[:, 0]
    assert num_fib.shape == (NUM_VOXELS,)
    assert set(np.unique(num_fib)) <= {0.0, 1.0, 2.0, 3.0}
    assert len(np.unique(num_fib)) > 1, "expected the count to vary across voxels"

    means = [exported.load(f"mean_f{i}samples.nii.gz")[:, 0] for i in (1, 2, 3)]
    expected = sum((m > FIBER_FRACTION_THRESHOLD).astype(np.float32) for m in means)
    np.testing.assert_allclose(num_fib, expected)


def test_fraction_ratios_are_finite(exported):
    """f1 is exactly 0 where stick 1 is masked off in every sample."""
    for name in ("mean_f2_f1_ratiosamples.nii.gz", "mean_f3_f1_ratiosamples.nii.gz"):
        ratio = exported.load(name)
        assert np.all(np.isfinite(ratio))


def test_reordered_stds_are_not_copies_of_the_samples(exported):
    """The reordered std_* maps used to be byte-identical to merged_*."""
    for name in ("f0", "f1", "f2", "f3", "fsum"):
        std = exported.load(f"std_{name}samples.nii.gz", sub="reordered")[:, 0]
        merged = exported.load(f"merged_{name}samples.nii.gz", sub="reordered")[:, 0]
        assert not np.allclose(std, merged), f"std_{name} is still a copy of merged"


def test_sorted_fractions_are_descending(exported):
    """The vectorised sort must still order the sticks by fraction."""
    f1, f2, f3 = (exported.load(f"mean_f{i}samples.nii.gz")[:, 0] for i in (1, 2, 3))
    # Means of per-sample sorted fractions stay ordered.
    assert np.all(f1 >= f2 - 1e-6)
    assert np.all(f2 >= f3 - 1e-6)


def test_sort_by_fractions_matches_the_reference_loop():
    """The argsort rewrite reproduces the original per-voxel Python loop."""
    rng = np.random.default_rng(1)
    fractions = rng.random((7, 5, 4)).astype(np.float32)
    mu1, mu2, mu3 = (rng.random((7, 5, 2)).astype(np.float32) for _ in range(3))

    expected_fractions = fractions.copy()
    expected = [np.zeros_like(mu1) for _ in range(3)]
    for i in range(fractions.shape[0]):
        for j in range(fractions.shape[1]):
            idx = np.argsort(-fractions[i, j, 1:])
            expected_fractions[i, j, 1:] = fractions[i, j, 1:][idx]
            mus_sorted = np.stack([mu1[i, j], mu2[i, j], mu3[i, j]], axis=0)[idx]
            for k in range(3):
                expected[k][i, j] = mus_sorted[k]

    order = np.argsort(-fractions[..., 1:], axis=-1, kind="stable")
    actual_fractions = fractions.copy()
    actual_fractions[..., 1:] = np.take_along_axis(fractions[..., 1:], order, axis=-1)
    mus = np.take_along_axis(
        np.stack([mu1, mu2, mu3], axis=-2), order[..., None], axis=-2
    )

    np.testing.assert_array_equal(actual_fractions, expected_fractions)
    for k in range(3):
        np.testing.assert_array_equal(mus[..., k, :], expected[k])


def test_conditional_mean_and_std_fall_back_when_nothing_is_kept():
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    keep = np.array([[True, True, False], [False, False, False]])
    fallback = values.mean(axis=1)

    mean = _conditional_mean(values, keep, fallback)
    np.testing.assert_allclose(mean, [1.5, fallback[1]])

    std = _conditional_std(values, keep, np.std(values, axis=1))
    np.testing.assert_allclose(std, [0.5, np.std(values[1])])


def test_divide_safe_returns_zero_on_a_vanishing_denominator():
    out = _divide_safe(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
    np.testing.assert_allclose(out, [0.0, 0.5])
    assert np.all(np.isfinite(out))


def test_spherical_to_cartesian_returns_unit_vectors():
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, np.pi, size=(4, 3))
    phi = rng.uniform(-np.pi, np.pi, size=(4, 3))
    vec = spherical_to_cartesian(theta, phi)
    assert vec.shape == (4, 3, 3)
    np.testing.assert_allclose(np.linalg.norm(vec, axis=-1), 1.0, atol=1e-6)


def test_map_over_voxels_chunks_without_changing_results():
    values = np.arange(60, dtype=np.float32).reshape(10, 3, 2)
    masks = np.ones((10, 3, 2), dtype=bool)

    def fn(value, mask):
        return {"doubled": value * 2, "summed": (value * mask).sum()}

    whole = map_over_voxels(fn, values, masks, batch_size=100)
    chunked = map_over_voxels(fn, values, masks, batch_size=3)

    for key in ("doubled", "summed"):
        np.testing.assert_allclose(whole[key], chunked[key])
    np.testing.assert_allclose(chunked["doubled"], values * 2)


def test_fraction_validity_checks_the_whole_simplex_draw():
    fractions = np.array(
        [[[0.4, 0.6], [0.4, 0.5], [-0.1, 1.1], [np.nan, np.nan]]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        _valid_fraction_samples(fractions), [[True, False, False, False]]
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_non_finite_theta_samples_export_as_nan_inside_brain(tmp_path, invalid):
    sim_type = Ball3StickSharedDiffusivity
    thetas = np.zeros((2, 3, sim_type.theta_dim), dtype=np.float32)
    thetas[0, 1, 0] = invalid
    thetas[1, :, 0] = invalid
    model_mask = np.ones((2, 3, 5), dtype=bool)
    brain_shape = (2, 1, 1)
    cfg = OmegaConf.create({
        "export": {
            "sort_by_fractions": True,
            "reorder_dyads": True,
            "export_stds": True,
            "condition_fsum_on_ball": True,
        }
    })
    orig_data = SimpleNamespace(
        affine=np.eye(4), shape=brain_shape + (1,), volume_slice=None
    )
    out_path = tmp_path / "non_finite"

    export_thetas_to_files_ball3stick(
        cfg,
        thetas,
        sim_type,
        model_mask,
        np.ones(2, dtype=bool),
        brain_shape,
        str(out_path),
        orig_data,
    )

    raw = np.asarray(nb.load(out_path / "raw_thetas.nii.gz").dataobj)
    assert np.all(np.isnan(raw[0, 0, 0, 1]))
    assert np.all(np.isnan(raw[1]))

    for name in (
        "mean_f0samples.nii.gz",
        "mean_dsamples.nii.gz",
        "mean_snrsamples.nii.gz",
    ):
        values = np.asarray(nb.load(out_path / name).dataobj)
        assert np.isnan(values[1, 0, 0])


def test_noise_only_fractions_export_as_nan_while_outside_stays_zero(tmp_path, caplog):
    sim_type = Ball3StickSharedDiffusivity
    thetas = np.zeros((2, 3, sim_type.theta_dim), dtype=np.float32)
    model_mask = np.ones((2, 3, 5), dtype=bool)
    model_mask[0, 1, :4] = False  # Noise-only has no valid fraction simplex.
    brain_shape = (3, 1, 1)  # The final spatial voxel is outside the brain.
    cfg = OmegaConf.create({
        "export": {
            "sort_by_fractions": True,
            "reorder_dyads": True,
            "export_stds": True,
            "condition_fsum_on_ball": True,
        }
    })
    orig_data = SimpleNamespace(
        affine=np.eye(4), shape=brain_shape + (1,), volume_slice=None
    )
    out_path = tmp_path / "invalid_fractions"

    with caplog.at_level("WARNING"):
        export_thetas_to_files_ball3stick(
            cfg,
            thetas,
            sim_type,
            model_mask,
            np.array([True, True, False]),
            brain_shape,
            str(out_path),
            orig_data,
        )

    merged = [
        np.asarray(nb.load(out_path / f"merged_f{i}samples.nii.gz").dataobj)
        for i in range(4)
    ]
    assert all(np.isnan(values[0, 0, 0, 1]) for values in merged)
    assert all(np.all(np.isfinite(values[1, 0, 0])) for values in merged)
    assert all(np.all(values[2, 0, 0] == 0.0) for values in merged)

    for name in (
        "mean_f0samples.nii.gz",
        "mean_num_fib_predsamples.nii.gz",
        "mean_f2_f1_ratiosamples.nii.gz",
    ):
        values = np.asarray(nb.load(out_path / name).dataobj)
        assert np.isnan(values[0, 0, 0])
        assert np.isfinite(values[1, 0, 0])
        assert values[2, 0, 0] == 0.0
    assert "1 posterior draws across 1 in-brain voxels are noise-only" in caplog.text
