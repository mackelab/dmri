import jax.numpy as jnp
import nibabel as nb
import numpy as np
from omegaconf import OmegaConf

from dmri.eval.eval_script import (
    _find_theta_samples_file,
    _has_model_selection_to_export,
    _load_theta_samples,
    _should_compute_feasible_model_probabilities,
)
from dmri.eval.export_metrics import (
    MetricSpec,
    _scatter_metric_values,
    _select_voxel_subset,
)


def test_voxel_subset_is_deterministic_and_scattered():
    spec = MetricSpec(
        key="test",
        type="ksd",
        output_filename="test.nii.gz",
        options={"voxel_subset_size": 3, "voxel_subset_seed": 4},
    )

    first = _select_voxel_subset(spec, 10)
    second = _select_voxel_subset(spec, 10)
    assert np.array_equal(first, second)
    assert first is not None

    scattered = _scatter_metric_values(
        np.arange(3, dtype=np.float32), first, num_voxels=10
    )
    assert np.array_equal(scattered[first], np.arange(3, dtype=np.float32))
    assert np.isnan(np.delete(scattered, first)).all()


def test_load_theta_samples_extracts_brain_voxels(tmp_path):
    brain_mask = np.array([True, False, True, False])
    samples = np.arange(4 * 2 * 3, dtype=np.float32).reshape(2, 2, 2, 3)
    path = tmp_path / "thetas.nii.gz"
    nb.save(nb.Nifti1Image(samples, np.eye(4)), path)

    loaded = _load_theta_samples(
        str(path),
        brain_mask,
        brain_shape=(2, 2),
        expected_num_samples=2,
        expected_theta_dim=3,
    )

    assert loaded.shape == (2, 2, 3)
    assert jnp.array_equal(loaded, samples.reshape(4, 2, 3)[brain_mask])


def test_find_theta_samples_uses_exported_raw_filename(tmp_path):
    path = tmp_path / "nested" / "raw_thetas.nii.gz"
    path.parent.mkdir()
    path.touch()

    assert _find_theta_samples_file(tmp_path) == str(path)


def test_model_selection_export_requires_generated_masks():
    assert not _has_model_selection_to_export(None, None)
    assert _has_model_selection_to_export(np.ones((1, 2)), None)
    assert _has_model_selection_to_export(None, np.ones((1, 2, 3)))


def test_feasible_probability_export_is_enabled_by_config():
    cfg = OmegaConf.create({"sample_mask": True})
    export_cfg = OmegaConf.create({
        "export_feasible_model_probabilities": True,
        "feasible_models": [[True, False]],
    })

    assert _should_compute_feasible_model_probabilities(cfg, export_cfg, "ball3stick")
    cfg.sample_mask = False
    assert not _should_compute_feasible_model_probabilities(
        cfg, export_cfg, "ball3stick"
    )
