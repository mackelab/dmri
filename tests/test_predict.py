from argparse import Namespace

import pytest

from dmri.predict import _hydra_overrides, _prepare_output, _validate_input


@pytest.fixture
def input_folder(tmp_path):
    for filename in ("data.nii.gz", "nodif_brain_mask.nii.gz", "bvals", "bvecs"):
        (tmp_path / filename).touch()
    return tmp_path


def test_validate_input_accepts_standard_folder(input_folder):
    assert _validate_input(input_folder) == input_folder.resolve()


def test_validate_input_reports_missing_files(tmp_path):
    with pytest.raises(ValueError, match="data.nii.gz"):
        _validate_input(tmp_path)


def test_prepare_output_requires_overwrite(input_folder):
    output = input_folder / "dmri_output"
    output.mkdir()
    with pytest.raises(ValueError, match="--overwrite"):
        _prepare_output(input_folder, "dmri_output", False)


def test_remote_hydra_overrides(input_folder):
    args = Namespace(
        model="model-a",
        repo_id="owner/repo",
        revision=None,
        cache_dir=None,
        local_checkpoint=None,
        local_files_only=False,
        seed=3,
        mask_samples=8,
        theta_samples=7,
    )
    overrides = _hydra_overrides(args, input_folder, input_folder / "out")
    assert "pretrained_repo_id=owner/repo" in overrides
    assert "checkpoint=best" in overrides
    assert "model_selection=average" in overrides
    assert "theta_sample.num_samples=7" in overrides
    assert "~export.metrics" in overrides


def test_local_checkpoint_overrides_remote(input_folder):
    checkpoint = input_folder / "checkpoint"
    checkpoint.mkdir()
    args = Namespace(
        model="ignored",
        repo_id="owner/repo",
        revision=None,
        cache_dir=None,
        local_checkpoint=checkpoint,
        local_files_only=False,
        seed=1,
        mask_samples=2,
        theta_samples=2,
    )
    overrides = _hydra_overrides(args, input_folder, input_folder / "out")
    assert f"path_checkpoint={input_folder}" in overrides
    assert "model_name=checkpoint" in overrides
    assert not any(value.startswith("pretrained_repo_id=") for value in overrides)
