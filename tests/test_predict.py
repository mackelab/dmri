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
    assert "checkpoint.pretrained.repo_id=owner/repo" in overrides
    assert "checkpoint.which=best" in overrides
    assert "evaluation/selection=average" in overrides
    assert "evaluation.sampling.theta.num_samples=7" in overrides
    assert "~evaluation.export.theta.metrics" in overrides


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
    assert f"checkpoint.path={input_folder}" in overrides
    assert "checkpoint.model_name=checkpoint" in overrides
    assert not any(
        value.startswith("checkpoint.pretrained.repo_id=") for value in overrides
    )


def _resolved(argv):
    from dmri.predict import _apply_defaults, _parser

    args = _parser().parse_args(argv)
    _apply_defaults(args)
    return args


def test_defaults_are_unchanged_without_a_quality_flag():
    """Adding presets must not change what a plain invocation does."""
    from dmri.predict import DEFAULT_QUALITY, QUALITY_PRESETS

    args = _resolved(["folder"])
    assert args.quality == DEFAULT_QUALITY
    assert args.num_steps == 40
    assert args.theta_samples == 50
    assert args.mask_samples == 50
    assert QUALITY_PRESETS[DEFAULT_QUALITY] == {"num_steps": 40, "samples": 50}


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_quality_presets_are_ordered_and_consistent(quality):
    from dmri.predict import QUALITY_PRESETS, _preset_cost

    args = _resolved(["folder", "--quality", quality])
    preset = QUALITY_PRESETS[quality]
    assert args.num_steps == preset["num_steps"]
    assert args.theta_samples == preset["samples"]
    # Each parameter sample is conditioned on a mask sample.
    assert args.mask_samples >= args.theta_samples
    assert _preset_cost("fast") < _preset_cost("balanced") < _preset_cost("high")


def test_explicit_flags_win_over_the_preset():
    args = _resolved(["folder", "--quality", "fast", "--num-steps", "5"])
    assert args.num_steps == 5, "an explicit flag must override the preset"
    assert args.theta_samples == 25, "unset values still come from the preset"


def test_preset_reaches_the_hydra_overrides(input_folder):
    from pathlib import Path

    from dmri.predict import _hydra_overrides

    args = _resolved([str(input_folder), "--quality", "fast"])
    args.local_checkpoint = None
    overrides = _hydra_overrides(args, input_folder, Path("/tmp/out"))

    assert "evaluation.sampling.theta.params.num_steps=20" in overrides
    assert "evaluation.sampling.theta.num_samples=25" in overrides
    assert "evaluation.sampling.mask.n_samples=25" in overrides


def test_prompting_is_skipped_when_nothing_can_answer(monkeypatch):
    """Scripts and CI must never block on a prompt."""
    from argparse import Namespace

    from dmri.predict import _can_prompt

    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert not _can_prompt(Namespace(non_interactive=False))
    assert not _can_prompt(Namespace(non_interactive=True))


def test_choose_accepts_a_number_a_name_or_the_default(monkeypatch):
    from dmri.predict import _choose

    options = [("alpha", ""), ("beta", ""), ("gamma", "")]
    answers = iter(["", "3", "beta", "nope", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    assert _choose("t", options, 1) == "beta", "empty answer takes the default"
    assert _choose("t", options, 0) == "gamma", "a number selects by position"
    assert _choose("t", options, 0) == "beta", "a name selects by value"
    # An invalid answer re-prompts rather than failing.
    assert _choose("t", options, 2) == "alpha"


def test_choose_falls_back_to_the_default_on_eof(monkeypatch):
    from dmri.predict import _choose

    def refuse(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", refuse)
    assert _choose("t", [("a", ""), ("b", "")], 1) == "b"


def test_interactive_setup_only_fills_what_is_missing(monkeypatch):
    from argparse import Namespace

    from dmri import predict

    monkeypatch.setattr(predict, "list_pretrained_models", lambda *a, **k: ["m1", "m2"])
    asked = []

    def record(title, options, default_index, **kwargs):
        asked.append(title)
        return options[default_index][0]

    monkeypatch.setattr(predict, "_choose", record)

    args = Namespace(model="m2", quality=None, repo_id="r", revision=None)
    predict._interactive_setup(args)
    assert args.model == "m2", "an explicitly chosen model is not re-asked"
    assert [t for t in asked if t.startswith("Model")] == []
    assert any(t.startswith("Quality") for t in asked)


def test_model_listing_failure_does_not_break_the_prompt(monkeypatch):
    from argparse import Namespace

    from dmri import predict

    def offline(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(predict, "list_pretrained_models", offline)
    models = predict._available_models(Namespace(repo_id="r", revision=None))
    assert models == [predict.DEFAULT_MODEL_NAME]


def test_console_is_quiet_until_a_cli_turns_it_on(capsys):
    """Library callers must not have output printed at them."""
    from dmri.eval import console

    console.set_enabled(False)
    try:
        console.say("should not appear")
        assert capsys.readouterr().out == ""

        console.set_enabled(True)
        console.say("hello %s", "world")
        assert capsys.readouterr().out == "hello world\n"
    finally:
        console.set_enabled(False)


def test_verbose_keeps_the_full_log():
    """--verbose must not disable Hydra's job logging."""
    from dmri.predict import _parser

    assert _parser().parse_args(["f"]).verbose is False
    assert _parser().parse_args(["f", "--verbose"]).verbose is True
