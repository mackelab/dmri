from argparse import Namespace

import numpy as np
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
    assert "evaluation.sampling.theta.num_samples=7" in overrides
    # Policy is a group selection now, not a pile of value strings.
    assert "predict/quality=balanced" in overrides
    assert "predict/model_mode=per-sample" in overrides


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


def _composed(argv, folder="/tmp/in", output="/tmp/out"):
    """Resolve a command line all the way through Hydra composition.

    Presets now live in `conf/predict/`, so the meaningful assertion is what the
    config resolves to, not which override strings predict happened to emit.
    """
    from pathlib import Path

    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from dmri.predict import _hydra_overrides

    args = _resolved(argv)
    overrides = _hydra_overrides(args, Path(folder), Path(output))
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base=None, config_module="conf"):
            return compose(
                config_name="predict",
                overrides=[o for o in overrides if not o.startswith("hydra")],
            )
    finally:
        GlobalHydra.instance().clear()


def _sampling(cfg):
    theta = cfg.evaluation.sampling.theta
    return {
        "num_steps": theta.params.num_steps,
        "theta_samples": theta.num_samples,
        "mask_samples": cfg.evaluation.sampling.mask.n_samples,
        "precision": cfg.evaluation.precision,
        "corrector": theta.corrector.name,
    }


def test_defaults_are_unchanged_without_a_quality_flag():
    """Moving the presets into config must not change a plain invocation.

    These are the numbers the Python `QUALITY_PRESETS` dict produced before the
    presets became `conf/predict/quality/*.yaml`.
    """
    from dmri.predict import DEFAULT_QUALITY

    cfg = _composed(["folder"])

    assert DEFAULT_QUALITY == "balanced"
    assert _sampling(cfg) == {
        "num_steps": 40,
        "theta_samples": 50,
        "mask_samples": 50,
        "precision": "fp32",
        "corrector": "auto",
    }
    assert cfg.evaluation.selection.name == "average"
    assert cfg.evaluation.pipeline.sample_mask is True


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        ("very-fast", (8, 10, 10, "fp16", "uncorrected")),
        ("fast", (20, 25, 25, "fp16", "uncorrected")),
        ("balanced", (40, 50, 50, "fp32", "auto")),
        ("high", (60, 100, 100, "fp32", "auto")),
    ],
)
def test_quality_presets_resolve_to_their_documented_values(quality, expected):
    """The contract each preset had before it moved into the config tree."""
    from dmri.predict import _preset_cost

    cfg = _composed(["folder", "--quality", quality])
    resolved = _sampling(cfg)

    assert tuple(resolved.values()) == expected
    # Each parameter sample is conditioned on a mask sample.
    assert resolved["mask_samples"] >= resolved["theta_samples"]
    assert (
        _preset_cost("very-fast")
        < _preset_cost("fast")
        < _preset_cost("balanced")
        < _preset_cost("high")
    )


def test_explicit_flags_win_over_the_preset():
    cfg = _composed([
        "folder",
        "--quality",
        "fast",
        "--num-steps",
        "5",
        "--precision",
        "fp32",
        "--corrector",
        "auto",
    ])
    resolved = _sampling(cfg)

    assert resolved["num_steps"] == 5, "an explicit flag must override the preset"
    assert resolved["theta_samples"] == 25, "unset values still come from the preset"
    assert resolved["precision"] == "fp32"
    assert resolved["corrector"] == "auto"


def test_set_reaches_keys_with_no_flag_and_wins_over_everything():
    """`--set` is the escape hatch; it is applied last on purpose."""
    cfg = _composed([
        "folder",
        "--quality",
        "fast",
        "--num-steps",
        "5",
        "--set",
        "evaluation.sampling.theta.params.num_steps=7",
        "--set",
        "evaluation.sampling.theta.params.t_max=60",
    ])

    assert cfg.evaluation.sampling.theta.params.num_steps == 7, "--set must win"
    assert cfg.evaluation.sampling.theta.params.t_max == 60, "no flag exists for t_max"


def test_set_rejects_a_value_without_an_equals_sign():
    from dmri.predict import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["folder", "--set", "not-an-override"])


@pytest.mark.parametrize("quality", ["very-fast", "fast"])
def test_low_cost_quality_uses_fp16_without_a_corrector(quality):
    cfg = _composed(["folder", "--quality", quality])

    assert cfg.evaluation.precision == "fp16"
    assert cfg.evaluation.sampling.theta.corrector.name == "uncorrected"


def _precision_cfg(precision):
    from omegaconf import OmegaConf

    return OmegaConf.create({"evaluation": {"precision": precision}})


def test_cpu_warns_about_slow_prediction_and_falls_back_from_fp16(monkeypatch):
    from dmri import predict

    warnings = []
    monkeypatch.setattr(predict.console, "warning", warnings.append)
    cfg = _precision_cfg("fp16")

    predict._adapt_precision_for_backend(cfg, "cpu")

    assert cfg.evaluation.precision == "fp32"
    assert any("few thousand voxels" in warning for warning in warnings)
    assert any("unsupported on CPU" in warning for warning in warnings)


def test_gpu_keeps_fp16_without_a_cpu_warning(monkeypatch):
    from dmri import predict

    warnings = []
    monkeypatch.setattr(predict.console, "warning", warnings.append)
    cfg = _precision_cfg("fp16")

    predict._adapt_precision_for_backend(cfg, "gpu")

    assert cfg.evaluation.precision == "fp16"
    assert warnings == []


def test_quality_is_selected_as_a_group_not_expanded_into_values(input_folder):
    """The point of the config tree: one override carries the whole preset."""
    from pathlib import Path

    from dmri.predict import _hydra_overrides

    args = _resolved([str(input_folder), "--quality", "very-fast"])
    args.local_checkpoint = None
    overrides = _hydra_overrides(args, input_folder, Path("/tmp/out"))

    assert "predict/quality=very-fast" in overrides
    # None of the preset's numbers are emitted; the tree supplies them.
    for key in (
        "evaluation.sampling.theta.params.num_steps",
        "evaluation.sampling.theta.num_samples",
        "evaluation.sampling.mask.n_samples",
        "evaluation.precision",
    ):
        assert not any(o.startswith(f"{key}=") for o in overrides), (
            f"{key} should come from predict/quality, not an override"
        )


def test_best_model_mode_uses_one_model_per_voxel():
    cfg = _composed(["folder", "--model-mode", "best"])

    assert cfg.evaluation.selection.name == "best"
    assert cfg.evaluation.pipeline.sample_mask is False
    assert cfg.evaluation.pipeline.select_models is True
    assert cfg.evaluation.pipeline.default_mask is None


@pytest.mark.parametrize(
    ("name", "mask"),
    [
        ("B1S", [True, True, False, False, True]),
        ("B2S", [True, True, True, False, True]),
        ("B3S", [True, True, True, True, True]),
    ],
)
def test_fixed_model_mode_resolves_the_selected_mask(name, mask):
    cfg = _composed(["folder", "--fixed-model", name.lower()])

    assert cfg.predict.fixed_model.name == name
    assert cfg.evaluation.selection.name == "none"
    assert cfg.evaluation.pipeline.sample_mask is False
    assert cfg.evaluation.pipeline.select_models is False
    assert cfg.evaluation.pipeline.default_mask == mask


def test_fixed_model_masks_match_the_best_selection_group():
    """`predict/fixed_model` and `evaluation/selection=ball3stick_best` list the
    same three models; a mismatch would mean predict and eval disagree."""
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base=None, config_module="conf"):
            best = compose(
                config_name="eval", overrides=["evaluation/selection=ball3stick_best"]
            )
    finally:
        GlobalHydra.instance().clear()

    fixed = [
        _composed(["folder", "--fixed-model", name]).evaluation.pipeline.default_mask
        for name in ("B1S", "B2S", "B3S")
    ]
    assert [list(row) for row in best.evaluation.selection.feasible_models] == [
        list(row) for row in fixed
    ]


def test_checkpoint_which_is_configurable_and_still_defaults_to_best():
    assert _composed(["folder"]).checkpoint.which == "best"
    assert (
        _composed(["folder", "--checkpoint-which", "latest"]).checkpoint.which
        == "latest"
    )


def test_prediction_export_drops_the_metrics_it_cannot_compute():
    """Prediction has no ground truth, so reconstruction_mse/ksd cannot run."""
    cfg = _composed(["folder"])

    assert cfg.evaluation.export.theta.name == "ball3stick_inference_results"
    assert "metrics" not in cfg.evaluation.export.theta


def test_model_mode_validation_is_strategy_specific():
    from dmri.predict import _validate_model_mode

    with pytest.raises(ValueError, match="--fixed-model is required"):
        _validate_model_mode(
            Namespace(
                model_mode="fixed",
                fixed_model=None,
                quality="balanced",
                mask_samples=1,
                theta_samples=2,
            )
        )

    with pytest.raises(ValueError, match="--mask-samples"):
        _validate_model_mode(
            Namespace(
                model_mode="per-sample",
                fixed_model=None,
                quality="balanced",
                mask_samples=1,
                theta_samples=2,
            )
        )

    _validate_model_mode(
        Namespace(
            model_mode="best",
            fixed_model=None,
            quality="balanced",
            mask_samples=1,
            theta_samples=2,
        )
    )


def test_validation_compares_against_the_preset_when_a_flag_is_absent():
    """--mask-samples 1 with the preset's 50 theta samples must still fail."""
    from dmri.predict import _validate_model_mode

    args = _resolved(["folder", "--mask-samples", "1"])
    with pytest.raises(ValueError, match="--mask-samples"):
        _validate_model_mode(args)


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


def test_choose_uses_arrow_key_dialog_on_a_terminal(monkeypatch):
    from dmri import predict

    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    monkeypatch.setattr(predict, "_arrow_choice", lambda *args: "gamma")

    options = [("alpha", "first"), ("beta", "second"), ("gamma", "third")]
    assert predict._choose("Model mode", options, 1) == "gamma"


def test_arrow_dialog_cancel_uses_the_default(monkeypatch):
    from dmri import predict

    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    monkeypatch.setattr(predict, "_arrow_choice", lambda *args: "balanced")

    assert predict._choose("Quality", [("fast", ""), ("balanced", "")], 1) == "balanced"


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
    assert not any(t.startswith("Model  (") for t in asked)
    assert any(t.startswith("Quality") for t in asked)
    assert any(t.startswith("Model mode") for t in asked)


def test_interactive_fixed_mode_asks_for_the_fixed_model(monkeypatch):
    from dmri import predict

    asked = []

    def record(title, options, default_index, **kwargs):
        asked.append(title)
        return options[default_index][0]

    monkeypatch.setattr(predict, "_choose", record)
    args = Namespace(
        model="m",
        quality="fast",
        model_mode="fixed",
        fixed_model=None,
        repo_id="r",
        revision=None,
    )
    predict._interactive_setup(args)

    assert args.fixed_model == "B3S"
    assert asked == ["Fixed model"]


def test_model_listing_failure_does_not_break_the_prompt(monkeypatch):
    from argparse import Namespace

    from dmri import predict

    def offline(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(predict, "list_pretrained_models", offline)
    models = predict._available_models(Namespace(repo_id="r", revision=None))
    assert models == [predict.DEFAULT_MODEL_NAME]


def test_known_models_have_family_descriptions():
    from dmri.predict import _model_description

    assert "Ball3Stick model family" in _model_description("b3s_2_4_6_128")
    assert "Multi-shell Ball3Stick" in _model_description("msb3s_2_4_6_128")
    assert _model_description("custom") == "Pretrained dMRI model"


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


def test_predict_logo_precedes_help(capsys):
    from dmri import console
    from dmri.predict import main

    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    output = capsys.readouterr().out
    assert output.startswith(console.LOGO)
    assert output.index(console.LOGO) < output.index("usage: dmri predict")


def _write_map(directory, name, array):
    import nibabel as nb

    directory.mkdir(parents=True, exist_ok=True)
    nb.save(nb.Nifti1Image(array, np.eye(4)), directory / name)


def test_write_viewer_collects_the_exported_maps(tmp_path):
    import numpy as np

    from dmri.predict import write_viewer

    results = tmp_path / "ball3stick_inference_results"
    volume = np.random.default_rng(0).random((8, 9, 6)).astype(np.float32)
    for name in ("mean_fsumsamples.nii.gz", "mean_f0samples.nii.gz"):
        _write_map(results, name, volume)
    # Not in the viewer config, so it must be ignored rather than break the viewer.
    _write_map(results, "merged_f0samples.nii.gz", volume)

    written = write_viewer(tmp_path)

    assert written is not None and written.exists()
    html = written.read_text()
    assert "mean_fsumsamples" in html and "mean_f0samples" in html
    # plotly escapes the "/" when embedding the JSON payload.
    assert "base64" in html and "data:image" in html.replace("\\u002f", "/"), (
        "slices are not embedded as images"
    )


def test_write_viewer_returns_none_without_maps(tmp_path):
    from dmri.predict import write_viewer

    assert write_viewer(tmp_path) is None


def test_write_viewer_skips_non_volume_maps(tmp_path):
    """4-D sample stacks must not be fed to the viewer."""
    import numpy as np

    from dmri.predict import write_viewer

    results = tmp_path / "ball3stick_inference_results"
    _write_map(
        results,
        "mean_fsumsamples.nii.gz",
        np.zeros((4, 4, 3, 5), dtype=np.float32),
    )
    assert write_viewer(tmp_path) is None


def test_no_viewer_flag_exists():
    from dmri.predict import _parser

    assert _parser().parse_args(["f"]).no_viewer is False
    assert _parser().parse_args(["f", "--no-viewer"]).no_viewer is True


def test_viewer_link_is_one_unbroken_uri(tmp_path, capsys):
    """The link must not be wrapped, or the terminal cannot linkify it.

    Console output is rendered through rich, which wraps to the terminal width
    by default and would split a long path across lines.
    """
    import numpy as np

    from dmri import console, predict

    results = tmp_path / "ball3stick_inference_results"
    _write_map(
        results,
        "mean_fsumsamples.nii.gz",
        np.random.default_rng(0).random((6, 6, 4)).astype(np.float32),
    )
    viewer = predict.write_viewer(tmp_path)
    assert viewer is not None

    console.configure(enabled=True)
    try:
        console.link("Viewer:    ", viewer)
    finally:
        console.configure(enabled=False)

    printed = capsys.readouterr().out
    uri = viewer.resolve().as_uri()
    assert uri in printed, f"the URI was broken up: {printed!r}"
    assert printed.count("\n") == 1, "the link spans more than one line"


def test_console_link_is_silent_when_disabled(tmp_path, capsys):
    from dmri import console

    target = tmp_path / "x.html"
    target.write_text("")
    console.configure(enabled=False)
    console.link("Viewer: ", target)
    assert capsys.readouterr().out == ""


def test_predict_does_not_launch_a_browser():
    """Opening a browser is intrusive over SSH and in CI; we only print a link."""
    import inspect

    from dmri import predict

    source = inspect.getsource(predict)
    assert "webbrowser" not in source
    assert "--no-open" not in source
