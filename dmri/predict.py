import argparse
import functools
import os
import shutil
import sys
from pathlib import Path

from dmri import console
from dmri.eval.precision import PRECISION_CHOICES, resolve_precision_for_backend
from dmri.hub import DEFAULT_MODEL_NAME, DEFAULT_REPO_ID, list_pretrained_models

STANDARD_FILES = ("data.nii.gz", "nodif_brain_mask.nii.gz", "bvals", "bvecs")

#: Root config for prediction. It inherits `eval.yaml` and layers the presets and
#: policy that used to be Python constants in this module.
CONFIG_NAME = "predict"
CONFIG_MODULE = "conf"

DEFAULT_QUALITY = "balanced"
DEFAULT_MODEL_MODE = "per-sample"


@functools.lru_cache(maxsize=1)
def _config_root() -> Path:
    """Directory of the packaged Hydra config tree."""
    import conf

    return Path(conf.__file__).parent


def _group_options(group: str) -> tuple[str, ...]:
    """Names of the options in a config group, sorted by file name."""
    return tuple(sorted(path.stem for path in (_config_root() / group).glob("*.yaml")))


def _compose(overrides=None):
    """Compose the prediction config without running anything.

    Used for `--help` text and the interactive picker, which need the preset
    numbers before Hydra takes over. The tree is the single source of truth, so
    these are read from it rather than duplicated here.
    """
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base=None, config_module=CONFIG_MODULE):
            return compose(config_name=CONFIG_NAME, overrides=list(overrides or []))
    finally:
        GlobalHydra.instance().clear()


@functools.cache
def _quality_preset(name: str) -> dict:
    """The resolved numbers behind one `predict/quality` option."""
    cfg = _compose([f"predict/quality={name}"])
    theta = cfg.evaluation.sampling.theta
    return {
        "num_steps": theta.params.num_steps,
        "samples": theta.num_samples,
        "mask_samples": cfg.evaluation.sampling.mask.n_samples,
        "precision": cfg.evaluation.precision,
        "corrector": theta.corrector.name,
    }


def quality_choices() -> tuple[str, ...]:
    """Quality presets, cheapest first."""
    return tuple(sorted(_group_options("predict/quality"), key=_preset_cost))


def model_mode_choices() -> tuple[str, ...]:
    options = _group_options("predict/model_mode")
    # Keep the documented order rather than the alphabetical one.
    preferred = ("per-sample", "best", "fixed")
    return tuple(name for name in preferred if name in options) + tuple(
        name for name in options if name not in preferred
    )


def fixed_model_choices() -> tuple[str, ...]:
    return _group_options("predict/fixed_model")


MODEL_DESCRIPTIONS = {
    "b3s_2_4_6_64": "Ball3Stick model family - compact single-shell model",
    "b3s_2_4_6_128": "Ball3Stick model family - single-shell dMRI",
    "msb3s_2_4_6_128": "Multi-shell Ball3Stick model family - multi-shell dMRI",
}


def _preset_cost(name):
    preset = _quality_preset(name)
    return preset["num_steps"] * preset["samples"]


def _quality_description(name):
    preset = _quality_preset(name)
    correction = "no corrector" if preset["corrector"] == "uncorrected" else "corrected"
    summary = (
        f"{preset['num_steps']} steps x {preset['samples']} samples, "
        f"{preset['precision']}, {correction}"
    )
    if name == DEFAULT_QUALITY:
        return f"{summary}  (default)"
    ratio = _preset_cost(name) / _preset_cost(DEFAULT_QUALITY)
    comparison = (
        f"~{1 / ratio:.0f}x less network work"
        if ratio < 1
        else f"~{ratio:.0f}x more network work"
    )
    return f"{summary}  ({comparison} than {DEFAULT_QUALITY})"


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _unit_float(value):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return value


def _fixed_model(value):
    value = value.upper()
    choices = fixed_model_choices()
    if value not in choices:
        raise argparse.ArgumentTypeError(
            f"choose one of {', '.join(choices)} for this model family"
        )
    return value


def _override(value):
    """A raw `key=value` Hydra override from `--set`."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected key=value, got {value!r} (e.g. evaluation.batch_size=4096)"
        )
    return value


def _parser():
    parser = argparse.ArgumentParser(
        prog="dmri predict",
        description="Run a pretrained DMRI model on an FSL/HCP-style data folder.",
    )
    parser.add_argument("folder", nargs="?", type=Path, help="Input data folder")
    parser.add_argument(
        "--model", help=f"Pretrained model name; defaults to {DEFAULT_MODEL_NAME}"
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo"
    )
    parser.add_argument("--revision", help="Hugging Face branch, tag, or commit")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory")
    parser.add_argument(
        "--local-checkpoint", type=Path, help="Use a local model bundle"
    )
    parser.add_argument(
        "--local-files-only", action="store_true", help="Disable downloads"
    )
    parser.add_argument(
        "--output-subdir", default="dmri_output", help="Output folder name"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output folder"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available pretrained models and exit",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--quality",
        choices=quality_choices(),
        help=(
            "Sampling preset: "
            + "; ".join(
                f"{name} ({_quality_description(name)})" for name in quality_choices()
            )
            + f". Defaults to {DEFAULT_QUALITY}."
        ),
    )
    parser.add_argument(
        "--mask-samples",
        type=_positive_int,
        help=(
            "Number of posterior model-mask samples in per-sample mode. "
            "Overrides --quality."
        ),
    )
    parser.add_argument(
        "--theta-samples",
        type=_positive_int,
        help="Number of parameter samples. Overrides --quality.",
    )
    parser.add_argument(
        "--model-mode",
        choices=model_mode_choices(),
        help=(
            "Model conditioning: posterior model per voxel/sample, best model "
            "per voxel, or one fixed model. Defaults to per-sample."
        ),
    )
    parser.add_argument(
        "--fixed-model",
        type=_fixed_model,
        metavar="{" + ",".join(fixed_model_choices()) + "}",
        help="Fixed Ball-and-Stick model; implies --model-mode=fixed.",
    )
    parser.add_argument(
        "--checkpoint-which",
        help=(
            "Which checkpoint to load: latest, best, or a step number. The "
            "pretrained bundles on the Hub currently ship only 'best'."
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=_override,
        help=(
            "Set any config key directly, e.g. "
            "--set evaluation.sampling.theta.params.t_max=60. Repeatable, and "
            "applied last so it wins over --quality and the other flags."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Add detailed evaluation logs while keeping the progress display.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip writing view_results.html alongside the maps.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; use defaults for anything not given on the command line.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        help="Voxels per batch. Defaults to a value chosen from GPU memory.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=_unit_float,
        help="Fraction of device memory JAX may preallocate (default 0.75).",
    )
    parser.add_argument(
        "--num-steps",
        type=_positive_int,
        help=(
            "ODE steps per posterior sample (default 40). Cost is linear in this; "
            "fewer steps is faster and less accurate."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        help=(
            "Numeric precision for the network forward pass. Half precision is "
            "faster and uses less memory. Defaults to fp16 for fast quality and "
            "fp32 otherwise; parameters and the sampler stay fp32."
        ),
    )
    parser.add_argument(
        "--corrector",
        choices=("auto", "none"),
        help="Theta correction; defaults to none for fast quality and auto otherwise.",
    )
    return parser


def _can_prompt(args) -> bool:
    """Only ask when someone is there to answer."""
    return (
        not args.non_interactive
        and sys.stdin is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def _arrow_choice(title, options, default_index):
    """Run a compact inline arrow-key selector."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    selected = [default_index]

    def content():
        rows = [("class:title", f"\n{title}\n")]
        for index, (value, description) in enumerate(options):
            active = index == selected[0]
            style = "class:selected" if active else ""
            marker = ">" if active else " "
            default = "  default" if index == default_index else ""
            rows.append((
                style,
                f" {marker} {index + 1}  {value:<14} {description}{default}\n",
            ))
        rows.append(("class:hint", "   Up/Down move  Enter select  1-9 jump\n"))
        return FormattedText(rows)

    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("k")
    def move_up(event):
        selected[0] = (selected[0] - 1) % len(options)

    @bindings.add("down")
    @bindings.add("j")
    def move_down(event):
        selected[0] = (selected[0] + 1) % len(options)

    @bindings.add("enter")
    def accept(event):
        event.app.exit(result=options[selected[0]][0])

    @bindings.add("c-c")
    @bindings.add("c-d")
    def use_default(event):
        event.app.exit(result=options[default_index][0])

    for index in range(min(len(options), 9)):
        key = str(index + 1)

        def choose_number(event, choice=index):
            event.app.exit(result=options[choice][0])

        bindings.add(key)(choose_number)

    application = Application(
        layout=Layout(
            Window(
                FormattedTextControl(content, focusable=True),
                always_hide_cursor=True,
            )
        ),
        key_bindings=bindings,
        style=Style.from_dict({
            "title": "bold",
            "selected": "bold cyan",
            "hint": "ansibrightblack",
        }),
        full_screen=False,
    )
    return application.run()


def _choose(title, options, default_index, stream=None):
    """Ask the user to pick one of ``options``; returns the chosen value.

    ``options`` is a sequence of ``(value, description)``. An empty answer takes
    the default, so pressing enter through the prompts is always valid.
    """
    stream = stream or sys.stdout
    if stream is sys.stdout and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _arrow_choice(title, options, default_index)
        except (EOFError, ImportError, OSError):
            # Unsupported terminals retain the number/name prompt below.
            pass

    print(f"\n{title}", file=stream)
    for index, (value, description) in enumerate(options, start=1):
        marker = " <-" if index - 1 == default_index else ""
        print(f"  {index}) {value:<16} {description}{marker}", file=stream)

    while True:
        try:
            answer = input(f"Choose [{default_index + 1}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=stream)
            return options[default_index][0]
        if not answer:
            return options[default_index][0]
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        for value, _ in options:
            if answer == value:
                return value
        print(f"  Please enter 1-{len(options)}, or a name.", file=stream)


def _available_models(args):
    """Return model names from the configured repo, falling back to the default.

    Network or filesystem errors are swallowed so the CLI can still offer the
    built-in default when offline.
    """
    try:
        models = list_pretrained_models(args.repo_id, revision=args.revision)
    except (ConnectionError, OSError):
        return [DEFAULT_MODEL_NAME]
    return models or [DEFAULT_MODEL_NAME]


def _model_description(name):
    return MODEL_DESCRIPTIONS.get(name, "Pretrained dMRI model")


def _interactive_setup(args) -> None:
    """Fill in whatever the user did not pass, by asking."""
    if args.model is None:
        models = _available_models(args)
        default_index = (
            models.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in models else 0
        )
        args.model = _choose(
            f"Model  (from {args.repo_id})",
            [(name, _model_description(name)) for name in models],
            default_index,
        )

    if args.quality is None:
        choices = quality_choices()
        args.quality = _choose(
            "Quality",
            [(name, _quality_description(name)) for name in choices],
            choices.index(DEFAULT_QUALITY),
        )

    if getattr(args, "model_mode", None) is None:
        args.model_mode = (
            "fixed"
            if getattr(args, "fixed_model", None) is not None
            else _choose(
                "Model mode",
                [
                    (
                        "per-sample",
                        "full uncertainty - model varies across posterior draws",
                    ),
                    ("best", "stable maps - highest-probability model per voxel"),
                    ("fixed", "controlled fit - one model throughout the brain"),
                ],
                0,
            )
        )

    if args.model_mode == "fixed" and getattr(args, "fixed_model", None) is None:
        args.fixed_model = _choose(
            "Fixed model",
            [
                ("B1S", "ball + 1 stick - simplest anisotropic model"),
                ("B2S", "ball + 2 sticks - two-way crossings"),
                ("B3S", "ball + 3 sticks - complex crossings"),
            ],
            2,
        )


def _apply_defaults(args) -> None:
    """Fill in the choices Hydra cannot make for us.

    The quality preset itself is resolved by Hydra from `conf/predict/quality`,
    so flags left unset stay None here and simply produce no override. What
    remains are the two couplings that no static config can express.
    """
    args.model = args.model or DEFAULT_MODEL_NAME
    args.quality = args.quality or DEFAULT_QUALITY
    if (
        getattr(args, "fixed_model", None) is not None
        and getattr(args, "model_mode", None) is None
    ):
        args.model_mode = "fixed"
    args.model_mode = getattr(args, "model_mode", None) or DEFAULT_MODEL_MODE

    # Mask samples follow theta samples unless asked for separately: drawing
    # fewer model masks than parameter samples would leave samples without one.
    if args.mask_samples is None and args.theta_samples is not None:
        preset = _quality_preset(args.quality)
        args.mask_samples = max(preset["mask_samples"], args.theta_samples)


def _adapt_precision_for_backend(cfg, backend=None) -> None:
    """Apply precision fallbacks and warn about impractical CPU runtimes.

    Operates on the composed config rather than the parsed arguments, because
    the precision usually comes from the quality preset and so is only known
    once Hydra has resolved it.
    """
    if backend is None:
        import jax

        backend = jax.default_backend()
    backend = str(backend).lower()

    requested_precision = cfg.evaluation.precision
    cfg.evaluation.precision = resolve_precision_for_backend(
        requested_precision, backend
    )
    if cfg.evaluation.precision != requested_precision:
        console.warning(
            f"{requested_precision} is unsupported on CPU; "
            f"using {cfg.evaluation.precision}."
        )
    if backend == "cpu":
        console.warning(
            "CPU prediction will be very slow, even for a few thousand voxels. "
            "Use a supported GPU accelerator when possible."
        )


def _effective_samples(args) -> tuple[int, int]:
    """The mask and theta sample counts this run will use.

    Flags left unset fall back to the quality preset, so validation sees the
    same numbers Hydra will resolve.
    """
    preset = _quality_preset(args.quality)
    mask = (
        args.mask_samples if args.mask_samples is not None else preset["mask_samples"]
    )
    theta = args.theta_samples if args.theta_samples is not None else preset["samples"]
    return mask, theta


def _validate_model_mode(args) -> None:
    if args.model_mode == "fixed" and args.fixed_model is None:
        raise ValueError("--fixed-model is required when --model-mode=fixed")
    if args.model_mode != "fixed" and args.fixed_model is not None:
        raise ValueError("--fixed-model can only be used with --model-mode=fixed")
    mask_samples, theta_samples = _effective_samples(args)
    if args.model_mode == "per-sample" and mask_samples < theta_samples:
        raise ValueError(
            "--mask-samples must be greater than or equal to --theta-samples "
            "when --model-mode=per-sample"
        )


def write_viewer(output_dir: Path, viewer_cfg=None) -> Path | None:
    """Write a self-contained HTML viewer for the exported maps.

    Args:
        output_dir: the prediction output folder, holding `*inference_results`.
        viewer_cfg: the `predict.viewer` config node -- which maps to show, how
            to render the dyads, and the file name. Defaults to the packaged
            `predict/viewer=default` group when omitted.

    Returns the path written, or None when there is nothing to show. Failing to
    build a viewer must never fail a prediction that already succeeded, so all
    errors here are swallowed after a warning.
    """
    import nibabel as nb
    import numpy as np

    from dmri.utils.viz import direction_colour, save_viewer, slice_viewer

    if viewer_cfg is None:
        viewer_cfg = _compose().predict.viewer

    def load(path: Path):
        if not path.is_file():
            return None
        array = np.asanyarray(nb.load(path).dataobj, dtype=np.float32)
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        return array

    volumes = {}
    for results_dir in sorted(output_dir.glob("*inference_results")):
        for name in viewer_cfg.maps:
            array = load(results_dir / name)
            if array is not None and array.ndim == 3:
                volumes[str(name).replace(".nii.gz", "")] = array
        for entry in viewer_cfg.dyads:
            dyads = load(results_dir / entry.file)
            if dyads is None or dyads.ndim != 4 or dyads.shape[-1] != 3:
                continue
            weight = load(results_dir / entry.weight) if entry.weight else None
            volumes[str(entry.label)] = direction_colour(dyads, weight)
    if not volumes:
        return None

    title = f"dmri predict - {output_dir.name}"
    figure = slice_viewer(volumes, title=title)
    return save_viewer(figure, output_dir / viewer_cfg.filename, title=title)


def _validate_input(folder: Path) -> Path:
    """Ensure the input folder exists and contains the required FSL/HCP files.

    Returns the resolved folder path.
    """
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")
    missing = [name for name in STANDARD_FILES if not (folder / name).is_file()]
    if missing:
        raise ValueError(
            f"Input folder is missing required files: {', '.join(missing)}"
        )
    return folder


def _prepare_output(folder: Path, name: str, overwrite: bool) -> Path:
    """Create or replace the output subdirectory inside ``folder``.

    Raises ValueError when the name is invalid or the folder already exists
    and ``overwrite`` is False.
    """
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("--output-subdir must be a single directory name")
    output_dir = folder / name
    if output_dir.exists():
        if not overwrite:
            raise ValueError(
                f"Output folder already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    return output_dir


def _hydra_overrides(args, folder: Path, output_dir: Path) -> list[str]:
    """Translate parsed arguments into Hydra overrides for `conf/predict.yaml`.

    Three kinds, in this order: per-run values that cannot live in config, the
    group selections behind `--quality`/`--model-mode`, and finally the flags
    and `--set` values that must beat whatever the preset chose.
    """
    model_mode = getattr(args, "model_mode", None) or DEFAULT_MODEL_MODE
    overrides = [
        f"evaluation.input.path={folder}",
        f"run.output_dir={output_dir}",
        f"checkpoint.model_name={args.model}",
        f"run.seed={args.seed}",
        f"hydra.run.dir={output_dir / '.hydra'}",
        # Policy lives in the tree; these just pick which option applies.
        f"predict/quality={getattr(args, 'quality', None) or DEFAULT_QUALITY}",
        f"predict/model_mode={model_mode}",
    ]
    if model_mode == "fixed":
        overrides.append(f"predict/fixed_model={args.fixed_model}")
    if getattr(args, "no_viewer", False):
        overrides.append("predict/viewer=none")

    checkpoint_which = getattr(args, "checkpoint_which", None)
    if checkpoint_which is not None:
        overrides.append(f"checkpoint.which={checkpoint_which}")

    # Explicit flags override the preset. Anything left as None is simply not
    # emitted, so the value composed from `predict/quality` stands.
    for value, key in (
        (getattr(args, "mask_samples", None), "evaluation.sampling.mask.n_samples"),
        (getattr(args, "theta_samples", None), "evaluation.sampling.theta.num_samples"),
        (
            getattr(args, "num_steps", None),
            "evaluation.sampling.theta.params.num_steps",
        ),
        (getattr(args, "batch_size", None), "evaluation.batch_size"),
        (getattr(args, "precision", None), "evaluation.precision"),
    ):
        if value is not None:
            overrides.append(f"{key}={value}")
    corrector = getattr(args, "corrector", None)
    if corrector is not None:
        overrides.append(
            "evaluation/theta/corrector@evaluation.sampling.theta.corrector="
            f"{corrector}"
        )
    if args.local_checkpoint is not None:
        checkpoint = args.local_checkpoint.expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError(f"Local checkpoint does not exist: {checkpoint}")
        overrides.extend([
            f"checkpoint.path={checkpoint.parent}",
            f"checkpoint.model_name={checkpoint.name}",
        ])
    else:
        overrides.extend([
            f"checkpoint.pretrained.repo_id={args.repo_id}",
            "checkpoint.pretrained.local_files_only="
            f"{str(args.local_files_only).lower()}",
        ])
        if args.revision:
            overrides.append(f"checkpoint.pretrained.revision={args.revision}")
        if args.cache_dir:
            overrides.append(
                "checkpoint.pretrained.cache_dir="
                f"{args.cache_dir.expanduser().resolve()}"
            )
    # Last, so an explicit --set wins over everything above.
    overrides.extend(getattr(args, "overrides", None) or [])
    return overrides


def main(argv=None):
    """Run prediction using the simple public interface."""
    console.print_logo()
    args = _parser().parse_args(argv)
    console.configure(enabled=True, verbose=args.verbose)
    if args.list_models:
        for model_name in list_pretrained_models(args.repo_id, revision=args.revision):
            print(model_name)
        return
    if args.folder is None:
        _parser().error("folder is required unless --list-models is used")

    try:
        folder = _validate_input(args.folder)
        if _can_prompt(args):
            print(f"\nInput:  {args.folder.expanduser().resolve()}")
            _interactive_setup(args)
        _apply_defaults(args)
        _validate_model_mode(args)
        output_dir = _prepare_output(folder, args.output_subdir, args.overwrite)
    except ValueError as error:
        _parser().error(str(error))

    if args.memory_fraction is not None:
        # Read when the XLA backend is first initialised, which has not happened
        # yet -- importing jax alone does not create a backend.
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory_fraction)

    try:
        overrides = _hydra_overrides(args, folder, output_dir)
    except ValueError as error:
        _parser().error(str(error))

    if not args.verbose:
        # Set at runtime, not through the environment: huggingface_hub reads
        # HF_HUB_DISABLE_PROGRESS_BARS when it is imported, which already
        # happened. Its bar counts cached files, so it only adds noise here.
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except ImportError:
            pass
        # The diagnostic log is verbose by design; this run shows the summary
        # above, the autotuning block and progress bars instead.
        overrides.append("hydra/job_logging=disabled")

    _run_with_overrides(overrides, folder, args)
    console.say(f"Prediction complete. Results are in {output_dir}")


def _summary_rows(cfg, folder, args):
    """Report what Hydra actually resolved, not what the flags asked for."""
    theta = cfg.evaluation.sampling.theta
    corrected = "no corrector" if theta.corrector.name == "uncorrected" else "corrected"
    mode = cfg.evaluation.selection.name
    if not cfg.evaluation.pipeline.select_models:
        mode = f"fixed ({cfg.predict.fixed_model.name})"
    elif cfg.evaluation.pipeline.sample_mask:
        mode = "per-sample"
    return [
        ("Input", folder),
        (
            "Model",
            args.local_checkpoint or f"{args.repo_id}/{cfg.checkpoint.model_name}",
        ),
        (
            "Sampling",
            f"{args.quality or DEFAULT_QUALITY}, {theta.params.num_steps} steps x "
            f"{theta.num_samples} samples, {cfg.evaluation.precision}, {corrected}",
        ),
        ("Model mode", mode),
        ("Output", cfg.run.output_dir),
    ]


def _run_with_overrides(overrides, folder, args):
    """Drive the Hydra entry point with a computed override list.

    `@hydra.main` reads `sys.argv`, so this is how it is invoked
    programmatically. Unlike the composition API it initialises `HydraConfig`,
    which `eval.yaml` needs for its `${hydra:runtime.cwd}` interpolation.
    """
    import hydra
    from omegaconf import DictConfig

    from dmri.eval.eval_script import run_eval

    @hydra.main(
        config_path="../conf", config_name=CONFIG_NAME + ".yaml", version_base=None
    )
    def _run(cfg: DictConfig):
        _adapt_precision_for_backend(cfg)
        console.summary("Prediction", _summary_rows(cfg, folder, args))
        run_eval(cfg)

        viewer_cfg = cfg.predict.viewer
        if not viewer_cfg.enabled:
            return
        output_dir = Path(cfg.run.output_dir)
        try:
            viewer = write_viewer(output_dir, viewer_cfg)
        except Exception as error:  # noqa: BLE001 - never fail a good prediction
            console.say(f"Could not write the HTML viewer: {error}")
            viewer = None
        if viewer is not None:
            console.link("Viewer:    ", viewer)

    original_argv = sys.argv
    try:
        sys.argv = ["dmri predict", *overrides]
        _run()
    finally:
        sys.argv = original_argv
