import argparse
import os
import shutil
import sys
from pathlib import Path

from dmri.eval.precision import PRECISION_CHOICES
from dmri.hub import DEFAULT_MODEL_NAME, DEFAULT_REPO_ID, list_pretrained_models

STANDARD_FILES = ("data.nii.gz", "nodif_brain_mask.nii.gz", "bvals", "bvecs")

#: Sampling presets. Cost is exactly linear in ``num_steps x samples``, so the
#: relative figures below are exact; the accuracy trade-off is not quantified
#: here and should be checked against a `high` run on your own data.
QUALITY_PRESETS = {
    "fast": {"num_steps": 20, "samples": 25},
    "balanced": {"num_steps": 40, "samples": 50},
    "high": {"num_steps": 60, "samples": 100},
}
DEFAULT_QUALITY = "balanced"


def _preset_cost(name):
    preset = QUALITY_PRESETS[name]
    return preset["num_steps"] * preset["samples"]


def _quality_description(name):
    preset = QUALITY_PRESETS[name]
    summary = f"{preset['num_steps']} steps x {preset['samples']} samples"
    if name == DEFAULT_QUALITY:
        return f"{summary}  (default)"
    ratio = _preset_cost(name) / _preset_cost(DEFAULT_QUALITY)
    comparison = f"~{1 / ratio:.0f}x faster" if ratio < 1 else f"~{ratio:.0f}x slower"
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
        choices=tuple(QUALITY_PRESETS),
        help=(
            "Sampling preset: "
            + "; ".join(
                f"{name} ({QUALITY_PRESETS[name]['num_steps']} steps x "
                f"{QUALITY_PRESETS[name]['samples']} samples)"
                for name in QUALITY_PRESETS
            )
            + f". Defaults to {DEFAULT_QUALITY}."
        ),
    )
    parser.add_argument(
        "--mask-samples",
        type=_positive_int,
        help="Number of model-mask samples. Overrides --quality.",
    )
    parser.add_argument(
        "--theta-samples",
        type=_positive_int,
        help="Number of parameter samples. Overrides --quality.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the full evaluation log instead of the summary and progress bars.",
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
        default="fp32",
        help=(
            "Numeric precision for the network forward pass. Half precision is "
            "faster and uses less memory; parameters and the sampler stay fp32."
        ),
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


def _choose(title, options, default_index, stream=None):
    """Ask the user to pick one of ``options``; returns the chosen value.

    ``options`` is a sequence of ``(value, description)``. An empty answer takes
    the default, so pressing enter through the prompts is always valid.
    """
    stream = stream or sys.stdout
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
    """Model names in the configured repo, falling back to the default name."""
    try:
        models = list_pretrained_models(args.repo_id, revision=args.revision)
    except Exception:
        return [DEFAULT_MODEL_NAME]
    return models or [DEFAULT_MODEL_NAME]


def _interactive_setup(args) -> None:
    """Fill in whatever the user did not pass, by asking."""
    if args.model is None:
        models = _available_models(args)
        default_index = (
            models.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in models else 0
        )
        args.model = _choose(
            f"Model  (from {args.repo_id})",
            [
                (name, "recommended default" if name == DEFAULT_MODEL_NAME else "")
                for name in models
            ],
            default_index,
        )

    if args.quality is None:
        args.quality = _choose(
            "Quality",
            [(name, _quality_description(name)) for name in QUALITY_PRESETS],
            list(QUALITY_PRESETS).index(DEFAULT_QUALITY),
        )


def _apply_defaults(args) -> None:
    """Resolve the quality preset, letting explicit flags win over it."""
    args.model = args.model or DEFAULT_MODEL_NAME
    args.quality = args.quality or DEFAULT_QUALITY
    preset = QUALITY_PRESETS[args.quality]

    if args.num_steps is None:
        args.num_steps = preset["num_steps"]
    if args.theta_samples is None:
        args.theta_samples = preset["samples"]
    if args.mask_samples is None:
        args.mask_samples = max(preset["samples"], args.theta_samples)


def _validate_input(folder: Path) -> Path:
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
    overrides = [
        f"evaluation.input.path={folder}",
        f"run.output_dir={output_dir}",
        f"checkpoint.model_name={args.model}",
        "checkpoint.which=best",
        f"run.seed={args.seed}",
        "evaluation/selection=average",
        f"evaluation.sampling.mask.n_samples={args.mask_samples}",
        f"evaluation.sampling.theta.num_samples={args.theta_samples}",
        "~evaluation.export.theta.metrics",
        f"hydra.run.dir={output_dir / '.hydra'}",
    ]
    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        overrides.append(f"evaluation.batch_size={batch_size}")
    overrides.append(
        f"evaluation.precision={getattr(args, 'precision', None) or 'fp32'}"
    )
    num_steps = getattr(args, "num_steps", None)
    if num_steps is not None:
        overrides.append(f"evaluation.sampling.theta.params.num_steps={num_steps}")
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
    return overrides


def main(argv=None):
    """Run prediction using the simple public interface."""
    args = _parser().parse_args(argv)
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
        if args.mask_samples < args.theta_samples:
            raise ValueError(
                "--mask-samples must be greater than or equal to --theta-samples"
            )
        output_dir = _prepare_output(folder, args.output_subdir, args.overwrite)
        overrides = _hydra_overrides(args, folder, output_dir)
    except ValueError as error:
        _parser().error(str(error))

    if args.memory_fraction is not None:
        # Read when the XLA backend is first initialised, which has not happened
        # yet -- importing jax alone does not create a backend.
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory_fraction)

    print()
    print(f"Input:     {folder}")
    print(f"Model:     {args.local_checkpoint or f'{args.repo_id}/{args.model}'}")
    print(
        f"Sampling:  {args.quality} "
        f"({args.num_steps} steps x {args.theta_samples} samples), {args.precision}"
    )
    print(f"Output:    {output_dir}")
    print(
        "\nThe first run on a new machine autotunes batch sizes, which takes a few "
        "minutes.\nThe result is cached, so later runs skip it."
    )

    from dmri.eval import console

    console.set_enabled(True)
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

    from dmri.eval.eval_script import main as eval_main

    original_argv = sys.argv
    try:
        sys.argv = ["dmri eval", *overrides]
        eval_main()
    finally:
        sys.argv = original_argv

    print(f"Prediction complete. Results are in {output_dir}")
