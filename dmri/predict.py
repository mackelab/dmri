import argparse
import shutil
import sys
from pathlib import Path

from dmri.hub import DEFAULT_MODEL_NAME, DEFAULT_REPO_ID, list_pretrained_models

STANDARD_FILES = ("data.nii.gz", "nodif_brain_mask.nii.gz", "bvals", "bvecs")


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parser():
    parser = argparse.ArgumentParser(
        prog="dmri predict",
        description="Run a pretrained DMRI model on an FSL/HCP-style data folder.",
    )
    parser.add_argument("folder", nargs="?", type=Path, help="Input data folder")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME, help="Pretrained model name"
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
    parser.add_argument("--mask-samples", type=_positive_int, default=50)
    parser.add_argument("--theta-samples", type=_positive_int, default=50)
    return parser


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
        f"data.path={folder}",
        f"output_dir={output_dir}",
        f"model_name={args.model}",
        "checkpoint=best",
        f"seed={args.seed}",
        "model_selection=average",
        f"mask_sample.n_samples={args.mask_samples}",
        f"theta_sample.num_samples={args.theta_samples}",
        "~export.metrics",
        f"hydra.run.dir={output_dir / '.hydra'}",
    ]
    if args.local_checkpoint is not None:
        checkpoint = args.local_checkpoint.expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError(f"Local checkpoint does not exist: {checkpoint}")
        overrides.extend([
            f"path_checkpoint={checkpoint.parent}",
            f"model_name={checkpoint.name}",
        ])
    else:
        overrides.extend([
            f"pretrained_repo_id={args.repo_id}",
            f"local_files_only={str(args.local_files_only).lower()}",
        ])
        if args.revision:
            overrides.append(f"pretrained_revision={args.revision}")
        if args.cache_dir:
            overrides.append(
                f"pretrained_cache_dir={args.cache_dir.expanduser().resolve()}"
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
        if args.mask_samples < args.theta_samples:
            raise ValueError(
                "--mask-samples must be greater than or equal to --theta-samples"
            )
        output_dir = _prepare_output(folder, args.output_subdir, args.overwrite)
        overrides = _hydra_overrides(args, folder, output_dir)
    except ValueError as error:
        _parser().error(str(error))

    print(f"Input:  {folder}")
    print(f"Model:  {args.local_checkpoint or f'{args.repo_id}/{args.model}'}")
    print(f"Output: {output_dir}")

    from dmri.eval.eval_script import main as eval_main

    original_argv = sys.argv
    try:
        sys.argv = ["dmri eval", *overrides]
        eval_main()
    finally:
        sys.argv = original_argv

    print(f"Prediction complete. Results are in {output_dir}")
