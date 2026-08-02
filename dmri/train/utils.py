import os
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from flax import nnx
from huggingface_hub import HfApi, snapshot_download
from omegaconf import OmegaConf

from dmri.config import build_artifact_config, load_artifact_config, normalize_config
from dmri.simulators.config import resolve_simulator_model
from dmri.train.build_model import build_model
from dmri.train.checkpointing import CheckpointManager


def load_cfg(path):
    path = os.path.abspath(os.fspath(path))

    for config_path in (
        os.path.join(path, "config.yaml"),
        os.path.join(path, ".hydra", "config.yaml"),
    ):
        if os.path.exists(config_path):
            return normalize_config(OmegaConf.load(config_path), "train")

    # Find most recent run directory (format: YYYY-MM-DD_HH-MM-SS)
    dirs = [
        d
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
        and len(d.split("_")) == 2
        and len(d.split("_")[0].split("-")) == 3
    ]

    if not dirs:
        raise ValueError(f"No run directories found in {path}")

    # Parse timestamps and find the most recent one
    timestamp_dirs = []
    for d in dirs:
        try:
            timestamp = datetime.strptime(d, "%Y-%m-%d_%H-%M-%S")
            timestamp_dirs.append((timestamp, d))
        except ValueError:
            continue

    if not timestamp_dirs:
        raise ValueError(f"No valid timestamped directories found in {path}")

    # Sort by timestamp (newest first)
    timestamp_dirs.sort(reverse=True)
    most_recent_dir = timestamp_dirs[0][1]

    # Try to load config from .hydra first, then fall back to 0/.hydra
    base_path = os.path.join(path, most_recent_dir)
    config_path = os.path.join(base_path, ".hydra", "config.yaml")

    if not os.path.exists(config_path):
        config_path = os.path.join(base_path, "0", ".hydra", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Could not find config.yaml in either .hydra or 0/.hydra directories in {base_path}"
            )

    return normalize_config(OmegaConf.load(config_path), "train")


def _checkpoint_source(path: Path, which: str | int) -> tuple[Path, Path]:
    checkpoints = path / "checkpoints"
    if which == "all":
        return checkpoints, Path("checkpoints")
    if which == "best":
        return checkpoints / "best", Path("checkpoints/best")

    if which == "latest":
        steps = [entry for entry in checkpoints.iterdir() if entry.name.isdigit()]
        if not steps:
            raise FileNotFoundError(f"No regular checkpoints found in {checkpoints}")
        source = max(steps, key=lambda entry: int(entry.name))
    elif isinstance(which, int):
        source = checkpoints / str(which)
    else:
        raise ValueError("which must be 'best', 'latest', 'all', or an integer step")
    return source, Path("checkpoints") / source.name


def bundle_checkpoint(
    path, output_dir, model_name=None, which="best_and_latest"
) -> Path:
    """Create a portable checkpoint bundle for a model repository.

    The bundle contains a root ``config.yaml`` and an Orbax checkpoint tree, so
    it can be restored without the original timestamped Hydra run directory.

    Args:
        path: Local training-result directory.
        output_dir: Directory in which to create the model subfolder.
        model_name: Subfolder name. Defaults to ``cfg.name``.
        which: ``"best_and_latest"`` (default), ``"best"``, ``"latest"``,
            ``"all"``, or an integer step.

    Returns:
        Path to the created model subfolder.
    """
    path = Path(path).expanduser().resolve()
    cfg = load_cfg(path)
    model_name = model_name or cfg.run.name
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be a single directory name")

    if which == "best_and_latest":
        sources = [
            _checkpoint_source(path, "best"),
            _checkpoint_source(path, "latest"),
        ]
    else:
        sources = [_checkpoint_source(path, which)]
    for source, _ in sources:
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint not found: {source}")

    model_dir = Path(output_dir).expanduser().resolve() / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    OmegaConf.save(cfg, model_dir / "config.yaml")
    OmegaConf.save(build_artifact_config(cfg), model_dir / "artifact.yaml")
    for source, relative_destination in sources:
        shutil.copytree(
            source,
            model_dir / relative_destination,
            ignore=shutil.ignore_patterns("*.orbax-checkpoint-tmp"),
        )
    return model_dir


def upload_checkpoint_to_hub(
    path,
    repo_id,
    *,
    model_name=None,
    which="best_and_latest",
    private=False,
    token=None,
    commit_message=None,
):
    """Upload a local run into a model subfolder in one Hugging Face repo.

    Authentication uses the cached Hugging Face token by default. Pass
    ``token`` explicitly for non-interactive or private-repository workflows.

    Returns:
        The Hugging Face commit information returned by ``upload_folder``.
    """
    with TemporaryDirectory() as temporary_dir:
        model_dir = bundle_checkpoint(path, temporary_dir, model_name, which)
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        return api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=model_dir,
            path_in_repo=model_dir.name,
            commit_message=commit_message
            or f"Upload {model_dir.name} checkpoint ({which})",
            ignore_patterns=[".DS_Store"],
        )


def download_checkpoint_from_hub(
    repo_id,
    model_name,
    *,
    revision=None,
    cache_dir=None,
    token=None,
    local_files_only=False,
) -> Path:
    """Download one model subfolder from a Hugging Face checkpoint repo.

    Only ``<model_name>/**`` is downloaded, allowing several variants to share
    one repository without downloading every checkpoint.
    """
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be a single directory name")

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=[f"{model_name}/**"],
        )
    )
    model_dir = snapshot_dir / model_name
    if not (model_dir / "config.yaml").exists():
        raise FileNotFoundError(
            f"Model {model_name!r} was not found in Hugging Face repo {repo_id!r}"
        )
    return model_dir.resolve()


def load_checkpoint(
    path=None,
    which="latest",
    partial_restore=False,
    *,
    repo_id=None,
    model_name=None,
    revision=None,
    cache_dir=None,
    token=None,
    local_files_only=False,
    precision=None,
):
    """Restore a checkpoint from a local result directory or Hugging Face.

    Provide ``path`` for a local checkpoint. For a remote bundle, provide both
    ``repo_id`` and ``model_name``; the selected subfolder is downloaded into
    the Hugging Face cache before the normal Orbax restore path is used.

    ``precision`` selects the compute dtype for the rebuilt model (see
    :mod:`dmri.eval.precision`); the default keeps the checkpoint's own setting.
    """
    if repo_id is not None:
        if path is not None:
            raise ValueError("Provide either path or repo_id, not both")
        if model_name is None:
            raise ValueError("model_name is required when loading from Hugging Face")
        path = download_checkpoint_from_hub(
            repo_id,
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    if path is None:
        raise ValueError("A local path or Hugging Face repo_id is required")

    path = os.path.abspath(os.fspath(path))
    cfg = load_cfg(path)
    artifact = load_artifact_config(path)
    build_cfg = (
        OmegaConf.create({
            "model": OmegaConf.to_container(artifact.model, resolve=False),
            "simulator": OmegaConf.to_container(artifact.simulator, resolve=False),
        })
        if artifact is not None
        else cfg
    )
    if precision is not None:
        # Must patch before build_model, which reads these keys.
        from dmri.eval.precision import apply_precision_to_cfg

        apply_precision_to_cfg(build_cfg, precision)
    sim_type = resolve_simulator_model(build_cfg)
    model = build_model(build_cfg, sim_type)
    model.eval()

    _graphdef, params, _state = nnx.split(model, nnx.Param, ...)

    checkpoint_dir = os.path.join(path, "checkpoints")
    continue_training = True
    checkpoint_manager = CheckpointManager(
        ckpt_dir=checkpoint_dir,
        max_to_keep=cfg.get("max_checkpoints", 5),
        keep_best=cfg.get("keep_best_checkpoint", True),
        recovery_threshold=cfg.get("recovery_threshold", float("inf")),
        continue_training=continue_training,
    )

    from_best = False
    if which == "latest":
        latest_step = checkpoint_manager.get_latest_step()
    elif which == "best":
        latest_step = None
        from_best = True
    elif isinstance(which, int):
        latest_step = which
    else:
        raise ValueError(f"Invalid checkpoint type: {which}")

    checkpoint = checkpoint_manager.restore_parameters(
        step=latest_step,
        params=params,
        from_best=from_best,
        partial_restore=partial_restore,
    )
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    return checkpoint, model, sim_type
