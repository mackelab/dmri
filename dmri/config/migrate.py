from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

CONFIG_SCHEMA_VERSION = 2
ConfigKind = Literal["train", "eval"]


class ConfigMigrationError(ValueError):
    """Raised when a configuration cannot be migrated without ambiguity."""


_MISSING = object()


def _plain(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        value = OmegaConf.to_container(cfg, resolve=False)
    else:
        value = deepcopy(cfg)
    if not isinstance(value, dict):
        raise ConfigMigrationError("The configuration root must be a mapping")
    return value


def _get(data: dict[str, Any], path: str, default: Any = _MISSING) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigMigrationError(
                f"Cannot create {path!r}: {part!r} is not a mapping"
            )
        current = child
    current[parts[-1]] = deepcopy(value)


def _delete(data: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def _move(
    source: dict[str, Any],
    target: dict[str, Any],
    old_path: str,
    new_path: str,
) -> None:
    old_value = _get(source, old_path)
    if old_value is _MISSING:
        return
    new_value = _get(target, new_path)
    if new_value is not _MISSING and new_value != old_value:
        raise ConfigMigrationError(
            f"Conflicting config values at {old_path!r} and {new_path!r}"
        )
    if new_value is _MISSING:
        _set(target, new_path, old_value)


def _migrate_common(source: dict[str, Any], target: dict[str, Any]) -> None:
    _move(source, target, "name", "run.name")
    _move(source, target, "seed", "run.seed")
    _move(source, target, "model", "model")
    _move(source, target, "simulator", "simulator")

    legacy_spelling = _get(source, "model.prefered_element_type")
    preferred = _get(target, "model.preferred_element_type")
    if legacy_spelling is not _MISSING:
        if preferred is not _MISSING and preferred != legacy_spelling:
            raise ConfigMigrationError(
                "Conflicting config values at 'model.prefered_element_type' and "
                "'model.preferred_element_type'"
            )
        if preferred is _MISSING:
            _set(target, "model.preferred_element_type", legacy_spelling)


def _migrate_train(source: dict[str, Any], target: dict[str, Any]) -> None:
    _migrate_common(source, target)
    _move(source, target, "train", "training")
    _move(source, target, "use_wandb", "tracking.enabled")
    _move(source, target, "wandb", "tracking.wandb")

    root_recovery = _get(source, "recovery_threshold")
    nested_recovery = _get(target, "training.recovery_threshold")
    if root_recovery is not _MISSING:
        if nested_recovery is not _MISSING and nested_recovery != root_recovery:
            raise ConfigMigrationError(
                "Conflicting config values at 'recovery_threshold' and "
                "'training.recovery_threshold'"
            )
        if nested_recovery is _MISSING:
            _set(target, "training.recovery_threshold", root_recovery)

    dataloader = _get(target, "training.dataloader")
    if isinstance(dataloader, dict):
        _migrate_dataloader(dataloader)

    for key in ("max_checkpoints", "keep_best_checkpoint"):
        _move(source, target, key, f"training.checkpoint.{key}")

    for path in (
        "name",
        "seed",
        "train",
        "use_wandb",
        "wandb",
        "recovery_threshold",
        "max_checkpoints",
        "keep_best_checkpoint",
        "model.prefered_element_type",
    ):
        _delete(target, path)


def _migrate_dataloader(dataloader: dict[str, Any]) -> None:
    aliases = (("train_params", "train_loader"), ("val_params", "val_loader"))
    for old_key, new_key in aliases:
        if old_key not in dataloader:
            continue
        if new_key in dataloader and dataloader[new_key] != dataloader[old_key]:
            raise ConfigMigrationError(
                f"Conflicting dataloader values at {old_key!r} and {new_key!r}"
            )
        dataloader.setdefault(new_key, deepcopy(dataloader[old_key]))

    flat_keys = ("batch_size", "shuffle", "drop_last")
    flat = {key: dataloader[key] for key in flat_keys if key in dataloader}
    if not flat:
        return
    train_loader = dataloader.setdefault("train_loader", {})
    if not isinstance(train_loader, dict):
        raise ConfigMigrationError("training.dataloader.train_loader must be a mapping")
    for key, value in flat.items():
        if key in train_loader and train_loader[key] != value:
            raise ConfigMigrationError(
                f"Conflicting dataloader values at {key!r} and train_loader.{key!r}"
            )
        train_loader.setdefault(key, value)


def _migrate_eval(source: dict[str, Any], target: dict[str, Any]) -> None:
    _migrate_common(source, target)
    # Legacy evaluation used the scalar root key ``checkpoint``. Schema v2 uses
    # that name for the complete checkpoint-source mapping.
    if "checkpoint" in target and not isinstance(target["checkpoint"], dict):
        target.pop("checkpoint")
    mappings = {
        "data": "evaluation.input",
        "mask_sample": "evaluation.sampling.mask",
        "theta_sample": "evaluation.sampling.theta",
        "model_selection": "evaluation.selection",
        "export": "evaluation.export.theta",
        "export_model_selection": "evaluation.export.model_selection",
        "batch_size": "evaluation.batch_size",
        "precision": "evaluation.precision",
        "sample_mask": "evaluation.pipeline.sample_mask",
        "sample_theta": "evaluation.pipeline.sample_theta",
        "select_models": "evaluation.pipeline.select_models",
        "default_mask": "evaluation.pipeline.default_mask",
        "reuse_theta_samples": "evaluation.pipeline.reuse_theta_samples",
        "output_dir": "run.output_dir",
    }
    for old_path, new_path in mappings.items():
        _move(source, target, old_path, new_path)

    checkpoint_mappings = {
        "results_root": "checkpoint.results_root",
        "results_folder": "checkpoint.results_folder",
        "path_checkpoint": "checkpoint.path",
        "model_name": "checkpoint.model_name",
        "params_name": "checkpoint.params_name",
        "checkpoint": "checkpoint.which",
        "pretrained_repo_id": "checkpoint.pretrained.repo_id",
        "pretrained_revision": "checkpoint.pretrained.revision",
        "pretrained_cache_dir": "checkpoint.pretrained.cache_dir",
        "local_files_only": "checkpoint.pretrained.local_files_only",
    }
    for old_path, new_path in checkpoint_mappings.items():
        if old_path == "checkpoint" and isinstance(source.get("checkpoint"), dict):
            continue
        _move(source, target, old_path, new_path)

    root_slice = _get(source, "slice")
    nested_slice = _get(target, "evaluation.input.slice")
    if root_slice is not _MISSING:
        if nested_slice is not _MISSING and nested_slice != root_slice:
            raise ConfigMigrationError(
                "Conflicting config values at 'slice' and 'evaluation.input.slice'"
            )
        if nested_slice is _MISSING:
            _set(target, "evaluation.input.slice", root_slice)

    old_flag = _get(source, "use_true_model_mask_for_synthetic")
    nested_flag = _get(target, "evaluation.input.use_true_model_mask_for_synthetic")
    if old_flag is not _MISSING:
        if nested_flag is not _MISSING and nested_flag != old_flag:
            raise ConfigMigrationError(
                "Conflicting synthetic mask settings at the root and input config"
            )
        if nested_flag is _MISSING:
            _set(
                target,
                "evaluation.input.use_true_model_mask_for_synthetic",
                old_flag,
            )

    legacy_paths = [*mappings, *checkpoint_mappings]
    legacy_paths.remove("checkpoint")
    for path in (
        *legacy_paths,
        "seed",
        "slice",
        "use_true_model_mask_for_synthetic",
        "name",
        "model.prefered_element_type",
    ):
        _delete(target, path)


def normalize_config(cfg: DictConfig | dict[str, Any], kind: ConfigKind) -> DictConfig:
    """Convert a legacy or current config into the canonical schema."""
    source = _plain(cfg)
    version = source.get("schema_version", 1)
    if version not in (1, CONFIG_SCHEMA_VERSION):
        raise ConfigMigrationError(
            f"Unsupported config schema version {version!r}; "
            f"this release supports 1 and {CONFIG_SCHEMA_VERSION}"
        )

    # Preserve Hydra metadata and extension keys while relocating all known
    # application fields into the canonical schema.
    target = deepcopy(source)

    if kind == "train":
        _migrate_train(source, target)
    elif kind == "eval":
        _migrate_eval(source, target)
    else:
        raise ValueError(f"Unknown config kind: {kind!r}")

    target["schema_version"] = CONFIG_SCHEMA_VERSION
    return OmegaConf.create(target)


def runtime_config(cfg: DictConfig | dict[str, Any], kind: ConfigKind) -> DictConfig:
    """Project canonical config into the paths used by current runtime modules."""
    canonical = normalize_config(cfg, kind)
    data = _plain(canonical)

    if kind == "train":
        data["name"] = _get(data, "run.name", "dmri")
        data["seed"] = _get(data, "run.seed", 0)
        data["train"] = deepcopy(_get(data, "training", {}))
        data["use_wandb"] = _get(data, "tracking.enabled", False)
        data["wandb"] = deepcopy(_get(data, "tracking.wandb", {}))
        checkpoint_cfg = _get(data, "training.checkpoint", {})
        if isinstance(checkpoint_cfg, dict):
            for key in ("max_checkpoints", "keep_best_checkpoint"):
                if key in checkpoint_cfg:
                    data[key] = checkpoint_cfg[key]
    else:
        aliases = {
            "data": "evaluation.input",
            "mask_sample": "evaluation.sampling.mask",
            "theta_sample": "evaluation.sampling.theta",
            "model_selection": "evaluation.selection",
            "export": "evaluation.export.theta",
            "export_model_selection": "evaluation.export.model_selection",
            "batch_size": "evaluation.batch_size",
            "precision": "evaluation.precision",
            "sample_mask": "evaluation.pipeline.sample_mask",
            "sample_theta": "evaluation.pipeline.sample_theta",
            "select_models": "evaluation.pipeline.select_models",
            "default_mask": "evaluation.pipeline.default_mask",
            "reuse_theta_samples": "evaluation.pipeline.reuse_theta_samples",
            "output_dir": "run.output_dir",
            "seed": "run.seed",
            "results_root": "checkpoint.results_root",
            "results_folder": "checkpoint.results_folder",
            "path_checkpoint": "checkpoint.path",
            "model_name": "checkpoint.model_name",
            "params_name": "checkpoint.params_name",
            "checkpoint": "checkpoint.which",
            "pretrained_repo_id": "checkpoint.pretrained.repo_id",
            "pretrained_revision": "checkpoint.pretrained.revision",
            "pretrained_cache_dir": "checkpoint.pretrained.cache_dir",
            "local_files_only": "checkpoint.pretrained.local_files_only",
        }
        # Read every source before writing any alias. An alias may shadow the
        # root of another's path -- `checkpoint` holds `checkpoint.which` -- so
        # interleaving reads and writes would make the result order-dependent.
        projected = {
            alias: deepcopy(value)
            for alias, path in aliases.items()
            if (value := _get(data, path)) is not _MISSING
        }
        data.update(projected)

    return OmegaConf.create(data)
