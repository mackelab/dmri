import pytest
from omegaconf import OmegaConf

from dmri.config import (
    ConfigMigrationError,
    build_artifact_config,
    normalize_config,
    runtime_config,
)


def test_legacy_training_config_migrates_to_canonical_schema():
    legacy = OmegaConf.create({
        "name": "legacy",
        "seed": 4,
        "use_wandb": True,
        "wandb": {"project": "test"},
        "model": {"prefered_element_type": "float32"},
        "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
        "train": {
            "recovery_threshold": 2.0,
            "dataloader": {"train_params": {"batch_size": 8}},
        },
    })

    cfg = normalize_config(legacy, "train")

    assert cfg.schema_version == 2
    assert cfg.run.name == "legacy"
    assert cfg.run.seed == 4
    assert cfg.tracking.enabled is True
    assert cfg.model.preferred_element_type == "float32"
    assert cfg.training.dataloader.train_loader.batch_size == 8


def test_legacy_eval_config_migrates_nested_paths():
    legacy = OmegaConf.create({
        "seed": 3,
        "model_name": "example",
        "data": {"source": "file"},
        "slice": {"x": 2},
        "theta_sample": {"num_samples": 10},
        "use_true_model_mask_for_synthetic": True,
    })

    cfg = normalize_config(legacy, "eval")

    assert cfg.run.seed == 3
    assert cfg.checkpoint.model_name == "example"
    assert cfg.evaluation.input.slice.x == 2
    assert cfg.evaluation.input.use_true_model_mask_for_synthetic is True
    assert cfg.evaluation.sampling.theta.num_samples == 10


def test_migration_rejects_conflicting_old_and_new_values():
    mixed = OmegaConf.create({
        "schema_version": 2,
        "name": "old",
        "run": {"name": "new"},
    })

    with pytest.raises(ConfigMigrationError, match="Conflicting config values"):
        normalize_config(mixed, "train")


def test_normalization_is_idempotent():
    current = OmegaConf.create({
        "schema_version": 2,
        "run": {"name": "example", "seed": 1},
        "training": {"track_ema": False},
    })

    once = normalize_config(current, "train")
    twice = normalize_config(once, "train")

    assert OmegaConf.to_container(once) == OmegaConf.to_container(twice)


def test_runtime_projection_keeps_consumers_isolated_from_migration():
    current = OmegaConf.create({
        "schema_version": 2,
        "run": {"name": "example", "seed": 1},
        "training": {"track_ema": False},
    })

    cfg = runtime_config(current, "train")

    assert cfg.name == "example"
    assert cfg.seed == 1
    assert cfg.train.track_ema is False


def test_current_eval_config_projects_to_runtime_paths():
    current = OmegaConf.create({
        "schema_version": 2,
        "run": {"seed": 8},
        "checkpoint": {"model_name": "example", "which": "best"},
        "evaluation": {
            "input": {"source": "file"},
            "sampling": {"theta": {"num_samples": 5}},
        },
    })

    cfg = runtime_config(current, "eval")

    assert cfg.seed == 8
    assert cfg.model_name == "example"
    assert cfg.checkpoint == "best"
    assert cfg.theta_sample.num_samples == 5


def test_eval_runtime_projection_is_order_independent():
    """An alias may shadow the root of another alias's path.

    ``checkpoint`` is projected from ``checkpoint.which``, so writing it before
    reading ``checkpoint.pretrained.*`` would leave those aliases unset and
    silently send prediction down the local-checkpoint branch.
    """
    current = OmegaConf.create({
        "schema_version": 2,
        "run": {"seed": 1},
        "checkpoint": {
            "model_name": "example",
            "params_name": "params_ema",
            "which": "best",
            "pretrained": {
                "repo_id": "owner/repo",
                "revision": "v1",
                "cache_dir": "/tmp/cache",
                "local_files_only": False,
            },
        },
    })

    cfg = runtime_config(current, "eval")

    assert cfg.checkpoint == "best"
    assert cfg.model_name == "example"
    assert cfg.params_name == "params_ema"
    assert cfg.pretrained_repo_id == "owner/repo"
    assert cfg.pretrained_revision == "v1"
    assert cfg.pretrained_cache_dir == "/tmp/cache"
    assert cfg.local_files_only is False


def test_artifact_contains_only_model_construction_config():
    cfg = OmegaConf.create({
        "schema_version": 2,
        "run": {"name": "example"},
        "model": {"name": "Model"},
        "simulator": {"sim_type": {"name": "Simulator"}},
        "training": {"optimizer": {"name": "adam"}},
    })

    artifact = build_artifact_config(cfg)

    assert artifact.artifact_version == 1
    assert artifact.model.name == "Model"
    assert artifact.simulator.sim_type.name == "Simulator"
    assert "training" not in artifact


def test_unknown_schema_version_is_rejected():
    with pytest.raises(ConfigMigrationError, match="Unsupported config schema"):
        normalize_config({"schema_version": 99}, "train")
