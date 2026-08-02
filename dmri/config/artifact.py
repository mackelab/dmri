from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dmri.config.migrate import CONFIG_SCHEMA_VERSION, normalize_config

ARTIFACT_VERSION = 1


def build_artifact_config(cfg: DictConfig | dict[str, Any]) -> DictConfig:
    """Extract the stable model-construction subset of a training config."""
    canonical = normalize_config(cfg, "train")
    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "parameter_tree_version": 1,
        "model": OmegaConf.to_container(canonical.model, resolve=False),
        "simulator": OmegaConf.to_container(canonical.simulator, resolve=False),
    }
    return OmegaConf.create(artifact)


def load_artifact_config(path: str | Path) -> DictConfig | None:
    artifact_path = Path(path) / "artifact.yaml"
    if not artifact_path.exists():
        return None
    artifact = OmegaConf.load(artifact_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {version!r}; expected {ARTIFACT_VERSION}"
        )
    return artifact
