from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

ROOT = Path(__file__).resolve().parents[1] / "conf"


def group_cases(config_name, groups):
    return [
        (config_name, group, path.stem)
        for group in groups
        for path in sorted((ROOT / group).glob("*.yaml"))
    ]


@pytest.fixture(autouse=True)
def clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def compose_config(name, overrides=None):
    with initialize_config_module(version_base=None, config_module="conf"):
        return compose(config_name=name, overrides=overrides or [])


def test_default_configs_compose():
    train = compose_config("train")
    evaluation = compose_config("eval")

    assert train.schema_version == 2
    assert train.training.optimizer.optimizer == "radam"
    assert evaluation.schema_version == 2
    assert evaluation.evaluation.sampling.theta.corrector.name == "auto"


@pytest.mark.parametrize(
    "preset",
    sorted(path.stem for path in (ROOT / "experiment" / "train").glob("*.yaml")),
)
def test_training_experiments_compose(preset):
    cfg = compose_config("train", [f"+experiment/train={preset}"])
    assert cfg.run.name


@pytest.mark.parametrize(
    "preset",
    sorted(path.stem for path in (ROOT / "experiment" / "eval").glob("*.yaml")),
)
def test_evaluation_experiments_compose(preset):
    cfg = compose_config("eval", [f"+experiment/eval={preset}"])
    assert cfg.evaluation.input.source


def test_new_cli_override_paths_compose():
    cfg = compose_config(
        "eval",
        [
            "evaluation/input=file",
            "evaluation/selection=average",
            "evaluation.input.path=/tmp/data",
            "checkpoint.model_name=example",
            "evaluation.sampling.theta.num_samples=7",
            "~evaluation.export.theta.metrics",
        ],
    )

    assert cfg.checkpoint.model_name == "example"
    assert cfg.evaluation.selection.name == "average"
    assert "metrics" not in cfg.evaluation.export.theta


@pytest.mark.parametrize(
    "config_name,group,option",
    group_cases(
        "train",
        (
            "model",
            "simulator",
            "training",
            "infrastructure/launcher",
            "infrastructure/partition",
        ),
    )
    + group_cases(
        "eval",
        (
            "evaluation/input",
            "evaluation/mask",
            "evaluation/theta",
            "evaluation/selection",
            "evaluation/export/theta",
            "evaluation/export/model_selection",
        ),
    ),
)
def test_top_level_config_groups_compose(config_name, group, option):
    cfg = compose_config(config_name, [f"{group}={option}"])
    assert cfg.schema_version == 2
