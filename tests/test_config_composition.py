from pathlib import Path

import jax
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from dmri.simulators.config import (
    resolve_acquisition_factories,
    resolve_simulator_model,
)

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


def test_predict_root_inherits_the_evaluation_tree():
    """`conf/predict.yaml` lists `eval` in its defaults rather than restating it.

    If that inheritance breaks, predict silently loses the whole evaluation
    tree, so pin a key from each group eval.yaml composes.
    """
    predict = compose_config("predict")
    evaluation = compose_config("eval")

    assert predict.schema_version == 2
    assert predict.evaluation.input.source == evaluation.evaluation.input.source
    assert predict.evaluation.sampling.mask.method == "naive"
    assert predict.evaluation.sampling.theta.params.t_max == 80
    assert predict.checkpoint.params_name == "params_ema"
    # And the predict-only layer is present on top.
    assert predict.predict.viewer.enabled is True
    assert predict.predict.fixed_model.name == "B3S"


@pytest.mark.parametrize(
    "preset",
    sorted(path.stem for path in (ROOT / "experiment" / "train").glob("*.yaml")),
)
def test_training_experiments_compose(preset):
    cfg = compose_config("train", [f"+experiment/train={preset}"])
    assert cfg.run.name


def test_all_model_rician_continuation_matches_legacy_architecture():
    cfg = compose_config(
        "train", ["+experiment/train=all_3_4_8_128_model_prior"]
    )

    assert cfg.run.name == "all_3_4_8_128_model_prior_rician_fixed"
    assert cfg.tracking.enabled is True
    assert cfg.tracking.wandb.project == "all"
    assert resolve_simulator_model(cfg).__name__ == "AllGaussianModelsParamCountPrior"
    assert len(resolve_acquisition_factories(cfg)) == 2
    assert cfg.model.model_dim == 128
    assert cfg.model.embedding_net.params.num_layers == 3
    assert cfg.model.model_selection_net.params.num_layers == 4
    assert cfg.model.inference_net.params.num_layers == 8
    assert cfg.training.continue_training is True
    assert cfg.training.partial_restore is False
    assert cfg.training.restart_optimizer is False
    assert cfg.training.optimizer.optimizer == "radam"
    assert cfg.training.optimizer.learning_rate == 1e-4
    assert cfg.training.ema_decay == 0.999
    assert cfg.training.eval.ksd is None


def test_convolved_two_gpu_continuation_matches_legacy_architecture():
    cfg = compose_config(
        "train",
        ["+experiment/train=all_3_6_8_128_model_prior_with_convs_2gpus"],
    )

    assert (
        cfg.run.name
        == "all_3_6_8_128_model_prior_with_convs_2gpus_rician_fixed"
    )
    assert cfg.tracking.enabled is True
    assert cfg.tracking.wandb.project == "all_conv"
    assert resolve_simulator_model(cfg).__name__ == "AllGaussianAndConvolvedModels"
    assert cfg.simulator.mask_prior.u_alpha == 1.0
    assert cfg.simulator.mask_prior.u_beta == 1.0
    assert len(resolve_acquisition_factories(cfg)) == 2
    assert cfg.model.model_dim == 128
    assert cfg.model.embedding_net.params.num_layers == 3
    assert cfg.model.model_selection_net.params.num_layers == 6
    assert cfg.model.inference_net.params.num_layers == 8
    assert cfg.training.continue_training is True
    assert cfg.training.partial_restore is False
    assert cfg.training.restart_optimizer is False
    assert cfg.training.dataloader.dataset.simulation_device.index == 1
    assert cfg.training.dataloader.train_loader.devices[0].index == 0
    assert cfg.training.optimizer.optimizer == "radam"
    assert cfg.training.optimizer.learning_rate == 1e-4
    assert cfg.training.ema_decay == 0.999
    assert cfg.training.eval.ksd is None


def test_updated_convolved_two_gpu_continuation_matches_frozen_run():
    cfg = compose_config(
        "train",
        [
            "+experiment/train="
            "updated_all_3_6_8_128_model_prior_with_convs_2gpus"
        ],
    )

    assert cfg.run.name.endswith("with_convs_2gpus_rician_fixed_continued")
    assert cfg.tracking.wandb.project == "all_conv"
    assert resolve_simulator_model(cfg).__name__ == "AllGaussianAndConvolvedModels"
    assert cfg.model.name == "DMRIInferenceModelConfigMaskPriorAmortizedPPP"
    assert cfg.model.embedding_net.params.num_layers == 3
    assert cfg.model.model_selection_net.params.num_layers == 6
    assert cfg.model.model_selection_net.params.outnorm is False
    assert cfg.model.model_selection_net.params.outlayer == "mlp"
    assert cfg.model.inference_net.params.num_layers == 8
    assert cfg.training.restart_optimizer is False
    assert cfg.training.optimizer.gradient_clip_value == 1.0
    assert cfg.training.dataloader.dataset.simulation_batch_size == 2048
    assert cfg.training.dataloader.dataset.buffer_size == 4_194_304
    assert cfg.training.cut_off_tsm == 0.0
    assert cfg.training.label_smoothing == 0.0
    assert cfg.training.eval.ksd.iters == 1


def test_updated_all_model_continuation_matches_frozen_run():
    cfg = compose_config(
        "train", ["+experiment/train=updated_all_3_4_8_128_model_prior"]
    )

    assert cfg.run.name == "updated_all_3_6_8_128_model_prior_rician_fixed_continued"
    assert cfg.tracking.wandb.project == "all"
    assert resolve_simulator_model(cfg).__name__ == "AllGaussianModelsParamCountPrior"
    assert cfg.model.name == "DMRIInferenceModelConfigMaskPriorAmortizedPPP"
    assert cfg.model.embedding_net.params.num_layers == 3
    assert cfg.model.model_selection_net.params.num_layers == 6
    assert cfg.model.model_selection_net.params.outnorm is False
    assert cfg.model.model_selection_net.params.outlayer == "mlp"
    assert cfg.model.inference_net.params.num_layers == 8
    assert cfg.training.restart_optimizer is False
    assert cfg.training.optimizer.learning_rate == 5e-4
    assert cfg.training.optimizer.gradient_clip_value == 1.0
    assert cfg.training.cut_off_tsm == 0.0
    assert cfg.training.label_smoothing == 0.0
    assert cfg.training.eval.ksd.iters == 1


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
    ("overrides", "selection", "sample_mask", "select_models", "default_mask"),
    [
        (
            ["evaluation/selection=average"],
            "average",
            True,
            True,
            None,
        ),
        (
            [
                "evaluation/selection=ball3stick_best",
                "evaluation.pipeline.sample_mask=false",
            ],
            "best",
            False,
            True,
            None,
        ),
        (
            [
                "evaluation/selection=none",
                "evaluation.pipeline.sample_mask=false",
                "evaluation.pipeline.select_models=false",
                "evaluation.pipeline.default_mask=[true,true,true,false,true]",
            ],
            "none",
            False,
            False,
            [True, True, True, False, True],
        ),
    ],
)
def test_prediction_model_modes_compose(
    overrides, selection, sample_mask, select_models, default_mask
):
    cfg = compose_config("eval", overrides)

    assert cfg.evaluation.selection.name == selection
    assert cfg.evaluation.pipeline.sample_mask is sample_mask
    assert cfg.evaluation.pipeline.select_models is select_models
    assert cfg.evaluation.pipeline.default_mask == default_mask


def test_simulator_class_path_can_be_overridden_without_new_config_file():
    cfg = compose_config(
        "train",
        ["simulator.model_class=dmri.simulators.models.Ball3StickNoise"],
    )

    assert resolve_simulator_model(cfg).__name__ == "Ball3StickNoise"


def test_score_experiment_uses_scalar_instead_of_duplicate_simulator_recipe():
    cfg = compose_config("train", ["+experiment/train=msb3s_2_4_6_128"])

    assert cfg.simulator.posterior_score is True
    assert cfg.simulator.model_class.endswith("Ball3StickSharedDiffusivity")


@pytest.mark.parametrize(
    "config_name,group,option",
    group_cases(
        "train",
        (
            "model",
            "simulator",
            "simulator/acquisition",
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
    )
    + group_cases(
        "predict",
        (
            "predict/quality",
            "predict/model_mode",
            "predict/fixed_model",
            "predict/viewer",
        ),
    ),
)
def test_top_level_config_groups_compose(config_name, group, option):
    cfg = compose_config(config_name, [f"{group}={option}"])
    assert cfg.schema_version == 2
    if group == "simulator":
        assert resolve_simulator_model(cfg).theta_dim > 0
    if group == "simulator/acquisition":
        factories = resolve_acquisition_factories(cfg)
        for factory in factories:
            acquisition = factory(jax.random.key(0))
            assert acquisition.bvals.shape[0] > 0
