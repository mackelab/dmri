import pytest
from omegaconf import OmegaConf

from dmri.config import load_artifact_config
from dmri.eval.data_sources import _get_acquisition_fns
from dmri.simulators.config import (
    resolve_acquisition_factories,
    resolve_simulator_model,
    simulator_model_config,
)
from dmri.simulators.models import AllGaussianModelsParamCountPrior, Ball3StickNoise
from dmri.train.build_simulator import build_simulator


def test_legacy_short_model_name_resolves_to_models_module():
    cfg = OmegaConf.create({"simulator": {"sim_type": {"name": "Ball3StickNoise"}}})

    assert resolve_simulator_model(cfg) is Ball3StickNoise
    assert simulator_model_config(cfg) == {
        "model_class": "dmri.simulators.models.Ball3StickNoise"
    }


def test_model_class_must_implement_multicompartment_contract():
    cfg = OmegaConf.create({"simulator": {"model_class": "builtins.dict"}})

    with pytest.raises(TypeError, match="must subclass MultiCompartment"):
        resolve_simulator_model(cfg)


def test_conflicting_model_class_and_legacy_name_fail():
    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "dmri.simulators.models.Ball3StickNoise",
            "sim_type": {"name": "BallStickZeppelinNoise"},
        }
    })

    with pytest.raises(ValueError, match="Conflicting simulator settings"):
        resolve_simulator_model(cfg)


def test_legacy_malformed_acquisition_layout_remains_supported():
    cfg = OmegaConf.create({
        "simulator": {
            "acquisition_scheme": {
                "name": "clinical",
                "schemes": [{"name": "clinical", "params": {"num_acquisitions": 35}}],
            }
        }
    })

    factories = resolve_acquisition_factories(cfg)

    assert len(factories) == 1
    assert callable(factories[0])


def test_conflicting_new_and_legacy_score_settings_fail():
    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "dmri.simulators.models.Ball3StickNoise",
            "posterior_score": False,
            "with_posterior_score": True,
            "acquisition_scheme": {
                "name": "hcp",
                "params": {
                    "schemes": [{"name": "hcp", "params": {"num_acquisitions": 105}}]
                },
            },
        }
    })

    with pytest.raises(ValueError, match="Conflicting simulator settings"):
        build_simulator(cfg)


def test_conflicting_new_and_legacy_mask_prior_settings_fail():
    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "dmri.simulators.models.Ball3StickNoise",
            "mask_prior": {"alpha": 2.0},
            "prior_mask_alpha": 1.0,
            "acquisition_scheme": {
                "name": "hcp",
                "params": {
                    "schemes": [{"name": "hcp", "params": {"num_acquisitions": 105}}]
                },
            },
        }
    })

    with pytest.raises(ValueError, match="Conflicting simulator mask-prior"):
        build_simulator(cfg)


def test_structural_mask_prior_arguments_are_derived_by_model_class():
    mask_prior = AllGaussianModelsParamCountPrior.create_mask_prior()

    assert mask_prior._sizes.shape == (
        len(AllGaussianModelsParamCountPrior.model_types),
    )


def test_frozen_artifact_simulator_shape_remains_resolvable(tmp_path):
    OmegaConf.save(
        {
            "artifact_version": 1,
            "config_schema_version": 2,
            "parameter_tree_version": 1,
            "model": {},
            "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
        },
        tmp_path / "artifact.yaml",
    )

    artifact = load_artifact_config(tmp_path)

    assert resolve_simulator_model(artifact) is Ball3StickNoise


def test_synthetic_evaluation_accepts_direct_acquisition_targets():
    config = OmegaConf.create({
        "acquisitions": [
            {
                "_target_": "dmri.simulators.random_hcp_acquisition",
                "_partial_": True,
                "num_acquisitions": 105,
            }
        ]
    })

    factories = _get_acquisition_fns(config)

    assert len(factories) == 1
    assert callable(factories[0])
