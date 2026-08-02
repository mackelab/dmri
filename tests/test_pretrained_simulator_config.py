from omegaconf import OmegaConf

from dmri.simulators.config import resolve_simulator_model, simulator_model_config
from dmri.simulators.models import Ball3StickNoise


def test_checkpoint_simulator_resolves_portable_model_class():
    cfg = OmegaConf.create({
        "simulator": {"model_class": "dmri.simulators.models.Ball3StickNoise"}
    })

    assert resolve_simulator_model(cfg) is Ball3StickNoise


def test_checkpoint_simulator_resolves_legacy_short_name():
    cfg = OmegaConf.create({"simulator": {"sim_type": {"name": "Ball3StickNoise"}}})

    assert resolve_simulator_model(cfg) is Ball3StickNoise
    assert simulator_model_config(cfg) == {
        "model_class": "dmri.simulators.models.Ball3StickNoise"
    }
