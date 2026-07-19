import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

from dmri.train.build_simulator import build_simulator
from dmri.train.checkpointing import CheckpointManager
from dmri.train.train_script import TrainState, apply_checkpoint_to_state
from dmri.train.utils import (
    bundle_checkpoint,
    download_checkpoint_from_hub,
    upload_checkpoint_to_hub,
)


def make_checkpoint_manager(threshold):
    manager = CheckpointManager.__new__(CheckpointManager)
    manager.recovery_threshold = threshold
    manager.prev_metric = None
    return manager


def test_recovery_threshold_handles_negative_metrics():
    manager = make_checkpoint_manager(1.5)

    assert not manager.should_recover(-10.0)
    assert not manager.should_recover(-6.0)
    assert manager.should_recover(-2.0)


def test_recovery_triggers_for_non_finite_metrics():
    manager = make_checkpoint_manager(2.0)

    assert manager.should_recover(float("nan"))
    assert manager.should_recover(float("inf"))

    disabled = make_checkpoint_manager(float("inf"))
    assert not disabled.should_recover(float("nan"))


def test_missing_optimizer_state_is_reinitialized():
    params = {"weight": jnp.array([1.0])}
    optimizer = optax.sgd(0.1)
    state = TrainState(
        params={"weight": jnp.array([0.0])},
        model_state={},
        opt_state=None,
        ema_state=None,
        rng=jax.random.key(0),
    )
    checkpoint = {"params": params, "step": 4}
    cfg = OmegaConf.create({"train": {"track_ema": False}})

    restored = apply_checkpoint_to_state(
        state,
        checkpoint,
        cfg,
        optimizer,
        ema_transform=None,
        log=logging.getLogger(__name__),
        rebuild_optimizer=False,
    )

    assert restored.step == 4
    assert restored.opt_state is not None


def test_simulator_accepts_explicit_mask_and_prior():
    cfg = OmegaConf.create({
        "simulator": {
            "sim_type": {"name": "Ball3StickNoise"},
            "acquisition_scheme": {
                "name": "hcp",
                "params": {
                    "schemes": [
                        {
                            "name": "hcp",
                            "params": {
                                "num_acquisitions": 105,
                                "typical_prob": 1.0,
                                "random_prob": 0.0,
                            },
                        }
                    ]
                },
            },
            "prior_mask_alpha": 1.0,
            "prior_mask_beta": 1.0,
            "with_posterior_score": False,
        }
    })
    sim_type, simulators = build_simulator(cfg)
    num_components = len(sim_type.model_types) + len(sim_type.noise_types)
    model_mask = jnp.ones(num_components, dtype=jnp.bool_)
    prior = jnp.array([0.25])

    result = simulators[0](
        jax.random.key(1), mask_prior_hyperparameter=prior, model_mask=model_mask
    )

    assert jnp.array_equal(result["model_mask"], model_mask)
    assert jnp.array_equal(result["mask_prior"], prior)


def make_checkpoint_tree(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best" / "12"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    (checkpoint / ".zarray").write_text("{}")
    OmegaConf.save({"name": "example-model"}, run_dir / "config.yaml")
    return run_dir


def test_bundle_checkpoint_creates_portable_layout(tmp_path):
    run_dir = make_checkpoint_tree(tmp_path)

    model_dir = bundle_checkpoint(run_dir, tmp_path / "bundle", which="best")

    assert model_dir.name == "example-model"
    assert (model_dir / "config.yaml").exists()
    assert (model_dir / "checkpoints" / "best" / "12" / "_CHECKPOINT_METADATA").exists()
    assert (model_dir / "checkpoints" / "best" / "12" / ".zarray").exists()


def test_upload_checkpoint_uses_model_subfolder(tmp_path, monkeypatch):
    run_dir = make_checkpoint_tree(tmp_path)
    calls = {}

    class FakeApi:
        def __init__(self, token):
            calls["token"] = token

        def create_repo(self, *args, **kwargs):
            calls["create"] = (args, kwargs)

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs
            assert (Path(kwargs["folder_path"]) / "config.yaml").exists()
            return "commit"

    monkeypatch.setattr("dmri.train.utils.HfApi", FakeApi)

    result = upload_checkpoint_to_hub(
        run_dir,
        "owner/dmri-pretrained",
        token="secret",
        private=True,
    )

    assert result == "commit"
    assert calls["create"][0] == ("owner/dmri-pretrained",)
    assert calls["create"][1]["private"] is True
    assert calls["upload"]["path_in_repo"] == "example-model"


def test_download_checkpoint_selects_model_subfolder(tmp_path, monkeypatch):
    model_dir = tmp_path / "snapshot" / "example-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text("name: example-model\n")
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return model_dir.parent

    monkeypatch.setattr("dmri.train.utils.snapshot_download", fake_snapshot_download)

    result = download_checkpoint_from_hub(
        "owner/dmri-pretrained", "example-model", revision="v1"
    )

    assert result == model_dir.resolve()
    assert calls["allow_patterns"] == ["example-model/**"]
    assert calls["revision"] == "v1"
