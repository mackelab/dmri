import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from omegaconf import OmegaConf

from dmri.train.build_simulator import build_simulator
from dmri.train.checkpointing import CheckpointManager
from dmri.train.dataset import SimulationDataset
from dmri.train.train_script import TrainState, apply_checkpoint_to_state
from dmri.train.utils import (
    bundle_checkpoint,
    download_checkpoint_from_hub,
    load_cfg,
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


def test_checkpoint_save_builds_args_without_persisting():
    manager = CheckpointManager.__new__(CheckpointManager)
    manager.keep_best = False

    class UnexpectedPersistence:
        def save(self, *args, **kwargs):
            raise AssertionError("checkpoint persistence should be disabled")

    manager.manager = UnexpectedPersistence()

    manager.save(
        step=1,
        params={"weight": jnp.array([1.0])},
        optimizer_state=None,
        loss=0.5,
        write_standard=False,
    )


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


def test_simulation_dataset_surfaces_background_failure():
    def simulator(key):
        return jnp.asarray([1.0], dtype=jnp.float32)

    dataset = SimulationDataset(
        simulator,
        simulation_batch_size=1,
        rng=jax.random.key(0),
        simulation_device=jax.devices("cpu")[0],
        jit_simulator=False,
        buffer_size=1,
    )
    try:
        dataset._stop_producer()

        def fail_batch():
            raise ValueError("simulator failed")

        dataset._produce_batch = fail_batch
        with dataset._condition:
            dataset._pending_refresh = dataset._batch_size
        dataset._start_producer()

        deadline = time.monotonic() + 5
        while dataset._producer_exception is None and time.monotonic() < deadline:
            time.sleep(0.01)

        with pytest.raises(
            RuntimeError, match="Background simulation failed"
        ) as exc_info:
            dataset[0]
        assert isinstance(exc_info.value.__cause__, ValueError)
    finally:
        dataset.close()


def test_new_simulator_config_builds_direct_class_and_acquisition():
    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "dmri.simulators.models.Ball3StickNoise",
            "posterior_score": False,
            "mask_prior": {},
            "acquisitions": [
                {
                    "_target_": ("dmri.simulators.random_hcp_acquisition"),
                    "_partial_": True,
                    "num_acquisitions": 105,
                    "typical_prob": 1.0,
                    "random_prob": 0.0,
                }
            ],
        }
    })

    model_class, simulators = build_simulator(cfg)
    result = simulators[0](jax.random.key(2))

    assert model_class.__name__ == "Ball3StickNoise"
    assert result["x"].shape == (105,)
    assert "target_score" not in result


def make_checkpoint_tree(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best" / "12"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    (checkpoint / ".zarray").write_text("{}")
    latest = run_dir / "checkpoints" / "15"
    latest.mkdir()
    (latest / "_CHECKPOINT_METADATA").write_text("{}")
    OmegaConf.save(
        {
            "name": "example-model",
            "model": {"name": "ExampleModel"},
            "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
        },
        run_dir / "config.yaml",
    )
    return run_dir


def test_bundle_checkpoint_creates_portable_layout(tmp_path):
    run_dir = make_checkpoint_tree(tmp_path)

    model_dir = bundle_checkpoint(run_dir, tmp_path / "bundle", which="best")

    assert model_dir.name == "example-model"
    assert (model_dir / "config.yaml").exists()
    assert (model_dir / "artifact.yaml").exists()
    assert (model_dir / "checkpoints" / "best" / "12" / "_CHECKPOINT_METADATA").exists()
    assert (model_dir / "checkpoints" / "best" / "12" / ".zarray").exists()

    config = OmegaConf.load(model_dir / "config.yaml")
    artifact = OmegaConf.load(model_dir / "artifact.yaml")
    assert config.schema_version == 2
    assert config.run.name == "example-model"
    assert artifact.artifact_version == 1
    assert "training" not in artifact
    assert artifact.simulator == {
        "model_class": "dmri.simulators.models.Ball3StickNoise"
    }
    assert "acquisitions" not in artifact.simulator


def test_bundle_checkpoint_defaults_to_best_and_latest(tmp_path):
    run_dir = make_checkpoint_tree(tmp_path)
    temporary = run_dir / "checkpoints" / "best" / "broken.orbax-checkpoint-tmp"
    temporary.mkdir()

    model_dir = bundle_checkpoint(run_dir, tmp_path / "bundle")

    assert (model_dir / "checkpoints" / "best" / "12").is_dir()
    assert (model_dir / "checkpoints" / "15").is_dir()
    assert not (model_dir / "checkpoints" / "best" / temporary.name).exists()


def test_load_cfg_migrates_legacy_timestamp_layout(tmp_path):
    hydra_dir = tmp_path / "2026-01-02_03-04-05" / ".hydra"
    hydra_dir.mkdir(parents=True)
    OmegaConf.save(
        {
            "name": "legacy",
            "seed": 9,
            "model": {"name": "ExampleModel"},
            "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
            "train": {"track_ema": False},
        },
        hydra_dir / "config.yaml",
    )

    cfg = load_cfg(tmp_path)

    assert cfg.schema_version == 2
    assert cfg.run.name == "legacy"
    assert cfg.run.seed == 9
    assert cfg.training.track_ema is False


def test_inference_restore_requests_ema_only_when_present(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "12" / "params_ema").mkdir(parents=True)
    calls = {}

    class FakeOrbaxManager:
        def latest_step(self):
            return 12

        def restore(self, step, args):
            calls["step"] = step
            calls["args"] = args
            return {"params": {"weight": 1}, "params_ema": {"weight": 2}}

    manager = CheckpointManager.__new__(CheckpointManager)
    manager.ckpt_dir = str(checkpoint_dir)
    manager.best_ckpt_dir = str(checkpoint_dir / "best")
    manager.manager = FakeOrbaxManager()
    manager.best_manager = FakeOrbaxManager()

    restored = manager.restore_parameters(None, {"weight": 0})

    assert calls["step"] == 12
    assert restored["params_ema"]["weight"] == 2
    assert restored["step"] == 12


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
            assert (Path(kwargs["folder_path"]) / "checkpoints" / "best").exists()
            assert (Path(kwargs["folder_path"]) / "checkpoints" / "15").exists()
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
