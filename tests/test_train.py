import logging

import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

from dmri.train.build_simulator import build_simulator
from dmri.train.checkpointing import CheckpointManager
from dmri.train.train_script import TrainState, apply_checkpoint_to_state


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
