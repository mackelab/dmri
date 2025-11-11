import jax
import jax.numpy as jnp
import pytest

from dmri.simulators.base import NoiseCompartment, SignalCompartment
from dmri.simulators.multi_compartment import MultiCompartment


class _DummySignal(SignalCompartment):
    theta_dim = 2

    def __init__(self, theta_vals):
        self.theta_vals = jnp.asarray(theta_vals)

    @classmethod
    def to_theta(cls, theta_vals) -> jnp.ndarray:
        return jnp.asarray(theta_vals)

    @classmethod
    def to_params(cls, theta, **kwargs):
        return (jnp.asarray(theta),)

    @classmethod
    def log_signal_fn(cls, acq, *args, **kwargs):
        raise NotImplementedError


class _DummyNoise(NoiseCompartment):
    theta_dim = 1

    def __init__(self, theta_vals):
        self.theta_vals = jnp.asarray(theta_vals)

    @classmethod
    def to_theta(cls, theta_vals) -> jnp.ndarray:
        return jnp.atleast_1d(theta_vals)

    @classmethod
    def to_params(cls, theta, **kwargs):
        return (jnp.atleast_1d(theta),)

    def noise(self, signal, rng):
        del rng
        return signal

    def log_likelihood(self, signal_pred, signal_obs):
        del signal_pred, signal_obs
        return jnp.array(0.0)


class _DummySimulator(MultiCompartment):
    model_types = [_DummySignal, _DummySignal, _DummySignal]
    noise_types = [_DummyNoise, _DummyNoise]
    fraction_prior = jnp.ones(3)


_MASKS = [
    jnp.array([True, True, True, True, True]),
    jnp.array([True, False, True, True, False]),
    jnp.array([False, True, True, False, True]),
]


def _assert_simulators_equal(sim_a, sim_b):
    assert jnp.allclose(sim_a.model_fractions, sim_b.model_fractions)
    assert len(sim_a.model_compartments) == len(sim_b.model_compartments)
    assert len(sim_a.noise_compartments) == len(sim_b.noise_compartments)

    for comp_a, comp_b in zip(sim_a.model_compartments, sim_b.model_compartments):
        assert jnp.allclose(comp_a.theta, comp_b.theta)

    for comp_a, comp_b in zip(sim_a.noise_compartments, sim_b.noise_compartments):
        assert jnp.allclose(comp_a.theta, comp_b.theta)


@pytest.mark.parametrize("mask", _MASKS)
def test_theta_mask_zeroes_inactive_parameters(mask):
    rng = jax.random.PRNGKey(0)
    theta = jax.random.normal(rng, (_DummySimulator.theta_dim,))
    theta_mask = _DummySimulator.theta_mask(mask)

    sim_full = _DummySimulator.from_theta(theta, model_mask=mask)
    sim_masked = _DummySimulator.from_theta(
        theta * theta_mask.astype(theta.dtype), model_mask=mask
    )

    assert theta_mask.dtype == jnp.bool_
    assert theta_mask.shape == (_DummySimulator.theta_dim,)
    _assert_simulators_equal(sim_full, sim_masked)
