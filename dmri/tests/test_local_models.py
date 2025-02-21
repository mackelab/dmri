import jax
import jax.numpy as jnp

import numpy as np

import dmri
from dmri.simulators import (
    acquisition_scheme,
    Dot,
    Ball,
    Stick,
    Dti,
    Zeppelin,
    WatsonStick,
    WatsonZeppelin,
    BinghamStick,
    BinghamZeppelin,
    NoddiB,
    NoddiW,
    SandiB,
    SandiW,
    Sphere,
    Cylinder,
)
from dmri.simulators.multi_compartment import (
    BallStick,
    Ball2Stick,
    Ball3Stick,
    BallStickZeppelin,
)


import pytest


@pytest.fixture(
    params=[
        Dot,
        Ball,
        Stick,
        Zeppelin,
        Dti,
        Sphere,
        Cylinder,
        WatsonStick,
        WatsonZeppelin,
        BinghamStick,
        BinghamZeppelin,
        NoddiB,
        NoddiW,
        SandiB,
        SandiW,
        BallStick,
        Ball2Stick,
        Ball3Stick,
        BallStickZeppelin,
    ]
)
def compartment_model(request):
    model_class = request.param
    theta = np.random.randn(model_class.theta_dim)
    return model_class.from_theta(theta)


def test_base_function(compartment_model):
    bvals = np.random.uniform(size=(10,)) * 1000
    bvecs = np.random.randn(10, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=-1, keepdims=True)

    acq = acquisition_scheme(bvals, bvecs)

    # Signal emulation
    signal = compartment_model.signal(acq)
    log_signal = compartment_model.log_signal(acq)

    assert signal.shape == (10,), f"Expected shape (10,) but got {signal.shape}"
    assert log_signal.shape == (10,), f"Expected shape (10,) but got {log_signal.shape}"
    assert jnp.all(jnp.isfinite(signal)), "Signal is not finite"

    # Pytree
    compartment_model_old = compartment_model
    flat, tree = jax.tree_util.tree_flatten(compartment_model)
    compartment_model = jax.tree_util.tree_unflatten(tree, flat)
    assert isinstance(compartment_model, type(compartment_model_old)), (
        "Failed to reconstruct the compartment model"
    )
    assert jnp.allclose(compartment_model.theta, compartment_model_old.theta), (
        "Failed to reconstruct the compartment model"
    )


def test_jitable(compartment_model):
    bvals = np.random.uniform(size=(10,)) * 1000
    bvecs = np.random.randn(10, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=-1, keepdims=True)

    acq = acquisition_scheme(bvals, bvecs)

    # Signal emulation
    signal = compartment_model.signal(acq)
    signal_jit = jax.jit(compartment_model.signal)(acq)

    assert jnp.allclose(signal, signal_jit, atol=1e-3), "JIT failed"
