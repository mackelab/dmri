import jax
import jax.numpy as jnp

import numpy as np

import dmri
from dmri.dmri_models.local_models.gaussian_models import Ball, Stick, Dti


import pytest


@pytest.fixture(params=[Ball, Stick, Dti])
def compartment_model(request):
    model_class = request.param
    theta = np.random.randn(model_class.theta_dim)
    return model_class.from_theta(theta)


def test_base_function(compartment_model):
    bvals = np.random.uniform(size=(10,)) * 5000
    bvecs = np.random.randn(10, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=-1, keepdims=True)

    # Signal emulation
    signal = compartment_model.signal(bvals, bvecs)
    log_signal = compartment_model.log_signal(bvals, bvecs)

    assert signal.shape == (10,), f"Expected shape (10,) but got {signal.shape}"
    assert log_signal.shape == (10,), f"Expected shape (10,) but got {log_signal.shape}"
    assert jnp.all(jnp.isfinite(signal)), "Signal is not finite"
    
    # Fitting
    
    
    
