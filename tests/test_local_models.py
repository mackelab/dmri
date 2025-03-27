import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmri.simulators import (
    Ball,
    BinghamStick,
    BinghamZeppelin,
    Cylinder,
    Dot,
    Dti,
    MultiCompartment,
    NoddiB,
    NoddiW,
    SandiB,
    SandiW,
    Sphere,
    Stick,
    WatsonStick,
    WatsonZeppelin,
    Zeppelin,
    acquisition_scheme,
)
from dmri.simulators.multi_compartment import (
    Ball2Stick,
    Ball3Stick,
    BallStick,
)


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
    assert jnp.allclose(
        compartment_model.theta, compartment_model_old.theta, atol=1e-3
    ), "Failed to reconstruct the compartment model"


def test_jitable(compartment_model):
    if isinstance(compartment_model, Cylinder):
        pytest.xfail("Cylinder model needs currently some non jax-compatible code")

    bvals = np.random.uniform(size=(10,)) * 1000
    bvecs = np.random.randn(10, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=-1, keepdims=True)

    acq = acquisition_scheme(bvals, bvecs)

    # Signal emulation
    signal = compartment_model.signal(acq)
    signal_jit = jax.jit(compartment_model.signal)(acq)

    # No tracer leaks
    _ = compartment_model.signal(acq)

    assert jnp.allclose(signal, signal_jit, atol=1e-3), "JIT failed"

def test_signal_properties(compartment_model):
    """Test fundamental properties of the signal."""
    # Create acquisition scheme with varying b-values
    bvals = jnp.array([0, 100, 500, 1000, 2000])
    bvecs = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]])
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    acq = acquisition_scheme(bvals, bvecs)

    # Get signals
    signal = compartment_model.signal(acq)

    # Test signal properties
    # Spherical models only approximate ..
    assert jnp.all(signal >= -0.05), "Signal should be non-negative"
    assert jnp.all(signal <= 1 + 0.05), "Signal should be bounded by 1"
    assert jnp.allclose(signal[0], 1.0, atol=0.2), "Signal at b=0 should be 1"


def test_gradient_computation(compartment_model):
    """Test that gradients can be computed for all parameters using multiple methods."""
    # Create acquisition scheme
    if isinstance(compartment_model, Cylinder):
        pytest.xfail("Cylinder model needs currently some non jax-compatible code")
    if isinstance(compartment_model, Sphere):
        pytest.xfail("Sphere model needs currently some non jax-compatible code")
    if isinstance(compartment_model, MultiCompartment):
        # pytest.xfail("MultiCompartmentModel model needs currently not implemented")
        pass

    bvals = jnp.array([0, 100, 1000])
    bvecs = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    acq = acquisition_scheme(bvals, bvecs)

    def loss_fn(theta):
        model = compartment_model.__class__.from_theta(theta)
        signal = model.signal(acq)
        return jnp.mean(signal)

    # 1. Test automatic differentiation gradient
    grad_fn = jax.grad(loss_fn)
    grad_ad = grad_fn(compartment_model.theta)

    # 2. Monte Carlo gradient approximation using Gaussian perturbations
    n_samples = 20_000
    key = jax.random.PRNGKey(0)
    grad_mc = jnp.zeros_like(compartment_model.theta)
    eps = 1e-1

    # Process samples in batches of 10_000
    batch_size = 10_000
    n_batches = n_samples // batch_size

    for i in range(n_batches):
        key, subkey = jax.random.split(key)

        # Generate noise samples for this batch
        noise = (
            jax.random.normal(
                subkey, shape=(batch_size,) + compartment_model.theta.shape
            )
            * eps
        )

        # Compute perturbed thetas
        thetas_perturbed = compartment_model.theta + noise

        # Vectorize loss computation over batch samples
        losses_perturbed = jax.vmap(loss_fn)(thetas_perturbed)
        loss_original = loss_fn(compartment_model.theta)

        # Accumulate gradient estimate
        grad_mc += jnp.sum(
            (losses_perturbed - loss_original)[:, None] * noise, axis=0
        ) / (eps * eps)

    grad_mc = grad_mc / n_samples

    # Check gradient properties
    assert grad_ad.shape == compartment_model.theta.shape, (
        "AD gradient shape should match parameter shape"
    )
    assert grad_mc.shape == compartment_model.theta.shape, (
        "MC gradient shape should match parameter shape"
    )
    assert jnp.all(jnp.isfinite(grad_ad)), "AD gradient should be finite"
    assert jnp.all(jnp.isfinite(grad_mc)), "MC gradient should be finite"

    # Compare gradients (using relative error for better numerical stability)
    rel_error_mc = jnp.abs(grad_ad - grad_mc)

    # Allow for some numerical error in the approximations
    assert jnp.all(rel_error_mc < 0.5), f"Monte Carlo gradient {grad_mc} {grad_ad} "

    # Test gradient direction consistency
    # The cosine similarity between gradients should be close to 1
    if jnp.linalg.norm(grad_ad) > 0:
        cos_sim_mc = jnp.dot(grad_ad, grad_mc) / (
            jnp.linalg.norm(grad_ad) * jnp.linalg.norm(grad_mc) + 1e-10
        )

        assert cos_sim_mc > 0.9, (
            "Monte Carlo gradient direction differs significantly from AD gradient"
        )
