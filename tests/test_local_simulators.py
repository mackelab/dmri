import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmri.simulators import (
    Ball,
    BallStickSharedDiffusivity,
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
from dmri.simulators.local_signal_models.ball import MultiShellBall
from dmri.simulators.local_signal_models.stick import MultiShellStick
from dmri.simulators.multi_compartment import (
    AllGaussianAndConvolvedModels,
    AllGaussianModels,
    Ball2Stick,
    Ball3Stick,
    Ball3StickSharedDiffusivity,
    Ball3StickSharedDiffusivityUniformFraction,
    BallStick,
    MultiShellBall3StickSharedDiffusivity,
    MultiShellBall3StickSharedDiffusivityGammaPrior,
    MultiShellBall3StickSharedDiffusivityUniformFraction,
)


@pytest.fixture(
    params=[
        Dot,
        Ball,
        MultiShellBall,
        Stick,
        MultiShellStick,
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
        BallStickSharedDiffusivity,
        Ball3StickSharedDiffusivity,
        Ball3StickSharedDiffusivityUniformFraction,
        MultiShellBall3StickSharedDiffusivity,
        MultiShellBall3StickSharedDiffusivityUniformFraction,
        MultiShellBall3StickSharedDiffusivityGammaPrior,
    ]
)
def compartment_model(request):
    model_class = request.param
    theta = np.random.randn(model_class.theta_dim)
    return model_class.from_theta(theta)

@pytest.fixture(
    params=[
        Ball3Stick,
        BallStickSharedDiffusivity,
        Ball3StickSharedDiffusivity,
        Ball3StickSharedDiffusivityUniformFraction,
        MultiShellBall3StickSharedDiffusivity,
        MultiShellBall3StickSharedDiffusivityUniformFraction,
        AllGaussianModels,
        AllGaussianAndConvolvedModels,
    ]
)
def multi_compartment_model(request):
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
    #if isinstance(compartment_model, Cylinder):
    #    pytest.xfail("Cylinder model needs currently some non jax-compatible code")

    bvals = np.random.uniform(size=(10,)) * 1000
    bvecs = np.random.randn(10, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=-1, keepdims=True)

    acq = acquisition_scheme(bvals, bvecs)

    # Signal emulation
    signal = compartment_model.signal(acq)
    signal_jit = jax.jit(compartment_model.signal)(acq)

    # No tracer leaks
    _ = compartment_model.signal(acq)

    assert jnp.allclose(signal, signal_jit, atol=1e-2, rtol=1e-3), "JIT failed"


def test_stabally_differentiable(compartment_model):
    bvals = np.random.uniform(size=(10,)) * 1000
    bvecs = np.random.randn(10, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=-1, keepdims=True)

    acq = acquisition_scheme(bvals, bvecs)

    # Signal emulation
    def ll(theta):
        model = compartment_model.__class__.from_theta(theta)
        signal = model.signal(acq)
        return jnp.mean((signal - 1) ** 2)

    grad_fn = jax.grad(ll)
    thetas = np.random.randn(100, compartment_model.theta_dim)
    grads = jax.vmap(grad_fn)(thetas)
    assert jnp.all(jnp.isfinite(grads)), "Gradient is not finite"


def test_signal_properties(compartment_model):
    """Test fundamental properties of the signal."""
    # Create acquisition scheme with varying b-values
    bvals = jnp.array([0.0, 100.0, 500.0, 1000.0, 2000.0])
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

def test_correct_theta_masking(multi_compartment_model):
    """Test that theta masking works correctly in multi-compartment models."""
    model = multi_compartment_model
    theta_dim = model.theta_dim
    num_compartments = model.num_compartments()

    model_mask = jnp.ones((num_compartments,), dtype=bool)
    theta_masks = model.theta_mask(model_mask=model_mask)

    assert len(theta_masks) == theta_dim, "Theta masks length should match theta dimension"
    assert jnp.all(theta_masks), "All theta components should be included when model_mask is all False"

    # Correct masked reconstruction
    def test_correct_rec(theta, model_mask):
        theta_mask = model.theta_mask(model_mask=model_mask)
        params1 = type(multi_compartment_model).from_theta(theta, model_mask=model_mask).params
        params2 = type(multi_compartment_model).from_theta(theta*theta_mask, model_mask=model_mask).params
        fraction1 = params1["model_fractions"]
        fraction2 = params2["model_fractions"]
        assert jnp.allclose(
            fraction1, fraction2
        ), "Model fractions should match after masking"
        comp1 = params1["model_compartments"] + params1.get("noise_compartments", [])
        comp2 = params2["model_compartments"] + params2.get("noise_compartments", [])
        # params1 and params2 should be the same at the active compartments
        for i, m in enumerate(model_mask):
            if m and comp1[i].theta_dim > 0:
                comp1_i, _ = jax.tree_util.tree_flatten(comp1[i].params)
                comp2_i, _ = jax.tree_util.tree_flatten(comp2[i].params)
                check = jax.tree_util.tree_reduce(
                    jnp.logical_and,
                    jax.tree_util.tree_map(
                        lambda a, b: jnp.allclose(a, b), comp1_i, comp2_i
                    ),
                    True,
                )
                assert check, f"Compartment {i} parameters should match after masking"
    for i in range(5):
        theta = np.random.randn(theta_dim)
        model_mask = np.random.choice([True, False], size=(num_compartments,))
        test_correct_rec(theta, model_mask)



def test_gradient_computation(compartment_model):
    """Test that gradients can be computed for all parameters using multiple methods."""
    # Create acquisition scheme
    #if isinstance(compartment_model, Cylinder):
    #    pytest.xfail("Cylinder model needs currently some non jax-compatible code")
    #if isinstance(compartment_model, Sphere):
    #    pytest.xfail("Sphere model needs currently some non jax-compatible code")
    #if isinstance(compartment_model, MultiCompartment):
        # pytest.xfail("MultiCompartmentModel model needs currently not implemented")
    #    pass

    bvals = jnp.array([0.0, 100.0, 1000.0])
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
    n_samples = 10_000
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
