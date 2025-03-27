import jax
import jax.numpy as jnp
import pytest
from typing import Callable, Optional, Tuple

from dmri.utils.transform import (
    dirichlet_to_normal,
    normal_to_dirichlet,
)

# Test configurations
GRADIENT_TEST_CONFIGS = [
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: jnp.sum(pi[:2]),  # loss_fn
        id="basic_gradient",
    ),
    pytest.param(
        jnp.array([2.0, 1.0, 0.5]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: jnp.sum(pi[:2]),  # loss_fn
        id="different_alphas",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        jnp.array([True, True, False]),  # mask
        lambda pi: jnp.sum(pi),  # loss_fn
        id="masked_gradient",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0, 0.0]),  # eps
        jnp.array([True, True, False, True]),  # mask
        lambda pi: jnp.sum(pi**2),  # loss_fn
        id="four_dim_masked",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.5, 0.2, 0.3]),  # eps
        jnp.array([False, True, False, True]),  # mask
        lambda pi: 2 * jnp.sum(pi**2),  # loss_fn
        id="four_dim_masked2",
    ),
    pytest.param(
        jnp.array([0.5, 2.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: jnp.sum(pi * jnp.array([1.0, 2.0, 3.0])),  # weighted loss
        id="weighted_sum",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: pi[0],  # first component only
        id="first_component",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: pi[2],  # last component only
        id="last_component",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: pi[1],  # middle component only
        id="middle_component",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([1.0, -1.0]),  # eps
        None,  # mask
        lambda pi: jnp.sum(pi * jnp.array([1.0, 0.0, -1.0])),  # alternating weights
        id="alternating_weights",
    ),
]

@pytest.mark.parametrize("alpha,eps,mask,loss_fn", GRADIENT_TEST_CONFIGS)
def test_normal_to_dirichlet_gradients(
    alpha: jnp.ndarray,
    eps: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray], float],
):
    """Test gradients of the normal to Dirichlet transformation using both AD and MC estimation."""

    def full_loss_fn(eps):
        pi = normal_to_dirichlet(alpha, eps, mask)
        return loss_fn(pi)

    # Compute AD gradient
    grad_fn = jax.grad(full_loss_fn)
    grad_ad = grad_fn(eps)

    # Monte Carlo gradient estimation
    n_samples = 10_000  # Increased samples for better accuracy
    key = jax.random.PRNGKey(0)
    grad_mc = jnp.zeros_like(eps)
    eps_mc = 1e-1  # Smaller perturbation for better accuracy

    # Process samples in batches
    batch_size = 10_000
    n_batches = n_samples // batch_size

    for i in range(n_batches):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=(batch_size,) + eps.shape) * eps_mc
        eps_perturbed = eps + noise
        losses_perturbed = jax.vmap(full_loss_fn)(eps_perturbed)
        loss_original = full_loss_fn(eps)
        grad_mc += jnp.sum(
            (losses_perturbed - loss_original)[:, None] * noise, axis=0
        ) / (eps_mc * eps_mc)

    grad_mc = grad_mc / n_samples

    # Check gradient properties
    assert jnp.all(jnp.isfinite(grad_ad)), "AD gradient should be finite"
    assert jnp.all(jnp.isfinite(grad_mc)), "MC gradient should be finite"
    assert grad_ad.shape == eps.shape, "AD gradient shape should match input shape"
    assert grad_mc.shape == eps.shape, "MC gradient shape should match input shape"

    # Compare gradients with tighter tolerance for weighted sums
    rel_error_mc = jnp.abs(grad_ad - grad_mc)
    assert jnp.all(rel_error_mc < 0.2), f"Monte Carlo gradient {grad_mc}  {grad_ad} "

    # Test gradient direction consistency with tighter tolerance
    if jnp.linalg.norm(grad_ad) > 0:
        cos_sim_mc = jnp.dot(grad_ad, grad_mc) / (
            jnp.linalg.norm(grad_ad) * jnp.linalg.norm(grad_mc) + 1e-10
        )
        assert cos_sim_mc > 0.9, (
            f"Monte Carlo gradient direction differs significantly from AD gradient. "
            f"Cosine similarity: {cos_sim_mc}"
        )

# Test configurations for roundtrip tests
ROUNDTRIP_TEST_CONFIGS = [
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.3, 0.3, 0.4]),  # pi
        None,  # mask
        id="basic_roundtrip",
    ),
    pytest.param(
        jnp.array([2.0, 1.0, 0.5]),  # alpha
        jnp.array([0.5, 0.3, 0.2]),  # pi
        None,  # mask
        id="different_alphas",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.5, 0.5, 0.0]),  # pi
        jnp.array([True, True, False]),  # mask
        id="masked_roundtrip",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.25, 0.25, 0.25, 0.25]),  # pi
        None,  # mask
        id="four_dimensional",
    ),
]


# Test configurations for edge cases
EDGE_CASE_CONFIGS = [
    pytest.param(
        jnp.array([1e-6, 1e-6, 1e-6]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        id="small_alpha",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([10.0, 10.0]),  # eps
        id="large_eps",
    ),
    pytest.param(
        jnp.array([100.0, 100.0, 100.0]),  # alpha
        jnp.array([0.0, 0.0]),  # eps
        id="large_alpha",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0]),  # alpha
        jnp.array([-3.0, -3.0]),  # eps
        id="negative_eps",
    ),
]

@pytest.mark.parametrize("alpha,eps", EDGE_CASE_CONFIGS)
def test_normal_to_dirichlet_edge_cases(alpha: jnp.ndarray, eps: jnp.ndarray):
    """Test edge cases for normal to Dirichlet transformation."""
    # Test forward transform
    pi = normal_to_dirichlet(alpha, eps)

    # Check properties
    assert jnp.all(jnp.isfinite(pi)), "Output should be finite"
    assert jnp.all(pi >= 0), "All probabilities should be non-negative"
    assert jnp.allclose(jnp.sum(pi), 1.0), "Probabilities should sum to 1"

    # Test gradient computation
    def loss_fn(eps):
        pi = normal_to_dirichlet(alpha, eps)
        return jnp.sum(pi**2)

    grad = jax.grad(loss_fn)(eps)
    assert jnp.all(jnp.isfinite(grad)), "Gradients should be finite"


@pytest.mark.parametrize("alpha,pi,mask", ROUNDTRIP_TEST_CONFIGS)
def test_normal_dirichlet_invertibility(
    alpha: jnp.ndarray, pi: jnp.ndarray, mask: Optional[jnp.ndarray]
):
    """Test that normal_to_dirichlet and dirichlet_to_normal are inverse operations.

    This test verifies that:
    1. Converting from normal space to Dirichlet space and back preserves the original normal parameters
    2. Converting from Dirichlet space to normal space and back preserves the original Dirichlet parameters
    """
    # Test direction 1: pi -> eps -> pi_recovered
    eps = dirichlet_to_normal(alpha, pi, mask)
    pi_recovered = normal_to_dirichlet(alpha, eps, mask)

    print(eps)
    print(pi_recovered)
    # The recovered pi should match the original pi
    assert jnp.allclose(pi, pi_recovered, rtol=1e-4, atol=1e-4), (
        "normal_to_dirichlet should be the inverse of dirichlet_to_normal"
    )

    # Test direction 2: eps -> pi -> eps_recovered
    eps_recovered = dirichlet_to_normal(alpha, pi_recovered, mask)

    # The recovered eps should match the original eps
    assert jnp.allclose(eps, eps_recovered, rtol=1e-4, atol=1e-4), (
        "dirichlet_to_normal should be the inverse of normal_to_dirichlet"
    )

    # Additional checks for numerical stability
    assert jnp.all(jnp.isfinite(eps)), "eps should be finite"
    assert jnp.all(jnp.isfinite(pi_recovered)), "pi_recovered should be finite"
    assert jnp.all(pi_recovered >= 0), "All probabilities should be non-negative"
    assert jnp.allclose(jnp.sum(pi_recovered), 1.0), "Probabilities should sum to 1"

# Test configurations for random cases
RANDOM_TEST_CONFIGS = [
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # pi
        None,  # mask
        id="five_dimensional_uniform",
    ),
    pytest.param(
        jnp.array([2.0, 1.5, 1.0, 0.5, 0.5]),  # alpha
        jnp.array([0.4, 0.3, 0.2, 0.05, 0.05]),  # pi
        None,  # mask
        id="five_dimensional_nonuniform",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]),  # pi
        None,  # mask
        id="six_dimensional_uniform",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]),  # pi
        None,  # mask
        id="seven_dimensional_uniform",
    ),
]

# Test configurations for random masked cases
RANDOM_MASKED_TEST_CONFIGS = [
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.3, 0.2, 0.2, 0.2, 0.1]),  # pi
        jnp.array([True, True, False, True, True]),  # mask
        id="five_dimensional_masked",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),  # pi
        jnp.array([True, False, True, True, False, True]),  # mask
        id="six_dimensional_masked",
    ),
]

@pytest.mark.parametrize(
    "alpha,pi,mask", RANDOM_TEST_CONFIGS + RANDOM_MASKED_TEST_CONFIGS
)
def test_random_normal_dirichlet_invertibility(
    alpha: jnp.ndarray, pi: jnp.ndarray, mask: Optional[jnp.ndarray]
):
    """Test invertibility with random configurations and higher dimensions."""
    # Test direction 1: pi -> eps -> pi_recovered
    eps = dirichlet_to_normal(alpha, pi, mask)
    pi_recovered = normal_to_dirichlet(alpha, eps, mask)

    # The recovered pi should match the original pi
    assert jnp.allclose(pi, pi_recovered, rtol=1e-4, atol=1e-4), (
        "normal_to_dirichlet should be the inverse of dirichlet_to_normal"
    )

    # Test direction 2: eps -> pi -> eps_recovered
    eps_recovered = dirichlet_to_normal(alpha, pi_recovered, mask)

    # The recovered eps should match the original eps
    assert jnp.allclose(eps, eps_recovered, rtol=1e-4, atol=1e-4), (
        "dirichlet_to_normal should be the inverse of normal_to_dirichlet"
    )

    # Additional checks for numerical stability
    assert jnp.all(jnp.isfinite(eps)), "eps should be finite"
    assert jnp.all(jnp.isfinite(pi_recovered)), "pi_recovered should be finite"
    assert jnp.all(pi_recovered >= 0), "All probabilities should be non-negative"
    assert jnp.allclose(jnp.sum(pi_recovered), 1.0), "Probabilities should sum to 1"

    # Test differentiability
    def loss_fn(eps):
        pi = normal_to_dirichlet(alpha, eps, mask)
        return jnp.sum(pi**2)

    grad = jax.grad(loss_fn)(eps)
    assert jnp.all(jnp.isfinite(grad)), "Gradients should be finite"
    assert grad.shape == eps.shape, "Gradient shape should match input shape"

# Test configurations for random gradient tests
RANDOM_GRADIENT_TEST_CONFIGS = [
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0, 0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: jnp.sum(pi**2),  # loss_fn
        id="five_dim_squared_sum",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: jnp.sum(
            pi * jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ),  # weighted sum
        id="six_dim_weighted_sum",
    ),
    pytest.param(
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # alpha
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # eps
        None,  # mask
        lambda pi: jnp.sum(
            pi * jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        ),  # alternating weights
        id="seven_dim_alternating",
    ),
]

@pytest.mark.parametrize("alpha,eps,mask,loss_fn", RANDOM_GRADIENT_TEST_CONFIGS)
def test_random_normal_to_dirichlet_gradients(
    alpha: jnp.ndarray,
    eps: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray], float],
):
    """Test gradients of the normal to Dirichlet transformation with random configurations."""

    def full_loss_fn(eps):
        pi = normal_to_dirichlet(alpha, eps, mask)
        return loss_fn(pi)

    # Compute AD gradient
    grad_fn = jax.grad(full_loss_fn)
    grad_ad = grad_fn(eps)

    # Monte Carlo gradient estimation
    n_samples = 10_000
    key = jax.random.PRNGKey(0)
    grad_mc = jnp.zeros_like(eps)
    eps_mc = 1e-1

    # Process samples in batches
    batch_size = 10_000
    n_batches = n_samples // batch_size

    for i in range(n_batches):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=(batch_size,) + eps.shape) * eps_mc
        eps_perturbed = eps + noise
        losses_perturbed = jax.vmap(full_loss_fn)(eps_perturbed)
        loss_original = full_loss_fn(eps)
        grad_mc += jnp.sum(
            (losses_perturbed - loss_original)[:, None] * noise, axis=0
        ) / (eps_mc * eps_mc)

    grad_mc = grad_mc / n_samples

    # Check gradient properties
    assert jnp.all(jnp.isfinite(grad_ad)), "AD gradient should be finite"
    assert jnp.all(jnp.isfinite(grad_mc)), "MC gradient should be finite"
    assert grad_ad.shape == eps.shape, "AD gradient shape should match input shape"
    assert grad_mc.shape == eps.shape, "MC gradient shape should match input shape"

    # Compare gradients with appropriate tolerance
    rel_error_mc = jnp.abs(grad_ad - grad_mc)
    assert jnp.all(rel_error_mc < 0.2), (
        f"Monte Carlo gradient {grad_mc} differs from AD gradient {grad_ad}"
    )

    # Test gradient direction consistency
    if jnp.linalg.norm(grad_ad) > 0:
        cos_sim_mc = jnp.dot(grad_ad, grad_mc) / (
            jnp.linalg.norm(grad_ad) * jnp.linalg.norm(grad_mc) + 1e-10
        )
        assert cos_sim_mc > 0.9, (
            f"Monte Carlo gradient direction differs significantly from AD gradient. "
            f"Cosine similarity: {cos_sim_mc}"
        )
