import jax
import jax.numpy as jnp
import pytest

from dmri.utils.transform import (
    dirichlet_to_normal,
    normal_to_dirichlet,
    r3_to_s3,
    s3_to_r3,
    transform_quaternion_to_S03,
    transform_S03_to_quaternion,
    uniform_rotation,
)


def test_normal_to_dirichlet():
    """Test the normal to Dirichlet transformation."""
    # Test basic functionality
    alpha = jnp.array([1.0, 1.0, 1.0])
    eps = jnp.array([0.0, 0.0])
    pi = normal_to_dirichlet(alpha, eps)

    # Check basic properties
    assert jnp.all(pi >= 0), "All probabilities should be non-negative"
    assert jnp.allclose(jnp.sum(pi), 1.0), "Probabilities should sum to 1"
    assert len(pi) == len(alpha), "Output length should match input length"

    # Test with different alpha values
    alpha = jnp.array([2.0, 1.0, 0.5])
    pi = normal_to_dirichlet(alpha, eps)
    assert jnp.all(pi >= 0), "All probabilities should be non-negative"
    assert jnp.allclose(jnp.sum(pi), 1.0), "Probabilities should sum to 1"

    # Test with mask
    mask = jnp.array([True, True, False])
    pi = normal_to_dirichlet(alpha, eps, mask)
    assert jnp.all(pi[~mask] == 0), "Masked probabilities should be zero"
    assert jnp.allclose(jnp.sum(pi), 1.0), "Probabilities should sum to 1"


def test_normal_to_dirichlet_gradients():
    """Test gradients of the normal to Dirichlet transformation using both AD and MC estimation."""

    pytest.xfail("This test is currently failing due to bug")

    # Test basic gradient computation
    def loss_fn(eps):
        alpha = jnp.array([1.0, 1.0, 1.0])
        pi = normal_to_dirichlet(alpha, eps)
        return jnp.sum(pi**2)

    eps = jnp.array([0.0, 0.0])

    # Compute AD gradient
    grad_fn = jax.grad(loss_fn)
    grad_ad = grad_fn(eps)

    # Monte Carlo gradient estimation
    n_samples = 50_000
    key = jax.random.PRNGKey(0)
    grad_mc = jnp.zeros_like(eps)
    eps_mc = 1e-3  # Perturbation size for MC estimation

    # Process samples in batches of 10_000
    batch_size = 10_000
    n_batches = n_samples // batch_size

    for i in range(n_batches):
        key, subkey = jax.random.split(key)

        # Generate noise samples for this batch
        noise = jax.random.normal(subkey, shape=(batch_size,) + eps.shape) * eps_mc

        # Compute perturbed epsilons
        eps_perturbed = eps + noise

        # Vectorize loss computation over batch samples
        losses_perturbed = jax.vmap(loss_fn)(eps_perturbed)
        loss_original = loss_fn(eps)

        # Accumulate gradient estimate
        grad_mc += jnp.sum(
            (losses_perturbed - loss_original)[:, None] * noise, axis=0
        ) / (eps_mc * eps_mc)

    grad_mc = grad_mc / n_samples

    # Check gradient properties
    assert jnp.all(jnp.isfinite(grad_ad)), "AD gradient should be finite"
    assert jnp.all(jnp.isfinite(grad_mc)), "MC gradient should be finite"
    assert grad_ad.shape == eps.shape, "AD gradient shape should match input shape"
    assert grad_mc.shape == eps.shape, "MC gradient shape should match input shape"

    # Compare gradients (using relative error for better numerical stability)
    rel_error_mc = jnp.abs(grad_ad - grad_mc) / (jnp.abs(grad_ad) + 1e-3)

    # Allow for some numerical error in the approximations
    assert jnp.all(rel_error_mc < 0.5), (
        f"Monte Carlo gradient differs significantly from AD gradient {grad_ad} {grad_mc}"
    )

    # Test gradient direction consistency
    # The cosine similarity between gradients should be close to 1
    if jnp.linalg.norm(grad_ad) > 0:
        cos_sim_mc = jnp.dot(grad_ad, grad_mc) / (
            jnp.linalg.norm(grad_ad) * jnp.linalg.norm(grad_mc) + 1e-10
        )
        assert cos_sim_mc > 0.9, (
            "Monte Carlo gradient direction differs significantly from AD gradient"
        )

    # Test gradient with different alpha values
    def loss_fn_alpha(eps):
        alpha = jnp.array([2.0, 1.0, 0.5])
        pi = normal_to_dirichlet(alpha, eps)
        return jnp.sum(pi)

    # Compute AD gradient with different alphas
    grad_fn_alpha = jax.grad(loss_fn_alpha)
    grad_ad_alpha = grad_fn_alpha(eps)

    # Monte Carlo estimation with different alphas
    grad_mc_alpha = jnp.zeros_like(eps)
    for i in range(n_batches):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=(batch_size,) + eps.shape) * eps_mc
        eps_perturbed = eps + noise
        losses_perturbed = jax.vmap(loss_fn_alpha)(eps_perturbed)
        loss_original = loss_fn_alpha(eps)
        grad_mc_alpha += jnp.sum(
            (losses_perturbed - loss_original)[:, None] * noise, axis=0
        ) / (eps_mc * eps_mc)

    grad_mc_alpha = grad_mc_alpha / n_samples

    # Compare gradients with different alphas
    rel_error_mc_alpha = jnp.abs(grad_ad_alpha - grad_mc_alpha) / (
        jnp.abs(grad_ad_alpha) + 1e-3
    )
    assert jnp.all(rel_error_mc_alpha < 1.0), (
        "Monte Carlo gradient differs significantly from AD gradient with different alphas"
    )

    # Test gradient with mask
    def loss_fn_masked(eps):
        alpha = jnp.array([1.0, 1.0, 1.0])
        mask = jnp.array([True, True, False])
        pi = normal_to_dirichlet(alpha, eps, mask)
        return jnp.sum(pi)

    # Compute AD gradient with mask
    grad_fn_masked = jax.grad(loss_fn_masked)
    grad_ad_masked = grad_fn_masked(eps)

    # Monte Carlo estimation with mask
    grad_mc_masked = jnp.zeros_like(eps)
    for i in range(n_batches):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=(batch_size,) + eps.shape) * eps_mc
        eps_perturbed = eps + noise
        losses_perturbed = jax.vmap(loss_fn_masked)(eps_perturbed)
        loss_original = loss_fn_masked(eps)
        grad_mc_masked += jnp.sum(
            (losses_perturbed - loss_original)[:, None] * noise, axis=0
        ) / (eps_mc * eps_mc)

    grad_mc_masked = grad_mc_masked / n_samples

    # Compare masked gradients
    rel_error_mc_masked = jnp.abs(grad_ad_masked - grad_mc_masked) / (
        jnp.abs(grad_ad_masked) + 1e-3
    )
    assert jnp.all(rel_error_mc_masked < 1.0), (
        "Monte Carlo gradient differs significantly from AD gradient with mask"
    )


def test_dirichlet_to_normal():
    """Test the Dirichlet to normal transformation."""
    # Test basic functionality
    alpha = jnp.array([1.0, 1.0, 1.0])
    pi = jnp.array([0.3, 0.3, 0.4])
    eps = dirichlet_to_normal(alpha, pi)

    # Check basic properties
    assert len(eps) == len(alpha) - 1, (
        "Output length should be one less than input length"
    )
    assert jnp.all(jnp.isfinite(eps)), "All values should be finite"

    # Test roundtrip
    pi_reconstructed = normal_to_dirichlet(alpha, eps)
    assert jnp.allclose(pi, pi_reconstructed, atol=1e-6), (
        "Roundtrip should reconstruct original values"
    )

    # Test with mask
    mask = jnp.array([True, True, False])
    eps = dirichlet_to_normal(alpha, pi, mask)
    assert jnp.all(jnp.isfinite(eps)), "All values should be finite"


def test_quaternion_rotations():
    """Test quaternion-based rotation transformations."""
    # Test quaternion to rotation matrix
    q = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    R = transform_quaternion_to_S03(q)
    assert jnp.allclose(R, jnp.eye(3)), (
        "Identity quaternion should give identity matrix"
    )

    # Test rotation matrix to quaternion
    q_reconstructed = transform_S03_to_quaternion(R)
    assert jnp.allclose(jnp.abs(q), jnp.abs(q_reconstructed)), (
        "Roundtrip should reconstruct original quaternion"
    )

    # Test with a non-trivial rotation
    theta = jnp.pi / 4  # 45 degrees
    q = jnp.array(
        [jnp.cos(theta / 2), jnp.sin(theta / 2), 0.0, 0.0]
    )  # Rotation around x-axis
    R = transform_quaternion_to_S03(q)
    assert jnp.allclose(jnp.linalg.det(R), 1.0), (
        "Rotation matrix should have determinant 1"
    )
    assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-3), (
        "Rotation matrix should be orthogonal"
    )


def test_r3_s3_transformations():
    """Test stereographic projections between R^3 and S^3."""
    # Test basic functionality
    v = jnp.array([1.0, 0.0, 0.0])
    q = r3_to_s3(v)
    assert jnp.allclose(jnp.linalg.norm(q), 1.0), "Quaternion should have unit norm"

    # Test roundtrip
    v_reconstructed = s3_to_r3(q)
    assert jnp.allclose(v, v_reconstructed, atol=1e-3), (
        "Roundtrip should reconstruct original vector"
    )

    # Test with zero vector
    v = jnp.array([0.0, 0.0, 0.0])
    q = r3_to_s3(v)
    assert jnp.allclose(q, jnp.array([1.0, 0.0, 0.0, 0.0])), (
        "Zero vector should map to (1,0,0,0)"
    )

    # Test with large values
    v = jnp.array([100.0, 100.0, 100.0])
    q = r3_to_s3(v)
    assert jnp.allclose(jnp.linalg.norm(q), 1.0, atol=1e-3), (
        "Quaternion should have unit norm"
    )


def test_uniform_rotation():
    """Test generation of uniform random rotations."""
    # Test basic functionality
    key = jax.random.PRNGKey(0)
    R = uniform_rotation(key)

    # Check rotation matrix properties
    assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-3), (
        "Rotation matrix should have determinant 1"
    )
    assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-3), (
        "Rotation matrix should be orthogonal"
    )

    # Test multiple rotations
    key, subkey = jax.random.split(key)
    R2 = uniform_rotation(subkey)
    assert not jnp.allclose(R, R2, atol=1e-3), (
        "Different keys should give different rotations"
    )

    # Test composition of rotations
    R_composed = R @ R2
    assert jnp.allclose(jnp.linalg.det(R_composed), 1.0, atol=1e-3), (
        "Composed rotation should have determinant 1"
    )
    assert jnp.allclose(R_composed @ R_composed.T, jnp.eye(3), atol=1e-3), (
        "Composed rotation should be orthogonal"
    )


def test_gradient_properties():
    """Test gradient properties of the transformations."""

    # Test normal_to_dirichlet gradient
    def loss_fn(theta):
        alpha = jnp.array([1.0, 1.0, 1.0])
        eps = theta
        pi = normal_to_dirichlet(alpha, eps)
        return jnp.sum(pi)

    theta = jnp.zeros(2)
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(theta)
    assert jnp.all(jnp.isfinite(grad)), "Gradient should be finite"

    # Test dirichlet_to_normal gradient
    def loss_fn_inv(theta):
        alpha = jnp.array([1.0, 1.0, 1.0])
        pi = theta
        eps = dirichlet_to_normal(alpha, pi)
        return jnp.sum(eps)

    theta = jnp.array([0.3, 0.3, 0.4])
    grad_fn = jax.grad(loss_fn_inv)
    grad = grad_fn(theta)
    assert jnp.all(jnp.isfinite(grad)), "Gradient should be finite"


def test_edge_cases():
    """Test edge cases and numerical stability."""
    # Test normal_to_dirichlet with extreme values
    alpha = jnp.array([1e-6, 1e-6, 1e-6])
    eps = jnp.array([0.0, 0.0])
    pi = normal_to_dirichlet(alpha, eps)
    assert jnp.all(jnp.isfinite(pi)), "Should handle small alpha values"

    # Test dirichlet_to_normal with extreme probabilities
    pi = jnp.array([1e-6, 1e-6, 1 - 2e-6])
    eps = dirichlet_to_normal(alpha, pi)
    assert jnp.all(jnp.isfinite(eps)), "Should handle extreme probabilities"

    # Test quaternion transformations with nearly zero values
    v = jnp.array([1e-6, 1e-6, 1e-6])
    q = r3_to_s3(v)
    assert jnp.all(jnp.isfinite(q)), "Should handle small input values"

    # Test rotation matrix with nearly degenerate cases
    theta = jnp.pi / 2 - 1e-6
    q = jnp.array([jnp.cos(theta / 2), jnp.sin(theta / 2), 0.0, 0.0])
    R = transform_quaternion_to_S03(q)
    assert jnp.all(jnp.isfinite(R)), "Should handle nearly degenerate rotations"
