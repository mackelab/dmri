import jax
import jax.numpy as jnp

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
