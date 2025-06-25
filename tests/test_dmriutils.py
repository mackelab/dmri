import pytest
import jax.numpy as jnp
import numpy as np
from dmri.utils.dmriutils import (
    fit_diffusion_tensor_linearized,
    cartesian_to_unitsphere,
    unitsphere_to_cartesian,
    normalize_bvecs,
    compute_FA,
    compute_MD,
    compute_RD,
    compute_AD,
    make_dyads,
    cart2sph,
    sph2cart,
)


@pytest.fixture
def sample_bvecs():
    """Create sample b-vectors for testing."""
    return jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def sample_bvals():
    """Create sample b-values for testing."""
    return jnp.array([0.0, 1000.0, 1000.0])


@pytest.fixture
def sample_signal():
    """Create sample signal for testing."""
    return jnp.array([1.0, 0.5, 0.5])


def test_fit_diffusion_tensor_linearized(sample_bvecs, sample_bvals, sample_signal):
    """Test the linearized diffusion tensor fitting."""
    logS = jnp.log(sample_signal)
    D = fit_diffusion_tensor_linearized(logS, sample_bvals, sample_bvecs)

    # Check shape
    assert D.shape == (3, 3)
    # Check symmetry
    assert jnp.allclose(D, D.T, atol=1e-6)
    # Check positive definiteness
    assert jnp.all(jnp.linalg.eigvals(D) + 1e-6 > 0)


def test_cartesian_spherical_conversion():
    """Test conversion between Cartesian and spherical coordinates."""
    # Test cartesian_to_unitsphere
    cart = jnp.array([1.0, 0.0, 0.0])
    sph = cartesian_to_unitsphere(cart)
    assert jnp.allclose(sph, jnp.array([jnp.pi / 2, 0.0]), atol=1e-6)

    # Test unitsphere_to_cartesian
    sph = jnp.array([jnp.pi / 2, 0.0])
    cart = unitsphere_to_cartesian(sph)
    assert jnp.allclose(cart, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_normalize_bvecs(sample_bvecs):
    """Test b-vector normalization."""
    # Test with already normalized vectors
    normalized = normalize_bvecs(sample_bvecs)
    assert jnp.allclose(normalized, sample_bvecs, atol=1e-6)

    # Test with non-normalized vectors
    non_normalized = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    normalized = normalize_bvecs(non_normalized)
    norms = jnp.linalg.norm(normalized, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)


def test_diffusion_metrics():
    """Test computation of diffusion metrics."""
    # Create a sample diffusion tensor
    D = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.3]])

    # Test FA
    fa = compute_FA(D)
    assert 0 <= fa <= 1

    # Test MD
    md = compute_MD(D)
    assert md > 0

    # Test RD
    rd = compute_RD(D)
    assert rd > 0
    assert rd < md

    # Test AD
    ad = compute_AD(D)
    assert ad > 0
    assert ad > rd


def test_make_dyads():
    """Test the make_dyads function."""
    # Create sample angles
    theta = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])
    phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])

    v, disp = make_dyads(theta, phi)

    # Check shape of v
    assert v.shape == (3,)

    # Check dispersion
    assert 0 <= disp <= 1

    # Test with percentile
    v, disp = make_dyads(theta, phi, percentile=95)
    assert 0 <= disp <= 180  # dispersion in degrees


def test_cart2sph_sph2cart():
    """Test conversion between Cartesian and spherical coordinates."""
    # Test cart2sph
    x, y, z = 1.0, 0.0, 0.0
    theta, phi = cart2sph(x, y, z)
    assert jnp.allclose(theta, jnp.pi / 2, atol=1e-6)
    assert jnp.allclose(phi, 0.0, atol=1e-6)

    # Test sph2cart
    theta, phi = jnp.pi / 2, 0.0
    cart = sph2cart(theta, phi)
    assert jnp.allclose(cart, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)
