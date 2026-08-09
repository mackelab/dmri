"""Regression tests for four defects found in the forward model.

Each test fails against the pre-fix code, so they pin the fix rather than the
current behaviour.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmri import simulators
from dmri.simulators.local_signal_models.signal_response_kernels import (
    StickKernel,
    ZeppelinKernel,
)
from dmri.simulators.noise_compartments import add_rician_noise


def test_rician_noise_draws_independent_quadratures():
    """Real and imaginary parts must come from different keys.

    Drawing both from one key made them identical, so the magnitude was
    `|s + n(1 + i)| = sqrt((s+n)^2 + n^2)` rather than Rician. Note the mean
    does not discriminate -- `E[(s+n_r)^2 + n_i^2] = s^2 + 2*sigma^2` either
    way -- so compare against the degenerate form directly.
    """
    key = jax.random.PRNGKey(0)
    signal = jnp.full((4096,), 0.5)
    sigma = 0.1

    noisy = add_rician_noise(key, signal, sigma)

    shared = jax.random.normal(key, shape=signal.shape) * sigma
    degenerate = jnp.abs((signal + shared) + 1j * shared)
    assert not jnp.allclose(noisy, degenerate), (
        "real and imaginary parts still come from one key"
    )

    # With independent quadratures the two are uncorrelated; sharing a key
    # forces correlation 1.
    expected_mean = np.sqrt(0.5**2 + 2 * sigma**2)
    assert abs(float(noisy.mean()) - expected_mean) < 0.02


def test_rician_noise_is_not_degenerate_across_keys():
    a = add_rician_noise(jax.random.PRNGKey(0), jnp.full((256,), 0.5), 0.1)
    b = add_rician_noise(jax.random.PRNGKey(1), jnp.full((256,), 0.5), 0.1)
    assert not jnp.allclose(a, b), "noise does not depend on the key"


@pytest.mark.parametrize("kernel", [StickKernel, ZeppelinKernel])
def test_kernel_diffusivity_bounds_are_ordered(kernel):
    """`StickKernel` had lam_min = 0.01 and lam_max = 0.0, i.e. inverted."""
    assert kernel.lam_min < kernel.lam_max, (
        f"{kernel.__name__} bounds inverted: {kernel.lam_min} !< {kernel.lam_max}"
    )
    assert kernel.lam_min == 0.0
    assert kernel.lam_max == 0.01


def test_zeppelin_kernel_passes_diffusivities_in_order():
    """The kernel transposed lam_par and lam_perp on the way to Zeppelin.

    A Zeppelin attenuates least perpendicular to `mu` when lam_par > lam_perp,
    so transposing the two inverts the anisotropy.
    """
    from dmri.simulators import Zeppelin

    bvals = jnp.asarray([1000.0, 1000.0])
    bvecs = jnp.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    acq = simulators.acquisition_scheme(bvals, bvecs)

    mu_cartesian = jnp.asarray([0.0, 0.0, 1.0])
    lam_par, lam_perp = 0.002, 0.0002

    from_kernel = ZeppelinKernel.kernel_fn(
        acq, mu_cartesian, lam_perp=lam_perp, lam_par=lam_par
    )
    # Ground truth straight from the model, in its own argument order.
    mu_spherical = jnp.asarray([0.0, 0.0])
    direct = jax.vmap(Zeppelin.signal_fn, in_axes=(0, None, None, None))(
        acq, mu_spherical, lam_par, lam_perp
    )
    np.testing.assert_allclose(from_kernel, direct, rtol=1e-6)

    # And the physics: parallel to mu attenuates more than perpendicular.
    assert from_kernel[0] < from_kernel[1]


def test_masked_off_compartment_cannot_poison_the_mixture():
    """`0 * NaN` is NaN, so an inactive compartment used to reach the mixture."""
    from dmri.simulators.multi_compartment import MultiCompartment

    signals = [jnp.asarray([1.0, 1.0]), jnp.asarray([jnp.nan, jnp.nan])]

    class _Fake:
        def __init__(self, value):
            self.value = value

        def signal(self, acq):
            return self.value

    mixed = MultiCompartment._mix_model_signals(
        None, [_Fake(s) for s in signals], jnp.asarray([1.0, 0.0])
    )
    assert jnp.all(jnp.isfinite(mixed)), "NaN from a zero-weighted compartment leaked"
    np.testing.assert_allclose(mixed, signals[0])


def test_nan_log_likelihood_is_not_silenced_to_a_perfect_fit():
    """A NaN log-likelihood must not be reported as 0.

    `jnp.nan_to_num` mapped NaN to 0, and this is a *log* likelihood, so an
    impossible parameter scored likelihood 1 -- the best possible value.
    """
    import inspect

    from dmri.simulators.multi_compartment import MultiCompartment

    source = inspect.getsource(MultiCompartment.log_likelihood)
    assert "nan_to_num" not in source, (
        "log_likelihood still silences NaN into a perfect score"
    )


def test_to_fod_removes_isotropic_components_and_matching_fractions():
    from dmri.simulators.local_signal_models import Ball, Stick
    from dmri.simulators.multi_compartment import MultiCompartment

    class MixedOrderModel(MultiCompartment):
        model_types = [Stick, Ball, Stick]
        noise_types = []
        fraction_prior = jnp.ones(3)

    model = MixedOrderModel(
        model_fractions=jnp.asarray([0.2, 0.3, 0.5]),
        model_compartments=[
            Stick(jnp.asarray([0.0, 0.0]), 0.001),
            Ball(0.001),
            Stick(jnp.asarray([1.0, 0.0]), 0.001),
        ],
        noise_compartments=[],
    )

    fod = model.to_fod(no_isotropic=True)

    assert len(fod.components) == 2
    np.testing.assert_allclose(fod.fractions, jnp.asarray([2 / 7, 5 / 7]))
