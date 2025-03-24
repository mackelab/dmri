from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

# from jax.scipy.stats import ks_2samp


def run_tarp(
    thetas_true: jnp.ndarray,
    posterior_samples: jnp.ndarray,
    references: Optional[jnp.ndarray] = None,
    distance: Callable = jnp.linalg.norm,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of the TARP method.
    """
    num_tarp_samples, dim_theta = thetas_true.shape

    if references is None:
        references = get_tarp_references(thetas_true)

    return _run_tarp(
        posterior_samples, thetas_true, references, distance, num_bins, z_score_theta
    )


def _run_tarp(
    posterior_samples: jnp.ndarray,
    thetas: jnp.ndarray,
    references: jnp.ndarray,
    distance: Callable = jnp.linalg.norm,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of the TARP diagnostic.
    """
    num_posterior_samples, num_tarp_samples, _ = posterior_samples.shape
    assert references.shape == thetas.shape

    if num_bins is None:
        num_bins = num_tarp_samples // 10

    if z_score_theta:
        lo = jnp.min(thetas, axis=0, keepdims=True)
        hi = jnp.max(thetas, axis=0, keepdims=True)
        posterior_samples = (posterior_samples - lo) / (hi - lo + 1e-10)
        thetas = (thetas - lo) / (hi - lo + 1e-10)

    sample_dists = distance(references - posterior_samples, axis=-1)
    theta_dists = distance(references - thetas, axis=-1)

    coverage_values = (
        jnp.sum(sample_dists < theta_dists, axis=0) / num_posterior_samples
    )
    hist, alpha_grid = jnp.histogram(coverage_values, bins=num_bins, density=True)
    ecp = jnp.cumsum(hist) / jnp.sum(hist)
    ecp = jnp.concatenate([jnp.array([0]), ecp])

    return ecp, alpha_grid


def get_tarp_references(thetas: jnp.ndarray) -> jnp.ndarray:
    """Returns reference points for the TARP diagnostic, sampled from a uniform."""
    lo = jnp.min(thetas, axis=0)
    hi = jnp.max(thetas, axis=0)
    return jax.random.uniform(
        jax.random.PRNGKey(0), shape=thetas.shape, minval=lo, maxval=hi
    )


def check_tarp(
    ecp: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[float, float]:
    """
    JAX implementation to check the TARP credibility levels.
    """
    midindex = alpha.shape[0] // 2
    atc = jnp.sum(ecp[midindex:] - alpha[midindex:])
    ks_prob = ks_2samp(ecp, alpha).pvalue
    return atc, ks_prob
