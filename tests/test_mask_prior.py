import jax
import jax.numpy as jnp
import numpy as np

from dmri.simulators.mask_prior import (
    BetaBernoulliMaskPrior,
    MarkovMaskPrior,
)


def test_beta_bernoulli_mask_prior_jittable():
    prior = BetaBernoulliMaskPrior(
        num_model_components=4,
        num_noise_components=2,
        alpha=2.0,
        beta=3.0,
        min_active_models=1,
    )

    def sample_pair(rng):
        sample = prior.sample(rng)
        return sample.hyperparameters, sample.model_mask

    sample_fn = jax.jit(sample_pair)
    key = jax.random.PRNGKey(0)
    hyper, mask = sample_fn(key)

    assert hyper.shape == (1,)
    assert mask.shape == (6,)
    assert mask.dtype == jnp.bool_

    log_prob_fn = jax.jit(
        lambda m, h: prior.log_prob(m, h)  # noqa: E731
    )
    lp = log_prob_fn(mask, hyper)
    assert jnp.isfinite(lp)


def test_markov_mask_prior_sampling_jittable():
    transition = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    start_prob = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prior = MarkovMaskPrior(
        num_model_components=3,
        num_noise_components=0,
        transition_matrix=transition,
        start_prob=start_prob,
        walk_length=2,
    )

    def sample_mask(rng):
        sample = prior.sample(rng)
        return sample.model_mask

    sample_fn = jax.jit(sample_mask)
    mask = sample_fn(jax.random.PRNGKey(42))

    assert mask.shape == (3,)
    assert mask.dtype == jnp.bool_
    # Walk length 2 implies exactly two visited nodes
    assert mask.sum() == 2
