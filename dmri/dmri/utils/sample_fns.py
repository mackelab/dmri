import jax
import jax.numpy as jnp


def sample_watson_ar_1(key, mu, kappa):
    """
    Draw a single sample from Watson(mu, kappa) on the unit sphere using
    acceptance-rejection from the uniform distribution on S^{d-1}.

    Arguments:
      key:    a jax.random.PRNGKey
      mu:     a jnp.ndarray of shape (d,) — will be normalized internally
      kappa:  a nonnegative float (concentration parameter)
    Returns:
      A jnp.ndarray of shape (d,) lying on the unit sphere, distributed ~ Watson(mu,kappa).
    """
    mu = mu / jnp.linalg.norm(mu)  # ensure mu is a unit vector

    def cond_fn(state):
        # state = (key, accepted, candidate)
        _, accepted, _ = state
        return jnp.logical_not(accepted)

    def body_fn(state):
        key, _, _ = state
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # 1) Propose x ~ Uniform(S^{d-1})
        z = jax.random.normal(subkey1, shape=mu.shape)
        x_proposal = z / jnp.linalg.norm(z)

        # 2) Acceptance probability
        log_accept_ratio = kappa * ((mu @ x_proposal) ** 2 - 1.0)
        u = jax.random.uniform(subkey2)

        accepted = jnp.log(u) <= log_accept_ratio
        return (key, accepted, x_proposal)

    # Initialize and run the while loop
    init_state = (key, False, jnp.zeros_like(mu))
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    _, _, x_accepted = final_state

    return x_accepted
