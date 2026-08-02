"""Deterministic reference outputs for every simulator model.

Used by ``tests/test_simulator_regression.py`` to pin the forward model's numeric
output across refactors. The capture and the check share this module so they
cannot drift apart.

Regenerate with::

    python tests/simulator_reference.py

Only regenerate when a change is *intended* to alter the simulator's output, and
say which models changed and why in the commit.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from dmri import simulators
from dmri.simulators.multi_compartment import _MODEL_CLASS_NAMES

REFERENCE_PATH = pathlib.Path(__file__).parent / "data" / "simulator_reference.npz"

#: Models whose graphs take minutes to compile; excluded to keep the suite usable.
SLOW_MODELS = frozenset({"AllGaussianAndConvolvedModels"})

MODEL_NAMES = tuple(sorted(_MODEL_CLASS_NAMES - SLOW_MODELS))


#: SSFP models read fields a plain scheme does not have.
SSFP_MODELS = frozenset({
    "SSFPBall3StickSharedDiffusivity",
    "SSFPBall3StickSharedDiffusivityBetterNorm",
})


def reference_ssfp_acquisition():
    """A fixed SSFP scheme, drawn once from the package's own factory."""
    return simulators.random_ssfp_acquisition(
        jax.random.PRNGKey(0), num_acquisitions=14
    )


def reference_acquisition():
    """A fixed acquisition scheme: 2 b0s plus two shells, deterministic bvecs."""
    bvals = jnp.asarray([0.0, 0.0] + [1000.0] * 6 + [2000.0] * 6)
    # Deterministic and well spread, without depending on a sampler.
    raw = jnp.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    bvecs = jnp.concatenate([jnp.zeros((2, 3)), raw, raw], axis=0)
    norms = jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvecs = jnp.where(norms > 0, bvecs / jnp.where(norms > 0, norms, 1.0), bvecs)
    return simulators.acquisition_scheme(bvals, bvecs)


def _theta_for(sim_type, seed):
    return jax.random.normal(jax.random.PRNGKey(seed), (sim_type.theta_dim,))


def outputs_for(name, seed=0):
    """Every numeric output of one model, as plain float32 arrays.

    Returns a mapping of ``"<name>.<quantity>" -> array``. Quantities that a
    model does not support are omitted rather than faked, so a model gaining or
    losing one shows up as a key change.
    """
    sim_type = getattr(simulators, name)
    theta = _theta_for(sim_type, seed)
    simulator = sim_type.from_theta(theta)
    acq = (
        reference_ssfp_acquisition() if name in SSFP_MODELS else reference_acquisition()
    )

    results = {
        f"{name}.theta": np.asarray(theta, dtype=np.float32),
        f"{name}.fractions": np.asarray(simulator.model_fractions, dtype=np.float32),
        f"{name}.signal": np.asarray(simulator.signal(acq), dtype=np.float32),
        f"{name}.log_signal": np.asarray(simulator.log_signal(acq), dtype=np.float32),
        f"{name}.theta_mask": np.asarray(
            sim_type.theta_mask(model_mask=None), dtype=np.float32
        ),
    }

    # A second draw through the same instance, to pin noise behaviour too.
    if sim_type.noise_types:
        noisy = simulator.signal(acq, rng=jax.random.PRNGKey(seed + 1))
        results[f"{name}.noisy_signal"] = np.asarray(noisy, dtype=np.float32)

    observed = jnp.abs(results[f"{name}.signal"])
    results[f"{name}.log_likelihood"] = np.asarray(
        simulator.log_likelihood(acq, observed), dtype=np.float32
    )
    return results


def capture(names=MODEL_NAMES, seed=0):
    """Reference outputs for ``names``, keyed ``"<model>.<quantity>"``."""
    captured = {}
    for name in names:
        captured.update(outputs_for(name, seed=seed))
    return captured


def load_reference():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"{REFERENCE_PATH} is missing; regenerate with "
            "`python tests/simulator_reference.py`"
        )
    with np.load(REFERENCE_PATH) as data:
        return {key: data[key] for key in data.files}


def main():
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    captured = capture()
    np.savez_compressed(REFERENCE_PATH, **captured)
    models = len({key.split(".", 1)[0] for key in captured})
    print(f"wrote {len(captured)} arrays for {models} models -> {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
