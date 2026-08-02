"""Pin the forward model's numeric output across refactors.

The simulator is what the pretrained checkpoints were trained against, so a
refactor that changes its output silently invalidates them. These tests compare
every model against a stored reference; see ``tests/simulator_reference.py`` for
how to regenerate it when a change is *meant* to alter the output.
"""

import numpy as np
import pytest
from simulator_reference import (
    MODEL_NAMES,
    load_reference,
    outputs_for,
)

# float32 through a long chain of transforms; tight enough to catch a real
# change, loose enough to survive XLA choosing different kernels.
RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def test_reference_covers_every_model(reference):
    """A new model must be added to the reference, not silently unguarded."""
    covered = {key.split(".", 1)[0] for key in reference}
    assert covered == set(MODEL_NAMES), (
        f"reference is stale: {set(MODEL_NAMES) ^ covered}"
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_output_is_unchanged(name, reference):
    """Every quantity of every model still matches the stored reference."""
    actual = outputs_for(name)

    expected_keys = {k for k in reference if k.startswith(f"{name}.")}
    assert set(actual) == expected_keys, (
        f"{name} gained or lost outputs: {set(actual) ^ expected_keys}"
    )

    for key, value in sorted(actual.items()):
        np.testing.assert_allclose(
            value,
            reference[key],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{key} changed",
        )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_output_is_finite(name):
    """No model should produce NaN or inf from an ordinary theta."""
    for key, value in outputs_for(name).items():
        if key.endswith(".log_signal") or key.endswith(".log_likelihood"):
            # Logs of a legitimately zero signal are -inf; only NaN is a defect.
            assert not np.isnan(value).any(), f"{key} contains NaN"
        else:
            assert np.isfinite(value).all(), f"{key} is not finite"


def test_repeated_construction_is_deterministic():
    """Same theta twice must give the same signal.

    This is the property the class-attribute mutation for shared parameters puts
    at risk, so it is worth pinning explicitly.
    """
    name = "Ball3StickSharedDiffusivity"
    first = outputs_for(name)
    second = outputs_for(name)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
