import pytest
import jax
import numpy as np
from dmri.train.dataloader import StreamDataLoader


def has_gpu():
    """Check if GPU is available without raising an error"""
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


@pytest.fixture
def single_simulator():
    def simulator(rng):
        # Simple simulator that returns a random array
        return jax.random.normal(rng, shape=(10,))

    return simulator


def create_loader(simulator_fn, **kwargs):
    """Helper function to create a loader with default CPU settings"""
    default_kwargs = {
        "batch_size": 16,
        "seed": 0,
        "num_batches": 4,
        "simulation_device": "cpu",
        "data_device": None,
        "jit_simulator": True,
    }
    default_kwargs.update(kwargs)
    if "rng" in default_kwargs and default_kwargs["rng"] is not None:
        default_kwargs.pop("seed", None)
    loader = StreamDataLoader(simulator_fn=simulator_fn, **default_kwargs)
    # probjax' DataLoader base class re-initialises its own RNG, so we need
    # to restore the JAX key before the first iteration.
    loader.reset()
    return loader


def test_single_simulator(single_simulator):
    loader = create_loader(single_simulator)

    batches = list(loader)
    assert len(batches) == 4
    for batch in batches:
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (16, 10)
        assert not np.all(np.isnan(batch))
    loader.close()


def test_dataloader_stats(single_simulator):
    loader = create_loader(single_simulator)

    _ = list(loader)
    stats = loader.get_stats()
    assert stats["batches_consumed"] == 4
    assert stats["batches_produced"] >= stats["batches_consumed"]
    assert stats["production_time"] >= 0
    assert stats["queue_wait_time"] >= 0

    loader.close()


def test_dataloader_reset_allows_reuse(single_simulator):
    loader = create_loader(single_simulator, seed=5)

    first_run = list(loader)
    loader.reset(seed=5)
    second_run = list(loader)

    assert len(first_run) == 4
    assert len(second_run) == 4
    for batch in second_run:
        assert batch.shape == (16, 10)

    loader.close()


def test_dataloader_seed_influences_outputs(single_simulator):
    loader_a = create_loader(single_simulator, seed=5)
    run_a = list(loader_a)
    loader_a.close()

    loader_b = create_loader(single_simulator, seed=42)
    run_b = list(loader_b)
    loader_b.close()

    assert len(run_a) == len(run_b) == 4
    assert any(not np.array_equal(a, b) for a, b in zip(run_a, run_b))


def test_dataloader_context_manager(single_simulator):
    with create_loader(single_simulator, num_batches=1) as loader:
        batch = next(iter(loader))
        assert batch.shape == (16, 10)

    with pytest.raises(RuntimeError):
        next(iter(loader))


@pytest.mark.skipif(not has_gpu(), reason="GPU not available")
def test_gpu_dataloader(single_simulator):
    loader = create_loader(
        single_simulator,
        data_device="gpu",
        simulation_device="gpu",
        num_batches=1,
    )

    batch = next(iter(loader))
    assert batch.shape == (16, 10)

    loader.close()


def test_dataloader_custom_rng(single_simulator):
    rng = jax.random.PRNGKey(7)
    loader = create_loader(single_simulator, rng=rng, num_batches=3)

    first_run = list(loader)
    loader.reset(rng=jax.random.PRNGKey(11))
    second_run = list(loader)

    assert len(first_run) == 3
    assert len(second_run) == 3
    assert any(not np.array_equal(a, b) for a, b in zip(first_run, second_run))

    loader.close()

    with pytest.raises(ValueError):
        StreamDataLoader(
            simulator_fn=single_simulator,
            batch_size=4,
            seed=0,
            rng=jax.random.PRNGKey(0),
            num_batches=1,
        )


def test_dataloader_infinite_stream(single_simulator):
    loader = StreamDataLoader(
        simulator_fn=single_simulator,
        batch_size=8,
        seed=0,
        simulation_device="cpu",
        data_device=None,
    )
    loader.reset()

    batches = []
    for idx, batch in enumerate(loader):
        batches.append(batch)
        if idx == 2:
            break

    assert len(batches) == 3
    assert all(b.shape == (8, 10) for b in batches)
    loader.close()
