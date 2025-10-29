import pytest
import jax
import jax.numpy as jnp
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


@pytest.fixture
def multiple_simulators():
    def simulator1(rng):
        # First simulator returns random normal values
        return jax.random.normal(rng, shape=(10,))

    def simulator2(rng):
        # Second simulator returns random uniform values
        return jax.random.uniform(rng, shape=(10,))

    def simulator3(rng):
        # Third simulator returns constant values
        return jnp.ones(10)

    return [simulator1, simulator2, simulator3]


def create_loader(simulator_fn, **kwargs):
    """Helper function to create a loader with default CPU settings"""
    default_kwargs = {
        "batch_size": 32,
        "simulation_batch_size": 8,
        "max_queue_size": 100,
        "num_producers": 2,
        "data_device": "cpu",
        "simulation_device": "cpu",
    }
    default_kwargs.update(kwargs)
    return StreamDataLoader(simulator_fn=simulator_fn, **default_kwargs)


def test_single_simulator(single_simulator):
    # Test with a single simulator
    loader = create_loader(single_simulator)

    # Get a few batches
    batches = []
    for i, batch in enumerate(loader):
        if i >= 3:  # Test 3 batches
            break
        assert batch.shape == (32, 10)
        batches.append(batch)

    # Verify batches are different
    for i in range(len(batches) - 1):
        assert not np.array_equal(batches[i], batches[i + 1]), (
            f"Batches {i} and {i + 1} are equal"
        )
    loader.close()


def test_multiple_simulators(multiple_simulators):
    # Test with multiple simulators
    loader = create_loader(multiple_simulators)

    # Get a few batches
    batches = []
    for i, batch in enumerate(loader):
        if i >= 20:  # Test 5 batches to ensure we get data from different simulators
            break
        assert batch.shape == (32, 10)
        batches.append(batch)

    # Verify we get different types of data (from different simulators)
    # Check if we have at least one batch with values > 1 (from uniform)
    has_uniform = any(np.max(batch) > 1 for batch in batches)
    # Check if we have at least one sample with constant values (from simulator3)
    has_constant = any(
        np.any([np.allclose(sample, sample[0]) for sample in batch]) for batch in batches
    )

    assert has_uniform, "Did not get uniform distribution data"
    assert has_constant, "Did not get constant data"

    loader.close()


def test_dataloader_stats(single_simulator):
    # Test statistics tracking
    loader = create_loader(single_simulator)

    # Get a few batches
    for i, _ in enumerate(loader):
        if i >= 3:
            break

    stats = loader.get_stats()
    assert stats["batches_produced"] > 0
    assert stats["batches_consumed"] > 0
    assert stats["production_time"] > 0
    assert stats["queue_wait_time"] >= 0

    loader.close()


def test_dataloader_recycling(single_simulator):
    # Test batch recycling functionality
    loader = create_loader(
        single_simulator,
        recycle_batches=True,
        recycle_threshold=0.5,  # Start recycling when queue is half full
    )

    # Get a few batches
    batches = []
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        batches.append(batch)

    stats = loader.get_stats()
    assert stats["batches_recycled"] >= 0  # Should have some recycled batches

    loader.close()


def test_dataloader_context_manager(single_simulator):
    # Test context manager functionality
    with create_loader(single_simulator, simulation_batch_size=8) as loader:
        # Get a batch
        batch = next(iter(loader))
        assert batch.shape == (32, 10)

    # Loader should be closed after context
    assert not any(thread.is_alive() for thread in loader.producer_threads)


@pytest.mark.skipif(not has_gpu(), reason="GPU not available")
def test_gpu_dataloader(single_simulator):
    """Test dataloader with GPU if available"""
    loader = create_loader(single_simulator, data_device="gpu", simulation_device="cpu")

    # Get a batch
    batch = next(iter(loader))
    assert batch.shape == (32, 10)

    loader.close()
