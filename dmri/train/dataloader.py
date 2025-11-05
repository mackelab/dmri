"""Streaming simulation dataloader built on top of probjax' ``DataLoader``.

The original implementation relied on a bespoke queueing system that supported
multiple simulator callables, background producer threads, and queue recycling.
The new implementation embraces the ``probjax.nn.io_util.DataLoader`` base
class and focuses on generating batches directly from a single simulator
function. This keeps the public surface compact while making it straightforward
to integrate with the rest of the probjax tooling.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import numpy as np
from probjax.nn.io_util import DataLoader

try:  # JAX changed the type alias for PRNG keys several times.
    from jax.random import KeyArray as PRNGKey
except ImportError:  # pragma: no cover - older JAX versions expose Array only.
    PRNGKey = jax.Array  # type: ignore[assignment]


SimOutput = Any


@dataclass
class _SimulationDataset:
    """Lightweight dataset placeholder expected by ``DataLoader``.

    The base class stores the dataset reference primarily for bookkeeping.
    Simulation-backed loaders never index into the dataset, so returning the
    index itself keeps things harmless if the parent class inspects samples.
    """

    virtual_size: int = 1_000_000

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, index: int) -> int:
        return index


def _resolve_device(kind: str, index: int) -> jax.Device:
    devices = jax.devices(kind)
    if not devices:
        raise RuntimeError(f"No JAX devices available for kind {kind!r}")
    if index >= len(devices):
        raise ValueError(f"Requested device index {index} for kind {kind!r}, but only {len(devices)} available")
    return devices[index]


class StreamDataLoader(DataLoader):
    """Generate simulation batches using a single simulator callable.

    Parameters
    ----------
    simulator_fn:
        Callable receiving a single PRNG key and returning the simulated sample.
    batch_size:
        Number of samples produced per batch.
    seed:
        Optional seed to derive the internal PRNG state. Ignored when ``rng`` is
        provided.
    rng:
        Optional explicit PRNG key. Mutually exclusive with ``seed``.
    num_batches:
        Optional finite length. When omitted the loader produces an infinite
        stream (caller is expected to stop iteration manually).
    simulation_device:
        Device kind used to JIT the simulator (``"cpu"``, ``"gpu"``, ...).
    data_device:
        Optional device kind to move the generated batch to. ``None`` keeps the
        data on the simulator device.
    device_index:
        Which device (of the selected kind) to use.
    jit_simulator:
        Whether to JIT compile the batched simulator.
    simulator_args / simulator_kwargs:
        Extra positional and keyword arguments forwarded to ``simulator_fn``.
    return_numpy:
        Whether to return numpy arrays instead of JAX device arrays.
    dataloader_kwargs:
        Forwarded to ``probjax.nn.io_util.DataLoader``.
    """

    def __init__(
        self,
        simulator_fn: Callable[..., SimOutput],
        *,
        batch_size: int,
        seed: Optional[int] = None,
        rng: Optional[PRNGKey] = None,
        num_batches: Optional[int] = None,
        simulation_device: str = "cpu",
        data_device: Optional[str] = None,
        device_index: int = 0,
        jit_simulator: bool = True,
        simulator_args: Optional[tuple[Any, ...]] = None,
        simulator_kwargs: Optional[Dict[str, Any]] = None,
        return_numpy: bool = True,
        **dataloader_kwargs: Any,
    ) -> None:
        if not callable(simulator_fn):
            raise TypeError("simulator_fn must be callable")
        if rng is not None and seed is not None:
            raise ValueError("Provide either rng or seed, not both.")

        if rng is None:
            seed = 0 if seed is None else int(seed)
            rng = jax.random.PRNGKey(seed)

        self._simulator_fn = simulator_fn
        self._simulator_args = simulator_args or ()
        self._simulator_kwargs = simulator_kwargs or {}
        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self._initial_rng = rng
        self._rng = rng
        self._num_batches = num_batches
        self._return_numpy = return_numpy

        self._simulation_device = _resolve_device(simulation_device, device_index)
        self._data_device = (
            _resolve_device(data_device, device_index) if data_device is not None else None
        )

        self._batched_simulator = self._build_batched_simulator(jit_simulator)

        self._stats = {
            "batches_produced": 0,
            "batches_consumed": 0,
            "production_time": 0.0,
            "queue_wait_time": 0.0,
            "batches_recycled": 0,
        }
        self._closed = False

        # ``DataLoader`` keeps a handle to the dataset even if subclasses do not
        # use it. Passing a virtual dataset with a large length avoids corner
        # cases where the base class might inspect indices.
        super().__init__(_SimulationDataset(), batch_size=self._batch_size, **dataloader_kwargs)

    def __iter__(self) -> Iterator[SimOutput]:
        if self._closed:
            raise RuntimeError("StreamDataLoader cannot be iterated after close()")

        for key in self._index_batches():
            batch = self._process_batch(key)
            self._stats["batches_consumed"] += 1
            yield batch

    def __len__(self) -> int:
        if self._num_batches is None:
            raise TypeError("StreamDataLoader has no finite length; specify num_batches.")
        return self._num_batches

    def __enter__(self) -> "StreamDataLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True

    def reset(self, *, seed: Optional[int] = None, rng: Optional[PRNGKey] = None) -> None:
        """Reset the underlying PRNG stream."""
        if rng is not None and seed is not None:
            raise ValueError("Provide either rng or seed, not both.")
        if rng is None:
            if seed is None:
                rng = self._initial_rng
            else:
                rng = jax.random.PRNGKey(int(seed))
        self._rng = rng
        self._closed = False

    def get_stats(self) -> Dict[str, Any]:
        """Return a shallow copy of the loader statistics."""
        return dict(self._stats)

    # --- probjax ``DataLoader`` hooks -------------------------------------------------

    def _index_batches(self) -> Iterator[PRNGKey]:
        produced = 0
        while self._num_batches is None or produced < self._num_batches:
            self._rng, key = jax.random.split(self._rng)
            produced += 1
            yield key

    def _process_batch(self, key: PRNGKey) -> SimOutput:
        keys = jax.random.split(key, self._batch_size)
        keys = jax.device_put(keys, self._simulation_device)

        start_time = time.perf_counter()
        batch = self._batched_simulator(keys)
        self._stats["batches_produced"] += 1
        self._stats["production_time"] += time.perf_counter() - start_time

        if self._data_device is not None:
            batch = jax.device_put(batch, self._data_device)

        batch = jax.device_get(batch)
        if self._return_numpy:
            batch = jax.tree_util.tree_map(np.asarray, batch)
        return batch

    # --- internal helpers ------------------------------------------------------------

    def _build_batched_simulator(self, jit_simulator: bool) -> Callable[[PRNGKey], SimOutput]:
        def single_call(key: PRNGKey) -> SimOutput:
            return self._simulator_fn(key, *self._simulator_args, **self._simulator_kwargs)

        batched = jax.vmap(single_call)
        if jit_simulator:
            return jax.jit(batched)
        return batched


__all__ = ["StreamDataLoader"]
