"""Streaming simulation dataloader built on top of probjax' ``DataLoader``.

The original implementation relied on a bespoke queueing system that supported
multiple simulator callables, background producer threads, and queue recycling.
The new implementation embraces the ``probjax.nn.io_util.DataLoader`` base
class and focuses on generating batches directly from a single simulator
function. This keeps the public surface compact while making it straightforward
to integrate with the rest of the probjax tooling.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Dict, Optional

import jax
import numpy as np
from probjax.nn.io_util import DataLoader

try:  # JAX changed the type alias for PRNG keys several times.
    from jax.random import KeyArray as PRNGKey
except ImportError:  # pragma: no cover - older JAX versions expose Array only.
    PRNGKey = jax.Array  # type: ignore[assignment]


SimOutput = Any


class _SimulationDataset:
    """Asynchronous dataset that streams simulations into a bounded buffer."""

    def __init__(
        self,
        simulator_fn: Callable[..., SimOutput],
        *,
        batch_size: int,
        rng: PRNGKey,
        simulation_device: jax.Device,
        data_device: Optional[jax.Device],
        jit_simulator: bool,
        simulator_args: tuple[Any, ...],
        simulator_kwargs: Dict[str, Any],
        return_numpy: bool,
        buffer_size: int,
        queue_timeout: Optional[float],
    ) -> None:
        self._virtual_size = 1_000_000
        self._simulator_fn = simulator_fn
        self._simulator_args = simulator_args
        self._simulator_kwargs = simulator_kwargs
        self._batch_size = batch_size
        self._simulation_device = simulation_device
        self._data_device = data_device
        self._return_numpy = return_numpy
        self._buffer_size = max(1, buffer_size)
        self._queue_timeout = queue_timeout

        self._initial_rng = rng
        self._sim_rng = jax.device_put(rng, simulation_device)

        self._queue: queue.Queue[Any] = queue.Queue(self._buffer_size)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._batched_simulator = self._build_batched_simulator(jit_simulator)
        self._producer: Optional[threading.Thread] = None

        self._stats = {
            "batches_produced": 0,
            "batches_consumed": 0,
            "production_time": 0.0,
            "queue_wait_time": 0.0,
            "batches_recycled": 0,
        }

        self._start_producer()

    def __len__(self) -> int:
        return self._virtual_size

    def __getitem__(self, index: int) -> Any:
        start = time.perf_counter()
        try:
            if self._queue_timeout is None:
                batch = self._queue.get()
            else:
                batch = self._queue.get(timeout=self._queue_timeout)
        except queue.Empty as err:
            raise RuntimeError("Timed out waiting for simulation batch.") from err
        wait_time = time.perf_counter() - start
        with self._lock:
            self._stats["queue_wait_time"] += wait_time
            self._stats["batches_consumed"] += 1
        return batch

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def reset(self, *, seed: Optional[int] = None, rng: Optional[PRNGKey] = None) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Provide either rng or seed, not both.")
        if rng is None:
            if seed is None:
                rng = self._initial_rng
            else:
                rng = jax.random.PRNGKey(int(seed))
        else:
            rng = jax.random.PRNGKey(int(rng)) if isinstance(rng, int) else rng
        self._initial_rng = rng
        self._stop_producer()
        self._sim_rng = jax.device_put(rng, self._simulation_device)
        self._queue = queue.Queue(self._buffer_size)
        self._start_producer()

    def close(self) -> None:
        self._stop_producer()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # --- internal helpers ------------------------------------------------------------

    def _start_producer(self) -> None:
        if self._producer is not None and self._producer.is_alive():
            return
        self._stop_event.clear()
        self._producer = threading.Thread(target=self._producer_main, daemon=True)
        self._producer.start()

    def _stop_producer(self) -> None:
        self._stop_event.set()
        if self._producer is not None and self._producer.is_alive():
            self._producer.join(timeout=1.0)
        self._producer = None

    def _producer_main(self) -> None:
        while not self._stop_event.is_set():
            try:
                batch, duration = self._produce_batch()
            except Exception:
                self._stop_event.set()
                raise

            placed = False
            while not placed and not self._stop_event.is_set():
                try:
                    self._queue.put(batch, timeout=0.1)
                    placed = True
                except queue.Full:
                    continue

            with self._lock:
                self._stats["batches_produced"] += 1
                self._stats["production_time"] += duration

    def _produce_batch(self) -> tuple[Any, float]:
        self._sim_rng, key = jax.random.split(self._sim_rng)
        keys = jax.random.split(key, self._batch_size)
        start = time.perf_counter()
        batch = self._batched_simulator(keys)
        duration = time.perf_counter() - start

        if self._data_device is not None:
            batch = jax.device_put(batch, self._data_device)

        batch = jax.device_get(batch)
        if self._return_numpy:
            batch = jax.tree_util.tree_map(np.asarray, batch)

        return batch, duration

    def _build_batched_simulator(self, jit_simulator: bool) -> Callable[[PRNGKey], SimOutput]:
        def single_call(key: PRNGKey) -> SimOutput:
            return self._simulator_fn(
                key, *self._simulator_args, **self._simulator_kwargs
            )

        batched = jax.vmap(single_call)
        if jit_simulator:
            return jax.jit(batched)
        return batched


def _resolve_device(kind: Optional[str], index: int) -> Optional[jax.Device]:
    if kind is None:
        return None
    if isinstance(kind, jax.Device):
        return kind
    try:
        devices = jax.devices(kind)
    except RuntimeError:
        devices = []
    if not devices:
        fallback = jax.devices()
        if fallback:
            if index >= len(fallback):
                raise ValueError(
                    f"Requested device index {index}, but only {len(fallback)} devices available"
                )
            return fallback[index]
        raise RuntimeError(f"No JAX devices available for kind {kind!r}")
    if index >= len(devices):
        raise ValueError(
            f"Requested device index {index} for kind {kind!r}, but only {len(devices)} available"
        )
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
        buffer_size: int = 8,
        queue_timeout: Optional[float] = None,
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

        self._simulation_device = _resolve_device(simulation_device, device_index)
        if self._simulation_device is None:
            raise RuntimeError("simulation_device could not be resolved to a JAX device.")
        self._data_device = _resolve_device(data_device, device_index)

        self._num_batches = num_batches
        self._return_numpy = return_numpy
        self._closed = False

        self._dataset = _SimulationDataset(
            simulator_fn=simulator_fn,
            batch_size=self._batch_size,
            rng=rng,
            simulation_device=self._simulation_device,
            data_device=self._data_device,
            jit_simulator=jit_simulator,
            simulator_args=self._simulator_args,
            simulator_kwargs=self._simulator_kwargs,
            return_numpy=self._return_numpy,
            buffer_size=buffer_size,
            queue_timeout=queue_timeout,
        )

        super().__init__(
            self._dataset, batch_size=self._batch_size, **dataloader_kwargs
        )

    def __iter__(self) -> Iterator[SimOutput]:
        if self._closed:
            raise RuntimeError("StreamDataLoader cannot be iterated after close()")

        super().__iter__()
        self._batches_yielded = 0
        return self

    def __next__(self) -> SimOutput:
        if self._num_batches is not None and self._batches_yielded >= self._num_batches:
            raise StopIteration
        batch = super().__next__()
        self._batches_yielded += 1
        return batch

    def __len__(self) -> int:
        if self._num_batches is None:
            raise TypeError(
                "StreamDataLoader has no finite length; specify num_batches."
            )
        return self._num_batches

    def __enter__(self) -> StreamDataLoader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._dataset.close()
        try:
            super().close()
        except AttributeError:
            pass
        self._closed = True

    def reset(
        self, *, seed: Optional[int] = None, rng: Optional[PRNGKey] = None
    ) -> None:
        """Reset the underlying PRNG stream."""
        self._dataset.reset(seed=seed, rng=rng)
        self._closed = False

    def get_stats(self) -> Dict[str, Any]:
        """Return a shallow copy of the loader statistics."""
        return self._dataset.get_stats()


__all__ = ["StreamDataLoader"]
