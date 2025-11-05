"""Streaming simulation dataloader built on top of probjax' ``DataLoader``.

The original implementation relied on a bespoke queueing system that supported
multiple simulator callables, background producer threads, and queue recycling.
The new implementation embraces the ``probjax.nn.io_util.DataLoader`` base
class and focuses on generating batches directly from a single simulator
function. This keeps the public surface compact while making it straightforward
to integrate with the rest of the probjax tooling.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from probjax.utils.typing import Device, RngKey

SimOutput = Any


class SimulationDataset:
    """Fixed-size dataset whose samples are refreshed asynchronously."""

    def __init__(
        self,
        simulator_fn: Callable[..., SimOutput],
        *,
        batch_size: int = 128,
        rng: RngKey,
        simulation_device: Device,
        data_device: Optional[Device],
        jit_simulator: bool,
        simulator_args: tuple[Any, ...],
        simulator_kwargs: Dict[str, Any],
        return_numpy: bool = False,
        buffer_size: int = 100_000,
        queue_timeout: Optional[float],
    ) -> None:
        del queue_timeout  # API compatibility; no longer used.

        self._simulator_fn = simulator_fn
        self._simulator_args = simulator_args
        self._simulator_kwargs = simulator_kwargs
        self._batch_size = int(batch_size)
        self._simulation_device = simulation_device
        self._data_device = data_device
        self._return_numpy = return_numpy

        self._buffer_batches = max(1, int(buffer_size))
        self._dataset_size = self._buffer_batches * self._batch_size

        self._initial_rng = rng
        self._sim_rng = jax.device_put(rng, simulation_device)

        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        self._batched_simulator = self._build_batched_simulator(jit_simulator)
        self._producer: Optional[threading.Thread] = None

        self._buffer: Optional[Any] = None
        self._buffer_leaves: Sequence[np.ndarray] = ()
        self._tree_def = None
        self._write_ptr = 0
        self._pending_refresh = 0

        self._stats = {
            "batches_produced": 0,
            "batches_consumed": 0,
            "production_time": 0.0,
            "queue_wait_time": 0.0,
            "batches_recycled": 0,
            "samples_served": 0,
            "samples_written": 0,
            "batches_requested": 0,
            "samples_requested": 0,
        }

        self._initialise_buffer()
        self._start_producer()

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, index: Any) -> Any:
        idxs, squeeze = self._normalise_indices(index)
        with self._lock:
            if self._buffer is None:
                raise RuntimeError("Simulation buffer not initialised.")
            leaves = [leaf[idxs] for leaf in self._buffer_leaves]
            self._stats["batches_requested"] += 1
            self._stats["samples_requested"] += idxs.shape[0]
        batch = jax.tree_util.tree_unflatten(self._tree_def, leaves)
        if squeeze:
            batch = jax.tree_util.tree_map(lambda x: x[0], batch)
        batch = self._convert_for_consumer(batch)
        self._request_refresh(idxs.shape[0])
        return batch

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def reset(
        self, *, seed: Optional[int] = None, rng: Optional[RngKey] = None
    ) -> None:
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
        self._sim_rng = jax.device_put(rng, self._simulation_device)
        self._stop_producer()
        with self._lock:
            self._pending_refresh = 0
        self._initialise_buffer()
        self._start_producer()

    def close(self) -> None:
        self._stop_producer()

    # --- internal helpers ------------------------------------------------------------

    def _initialise_buffer(self) -> None:
        self._stop_event.clear()
        self._write_ptr = 0

        batch, duration = self._produce_batch()
        batch_host = self._to_host(batch)
        batch_leaves = jax.tree_util.tree_leaves(batch_host)

        tree_def = jax.tree_util.tree_structure(batch_host)
        buffer = jax.tree_util.tree_map(
            lambda x: np.empty(
                (self._dataset_size,) + np.asarray(x).shape[1:],
                dtype=np.asarray(x).dtype,
            ),
            batch_host,
        )
        buffer_leaves = jax.tree_util.tree_leaves(buffer)

        if not buffer_leaves:
            raise RuntimeError("Simulator returned an empty batch.")

        for buf_leaf, data_leaf in zip(buffer_leaves, batch_leaves, strict=True):
            reshaped = buf_leaf.reshape(
                (self._buffer_batches, self._batch_size) + data_leaf.shape[1:]
            )
            reshaped[:] = data_leaf

        with self._lock:
            self._buffer = buffer
            self._buffer_leaves = buffer_leaves
            self._tree_def = tree_def
            self._stats["batches_produced"] += 1
            self._stats["production_time"] += duration
            self._stats["samples_written"] += self._batch_size
            self._stats["batches_recycled"] += max(0, self._buffer_batches - 1)
            self._pending_refresh = max(0, self._buffer_batches - 1) * self._batch_size

        with self._condition:
            self._condition.notify_all()

    def _start_producer(self) -> None:
        if self._producer is not None and self._producer.is_alive():
            return
        self._stop_event.clear()
        self._producer = threading.Thread(target=self._producer_main, daemon=True)
        self._producer.start()

    def _stop_producer(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._producer is not None and self._producer.is_alive():
            self._producer.join(timeout=1.0)
        self._producer = None

    def _producer_main(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while (
                    not self._stop_event.is_set()
                    and self._pending_refresh < self._batch_size
                ):
                    self._condition.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                self._pending_refresh -= self._batch_size
            try:
                batch, duration = self._produce_batch()
            except Exception:
                self._stop_event.set()
                raise

            batch_host = self._to_host(batch)
            with self._lock:
                if self._buffer is None:
                    continue
                self._write_batch(batch_host)
                self._stats["batches_produced"] += 1
                self._stats["production_time"] += duration

    def _produce_batch(self) -> tuple[Any, float]:
        self._sim_rng, key = jax.random.split(self._sim_rng)
        keys = jax.random.split(key, self._batch_size)
        start = time.perf_counter()
        batch = self._batched_simulator(keys)
        duration = time.perf_counter() - start
        return batch, duration

    def _build_batched_simulator(
        self, jit_simulator: bool
    ) -> Callable[[RngKey], SimOutput]:
        def single_call(key: RngKey) -> SimOutput:
            return self._simulator_fn(
                key, *self._simulator_args, **self._simulator_kwargs
            )

        batched = jax.vmap(single_call)
        if jit_simulator:
            return jax.jit(batched)
        return batched

    def _to_host(self, batch: Any) -> Any:
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), batch)

    def _write_batch(self, batch_host: Any) -> None:
        if self._buffer is None:
            raise RuntimeError("Simulation buffer not initialised.")
        indices = self._reserve_indices(self._batch_size)
        batch_leaves = jax.tree_util.tree_leaves(batch_host)
        for buf_leaf, data_leaf in zip(self._buffer_leaves, batch_leaves, strict=True):
            buf_leaf[indices] = data_leaf
        self._stats["samples_written"] += len(indices)

    def _reserve_indices(self, count: int) -> np.ndarray:
        start = self._write_ptr
        end = (start + count) % self._dataset_size
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        if start < end or end == 0:
            idxs = np.arange(start, start + count, dtype=np.int64) % self._dataset_size
        else:
            first = np.arange(start, self._dataset_size, dtype=np.int64)
            second = np.arange(0, end, dtype=np.int64)
            idxs = np.concatenate([first, second])
        self._write_ptr = end
        return idxs

    def _convert_for_consumer(self, batch: Any) -> Any:
        if self._return_numpy:
            return batch
        if self._data_device is not None:
            return jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self._data_device), batch
            )
        return jax.tree_util.tree_map(jnp.asarray, batch)

    def _normalise_indices(self, index: Any) -> tuple[np.ndarray, bool]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._dataset_size)
            idxs = np.arange(start, stop, step, dtype=np.int64)
            squeeze = False
        elif isinstance(index, (list, tuple, np.ndarray)):
            arr = np.asarray(index, dtype=np.int64)
            idxs = np.mod(arr, self._dataset_size)
            squeeze = False
        else:
            idx = int(index)
            idxs = np.array([(idx % self._dataset_size)], dtype=np.int64)
            squeeze = True
        return idxs, squeeze

    def _request_refresh(self, count: int) -> None:
        if count <= 0:
            return
        with self._condition:
            self._pending_refresh += count
            self._condition.notify_all()

    def mark_batch_served(self, sample_count: int) -> None:
        with self._lock:
            self._stats["batches_consumed"] += 1
            self._stats["samples_served"] += int(sample_count)
