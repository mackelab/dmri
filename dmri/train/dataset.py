"""Streaming simulation dataloader built on top of probjax' ``DataLoader``.

The original implementation relied on a bespoke queueing system that supported
multiple simulator callables, background producer threads, and queue recycling.
The new implementation embraces the ``probjax.nn.io_util.DataLoader`` base
class and focuses on generating batches directly from a single simulator
function. This keeps the public surface compact while making it straightforward
to integrate with the rest of the probjax tooling.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from probjax.nn.io_util import DataLoader
from probjax.utils.typing import Device, RngKey

SimOutput = Any


class SimulationDataset:
    """Fixed-size dataset whose samples are refreshed asynchronously."""

    def __init__(
        self,
        simulator_fn: Callable[..., SimOutput],
        *,
        simulation_batch_size: int = 128,
        rng: RngKey,
        simulation_device: Device,
        jit_simulator: bool = True,
        return_numpy: bool = False,
        buffer_size: int = 8192,
    ) -> None:
        self._simulator_fn = simulator_fn
        self._batch_size = int(simulation_batch_size)
        self._simulation_device = simulation_device
        self._return_numpy = return_numpy

        # Round buffer up to a whole number of batches for clean ring writes.
        self._buffer_batches = max(1, math.ceil(int(buffer_size) / self._batch_size))
        self._dataset_size = self._buffer_batches * self._batch_size

        self._initial_rng = rng
        self._sim_rng = jax.device_put(rng, simulation_device)

        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        self._batched_simulator = self._build_batched_simulator(jit_simulator)
        self._producer: threading.Thread | None = None

        self._buffer: Any | None = None
        self._buffer_leaves: Sequence[np.ndarray] = ()
        self._tree_def = None
        self._write_ptr = 0
        self._pending_refresh = 0

        self._stats = {
            "batches_produced": 0,
            "production_time": 0.0,
            "batches_recycled": 0,
            "samples_written": 0,
            "batches_requested": 0,
            "samples_requested": 0,
        }

        self._initialise_buffer()
        self._start_producer()

    def __del__(self) -> None:
        """Cleanup: stop producer thread when object is deleted."""
        try:
            self.close()
        except Exception:
            # Suppress exceptions during cleanup to avoid errors in __del__
            pass

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

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def reset(self, *, seed: int | None = None, rng: RngKey | None = None) -> None:
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
        """Stop the producer thread and clean up resources."""
        self._stop_producer()
        with self._lock:
            self._buffer = None
            self._buffer_leaves = ()
            self._tree_def = None

    def set_data(self, data: Any) -> None:
        """Replace the internal buffer with user-provided data.

        Parameters
        ----------
        data : Any
            A PyTree of arrays with a leading sample dimension. Leaves must be
            array-like and broadcast-consistent in their first dimension.
        start_producer : bool, default False
            If True, (re)start the background producer after setting the buffer.
            By default we keep the dataset static and the producer stopped.
        """
        # Stop producer while we mutate the buffer
        self._stop_producer()

        # Convert to host NumPy, validate tree & batch dimension
        data_host = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), data)
        leaves = jax.tree_util.tree_leaves(data_host)
        if not leaves:
            raise ValueError("set_data: Provided data has no leaves.")

        # Infer N (samples) and validate consistent leading dim
        try:
            N = int(leaves[0].shape[0])
        except Exception as e:
            raise ValueError("set_data: Could not infer leading dimension.") from e
        for i, lf in enumerate(leaves[1:], start=1):
            if lf.shape[0] != N:
                raise ValueError(
                    f"set_data: Leaf 0 has N={N} but leaf {i} has N={lf.shape[0]}."
                )

        if N <= 0:
            raise ValueError("set_data: Need at least one sample.")

        # Round up to a whole number of batches
        batch_size = self._batch_size
        buffer_batches = max(1, math.ceil(N / batch_size))
        dataset_size = buffer_batches * batch_size

        # Build new buffer with rounded size and copy data (pad by wrap if needed)
        tree_def = jax.tree_util.tree_structure(data_host)
        new_buffer = jax.tree_util.tree_map(
            lambda x: np.empty((dataset_size,) + x.shape[1:], dtype=x.dtype),
            data_host,
        )
        new_buffer_leaves = jax.tree_util.tree_leaves(new_buffer)

        if N == dataset_size:
            # Exact fit
            for buf_leaf, data_leaf in zip(new_buffer_leaves, leaves, strict=True):
                buf_leaf[:] = data_leaf
        else:
            # Copy the N samples, then pad by wrapping from the start
            for buf_leaf, data_leaf in zip(new_buffer_leaves, leaves, strict=True):
                buf_leaf[:N] = data_leaf
                remaining = dataset_size - N
                if remaining > 0:
                    # Wrap (repeat from the start) to fill the last partial batch
                    wrap_src = data_leaf[:remaining % N if N != 0 else 0] if remaining > N else data_leaf[:remaining]
                    # If remaining > N, tile then slice (avoids large loops)
                    if remaining > N:
                        reps = (remaining + N - 1) // N
                        tiled = np.concatenate([data_leaf] * reps, axis=0)
                        buf_leaf[N:] = tiled[:remaining]
                    else:
                        buf_leaf[N:] = wrap_src

        # Reset internal state and stats
        with self._lock:
            self._buffer = new_buffer
            self._buffer_leaves = new_buffer_leaves
            self._tree_def = tree_def
            self._write_ptr = 0
            self._pending_refresh = 0

            # Update size bookkeeping to match the new buffer
            self._buffer_batches = buffer_batches
            self._dataset_size = dataset_size

            # Reset (only) counters that logically depend on production
            self._stats.update({
                "batches_produced": 0,
                "production_time": 0.0,
                "batches_recycled": 0,
                "samples_written": dataset_size,
                "batches_requested": 0,
                "samples_requested": 0,
            })

        # Optionally restart the producer (kept off by default for a fixed dataset)
        self._start_producer()


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
            # Request refresh of entire buffer so producer starts working immediately
            self._pending_refresh = self._dataset_size

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
        # Split keeps arrays on same device as input (simulation_device)
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
            return self._simulator_fn(key)

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
        return jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, device=self._simulation_device), batch
        )

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


def _config_to_mapping(config: Any) -> dict[str, Any]:
    """Convert OmegaConf/Mapping configs into a plain dictionary."""
    if config is None:
        return {}
    if isinstance(config, (DictConfig, ListConfig)):
        container = OmegaConf.to_container(config, resolve=True)
    else:
        container = config
    if container is None:
        return {}
    if isinstance(container, Mapping):
        return dict(container)
    raise TypeError(f"Unsupported dataloader config type {type(config)}")


def instantiate_dataloader(
    dataset: Any,
    loader_cfg: Any,
    *,
    seed: Optional[int] = None,
    default_shuffle: Optional[bool] = None,
    default_drop_last: Optional[bool] = None,
) -> DataLoader:
    """Instantiate a probjax DataLoader with normalised parameters."""
    params = _config_to_mapping(loader_cfg)

    batch_size = params.get("batch_size")
    if batch_size is None:
        raise ValueError("Dataloader configuration must include batch_size.")
    params["batch_size"] = int(batch_size)

    if default_shuffle is not None and "shuffle" not in params:
        params["shuffle"] = bool(default_shuffle)
    if default_drop_last is not None and "drop_last" not in params:
        params["drop_last"] = bool(default_drop_last)

    if seed is not None and params.get("seed") is None:
        params["seed"] = int(seed)

    return DataLoader(dataset, **params)
