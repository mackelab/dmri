import queue
import random  # Add Python's random module
import threading
import time
from functools import partial
from typing import Any, Optional

import jax
import numpy as np


class StreamDataLoader:
    """
    A simple JAX DataLoader-like class that:
      1) Runs background threads to produce CPU data batches.
      2) Prefetches each batch to the GPU (or specified device)
         just-in-time while the GPU is busy training on the previous batch.
    """

    def __init__(
        self,
        simulator_fn,
        batch_size=256,
        simulation_batch_size: Optional[int] = None,
        max_queue_size=10_000,
        seed=0,
        data_device="gpu",
        simulation_device="cpu",
        device_idx=0,
        daemon=True,
        num_producers=1,
        use_inplace_updates=False,  # New parameter for in-place updates
        ring_size=2,  # New parameter for ring buffer size
        queue_timeout=None,  # New parameter for queue timeout
        prefetch_depth=8,  # New parameter for controlling prefetch depth
        recycle_batches=True,  # Whether to recycle batches back into queue
        recycle_threshold=0.5,  # Threshold below which to start recycling (fraction of max queue size)
    ):
        """
        Args:
            simulator_fn: A function simulator(rng) -> batch_of_data
                          or a list of such functions.
                          We'll vmap/jit over it in a background thread.
            rng:          JAX PRNGKey to seed the simulator.
            batch_size:   Number of items per batch.
            simulation_batch_size: Number of items generated per simulator call.
                                   If None, defaults to batch_size.
            max_queue_size: Max items the queue can hold.
            device:       "cpu" or "gpu" (passed to jax.devices()).
            device_idx:   Index of the device (0 for first GPU, etc.).
            daemon:       Whether the producer threads are daemonized.
            num_producers: Number of producer threads to use.
            queue_timeout: Timeout in seconds for queue operations (None = no timeout).
            prefetch_depth: Number of batches to prefetch to device (default=2).
            recycle_batches: Whether to put consumed batches back into the queue.
            recycle_threshold: Queue fullness fraction below which to start recycling (default=0.1).
        """
        # Set Python's random seed
        random.seed(seed)

        # Handle both single simulator and list of simulators
        if isinstance(simulator_fn, list):
            self.simulators = simulator_fn
            self.num_simulators = len(simulator_fn)
        else:
            self.simulators = [simulator_fn]
            self.num_simulators = 1

        self.rng = jax.random.key(seed)
        self.batch_size = batch_size
        self.simulation_batch_size = simulation_batch_size or batch_size
        if self.simulation_batch_size <= 0:
            raise ValueError("simulation_batch_size must be positive.")
        self.data_device = jax.devices(data_device)[device_idx]
        self.simulation_device = jax.devices(simulation_device)[device_idx]
        self.num_producers = max(1, num_producers)  # Ensure at least 1 producer
        self.use_inplace_updates = use_inplace_updates
        self.ring_size = ring_size
        self.recycle_batches = recycle_batches
        self.recycle_threshold = recycle_threshold  # Store the recycle threshold
        self._simulator_cycle: list[int] = []
        self._simulator_cycle_lock = threading.Lock()
        self._recycle_buffer: list[Any] = []
        self._recycle_lock = threading.Lock()

        # Compile the batch simulator once during initialization
        @partial(jax.jit, device=self.simulation_device, static_argnums=(1,))
        def batch_simulator(rng_key, simulator_idx):
            rngs = jax.random.split(rng_key, self.simulation_batch_size)
            return jax.vmap(self.simulators[simulator_idx])(rngs)

        self.batch_simulator = batch_simulator

        # Ensure compilation is complete before proceeding
        # Use a dummy key to trigger compilation for each simulator
        dummy_key = jax.random.key(0)
        for i in range(self.num_simulators):
            _ = self.batch_simulator(dummy_key, i)[0].block_until_ready()

        # For the background threads
        self.event = threading.Event()
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.queue_timeout = queue_timeout
        self.paused = threading.Event()  # For pause/resume functionality
        self.stats = {
            "batches_produced": 0,
            "batches_consumed": 0,
            "batches_recycled": 0,  # Counter for recycled batches
            "production_time": 0.0,
            "queue_wait_time": 0.0,
        }
        self.thread_exceptions = queue.Queue()
        self.prefetch_depth = max(1, prefetch_depth)  # Ensure at least 1

        # Create and start multiple producer threads
        self.producer_threads = []
        # Split the RNG for each thread
        thread_rngs = jax.random.split(self.rng, self.num_producers)

        for i in range(self.num_producers):
            thread_rng = thread_rngs[i]
            thread = threading.Thread(
                target=self._producer_loop,
                args=(thread_rng,),
                daemon=daemon,
                name=f"producer-{i}",
            )
            self.producer_threads.append(thread)
            thread.start()

        # Wait until we have at least one batch in the queue
        # This ensures we don't start with recycled data
        max_wait_time = queue_timeout or 100.0  # Maximum time to wait in seconds
        start_time = time.time()
        while self.queue.empty() and time.time() - start_time < max_wait_time:
            time.sleep(0.1)

        if self.queue.empty():
            raise RuntimeError(
                f"Failed to initialize dataloader: no data produced within timeout ({max_wait_time}s)"
            )

    def _producer_loop(self, thread_rng):
        """
        The background loop that repeatedly:
          - Splits the RNG
          - Produces a batch on the device
          - Transfers it back to CPU (numpy array)
          - Puts it into the CPU queue

        Args:
            thread_rng: The initial RNG key for this thread
        """
        try:
            key = jax.device_put(thread_rng, self.simulation_device)
            while not self.event.is_set():
                if self.paused.is_set():
                    time.sleep(0.01)
                    continue

                self._drain_recycled_segments()

                if self.queue.full():
                    time.sleep(0.01)
                    continue

                start_time = time.time()
                key, rng_sub = jax.random.split(key)
                simulator_idx = self._next_simulator_index()
                data = self.batch_simulator(rng_sub, simulator_idx)

                data_cpu = jax.tree_util.tree_map(np.asarray, data)
                data_cpu = jax.tree_util.tree_map(
                    lambda x: np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=0.0),
                    data_cpu,
                )

                production_time = time.time() - start_time
                with threading.Lock():
                    self.stats["production_time"] += production_time

                timeout = self._queue_timeout()
                while not self.event.is_set():
                    queue_start = time.time()
                    try:
                        self.queue.put(data_cpu, timeout=timeout)
                        with threading.Lock():
                            self.stats["batches_produced"] += 1
                            self.stats["queue_wait_time"] += time.time() - queue_start
                        break
                    except queue.Full:
                        if self.event.is_set():
                            break
                        time.sleep(0.01)

        except Exception as e:
            # Capture exceptions from threads
            self.thread_exceptions.put((threading.current_thread().name, e))
            raise  # Re-raise to see in thread

    def _next_simulator_index(self) -> int:
        """
        Return the next simulator index using a shuffled cycle to ensure coverage
        while keeping random ordering across threads.
        """
        with self._simulator_cycle_lock:
            if not self._simulator_cycle:
                indices = list(range(self.num_simulators))
                random.shuffle(indices)
                self._simulator_cycle.extend(indices)
            return self._simulator_cycle.pop()

    def _recycle_batch(self, batch):
        """
        Recycle a batch by putting it back into the queue.
        Only recycles when queue is below the configured threshold of its capacity.
        """
        if not self.recycle_batches:
            return

        segments = []
        total_size = self._tree_batch_size(batch)
        start = 0
        while start < total_size:
            end = min(start + self.simulation_batch_size, total_size)
            segments.append(self._tree_slice(batch, start, end))
            start = end

        with self._recycle_lock:
            self._recycle_buffer.extend(segments)
            random.shuffle(self._recycle_buffer)

        if segments:
            with threading.Lock():
                self.stats["batches_recycled"] += len(segments)

    def _cpu_data_stream(self):
        """
        A generator that assembles full training batches from simulation-sized
        segments produced by background threads.
        """
        pending_batch = None
        pending_size = 0

        while not self.event.is_set():
            # Flush pending segments into a full batch if possible
            if pending_batch is not None and pending_size >= self.batch_size:
                full_batch = self._tree_slice(pending_batch, 0, self.batch_size)
                remaining_size = pending_size - self.batch_size
                pending_batch = (
                    self._tree_slice(pending_batch, self.batch_size, pending_size)
                    if remaining_size > 0
                    else None
                )
                pending_size = remaining_size
                with threading.Lock():
                    self.stats["batches_consumed"] += 1
                yield full_batch
                if self.recycle_batches:
                    self._recycle_batch(full_batch)
                continue

            # Otherwise, pull the next simulation segment
            if not self.thread_exceptions.empty():
                thread_name, exception = self.thread_exceptions.get()
                raise RuntimeError(
                    f"Exception in producer thread {thread_name}: {exception}"
                )

            self._drain_recycled_segments()

            timeout = self._queue_timeout()
            try:
                segment = self.queue.get(timeout=timeout)
                segment_size = self._tree_batch_size(segment)
                if pending_batch is None:
                    pending_batch = segment
                    pending_size = segment_size
                else:
                    pending_batch = self._tree_concat(pending_batch, segment)
                    pending_size += segment_size
            except queue.Empty:
                if self.event.is_set():
                    break
                time.sleep(0.01)

        # Move any recycled segments into the pending buffer before final drain
        self._drain_recycled_segments()
        if self.recycle_batches and self._recycle_buffer:
            with self._recycle_lock:
                while self._recycle_buffer:
                    segment = self._recycle_buffer.pop()
                    if pending_batch is None:
                        pending_batch = segment
                        pending_size = self._tree_batch_size(segment)
                    else:
                        pending_batch = self._tree_concat(pending_batch, segment)
                        pending_size += self._tree_batch_size(segment)

        # Drain any remaining full batch when shutting down
        while pending_batch is not None and pending_size >= self.batch_size:
            full_batch = self._tree_slice(pending_batch, 0, self.batch_size)
            remaining_size = pending_size - self.batch_size
            pending_batch = (
                self._tree_slice(pending_batch, self.batch_size, pending_size)
                if remaining_size > 0
                else None
            )
            pending_size = remaining_size
            with threading.Lock():
                self.stats["batches_consumed"] += 1
            yield full_batch
            if self.recycle_batches:
                self._recycle_batch(full_batch)

    # --------------------------------------------------------------------------
    # Method A: Device put in prefetch generator
    # --------------------------------------------------------------------------

    def _prefetch_data_stream(self, cpu_stream):
        """
        Generator that:
          - Pulls CPU batches from 'cpu_stream'
          - Asynchronously copies them to `self.device` using `device_put`
          - Prefetches multiple batches to better overlap computation with I/O
          - Yields batches that are already transferred to the device
          - Continues trying to get data even when the stream is temporarily empty
            (which may return backup batches if enabled)

        Note: JAX's device_put is non-blocking by default, which means data transfer
        occurs in the background while computation proceeds.
        """
        stream_iter = iter(cpu_stream)
        prefetch_queue = queue.Queue(maxsize=self.prefetch_depth)

        def prefetch_worker():
            """Background thread that continuously tries to maintain prefetch depth"""
            while not self.event.is_set():
                if prefetch_queue.qsize() < self.prefetch_depth:
                    try:
                        cpu_batch = next(stream_iter)
                        batch_on_device = jax.device_put(cpu_batch, self.data_device)
                        prefetch_queue.put(batch_on_device, timeout=0.1)
                    except (StopIteration, queue.Full):
                        # Brief pause to avoid busy loop
                        time.sleep(0.001)
                else:
                    # Brief pause if queue is full
                    time.sleep(0.001)

        # Start prefetch worker thread
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()
        # Block until the prefetch queue is filled
        while prefetch_queue.qsize() < self.prefetch_depth:
            time.sleep(0.1)
        current_batch = prefetch_queue.get(timeout=0.05)
        yield current_batch
        # Main loop - yield batches as soon as they're available
        while not self.event.is_set():
            # print(prefetch_queue.qsize())
            try:
                # Get batch with minimal timeout to avoid blocking
                current_batch = prefetch_queue.get(timeout=0.05)
                if (
                    prefetch_queue.qsize()
                    < self.prefetch_depth * self.recycle_threshold
                ):
                    prefetch_queue.put(current_batch)
                yield current_batch
            except queue.Empty:
                # Just return whatever is available
                yield current_batch

    # --------------------------------------------------------------------------
    # Method B: In-place updates with ring buffer
    # --------------------------------------------------------------------------

    @staticmethod
    def _tree_batch_size(batch):
        first_leaf = jax.tree_util.tree_leaves(batch)[0]
        return first_leaf.shape[0]

    @staticmethod
    def _tree_concat(batch_a, batch_b):
        if batch_a is None:
            return batch_b
        return jax.tree_util.tree_map(
            lambda x, y: np.concatenate([x, y], axis=0), batch_a, batch_b
        )

    @staticmethod
    def _tree_slice(batch, start, end):
        return jax.tree_util.tree_map(lambda x: np.array(x[start:end], copy=True), batch)

    @staticmethod
    def _copy_inplace(dst, src):
        """
        In-place update: copy src -> dst with .at[..].set(...)
        Works with PyTree structures.
        """

        # Handle PyTree structures - apply the operation to each leaf node
        def copy_leaf(d, s):
            return jax.jit(lambda x, y: x.at[:].set(y), donate_argnums=(0,))(d, s)

        return jax.tree_util.tree_map(copy_leaf, dst, src)

    def _init_ring_buffers(self, first_batch_cpu):
        """
        Allocates ring_size device arrays to match shape/dtype of the first batch.
        Now handles PyTree structures.
        """
        self.gpu_ring_buffers = []

        # Function to create zero array matching shape and dtype
        def create_zeros_like(arr):
            return jax.device_put(
                np.zeros(arr.shape, dtype=arr.dtype), self.data_device
            )

        for _ in range(self.ring_size):
            # Create a zero structure with the same PyTree structure
            zero_pytree = jax.tree_util.tree_map(create_zeros_like, first_batch_cpu)
            self.gpu_ring_buffers.append(zero_pytree)

        self._ring_initialized = True

    def _prefetch_data_stream_inplace(self, cpu_stream):
        """
        Generator that:
          - Allocates ring buffers on first batch (if not done)
          - Copies CPU data into ring buffers in-place
          - Yields them, prefetching the next
          - Continues trying to get data even when the stream is temporarily empty
            (which may return backup batches if enabled)
        """
        stream_iter = iter(cpu_stream)
        ring_idx = 0

        # Try to get first batch
        try:
            first_batch_cpu = next(stream_iter)
        except StopIteration:
            # No data available, try again once
            try:
                time.sleep(0.01)  # Brief pause to allow data production
                first_batch_cpu = next(stream_iter)
            except StopIteration:
                return  # Still no data, exit

        # Init ring buffer
        self._init_ring_buffers(first_batch_cpu)

        # Copy first batch in-place
        buf = self._copy_inplace(self.gpu_ring_buffers[ring_idx], first_batch_cpu)
        self.gpu_ring_buffers[ring_idx] = buf
        yield buf
        ring_idx = (ring_idx + 1) % self.ring_size

        # Main loop - continuously get batches while keeping track of the ring buffer
        while not self.event.is_set():
            try:
                # Try to get next batch (may be from backup if queue is empty)
                batch_cpu = next(stream_iter)

                # Copy to current ring buffer slot and yield
                buf = self._copy_inplace(self.gpu_ring_buffers[ring_idx], batch_cpu)
                self.gpu_ring_buffers[ring_idx] = buf
                yield buf

                # Move to next ring buffer slot
                ring_idx = (ring_idx + 1) % self.ring_size

            except StopIteration:
                # No batches available right now
                if self.event.is_set():
                    # If shutting down, exit
                    break

                # Brief pause to avoid busy loop and let backup mechanism work
                time.sleep(0.01)
                # Continue trying in the next iteration

    def __iter__(self):
        """
        Returns a generator that yields GPU batches, prefetching them in parallel.
        """
        cpu_stream = self._cpu_data_stream()

        # `yield from` the prefetch generator
        if self.use_inplace_updates:
            device_stream = self._prefetch_data_stream_inplace(cpu_stream)
        else:
            device_stream = self._prefetch_data_stream(cpu_stream)

        # Initialize the device stream
        _ = next(device_stream)

        while True:
            yield next(device_stream)

    def close(self):
        """
        Signal all producer threads to exit and wait for them to join.
        """
        self.event.set()
        for thread in self.producer_threads:
            if thread.is_alive():
                # Set a timeout for joining threads
                thread.join(timeout=1.0)
                if thread.is_alive():
                    # Final attempt without timeout to ensure clean shutdown
                    thread.join()

    def __del__(self):
        """
        Make sure the background threads are stopped if the loader is garbage-collected.
        """
        try:
            if hasattr(self, "event"):
                self.close()
        except Exception:
            pass  # Ignore any errors during cleanup

    def queue_size(self):
        """Return the current size of the queue."""
        return self.queue.qsize()

    def queue_empty(self):
        """Check if the queue is empty."""
        return self.queue.empty()

    def queue_full(self):
        """Check if the queue is full."""
        return self.queue.full()

    def get_stats(self):
        """Return statistics about the dataloader operation."""
        return {k: v for k, v in self.stats.items()}

    def pause(self):
        """Temporarily pause data production."""
        self.paused.set()

    def resume(self):
        """Resume data production after pausing."""
        self.paused.clear()

    def _drain_recycled_segments(self) -> None:
        """Push recycled simulation segments back into the queue when there is space."""
        if not self.recycle_batches:
            return
        if not self._recycle_buffer:
            return

        with self._recycle_lock:
            while self._recycle_buffer and not self.queue.full():
                segment = self._recycle_buffer.pop()
                try:
                    self.queue.put_nowait(segment)
                except queue.Full:
                    # Put it back and stop if queue became full mid-loop
                    self._recycle_buffer.append(segment)
                    break

    def _queue_timeout(self) -> float:
        """Return a finite timeout value for queue operations."""
        return self.queue_timeout if self.queue_timeout is not None else 0.1

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting a context manager block."""
        self.close()
        return False  # Don't suppress exceptions
