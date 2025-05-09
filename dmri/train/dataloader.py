import queue
import threading
import time
from collections import deque
from functools import partial

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
        recycle_threshold=0.1,  # Threshold below which to start recycling (fraction of max queue size)
    ):
        """
        Args:
            simulator_fn: A function simulator(rng) -> batch_of_data
                          or a jitted function that produces a single item.
                          We'll vmap/jit over it in a background thread.
            rng:          JAX PRNGKey to seed the simulator.
            batch_size:   Number of items per batch.
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
        self.simulator_fn = simulator_fn
        self.rng = jax.random.key(seed)
        self.batch_size = batch_size
        self.data_device = jax.devices(data_device)[device_idx]
        self.simulation_device = jax.devices(simulation_device)[device_idx]
        self.num_producers = max(1, num_producers)  # Ensure at least 1 producer
        self.use_inplace_updates = use_inplace_updates
        self.ring_size = ring_size
        self.recycle_batches = recycle_batches
        self.recycle_threshold = recycle_threshold  # Store the recycle threshold

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
            # Wait some time to ensure all threads are started
            time.sleep(0.1)

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
            # We'll jit+vmap the user simulator to produce batch_size items at once
            @partial(jax.jit, device=self.simulation_device)
            def batch_simulator(rng_key):
                rngs = jax.random.split(rng_key, self.batch_size)
                return jax.vmap(self.simulator_fn)(rngs)

            key = jax.device_put(thread_rng, self.simulation_device)
            while not self.event.is_set():
                # Wait if production is paused
                if self.paused.is_set():
                    time.sleep(0.1)
                    continue

                # Generate a batch
                start_time = time.time()
                key, rng_sub = jax.random.split(key)
                data = batch_simulator(rng_sub)

                # PyTree-friendly conversion to CPU
                # This handles cases where data is a nested structure (PyTree)
                data_cpu = data  # jax.tree_map(np.array, data)

                production_time = time.time() - start_time
                with threading.Lock():
                    self.stats["production_time"] += production_time

                # Blocks if queue is full
                queue_start = time.time()
                try:
                    self.queue.put(data_cpu, timeout=self.queue_timeout)
                    with threading.Lock():
                        self.stats["batches_produced"] += 1
                        self.stats["queue_wait_time"] += time.time() - queue_start
                except queue.Full:
                    # Just continue trying if we hit a timeout
                    continue
        except Exception as e:
            # Capture exceptions from threads
            self.thread_exceptions.put((threading.current_thread().name, e))
            raise  # Re-raise to see in thread

    def _recycle_batch(self, batch):
        """
        Recycle a batch by putting it back into the queue.
        Only recycles when queue is below the configured threshold of its capacity.
        """
        if not self.recycle_batches:
            return

        # Only recycle if queue is below the configured threshold capacity
        current_fullness = self.queue.qsize() / self.queue.maxsize
        if current_fullness >= self.recycle_threshold:
            return

        try:
            # Use non-blocking put to avoid deadlocks if queue is full
            self.queue.put_nowait(batch)
            with threading.Lock():
                self.stats["batches_recycled"] += 1
        except queue.Full:
            # Queue is full, drop the batch
            pass

    def _cpu_data_stream(self):
        """
        A generator that yields batches from the CPU queue.
        Recycles batches back into the queue if enabled.
        Ends if the event is set or a thread exception occurred.
        """
        while not self.event.is_set():
            # Check for thread exceptions
            if not self.thread_exceptions.empty():
                thread_name, exception = self.thread_exceptions.get()
                raise RuntimeError(
                    f"Exception in producer thread {thread_name}: {exception}"
                )

            # Try to get batch from queue
            try:
                batch_cpu = self.queue.get(timeout=self.queue_timeout)
                with threading.Lock():
                    self.stats["batches_consumed"] += 1

                yield batch_cpu

                # Recycle the batch by putting it back into the queue
                if self.recycle_batches:
                    self._recycle_batch(batch_cpu)

            except queue.Empty:
                # If queue is empty, check if we should exit
                if self.event.is_set():
                    break
                # Otherwise, continue to try again
                continue

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
        prefetch_queue = deque(maxlen=self.prefetch_depth)

        # Initial prefetching phase - fill the prefetch queue
        for _ in range(self.prefetch_depth):
            try:
                cpu_batch = next(stream_iter)
                # Non-blocking device transfer
                batch_on_device = jax.device_put(cpu_batch, self.data_device)
                prefetch_queue.append(batch_on_device)
            except StopIteration:
                break

        # If no batches were prefetched, we're done
        if not prefetch_queue:
            return

        # Main loop - yield current batch while prefetching next
        while prefetch_queue or not self.event.is_set():
            # If we have batches to yield, do so
            if prefetch_queue:
                # Get the next batch to yield (oldest prefetched batch)
                current_batch = prefetch_queue.popleft()
                yield current_batch
            else:
                # No more batches in queue but not shutting down yet
                # Small wait to avoid busy loop
                time.sleep(0.01)

            # Try to prefetch more to maintain prefetch_depth
            while len(prefetch_queue) < self.prefetch_depth and not self.event.is_set():
                try:
                    cpu_batch = next(stream_iter)
                    batch_on_device = jax.device_put(cpu_batch, self.data_device)
                    prefetch_queue.append(batch_on_device)
                except StopIteration:
                    # Stream temporarily exhausted, try again next loop
                    # This allows getting backup batches if the queue was empty
                    break

    # --------------------------------------------------------------------------
    # Method B: In-place updates with ring buffer
    # --------------------------------------------------------------------------
    @staticmethod
    def _copy_inplace(dst, src):
        """
        In-place update: copy src -> dst with .at[..].set(...)
        Works with PyTree structures.
        """

        # Handle PyTree structures - apply the operation to each leaf node
        def copy_leaf(d, s):
            return jax.jit(lambda x, y: x.at[:].set(y), donate_argnums=(0,))(d, s)

        return jax.tree_map(copy_leaf, dst, src)

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
            zero_pytree = jax.tree_map(create_zeros_like, first_batch_cpu)
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
            yield from self._prefetch_data_stream_inplace(cpu_stream)
        else:
            yield from self._prefetch_data_stream(cpu_stream)

    def close(self):
        """
        Signal all producer threads to exit and wait for them to join.
        """
        self.event.set()
        for thread in self.producer_threads:
            if thread.is_alive():
                thread.join()

    def __del__(self):
        """
        Make sure the background threads are stopped if the loader is garbage-collected.
        """
        self.close()

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

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting a context manager block."""
        self.close()
        return False  # Don't suppress exceptions
