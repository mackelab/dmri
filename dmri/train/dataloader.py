import threading
import queue
import time
from functools import partial
from contextlib import contextmanager

import jax
import jax.numpy as jnp
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
        """
        self.simulator_fn = simulator_fn
        self.rng = jax.random.key(seed)
        self.batch_size = batch_size
        self.data_device = jax.devices(data_device)[device_idx]
        self.simulation_device = jax.devices(simulation_device)[device_idx]
        self.num_producers = max(1, num_producers)  # Ensure at least 1 producer
        self.use_inplace_updates = use_inplace_updates
        self.ring_size = ring_size

        # For the background threads
        self.event = threading.Event()
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.queue_timeout = queue_timeout
        self.paused = threading.Event()  # For pause/resume functionality
        self.stats = {
            "batches_produced": 0,
            "batches_consumed": 0,
            "production_time": 0.0,
            "queue_wait_time": 0.0,
        }
        self.thread_exceptions = queue.Queue()

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
                data_cpu = jax.tree_map(np.array, data)

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

    def _cpu_data_stream(self):
        """
        A generator that yields batches from the CPU queue.
        Ends if the event is set or a thread exception occurred.
        """
        while not self.event.is_set():
            # Check for thread exceptions
            if not self.thread_exceptions.empty():
                thread_name, exception = self.thread_exceptions.get()
                raise RuntimeError(
                    f"Exception in producer thread {thread_name}: {exception}"
                )

            # Blocks if queue is empty
            try:
                batch_cpu = self.queue.get(timeout=self.queue_timeout)
                with threading.Lock():
                    self.stats["batches_consumed"] += 1
                yield batch_cpu
            except queue.Empty:
                # If we hit a timeout, check if we should exit
                if self.event.is_set():
                    break
                continue

    # --------------------------------------------------------------------------
    # Method A: Device put in prefetch generator
    # --------------------------------------------------------------------------

    def _prefetch_data_stream(self, cpu_stream):
        """
        Generator that:
          - Pulls CPU batches from 'cpu_stream'
          - Asynchronously copies them to `self.device` using `device_put`
          - Yields them (already on the device)
          - Prefetches the next batch in parallel
        """
        stream_iter = iter(cpu_stream)

        # Try to grab first batch
        try:
            # PyTree-friendly device put
            next_batch_on_gpu = jax.device_put(next(stream_iter), self.data_device)
        except StopIteration:
            return  # If no data, we're done

        while True:
            current_batch = next_batch_on_gpu  # The batch to yield
            # Prefetch the next batch
            try:
                cpu_batch = next(stream_iter)
                next_batch_on_gpu = jax.device_put(cpu_batch, self.data_device)
            except StopIteration:
                next_batch_on_gpu = None

            yield current_batch

            if next_batch_on_gpu is None:
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
                np.zeros(arr.shape, dtype=arr.dtype),
                self.data_device
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
        """
        stream_iter = iter(cpu_stream)
        ring_idx = 0

        # Get the first batch
        try:
            first_batch_cpu = next(stream_iter)
        except StopIteration:
            return

        # Init ring buffer
        self._init_ring_buffers(first_batch_cpu)

        # Copy first batch in-place
        buf = self._copy_inplace(self.gpu_ring_buffers[ring_idx], first_batch_cpu)
        self.gpu_ring_buffers[ring_idx] = buf
        yield buf
        ring_idx = (ring_idx + 1) % self.ring_size

        # Subsequent batches
        for batch_cpu in stream_iter:
            buf = self._copy_inplace(self.gpu_ring_buffers[ring_idx], batch_cpu)
            self.gpu_ring_buffers[ring_idx] = buf
            yield buf
            ring_idx = (ring_idx + 1) % self.ring_size

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
