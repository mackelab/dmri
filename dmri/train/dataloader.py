import queue
import threading
import time
from collections import deque
from functools import partial

import jax
import numpy as np


class StreamDataLoader:
    """
    A JAX DataLoader-like class that:
      1) Runs background threads to produce and update a shared data corpus.
      2) Draws batches from the corpus for training.
      3) Prefetches each batch to the GPU (or specified device)
         just-in-time while the GPU is busy training on the previous batch.
    """

    def __init__(
        self,
        simulator_fn,
        batch_size=256,
        corpus_size=10_000,  # Size of the shared corpus
        update_batch_size=64,  # Smaller batch size for corpus updates
        max_queue_size=10_000,
        seed=0,
        data_device="gpu",
        simulation_device="cpu",
        device_idx=0,
        daemon=True,
        num_producers=1,
        use_inplace_updates=False,
        ring_size=2,
        queue_timeout=None,
        prefetch_depth=3,
        update_lock_free=True,  # New option to enable lock-free corpus updates
        pre_batch_size=5,       # Number of pre-generated batches per producer
        enable_profiling=False, # Enable detailed performance profiling
    ):
        """
        Args:
            simulator_fn: A function simulator(rng) -> batch_of_data
                          or a jitted function that produces a single item.
                          We'll vmap/jit over it in a background thread.
            batch_size:   Number of items per batch for training.
            corpus_size:  Total size of the shared corpus.
            update_batch_size: Number of items per update batch (smaller than batch_size).
            max_queue_size: Max items the queue can hold.
            data_device:  "cpu" or "gpu" (passed to jax.devices()).
            simulation_device: Device to run simulations on.
            device_idx:   Index of the device (0 for first GPU, etc.).
            daemon:       Whether the producer threads are daemonized.
            num_producers: Number of producer threads to use.
            use_inplace_updates: Whether to use in-place updates for prefetching.
            ring_size:    Size of the ring buffer for in-place updates.
            queue_timeout: Timeout in seconds for queue operations (None = no timeout).
            prefetch_depth: Number of batches to prefetch to device.
            update_lock_free: Whether to use lock-free updates for the corpus.
            pre_batch_size: Number of pre-generated batches per producer.
            enable_profiling: Whether to enable detailed performance profiling.
        """
        self.simulator_fn = simulator_fn
        self.rng = jax.random.key(seed)
        self.batch_size = batch_size
        self.corpus_size = corpus_size
        self.update_batch_size = min(update_batch_size, batch_size)  # Ensure update_batch_size <= batch_size
        self.data_device = jax.devices(data_device)[device_idx]
        self.simulation_device = jax.devices(simulation_device)[device_idx]
        self.num_producers = max(1, num_producers)  # Ensure at least 1 producer
        self.use_inplace_updates = use_inplace_updates
        self.ring_size = ring_size
        self.update_lock_free = update_lock_free
        self.pre_batch_size = pre_batch_size
        self.enable_profiling = enable_profiling

        # For the background threads
        self.event = threading.Event()
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.queue_timeout = queue_timeout
        self.paused = threading.Event()  # For pause/resume functionality

        # Statistics tracking
        self.stats = {
            "batches_produced": 0,
            "batches_consumed": 0,
            "production_time": 0.0,
            "queue_wait_time": 0.0,
            "corpus_updates": 0,
        }

        # Detailed performance profiling
        if self.enable_profiling:
            self.profiling_stats = {
                "simulation_time": 0.0,
                "update_time": 0.0,
                "extraction_time": 0.0,
                "lock_wait_time": 0.0,
                "lock_held_time": 0.0,
            }

        self.thread_exceptions = queue.Queue()
        self.prefetch_depth = max(1, prefetch_depth)  # Ensure at least 1

        # Initialize the shared corpus
        self.corpus_lock = threading.Lock()
        self.corpus = None  # Will be initialized with the first batch
        self.corpus_indices = {}  # Maps thread ID to its assigned section of the corpus

        # Create separate locks for each section of the corpus if using fine-grained locking
        if self.update_lock_free:
            self.section_locks = [threading.Lock() for _ in range(self.num_producers)]
        else:
            self.section_locks = None

        # Pre-generated batch indices for each thread
        # Use a single contiguous array for all threads with thread-specific sections
        # This improves memory alignment and locality
        total_indices = self.num_producers * self.pre_batch_size * self.batch_size
        self.batch_indices_pool = np.random.randint(0, self.corpus_size, size=total_indices)
        # Reshape to (num_producers, pre_batch_size, batch_size) for easier access
        self.batch_indices_pool = self.batch_indices_pool.reshape(
            self.num_producers, self.pre_batch_size, self.batch_size
        )
        self.batch_indices_locks = [threading.Lock() for _ in range(self.num_producers)]

        # Pre-initialize corpus with a pilot run
        init_rng, thread_rngs = jax.random.split(self.rng)

        # Generate a sample batch to initialize the corpus
        @partial(jax.jit, device=self.simulation_device)
        def init_simulator(rng_key):
            rngs = jax.random.split(rng_key, self.update_batch_size)
            return jax.vmap(self.simulator_fn)(rngs)

        # Initialize corpus upfront to avoid race conditions
        first_batch = init_simulator(init_rng)
        self._init_corpus(first_batch)

        thread_rngs = jax.random.split(thread_rngs, self.num_producers)

        # Create and start multiple producer threads
        self.producer_threads = []

        for i in range(self.num_producers):
            thread_rng = thread_rngs[i]
            thread = threading.Thread(
                target=self._producer_loop,
                args=(thread_rng, i),
                daemon=daemon,
                name=f"producer-{i}",
            )
            self.producer_threads.append(thread)
            thread.start()

    def _init_corpus(self, sample_batch):
        """
        Initialize the corpus with the first batch of data.
        Each producer thread will be assigned a section of the corpus to update.
        """
        with self.corpus_lock:
            if self.corpus is not None:
                return  # Already initialized

            # Create a corpus with the same structure as the sample batch
            def create_corpus_array(arr):
                # Create an array with shape (corpus_size, *arr.shape[1:])
                shape = (self.corpus_size,) + arr.shape[1:]
                return np.zeros(shape, dtype=arr.dtype)

            self.corpus = jax.tree_map(create_corpus_array, sample_batch)

            # Assign sections of the corpus to each producer thread
            section_size = self.corpus_size // self.num_producers
            for i in range(self.num_producers):
                start_idx = i * section_size
                end_idx = start_idx + section_size if i < self.num_producers - 1 else self.corpus_size
                self.corpus_indices[i] = (start_idx, end_idx)

            # Initialize the corpus with the sample batch
            for i in range(min(self.corpus_size, self.batch_size)):
                # Check if we're dealing with JAX arrays or NumPy arrays
                def update_array(c, s):
                    if hasattr(c, 'at'):
                        # JAX array
                        return c.at[i].set(s[i % self.batch_size])
                    else:
                        # NumPy array
                        c[i] = s[i % self.batch_size]
                        return c

                self.corpus = jax.tree_map(update_array, self.corpus, sample_batch)

    def _get_batch_indices(self, thread_id):
        """Get pre-generated batch indices for a thread"""
        with self.batch_indices_locks[thread_id]:
            # Get a random set of pre-generated indices
            idx = np.random.randint(0, self.pre_batch_size)
            # Access the thread's section of the contiguous array
            batch_indices = self.batch_indices_pool[thread_id, idx].copy()

            # Update the indices for future use
            self.batch_indices_pool[thread_id, idx] = np.random.randint(
                0, self.corpus_size, size=self.batch_size
            )

        return batch_indices

    def _producer_loop(self, thread_rng, thread_id):
        """
        The background loop that repeatedly:
          - Splits the RNG
          - Produces a smaller batch on the device
          - Updates its assigned section of the corpus
          - Puts a batch from the corpus into the queue for training

        Args:
            thread_rng: The initial RNG key for this thread
            thread_id: The ID of this producer thread
        """
        try:
            # We'll jit+vmap the user simulator to produce update_batch_size items at once
            @partial(jax.jit, device=self.simulation_device)
            def batch_simulator(rng_key):
                rngs = jax.random.split(rng_key, self.update_batch_size)
                return jax.vmap(self.simulator_fn)(rngs)

            # Pre-compile the batch extraction function - removed static_argnums
            @jax.jit
            def extract_batch(corpus, indices):
                """Efficiently extract a batch from the corpus using pre-compiled function"""
                return jax.tree_map(lambda x: x[indices], corpus)

            # Pre-compile the batch update function for efficiency
            @jax.jit
            def update_batch(corpus, indices, data):
                """Efficiently update multiple indices in the corpus at once"""
                def update_array(arr, data_arr):
                    return arr.at[indices].set(data_arr)
                return jax.tree_map(update_array, corpus, data)

            key = jax.device_put(thread_rng, self.simulation_device)
            corpus_start, corpus_end = self.corpus_indices[thread_id]
            corpus_section_size = corpus_end - corpus_start

            # No need to initialize corpus here, it's done in __init__

            while not self.event.is_set():
                # Wait if production is paused
                if self.paused.is_set():
                    time.sleep(0.1)
                    continue

                start_time = time.time()

                # Step 1: Generate a smaller batch for corpus update
                sim_start = time.time()
                key, rng_sub = jax.random.split(key)
                update_data = batch_simulator(rng_sub)
                sim_time = time.time() - sim_start

                if self.enable_profiling:
                    with threading.Lock():
                        self.profiling_stats["simulation_time"] += sim_time

                # Step 2: Update the corpus section assigned to this thread
                update_start = time.time()
                update_indices = np.random.randint(
                    corpus_start, corpus_end, size=self.update_batch_size
                )

                lock_wait_start = time.time()

                # Choose the appropriate locking strategy
                if self.update_lock_free:
                    # Only lock this thread's section of the corpus
                    with self.section_locks[thread_id]:
                        lock_held_start = time.time()

                        # Get a reference to just this thread's section of the corpus
                        with self.corpus_lock:
                            corpus_section = self.corpus

                        # Update the section (only this thread writes to these indices)
                        updated_section = update_batch(corpus_section, update_indices, update_data)

                        # Update the shared corpus with the updated section
                        with self.corpus_lock:
                            self.corpus = updated_section
                            self.stats["corpus_updates"] += 1

                        if self.enable_profiling:
                            lock_held_time = time.time() - lock_held_start
                            with threading.Lock():
                                self.profiling_stats["lock_held_time"] += lock_held_time
                else:
                    # Use the global lock for updates
                    with self.corpus_lock:
                        lock_held_start = time.time()

                        # Update all indices in a single vectorized operation
                        self.corpus = update_batch(self.corpus, update_indices, update_data)
                        self.stats["corpus_updates"] += 1

                        if self.enable_profiling:
                            lock_held_time = time.time() - lock_held_start
                            with threading.Lock():
                                self.profiling_stats["lock_held_time"] += lock_held_time

                update_time = time.time() - update_start
                lock_wait_time = lock_held_start - lock_wait_start

                if self.enable_profiling:
                    with threading.Lock():
                        self.profiling_stats["update_time"] += update_time
                        self.profiling_stats["lock_wait_time"] += lock_wait_time

                # Step 3: Create a training batch from the corpus using pre-generated indices
                extract_start = time.time()
                batch_indices = self._get_batch_indices(thread_id)

                # Convert NumPy batch_indices to JAX array
                batch_indices_jax = jax.numpy.array(batch_indices)

                # Take a snapshot of the corpus to avoid holding the lock during extraction
                with self.corpus_lock:
                    corpus_snapshot = self.corpus

                # Extract the batch from the corpus snapshot - no lock needed
                data_cpu = extract_batch(corpus_snapshot, batch_indices_jax)
                extract_time = time.time() - extract_start

                if self.enable_profiling:
                    with threading.Lock():
                        self.profiling_stats["extraction_time"] += extract_time

                production_time = time.time() - start_time
                with threading.Lock():
                    self.stats["production_time"] += production_time

                # Step 4: Put the batch into the queue for training
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
          - Prefetches multiple batches to better overlap computation with I/O
          - Yields batches that are already transferred to the device

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
        while prefetch_queue:
            # Get the next batch to yield (oldest prefetched batch)
            current_batch = prefetch_queue.popleft()

            # Prefetch one more to maintain prefetch_depth
            try:
                cpu_batch = next(stream_iter)
                batch_on_device = jax.device_put(cpu_batch, self.data_device)
                prefetch_queue.append(batch_on_device)
            except StopIteration:
                pass  # No more batches to prefetch

            yield current_batch

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
        if hasattr(self, 'producer_threads'):
            for thread in self.producer_threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)  # Wait up to 1 second
                    if thread.is_alive():
                        # Force terminate if thread didn't exit cleanly
                        try:
                            thread._stop()  # Force thread termination
                        except Exception:
                            pass  # Ignore errors during force termination

    def __del__(self):
        """
        Make sure the background threads are stopped if the loader is garbage-collected.
        """
        try:
            if hasattr(self, 'event'):
                self.event.set()
            if hasattr(self, 'producer_threads'):
                for thread in self.producer_threads:
                    if thread.is_alive():
                        thread.join()
        except Exception:
            pass  # Ignore errors during cleanup

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

    def get_profiling_stats(self):
        """Return detailed profiling statistics if enabled."""
        if not self.enable_profiling:
            return {"profiling_disabled": True}
        return {k: v for k, v in self.profiling_stats.items()}
