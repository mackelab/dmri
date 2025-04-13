import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
from contextlib import contextmanager
from dmri.train.dataloader import StreamDataLoader
import matplotlib.pyplot as plt
from tabulate import tabulate



def create_simulator(data_shape=(32, 32, 3), complexity=1):
    """
    Create a simulator function that generates random data with configurable complexity.

    Args:
        data_shape: Shape of the data to generate
        complexity: Number of operations to perform (higher = more compute)

    Returns:
        A JIT-compiled function that returns JAX arrays
    """
    # Using jit to ensure the output is always a JAX array, not numpy
    @jax.jit
    def simulator(key):
        # Generate random data
        x = jax.random.normal(key, data_shape)

        # Add some compute to simulate a more complex generator
        for _ in range(complexity):
            x = jnp.sin(x) + jnp.cos(x)
            x = jnp.tanh(x)

        return x

    return simulator


def safe_close_dataloader(dataloader):
    """Safely close the dataloader, handling potential missing attributes"""
    dataloader.close()



# Use a plain class without .at[] for numpy arrays
class SimpleNumpyCompatibleClass:
    """A simple class that provides JAX-like API for NumPy arrays"""

    def __init__(self, simulator_fn, batch_size=128, **kwargs):
        self.simulator_fn = simulator_fn
        self.batch_size = batch_size
        self.stats = {}

    def __iter__(self):
        key = jax.random.key(0)
        for i in range(10):  # Produce a few batches
            key, subkey = jax.random.split(key)
            yield self.simulator_fn(subkey)

    def get_stats(self):
        return {"batches_processed": 10}

    def close(self):
        pass


def profile_dataloader(
    simulator_fn,
    run_duration=10,
    batch_size=128,
    corpus_size=1000,
    update_batch_size=32,
    num_producers=4,
    update_lock_free=True,
    prefetch_depth=3,
    ring_size=2,
):
    """
    Profile the StreamDataLoader performance with given configuration.

    Args:
        simulator_fn: Function that generates data samples (JAX arrays)
        run_duration: Duration to run the profiling in seconds
        batch_size: Batch size for training
        corpus_size: Size of the data corpus
        update_batch_size: Size of update batches
        num_producers: Number of producer threads
        update_lock_free: Whether to use lock-free updates
        prefetch_depth: Prefetch queue depth
        ring_size: Size of ring buffer for in-place updates

    Returns:
        Dictionary of profiling results
    """
    print(f"Initializing StreamDataLoader with batch_size={batch_size}, corpus_size={corpus_size}, num_producers={num_producers}")

    # Initialize dataloader with profiling enabled
    dataloader = StreamDataLoader(
        simulator_fn=simulator_fn,
        batch_size=batch_size,
        corpus_size=corpus_size,
        update_batch_size=update_batch_size,
        max_queue_size=100,
        num_producers=num_producers,
        update_lock_free=update_lock_free,
        enable_profiling=True,
        prefetch_depth=prefetch_depth,
        ring_size=ring_size,
    )

    # Run for specified duration
    start_time = time.time()
    batches_processed = 0
    last_print_time = start_time

    print(f"Starting profiling run for {run_duration} seconds...")

    # Process batches for the specified duration
    try:
        # Create a thread to iterate over the dataloader
        import threading
        import queue

        # Queue to store batches
        batch_queue = queue.Queue(maxsize=100)

        # Flag to signal when to stop
        stop_flag = threading.Event()

        def dataloader_thread_func():
            try:
                for batch in dataloader:
                    if stop_flag.is_set():
                        break
                    batch_queue.put(batch)
            except Exception as e:
                print(f"Error in dataloader thread: {e}")
                import traceback
                traceback.print_exc()

        # Start the dataloader thread
        dataloader_thread = threading.Thread(target=dataloader_thread_func)
        dataloader_thread.daemon = True  # Make it a daemon thread
        dataloader_thread.start()

        # Process batches from the queue
        while not stop_flag.is_set():
            # Check if we've run long enough
            if time.time() - start_time >= run_duration:
                print(f"Reached target duration of {run_duration} seconds")
                stop_flag.set()
                break

            # Try to get a batch with a timeout
            try:
                batch = batch_queue.get(timeout=1.0)
                batches_processed += 1

                # Print progress every second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    elapsed = current_time - start_time
                    print(f"Processed {batches_processed} batches in {elapsed:.1f} seconds ({batches_processed/elapsed:.1f} batches/sec)")
                    last_print_time = current_time

                # Add a small simulated training step
                time.sleep(0.01)  # Simulate 10ms of GPU training time
            except queue.Empty:
                # No batch available, check if we should continue
                if time.time() - start_time >= run_duration:
                    print(f"Reached target duration of {run_duration} seconds")
                    stop_flag.set()
                    break
                continue
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure to close the dataloader
        print("Closing dataloader...")
        safe_close_dataloader(dataloader)
        print("Dataloader closed")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Profiling completed in {elapsed_time:.1f} seconds, processed {batches_processed} batches")

    # Get profiling stats
    try:
        stats = dataloader.get_stats()
        profiling_stats = dataloader.get_profiling_stats()

        # Calculate total simulations produced
        total_simulations = 0
        if "corpus_updates" in stats:
            total_simulations = stats["corpus_updates"] * update_batch_size
        elif "batches_produced" in stats:
            total_simulations = stats["batches_produced"] * batch_size

        print(f"Total simulations produced: {total_simulations}")
    except Exception as e:
        print(f"Error getting stats: {e}")
        stats = {"error": str(e)}
        profiling_stats = {"error": str(e)}
        total_simulations = 0

    # Consolidate results
    results = {
        "config": {
            "batch_size": batch_size,
            "corpus_size": corpus_size,
            "update_batch_size": update_batch_size,
            "num_producers": num_producers,
            "update_lock_free": update_lock_free,
            "prefetch_depth": prefetch_depth,
            "ring_size": ring_size,
        },
        "performance": {
            "elapsed_time": elapsed_time,
            "batches_processed": batches_processed,
            "batches_per_second": batches_processed / elapsed_time if elapsed_time > 0 else 0,
            "avg_batch_time_ms": (elapsed_time / batches_processed) * 1000 if batches_processed > 0 else 0,
            "total_simulations": total_simulations,
            "simulations_per_second": total_simulations / elapsed_time if elapsed_time > 0 else 0,
        },
        "dataloader_stats": stats,
        "profiling_stats": profiling_stats,
    }

    return results


def analyze_results(results):
    """
    Analyze and print profiling results.

    Args:
        results: Results from profile_dataloader
    """
    config = results["config"]
    perf = results["performance"]
    stats = results["dataloader_stats"]
    profiling = results["profiling_stats"]

    print("\n" + "="*80)
    print(f"DATALOADER PROFILING RESULTS")
    print("="*80)

    # Configuration
    print("\nConfiguration:")
    config_table = [[k, v] for k, v in config.items()]
    print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="pretty"))

    # Performance Summary
    print("\nPerformance Summary:")
    perf_table = [[k, f"{v:.2f}" if isinstance(v, float) else v] for k, v in perf.items()]
    print(tabulate(perf_table, headers=["Metric", "Value"], tablefmt="pretty"))

    # Detailed Statistics
    print("\nDataloader Statistics:")
    stats_table = [[k, v] for k, v in stats.items()]
    print(tabulate(stats_table, headers=["Statistic", "Value"], tablefmt="pretty"))

    # Profiling Breakdown
    if profiling and "profiling_disabled" not in profiling:
        print("\nProfiling Breakdown:")
        # Calculate percentages of time spent in each phase
        total_time = sum(v for k, v in profiling.items())
        if total_time > 0:
            profiling_table = [
                [k, f"{v:.2f}s", f"{(v/total_time)*100:.1f}%"]
                for k, v in profiling.items()
            ]
            print(tabulate(profiling_table, headers=["Phase", "Time", "Percentage"], tablefmt="pretty"))

            # Print any potential bottlenecks
            print("\nPotential Bottlenecks:")
            bottlenecks = []

            if profiling.get("simulation_time", 0) / total_time > 0.5:
                bottlenecks.append(["Simulation", "High simulation time - consider a simpler simulator or reducing update_batch_size"])

            if profiling.get("lock_wait_time", 0) / total_time > 0.3:
                bottlenecks.append(["Lock Contention", "High lock wait time - try increasing corpus_size or reducing producer count"])

            if profiling.get("lock_held_time", 0) / total_time > 0.4:
                bottlenecks.append(["Lock Duration", "Locks held for too long - optimize update operations"])

            if not bottlenecks:
                bottlenecks.append(["None detected", "Dataloader seems well-balanced"])

            print(tabulate(bottlenecks, headers=["Bottleneck", "Recommendation"], tablefmt="pretty"))
    else:
        print("\nDetailed profiling information not available.")


def compare_configurations(simulator_fn, run_duration=10):
    """
    Compare different dataloader configurations and plot results.

    Args:
        simulator_fn: Function that generates data samples
        run_duration: Duration to run each configuration
    """
    configurations = [
        {"name": "Default", "batch_size": 128, "corpus_size": 1000, "update_batch_size": 32, "num_producers": 4, "update_lock_free": True},
        {"name": "Large Corpus", "batch_size": 128, "corpus_size": 10000, "update_batch_size": 32, "num_producers": 4, "update_lock_free": True},
        {"name": "Small Updates", "batch_size": 128, "corpus_size": 1000, "update_batch_size": 16, "num_producers": 4, "update_lock_free": True},
        {"name": "Global Lock", "batch_size": 128, "corpus_size": 1000, "update_batch_size": 32, "num_producers": 4, "update_lock_free": False},
        {"name": "Many Producers", "batch_size": 128, "corpus_size": 1000, "update_batch_size": 32, "num_producers": 8, "update_lock_free": True},
    ]

    results = []
    for config in configurations:
        print(f"\nTesting configuration: {config['name']}")
        result = profile_dataloader(
            simulator_fn=simulator_fn,
            run_duration=run_duration,
            batch_size=config["batch_size"],
            corpus_size=config["corpus_size"],
            update_batch_size=config["update_batch_size"],
            num_producers=config["num_producers"],
            update_lock_free=config["update_lock_free"],
        )
        results.append((config["name"], result))

    # Plot comparison
    names = [name for name, _ in results]
    throughputs = [res["performance"]["batches_per_second"] for _, res in results]

    plt.figure(figsize=(10, 6))
    plt.bar(names, throughputs)
    plt.title("Dataloader Throughput Comparison")
    plt.xlabel("Configuration")
    plt.ylabel("Batches per Second")
    plt.savefig("dataloader_comparison.png")
    print(f"\nComparison chart saved to dataloader_comparison.png")

    # Print summary table
    summary = []
    for name, res in results:
        summary.append([
            name,
            f"{res['performance']['batches_per_second']:.2f}",
            f"{res['performance']['avg_batch_time_ms']:.2f}",
            res['config']['corpus_size'],
            res['config']['update_batch_size'],
            res['config']['num_producers'],
            "Yes" if res['config']['update_lock_free'] else "No",
        ])

    print("\nConfiguration Comparison Summary:")
    print(tabulate(summary, headers=[
        "Configuration",
        "Throughput (batches/s)",
        "Avg Batch Time (ms)",
        "Corpus Size",
        "Update Batch Size",
        "Producers",
        "Lock-Free Updates"
    ], tablefmt="pretty"))


def main():
    parser = argparse.ArgumentParser(description="Profile the StreamDataLoader performance")
    parser.add_argument("--mode", choices=["single", "compare"], default="single", help="Profiling mode")
    parser.add_argument("--duration", type=int, default=20, help="Duration of profiling run in seconds")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--corpus-size", type=int, default=10_000, help="Corpus size")
    parser.add_argument("--update-batch-size", type=int, default=32, help="Update batch size")
    parser.add_argument("--num-producers", type=int, default=4, help="Number of producer threads")
    parser.add_argument("--update-lock-free", action="store_true", help="Use lock-free updates")
    parser.add_argument("--prefetch-depth", type=int, default=3, help="Prefetch queue depth")
    parser.add_argument("--ring-size", type=int, default=2, help="Ring buffer size")
    parser.add_argument("--complexity", type=int, default=5, help="Simulator complexity (1-10)")
    parser.add_argument("--data-shape", type=str, default="32,32,3", help="Data shape (comma-separated)")

    args = parser.parse_args()

    # Parse data shape
    try:
        data_shape = tuple(int(dim) for dim in args.data_shape.split(","))
    except ValueError:
        print("Error parsing data shape. Using default (32,32,3)")
        data_shape = (32, 32, 3)

    # Create simulator function - properly JIT-compiled to ensure JAX arrays
    simulator = create_simulator(data_shape=data_shape, complexity=args.complexity)

    if args.mode == "single":
        # Run single profiling
        results = profile_dataloader(
            simulator_fn=simulator,
            run_duration=args.duration,
            batch_size=args.batch_size,
            corpus_size=args.corpus_size,
            update_batch_size=args.update_batch_size,
            num_producers=args.num_producers,
            update_lock_free=args.update_lock_free,
            prefetch_depth=args.prefetch_depth,
            ring_size=args.ring_size,
        )
        analyze_results(results)
    else:
        # Run comparison
        compare_configurations(simulator, run_duration=args.duration)


if __name__ == "__main__":
    main()
