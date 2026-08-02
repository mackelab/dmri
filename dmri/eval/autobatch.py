"""Hardware-aware batch-size selection for the evaluation/prediction path.

A null ``eval_batch_size`` means *use the largest batch this machine can run*.
Nothing caps it but the device's memory and the size of the data, so the same
config makes full use of a laptop GPU and of a multi-GPU node.

Resolution order (first hit wins):

1. an explicit override (CLI flag or ``DMRI_BATCH_SIZE``), used verbatim,
2. an explicit ``eval_batch_size`` in the config, also used verbatim,
3. a cached value from a previous run with the same model, precision and devices,
4. a seed guess from device memory, refined against XLA's memory analysis of the
   compiled function and validated by a compile at the chosen size.

The budget is per device and every device holds an equal shard, so the resolved
batch is ``num_devices x per_device_batch``.
"""

import functools
import hashlib
import json
import os
import pathlib
from collections.abc import Sequence
from importlib import metadata

import jax
import jax.numpy as jnp
import numpy as np

# XLA preallocates 75% of device memory, so an 80GB card exposes ~60GiB, which
# fits ~30k voxels for this pipeline -- the prior that seeds the probe.
H100_REFERENCE_BUDGET = int(0.75 * 80 * 1024**3)
BYTES_PER_VOXEL_PRIOR = H100_REFERENCE_BUDGET / 30_000

# Used only when the host's available memory cannot be measured at all.
CPU_FALLBACK_BUDGET = 4 * 1024**3
# Share of available host memory a batch may claim. The host also holds the
# signal, the theta output and numpy working copies at the same time.
HOST_MEMORY_FRACTION = 0.5
# Even a busy machine should manage a small batch rather than give up.
MIN_HOST_BUDGET = 512 * 1024**2

DEFAULT_MIN_BATCH = 256
# Round batch sizes to a multiple of this so runs do not trace many shapes.
SHAPE_QUANTUM = 256
PROBE_SIZES = (8, 1024)
# XLA's estimate excludes allocator fragmentation; the OOM halving is the net.
SAFETY_FACTOR = 0.9
# How many measure-and-predict rounds to spend closing on the ceiling.
REFINE_ROUNDS = 4
# Most a single probe may grow over the last size known to fit. Generous,
# because the affine fit overestimates cost and so predicts conservatively;
# the cap only guards against a badly wrong fit leaping into an OOM.
MAX_PROBE_GROWTH = 16.0
# Stop refining once the predicted gain is this small. Chasing the last few
# percent costs a full compile per step for no practical gain.
BISECT_TOLERANCE = 0.10
# Rows needed per SM before the device is reasonably busy.
COMPUTE_FLOOR_PER_CORE = 32

ENV_OVERRIDE = "DMRI_BATCH_SIZE"
CACHE_FILENAME = "dmri_batch_size.json"


def available_devices(kind):
    """Devices of ``kind``, or an empty tuple when that backend is absent.

    ``jax.devices(kind)`` raises rather than returning an empty list when the
    backend is not present at all, which is the normal case on a CPU-only host.
    """
    try:
        return tuple(jax.devices(kind))
    except RuntimeError:
        return ()


#: Axis name of the 1-D mesh the evaluation batch is split over.
BATCH_AXIS = "batch"


def make_eval_mesh(devices):
    """A 1-D mesh over ``devices`` whose single axis is the voxel batch."""
    devices = tuple(devices) if devices else (jax.devices()[0],)
    return jax.sharding.Mesh(np.asarray(devices), (BATCH_AXIS,))


def batch_sharding_for(devices):
    """Sharding that splits the leading (voxel) axis across ``devices``."""
    mesh = make_eval_mesh(devices)
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(BATCH_AXIS))


def replicated_sharding_for(devices):
    """Sharding that places a full copy on every device (for model parameters)."""
    mesh = make_eval_mesh(devices)
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _round_down(value, quantum=SHAPE_QUANTUM, minimum=DEFAULT_MIN_BATCH):
    """Round ``value`` down to a multiple of ``quantum``, never below ``minimum``."""
    value = int(value)
    if value <= minimum:
        return minimum
    return max(minimum, (value // quantum) * quantum)


def available_host_memory():
    """Bytes that can be allocated on the host **without swapping**.

    Swapping is far slower than simply using a smaller batch, so this budgets
    from what is currently available rather than from total RAM.
    """
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    # The kernel's own estimate: free memory plus the page cache
                    # it can reclaim, and it excludes swap.
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        # Portable fallback (works on macOS). More conservative than
        # MemAvailable: free pages only, ignoring reclaimable cache.
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def host_memory_budget():
    """Host bytes to spend on a batch.

    Only a fraction of what is available, because unlike a GPU the host pool
    simultaneously holds the flattened signal, the theta output and numpy
    working copies during export -- the batch is not alone there.
    """
    available = available_host_memory()
    if not available or available <= 0:
        return CPU_FALLBACK_BUDGET
    return max(int(available * HOST_MEMORY_FRACTION), MIN_HOST_BUDGET)


def device_memory_budget(device):
    """Free bytes usable for a batch on ``device``.

    Returns the allocator limit minus what is already resident (model params and
    the flattened signal array both live on the device before batching starts).
    On CPU there is no allocator limit to read, so the host's available memory
    is measured instead.
    """
    try:
        stats = device.memory_stats()
    except RuntimeError:
        stats = None
    if not stats:
        return host_memory_budget()
    limit = stats.get("bytes_limit")
    if not limit:
        return host_memory_budget()
    in_use = stats.get("bytes_in_use", 0) or 0
    return max(int(limit) - int(in_use), 0)


def device_core_count(device):
    """Number of compute cores (SMs) on ``device``, or None when unavailable."""
    count = getattr(device, "core_count", None)
    return int(count) if count else None


def cache_dir():
    """Directory holding the batch-size cache, alongside the JAX compile cache."""
    configured = jax.config.jax_compilation_cache_dir
    return configured or ".jax_cache"


def _cache_path():
    return os.path.join(cache_dir(), CACHE_FILENAME)


@functools.lru_cache(maxsize=1)
def code_fingerprint():
    """Marker that changes whenever the code defining the graphs changes.

    A cached batch size is only valid for the computation it was measured on, so
    editing the model or the sampler must invalidate it. Hashing the source costs
    ~2ms; hashing the lowered HLO would be exact but costs ~6s per run.
    """
    digest = hashlib.sha256()
    try:
        import dmri

        root = pathlib.Path(dmri.__file__).parent
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    except OSError:
        # A fingerprint we cannot compute simply means less cache reuse.
        digest.update(b"unknown")

    for package in ("jax", "jaxlib", "flax", "probjax"):
        try:
            digest.update(f"{package}={metadata.version(package)}".encode())
        except metadata.PackageNotFoundError:
            pass
    return digest.hexdigest()[:16]


def make_cache_key(*parts):
    """Stable key from any hashable description of the workload.

    Always mixes in the code fingerprint, so a size measured on one graph is
    never reused for another.
    """
    payload = "|".join(repr(part) for part in parts)
    return hashlib.sha256(f"{code_fingerprint()}|{payload}".encode()).hexdigest()[:32]


def load_cached_batch_size(cache_key):
    try:
        with open(_cache_path()) as handle:
            cache = json.load(handle)
    except (OSError, ValueError):
        return None
    value = cache.get(cache_key) if isinstance(cache, dict) else None
    return int(value) if isinstance(value, int) and value > 0 else None


def store_cached_batch_size(cache_key, batch_size):
    """Persist ``batch_size``; failures are non-fatal, the probe just reruns."""
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path) as handle:
                cache = json.load(handle)
            if not isinstance(cache, dict):
                cache = {}
        except (OSError, ValueError):
            cache = {}
        cache[cache_key] = int(batch_size)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except OSError:
        pass


def explicit_override():
    """Batch size forced through the environment, if any."""
    raw = os.environ.get(ENV_OVERRIDE)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def heuristic_batch_size(
    budget,
    core_count=None,
    min_batch=DEFAULT_MIN_BATCH,
    max_batch=None,
):
    """Seed guess from device memory alone, without compiling anything.

    A starting point for the probe and the fallback when probing fails; not a
    ceiling.
    """
    guess = budget / BYTES_PER_VOXEL_PRIOR

    if core_count:
        # Keep the device busy, but never above what memory allows.
        compute_floor = COMPUTE_FLOOR_PER_CORE * core_count
        guess = max(guess, min(compute_floor, guess if guess > 0 else compute_floor))

    guess = max(min_batch, guess)
    if max_batch is not None:
        guess = min(guess, max_batch)
    return _round_down(guess, minimum=min(min_batch, int(max_batch or min_batch)))


def _analyzed_bytes(stats):
    total = (
        (stats.temp_size_in_bytes or 0)
        + (stats.argument_size_in_bytes or 0)
        + (stats.output_size_in_bytes or 0)
        - (stats.alias_size_in_bytes or 0)
    )
    return max(total, 0)


def _probe_at(fn, batch_size, example_args, key_example):
    """Compile ``fn`` for ``batch_size`` rows and return its analyzed byte cost.

    Releases everything it allocated before returning. A retained executable
    counts against the device for the next probe, which would make each
    measurement depend on the ones before it and on the order they ran in.
    """

    def tile(example):
        example = jnp.asarray(example)
        return jnp.zeros((batch_size, *example.shape[1:]), dtype=example.dtype)

    args = keys = compiled = None
    try:
        args = [tile(arg) for arg in example_args]
        keys = jax.random.split(jnp.asarray(key_example), batch_size)
        compiled = jax.jit(fn).lower(keys, *args).compile()
        return _analyzed_bytes(compiled.memory_analysis())
    finally:
        del args, keys, compiled
        jax.clear_caches()


def probe_batch_size(
    fn,
    example_args,
    key_example,
    budget,
    max_batch,
    min_batch=DEFAULT_MIN_BATCH,
    probe_sizes=PROBE_SIZES,
    refine_rounds=REFINE_ROUNDS,
    logger=None,
):
    """Find the largest batch that fits in ``budget`` using XLA's memory analysis.

    Approaches the ceiling from below. A compile that exceeds device memory
    fails, and fails slowly -- roughly twice the cost of one that succeeds --
    and leaves allocations behind that skew the next measurement. So rather than
    probing upward until something breaks, each measurement refines an affine
    cost model and predicts the next size, with growth capped so the prediction
    creeps up instead of leaping past the limit.

    Returns None if anything goes wrong, leaving the caller on its heuristic.
    """
    usable_budget = budget * SAFETY_FACTOR
    small, large = probe_sizes

    def measure(size):
        """Analyzed bytes at ``size``, or None if it could not be compiled."""
        try:
            return _probe_at(fn, size, example_args, key_example)
        except Exception as error:
            if not _looks_like_oom(error) and logger is not None:
                logger.info(f"Batch-size probe failed at {size} ({error}).")
            return None

    def predict(anchor_size, anchor_bytes):
        """Largest size the affine fit through two points says will fit."""
        slope = (anchor_bytes - bytes_small) / max(anchor_size - small, 1)
        if slope <= 0:
            return None
        intercept = anchor_bytes - slope * anchor_size
        usable = usable_budget - intercept
        if usable <= 0:
            return min_batch
        return _clamp(usable / slope, min_batch, max_batch)

    bytes_small = measure(small)
    bytes_large = measure(large)
    if bytes_small is None or bytes_large is None:
        return None

    if bytes_large > usable_budget:
        # Even the second probe size does not fit, so search downward instead of
        # treating it as a known-good floor.
        candidate = predict(large, bytes_large)
        if candidate is None or candidate >= large:
            return _clamp(min_batch, min_batch, max_batch)
        measured = measure(candidate)
        while measured is None or measured > usable_budget:
            candidate = _clamp(candidate // 2, min_batch, max_batch)
            if candidate <= min_batch:
                return _clamp(min_batch, min_batch, max_batch)
            measured = measure(candidate)
        return candidate

    best = _clamp(large, min_batch, max_batch)
    candidate = predict(large, bytes_large)
    if candidate is None:
        return None

    for _ in range(refine_rounds):
        # Creep towards the ceiling: the fit is least reliable furthest from
        # the points it was built on, so do not leap.
        capped = _clamp(min(candidate, best * MAX_PROBE_GROWTH), min_batch, max_batch)
        if capped <= best:
            break

        measured = measure(capped)
        if measured is None:
            # Too big after all; back off and try once more from lower down.
            candidate = _clamp(best + (capped - best) // 2, min_batch, max_batch)
            if candidate <= best:
                break
            continue

        if measured > usable_budget:
            candidate = predict(capped, measured)
            if candidate is None or candidate <= best:
                break
            continue

        best = capped
        if capped >= max_batch:
            break
        candidate = predict(capped, measured)
        if candidate is None:
            break
        if candidate <= best * (1 + BISECT_TOLERANCE):
            break

    return best


def _clamp(value, min_batch, max_batch):
    value = _round_down(max(min_batch, value), minimum=min_batch)
    if max_batch is not None:
        value = min(value, int(max_batch))
    return max(value, min(min_batch, int(max_batch) if max_batch else min_batch))


def _looks_like_oom(error):
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("out of memory", "resource_exhausted", "failed to allocate")
    )


def resolve_batch_size(
    fn,
    *example_args,
    key_example,
    configured=None,
    devices=None,
    cache_key=None,
    logger=None,
    override=None,
    min_batch=DEFAULT_MIN_BATCH,
    probe=True,
    num_voxels=None,
):
    """Pick a batch size for running ``fn`` over a leading voxel axis.

    ``configured`` of None means *determine the largest workable batch*; any
    explicit value is used verbatim. ``fn`` is called as ``fn(keys, *args)`` with
    a leading batch axis, matching
    :func:`dmri.eval.sampling_methods.eval_in_batches`.
    """
    print_fn = logger.info if logger is not None else print

    override = override if override is not None else explicit_override()
    if override:
        print_fn(f"Using explicit batch size {override}")
        return int(override)
    if configured is not None:
        print_fn(f"Using configured batch size {int(configured)}")
        return int(configured)

    device = _primary_device(devices)
    num_devices = len(devices) if isinstance(devices, Sequence) and devices else 1

    # The budget is per device and every device holds an equal shard, so total
    # capacity grows with the device count.
    per_device_budget = device_memory_budget(device)
    core_count = device_core_count(device)

    # Never batch beyond the data itself.
    if num_voxels is None and example_args:
        num_voxels = int(np.shape(example_args[0])[0]) or None
    max_batch = num_voxels

    if cache_key is not None:
        cached = load_cached_batch_size(cache_key)
        if cached:
            cached = min(cached, max_batch) if max_batch else cached
            print_fn(f"Using cached batch size {cached}")
            return _align_to_devices(cached, num_devices, min_batch)

    per_device_max = (
        max(1, max_batch // num_devices) if max_batch and num_devices > 1 else max_batch
    )
    guess = heuristic_batch_size(
        per_device_budget, core_count, min_batch, max_batch=per_device_max
    )
    print_fn(
        f"Heuristic batch size {guess * num_devices} "
        f"(budget {per_device_budget / 1024**3:.1f}GiB/device, cores {core_count}, "
        f"{num_devices} device(s))"
    )

    per_device_batch = guess
    if probe:
        refined = probe_batch_size(
            fn,
            example_args,
            key_example,
            per_device_budget,
            per_device_max,
            min_batch=min_batch,
            logger=logger,
        )
        if refined:
            print_fn(
                f"Probed batch size {refined * num_devices} (was {guess * num_devices})"
            )
            per_device_batch = refined

    batch_size = _align_to_devices(
        per_device_batch * num_devices, num_devices, min_batch
    )
    if max_batch:
        batch_size = min(batch_size, max_batch)

    if cache_key is not None:
        store_cached_batch_size(cache_key, batch_size)
    return batch_size


def _align_to_devices(batch_size, num_devices, min_batch):
    """Keep the batch divisible by the device count so shards stay even."""
    del min_batch  # kept for call-site symmetry
    if num_devices <= 1:
        return max(int(batch_size), 1)
    aligned = (int(batch_size) // num_devices) * num_devices
    return max(aligned, num_devices)


def _primary_device(devices):
    if isinstance(devices, str):
        available = available_devices(devices)
        return available[0] if available else jax.devices()[0]
    if devices:
        return list(devices)[0]
    for kind in ("gpu", "tpu"):
        available = available_devices(kind)
        if available:
            return available[0]
    return jax.devices()[0]
