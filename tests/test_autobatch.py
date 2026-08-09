"""Tests for hardware-aware batch-size selection and padded batching."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmri.eval import autobatch
from dmri.eval.autobatch import (
    H100_REFERENCE_BUDGET,
    device_memory_budget,
    heuristic_batch_size,
    load_cached_batch_size,
    make_cache_key,
    resolve_batch_size,
    store_cached_batch_size,
)
from dmri.eval.sampling_methods import eval_in_batches

CONFIGURED_MAX = 30_000


def _toy_fn(keys, x):
    """A small compilable workload whose memory grows with the batch."""
    h = jnp.tanh(x @ jnp.ones((x.shape[-1], 64), dtype=x.dtype))
    return h.sum(-1)


class FakeDevice:
    """A device reporting a fixed allocator budget and core count."""

    def __init__(self, limit, in_use=0, core_count=68, kind="fake"):
        self._stats = {"bytes_limit": limit, "bytes_in_use": in_use}
        self.core_count = core_count
        self.device_kind = kind

    def memory_stats(self):
        return self._stats


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(autobatch, "cache_dir", lambda: str(tmp_path))
    return tmp_path


def test_device_memory_budget_subtracts_what_is_resident():
    device = FakeDevice(limit=10_000, in_use=4_000)
    assert device_memory_budget(device) == 6_000


def test_device_memory_budget_measures_the_host_without_stats():
    """CPU has no allocator limit to read, so measure available host memory."""

    class NoStats:
        def memory_stats(self):
            return None

    budget = device_memory_budget(NoStats())
    assert budget == autobatch.host_memory_budget()
    available = autobatch.available_host_memory()
    if available:
        assert budget < available, "must leave headroom rather than risk swapping"


def test_cache_round_trip(cache_dir):
    key = make_cache_key("stage", "model", 288, 50)
    assert load_cached_batch_size(key) is None
    store_cached_batch_size(key, 4096)
    assert load_cached_batch_size(key) == 4096


def test_corrupt_cache_is_ignored(cache_dir):
    (cache_dir / autobatch.CACHE_FILENAME).write_text("{not json")
    assert load_cached_batch_size(make_cache_key("x")) is None
    # A later write still succeeds and produces valid JSON.
    store_cached_batch_size(make_cache_key("x"), 512)
    payload = json.loads((cache_dir / autobatch.CACHE_FILENAME).read_text())
    assert payload[make_cache_key("x")] == 512


def test_explicit_override_skips_probing_and_cache(cache_dir):
    def fail(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("probe should not run for an explicit override")

    resolved = resolve_batch_size(
        fail,
        np.zeros((10, 3), dtype=np.float32),
        key_example=jax.random.PRNGKey(0),
        configured=None,
        cache_key=make_cache_key("override"),
        override=1234,
    )
    assert resolved == 1234
    # Nothing was cached, so a later call is free to resolve differently.
    assert load_cached_batch_size(make_cache_key("override")) is None


def test_resolve_uses_the_cache_before_probing(cache_dir):
    key = make_cache_key("cached")
    store_cached_batch_size(key, 2048)

    def fail(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("probe should not run on a cache hit")

    resolved = resolve_batch_size(
        fail,
        np.zeros((10, 3), dtype=np.float32),
        key_example=jax.random.PRNGKey(0),
        configured=None,
        cache_key=key,
        num_voxels=10**6,
    )
    assert resolved == 2048


def test_resolve_falls_back_when_the_probe_fails(cache_dir):
    def unprobeable(keys, x):
        raise RuntimeError("cannot compile")

    resolved = resolve_batch_size(
        unprobeable,
        np.zeros((10, 3), dtype=np.float32),
        key_example=jax.random.PRNGKey(0),
        configured=None,
        cache_key=make_cache_key("broken"),
        num_voxels=10**6,
    )
    assert resolved > 0


@pytest.mark.parametrize(
    ("total", "batch_size"),
    [(1000, 300), (1000, 1000), (100, 300), (900, 300), (1, 300), (7, 4)],
)
def test_padding_keeps_results_correct(total, batch_size):
    """A ragged final batch must not change the output."""
    data = np.arange(total * 3, dtype=np.float32).reshape(total, 3)

    def fn(keys, x):
        return x * 2.0

    out = eval_in_batches(
        fn, jax.random.PRNGKey(0), data, batch_size=batch_size, min_batch_size=1
    )
    assert out.shape == data.shape
    np.testing.assert_allclose(out, data * 2.0)


def test_only_one_shape_is_traced():
    """Padding exists so the ragged tail does not force a second compile."""
    shapes = []

    @jax.jit
    def compiled(keys, x):
        return x * 2.0

    def fn(keys, x):
        shapes.append(tuple(x.shape))
        return compiled(keys, x)

    data = np.arange(1000 * 3, dtype=np.float32).reshape(1000, 3)
    eval_in_batches(fn, jax.random.PRNGKey(0), data, batch_size=300, min_batch_size=1)

    assert len(shapes) == 4, "expected ceil(1000/300) batches"
    assert len(set(shapes)) == 1, f"traced more than one shape: {set(shapes)}"


def test_batch_larger_than_the_data_is_not_padded_up():
    """A single short batch should not be inflated to the full batch size.

    It is still rounded up to a multiple of the device count so the shards
    divide evenly, which is the only padding allowed here.
    """
    shapes = []

    def fn(keys, x):
        shapes.append(tuple(x.shape))
        return x * 2.0

    num_devices = len(jax.devices())
    data = np.ones((17, 2), dtype=np.float32)
    out = eval_in_batches(fn, jax.random.PRNGKey(0), data, batch_size=30_000)

    expected_rows = 17 + (-17 % num_devices)
    assert shapes == [(expected_rows, 2)]
    assert expected_rows < 30_000, "the batch was inflated to the full batch size"
    # Padding never leaks into the result.
    np.testing.assert_allclose(out, data * 2.0)


def test_oom_detection_matches_jax_resource_exhausted_messages():
    from dmri.eval.sampling_methods import _is_oom

    assert _is_oom(RuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating 1234"))
    assert _is_oom(RuntimeError("Failed to allocate request for 8.00GiB"))
    assert not _is_oom(RuntimeError("some unrelated failure"))


def test_padded_batches_survive_multiple_arguments():
    data = np.arange(70, dtype=np.float32).reshape(35, 2)
    extra = np.arange(35, dtype=np.float32).reshape(35, 1)

    def fn(keys, x, y):
        return x + y

    out = eval_in_batches(
        fn, jax.random.PRNGKey(0), data, extra, batch_size=8, min_batch_size=1
    )
    np.testing.assert_allclose(out, data + extra)


def test_probe_estimates_scale_with_memory():
    """A bigger budget must not yield a smaller probed batch size."""
    example = (jnp.zeros((1, 32), dtype=jnp.float32),)
    key = jax.random.PRNGKey(0)
    small = autobatch.probe_batch_size(
        _toy_fn, example, key, budget=2 * 1024**3, max_batch=10**7
    )
    large = autobatch.probe_batch_size(
        _toy_fn, example, key, budget=64 * 1024**3, max_batch=10**7
    )
    assert small is not None and large is not None
    assert large > small, "a 32x larger budget must admit a larger batch"


def test_heuristic_has_no_ceiling_and_scales_with_memory():
    """`eval_batch_size: null` means find the max, so nothing caps the guess."""
    tiny = heuristic_batch_size(int(7.2 * 1024**3))
    huge = heuristic_batch_size(H100_REFERENCE_BUDGET * 4)

    assert 0 < tiny < huge
    # A datacenter-class budget is no longer clamped to the old 30k default.
    assert huge > CONFIGURED_MAX
    assert tiny % autobatch.SHAPE_QUANTUM == 0


def test_heuristic_respects_the_data_size():
    """Never batch more voxels than there are."""
    assert heuristic_batch_size(H100_REFERENCE_BUDGET * 4, max_batch=1000) <= 1000


def test_configured_value_is_used_verbatim(cache_dir):
    """An explicit config value pins the batch: no probe, no clamping."""

    def fail(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("probe should not run for a configured value")

    resolved = resolve_batch_size(
        fail,
        np.zeros((10, 3), dtype=np.float32),
        key_example=jax.random.PRNGKey(0),
        configured=4321,
        cache_key=make_cache_key("configured"),
    )
    assert resolved == 4321


def test_resolve_never_exceeds_the_voxel_count(cache_dir):
    resolved = resolve_batch_size(
        _toy_fn,
        jnp.zeros((1, 32), dtype=jnp.float32),
        key_example=jax.random.PRNGKey(0),
        configured=None,
        cache_key=None,
        num_voxels=777,
    )
    assert 0 < resolved <= 777


def test_probe_stays_within_the_budget():
    """The refinement must not hand back a batch whose own compile overshoots."""
    budget = 512 * 1024**2
    example = (jnp.zeros((1, 32), dtype=jnp.float32),)
    resolved = autobatch.probe_batch_size(
        _toy_fn, example, jax.random.PRNGKey(0), budget=budget, max_batch=10**7
    )
    assert resolved is not None
    measured = autobatch._probe_at(_toy_fn, resolved, example, jax.random.PRNGKey(0))
    assert measured <= budget, "probed batch exceeds the memory budget it fitted"


def test_batch_scales_with_device_count(cache_dir):
    """Each device holds a shard, so total capacity grows with device count."""
    devices = jax.devices()
    one = resolve_batch_size(
        _toy_fn,
        jnp.zeros((1, 32), dtype=jnp.float32),
        key_example=jax.random.PRNGKey(0),
        configured=None,
        devices=devices[:1],
        cache_key=None,
        num_voxels=10**6,
        probe=False,
    )
    many = resolve_batch_size(
        _toy_fn,
        jnp.zeros((1, 32), dtype=jnp.float32),
        key_example=jax.random.PRNGKey(0),
        configured=None,
        devices=devices,
        cache_key=None,
        num_voxels=10**6,
        probe=False,
    )
    assert many == one * len(devices)
    assert many % len(devices) == 0, "shards must divide evenly"


def test_sharding_helpers_cover_every_device():
    devices = jax.devices()
    batch = autobatch.batch_sharding_for(devices)
    replicated = autobatch.replicated_sharding_for(devices)
    assert batch.mesh.size == len(devices)
    assert batch.spec[0] == autobatch.BATCH_AXIS
    assert replicated.spec == () or all(a is None for a in replicated.spec)


def test_multi_device_matches_single_device():
    """Sharded and unsharded runs must agree exactly."""
    devices = jax.devices()
    data = np.arange(600 * 3, dtype=np.float32).reshape(600, 3)

    single = eval_in_batches(
        _toy_fn, jax.random.PRNGKey(0), data, batch_size=128, devices=devices[:1]
    )
    sharded = eval_in_batches(
        _toy_fn, jax.random.PRNGKey(0), data, batch_size=128, devices=devices
    )
    np.testing.assert_array_equal(single, sharded)


def test_random_draws_are_independent_of_the_batch_size():
    """Keys are assigned per row up front, so batching cannot change the draws.

    Without this the sampled maps would depend on the auto-resolved batch size,
    and therefore on which GPU the run happened to land on.
    """

    @jax.jit
    def draw(keys, x):
        return jax.vmap(lambda k: jax.random.normal(k, (4,)))(keys) + x[:, :1]

    data = np.zeros((250, 1), dtype=np.float32)
    reference = eval_in_batches(
        draw, jax.random.PRNGKey(0), data, batch_size=250, min_batch_size=1
    )
    for batch_size in (7, 32, 64, 249, 1000):
        out = eval_in_batches(
            draw, jax.random.PRNGKey(0), data, batch_size=batch_size, min_batch_size=1
        )
        np.testing.assert_allclose(
            out,
            reference,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"draws changed at batch_size={batch_size}",
        )
    # A different seed must still give different draws.
    other = eval_in_batches(
        draw, jax.random.PRNGKey(1), data, batch_size=64, min_batch_size=1
    )
    assert not np.array_equal(other, reference)


def test_random_draws_are_independent_of_the_device_count():
    @jax.jit
    def draw(keys, x):
        return jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys) + x[:, :1]

    data = np.zeros((120, 1), dtype=np.float32)
    devices = jax.devices()
    single = eval_in_batches(
        draw, jax.random.PRNGKey(0), data, batch_size=64, devices=devices[:1]
    )
    sharded = eval_in_batches(
        draw, jax.random.PRNGKey(0), data, batch_size=64, devices=devices
    )
    np.testing.assert_array_equal(single, sharded)


def test_host_budget_tracks_mem_available(tmp_path, monkeypatch):
    """On CPU the budget must come from real availability, never from swap."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       65740236 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:   20971520 kB\n"  # 20 GiB
        "SwapTotal:       8388604 kB\n"
    )
    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return real_open(meminfo, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert autobatch.available_host_memory() == 20 * 1024**3

    budget = autobatch.host_memory_budget()
    # Only a fraction, because the host also holds the signal and the outputs.
    assert budget == int(20 * 1024**3 * autobatch.HOST_MEMORY_FRACTION)
    assert budget < autobatch.available_host_memory(), "must not budget all of RAM"


def test_host_budget_falls_back_when_meminfo_is_unreadable(monkeypatch):
    monkeypatch.setattr(autobatch, "available_host_memory", lambda: None)
    assert autobatch.host_memory_budget() == autobatch.CPU_FALLBACK_BUDGET


def test_scarce_host_memory_yields_a_small_batch(monkeypatch):
    """A machine under pressure should shrink the batch, not start swapping."""
    monkeypatch.setattr(autobatch, "available_host_memory", lambda: 256 * 1024**2)
    scarce = autobatch.host_memory_budget()
    monkeypatch.setattr(autobatch, "available_host_memory", lambda: 64 * 1024**3)
    plentiful = autobatch.host_memory_budget()

    assert scarce < plentiful
    assert autobatch.heuristic_batch_size(scarce) < autobatch.heuristic_batch_size(
        plentiful
    )


def test_cpu_device_budget_uses_host_memory(monkeypatch):
    class NoStats:
        def memory_stats(self):
            return None

    monkeypatch.setattr(autobatch, "host_memory_budget", lambda: 12345)
    assert autobatch.device_memory_budget(NoStats()) == 12345


def test_eval_in_batches_handles_pytree_results():
    """The fused mask pass returns two arrays; batching must trim both."""

    def fn(keys, x):
        return x * 2.0, x.sum(axis=-1)

    data = np.arange(70 * 2, dtype=np.float32).reshape(70, 2)
    doubled, summed = eval_in_batches(
        fn, jax.random.PRNGKey(0), data, batch_size=16, min_batch_size=1
    )
    assert doubled.shape == (70, 2) and summed.shape == (70,)
    np.testing.assert_allclose(doubled, data * 2.0)
    np.testing.assert_allclose(summed, data.sum(axis=-1))


def test_cache_key_includes_the_code_fingerprint(monkeypatch):
    """A size measured on one graph must not survive a code change."""
    autobatch.code_fingerprint.cache_clear()
    key_before = make_cache_key("stage", "model")

    monkeypatch.setattr(autobatch, "code_fingerprint", lambda: "different")
    assert make_cache_key("stage", "model") != key_before


def test_code_fingerprint_follows_the_source(tmp_path, monkeypatch):
    import pathlib

    import dmri

    autobatch.code_fingerprint.cache_clear()
    before = autobatch.code_fingerprint()

    target = pathlib.Path(dmri.__file__).parent / "eval" / "autobatch.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# a change that could alter the graph\n")
        autobatch.code_fingerprint.cache_clear()
        assert autobatch.code_fingerprint() != before
    finally:
        target.write_bytes(original)
        autobatch.code_fingerprint.cache_clear()

    assert autobatch.code_fingerprint() == before, (
        "restoring the source must restore it"
    )


def test_code_fingerprint_is_cheap():
    """It runs on every cache lookup, so it must not cost real time."""
    import time

    autobatch.code_fingerprint.cache_clear()
    started = time.perf_counter()
    autobatch.code_fingerprint()
    assert time.perf_counter() - started < 0.5


def test_progress_bar_only_appears_for_a_terminal(monkeypatch):
    """A bar in a log file or a pipe is noise, so only draw it for a person."""
    from dmri import console
    from dmri.eval.sampling_methods import _voxel_progress

    monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
    console.configure(enabled=True)
    plain = _voxel_progress("Sampling", 100)
    assert plain is not None
    plain.close()

    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    console.configure(enabled=True)
    assert _voxel_progress(None, 100) is None, "no description means no bar"

    bar = _voxel_progress("Sampling", 100)
    assert bar is not None
    bar.close()
    console.set_enabled(False)


def test_batching_falls_back_to_log_lines_without_a_bar(monkeypatch):
    """When no bar is shown the per-batch log lines must still be emitted."""
    monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
    lines = []

    class Recorder:
        def info(self, message, *args):
            lines.append(message % args if args else message)

    data = np.ones((30, 2), dtype=np.float32)
    eval_in_batches(
        lambda keys, x: x * 2.0,
        jax.random.PRNGKey(0),
        data,
        batch_size=10,
        logger=Recorder(),
        min_batch_size=1,
        desc="Sampling",
    )
    assert sum("Evaluating batch" in line for line in lines) == 3


def test_batching_closes_progress_after_an_error(monkeypatch):
    from dmri.eval import sampling_methods

    class Progress:
        closed = False

        def update(self, advance):
            pass

        def reset(self):
            pass

        def close(self):
            self.closed = True

    progress = Progress()
    monkeypatch.setattr(sampling_methods, "_voxel_progress", lambda *args: progress)

    def fail(keys, x):
        raise RuntimeError("compiler failure")

    with pytest.raises(RuntimeError, match="compiler failure"):
        eval_in_batches(
            fail,
            jax.random.PRNGKey(0),
            np.ones((4, 2), dtype=np.float32),
            batch_size=2,
            min_batch_size=1,
            desc="Sampling",
        )
    assert progress.closed


class _SyntheticCost:
    """A workload whose memory is a known affine function of the batch size.

    Per-voxel cost is deliberately higher at tiny batches, which is what makes a
    linear fit from small probes overestimate and undershoot the real ceiling.
    """

    def __init__(self, per_voxel, overhead, small_batch_penalty=0.0):
        self.per_voxel = per_voxel
        self.overhead = overhead
        self.small_batch_penalty = small_batch_penalty
        self.probes = []

    def __call__(self, size):
        self.probes.append(size)
        penalty = self.small_batch_penalty / max(size, 1)
        return self.overhead + size * (self.per_voxel + penalty)


def _probe_with(cost, budget, max_batch=10**7, monkeypatch=None):
    monkeypatch.setattr(autobatch, "_probe_at", lambda fn, size, args, key: cost(size))
    return autobatch.probe_batch_size(
        object(),
        (np.zeros((1, 4), np.float32),),
        jax.random.PRNGKey(0),
        budget,
        max_batch,
    )


def test_probe_finds_the_ceiling_despite_a_misleading_small_batch_fit(monkeypatch):
    """The seed underestimates; the search must grow to recover the headroom."""
    budget = 1_000_000
    cost = _SyntheticCost(per_voxel=50, overhead=1000, small_batch_penalty=200_000)
    resolved = _probe_with(cost, budget, monkeypatch=monkeypatch)

    ceiling = (budget * autobatch.SAFETY_FACTOR - 1000) / 50
    assert resolved <= ceiling, "picked a batch that does not fit"
    assert resolved > 0.7 * ceiling, (
        f"left too much on the table: {resolved} of {ceiling:.0f}"
    )
    assert max(cost.probes) <= ceiling * 2, "probed absurdly far past the ceiling"


def test_probe_shrinks_when_the_seed_does_not_fit(monkeypatch):
    """The opposite error: the seed overestimates and must be walked back."""
    budget = 1_000_000
    cost = _SyntheticCost(per_voxel=500, overhead=1000)
    resolved = _probe_with(cost, budget, monkeypatch=monkeypatch)

    assert cost(resolved) <= budget * autobatch.SAFETY_FACTOR, "resolved size overflows"


def test_probe_never_exceeds_the_data_size(monkeypatch):
    cost = _SyntheticCost(per_voxel=1, overhead=0)
    resolved = _probe_with(cost, 10**12, max_batch=5_000, monkeypatch=monkeypatch)
    assert resolved <= 5_000


def test_probe_result_always_fits_its_own_budget(monkeypatch):
    """Whatever the cost curve, the answer must satisfy the budget it was given."""
    budget = 2_000_000
    for per_voxel, overhead, penalty in [
        (10, 0, 0),
        (137, 5_000, 1_000_000),
        (1, 1_900_000, 0),
        (900, 0, 50_000),
    ]:
        cost = _SyntheticCost(per_voxel, overhead, penalty)
        resolved = _probe_with(cost, budget, monkeypatch=monkeypatch)
        assert (
            resolved >= autobatch.DEFAULT_MIN_BATCH
            or resolved == autobatch.DEFAULT_MIN_BATCH
        )
        if resolved > autobatch.DEFAULT_MIN_BATCH:
            assert cost(resolved) <= budget * autobatch.SAFETY_FACTOR, (
                f"per_voxel={per_voxel} overhead={overhead}: {resolved} overflows"
            )


def test_probe_compile_count_is_bounded(monkeypatch):
    """Each probe is a real compile, so the search must not run away."""
    cost = _SyntheticCost(per_voxel=50, overhead=1000, small_batch_penalty=200_000)
    _probe_with(cost, 1_000_000, monkeypatch=monkeypatch)
    assert len(cost.probes) <= 12, f"took {len(cost.probes)} compiles"
