"""Tests for evaluation-time precision selection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from dmri.eval.precision import (
    FP32,
    PRECISION_CHOICES,
    PRECISION_PRESETS,
    apply_precision_to_cfg,
    is_half,
    normalize_precision,
    resolve_precision_for_backend,
)


def test_default_is_fp32_and_changes_nothing():
    assert normalize_precision(None) == FP32
    assert not is_half(FP32)
    assert PRECISION_PRESETS[FP32] == {}

    cfg = OmegaConf.create({"model": {"model_dim": 64}})
    apply_precision_to_cfg(cfg, FP32)
    assert "dtype" not in cfg.model


@pytest.mark.parametrize("name", ["bf16", "fp16"])
def test_half_presets_keep_params_and_accumulation_in_fp32(name):
    preset = PRECISION_PRESETS[name]
    assert preset["param_dtype"] == "float32", "weights must stay fp32"
    assert preset["preferred_element_type"] == "float32", "accumulate in fp32"
    assert preset["dtype"] in ("bfloat16", "float16")
    # An explicit dot-algorithm preset is deliberately not set: F16_F16_F32 is
    # rejected by the attention kernels on Ampere.
    assert "precision" not in preset
    assert is_half(name)


@pytest.mark.parametrize("name", PRECISION_CHOICES)
def test_apply_precision_patches_model_cfg(name):
    cfg = OmegaConf.create({"model": {"model_dim": 64}})
    resolved = apply_precision_to_cfg(cfg, name, platform="gpu")
    assert resolved == name
    for key, value in PRECISION_PRESETS[name].items():
        assert cfg.model[key] == value
    # Unrelated keys survive.
    assert cfg.model.model_dim == 64


def test_fp16_falls_back_to_fp32_on_cpu():
    assert resolve_precision_for_backend("fp16", "cpu") == FP32
    assert resolve_precision_for_backend("fp16", "gpu") == "fp16"
    assert resolve_precision_for_backend("bf16", "cpu") == "bf16"

    cfg = OmegaConf.create({"model": {"model_dim": 64}})
    resolved = apply_precision_to_cfg(cfg, "fp16", platform="cpu")

    assert resolved == FP32
    assert "dtype" not in cfg.model


def test_aliases_and_unknown_values():
    assert normalize_precision("float32") == FP32
    assert normalize_precision("bfloat16") == "bf16"
    assert normalize_precision("half") == "fp16"
    assert normalize_precision("FP16") == "fp16"
    with pytest.raises(ValueError, match="Unknown precision"):
        normalize_precision("int8")


def test_precision_reaches_layers_but_not_parameters():
    """The preset must set the compute dtype while weights stay float32."""
    for name in ("bf16", "fp16"):
        preset = PRECISION_PRESETS[name]
        layer = nnx.Linear(
            8,
            8,
            rngs=nnx.Rngs(0),
            dtype=jnp.dtype(preset["dtype"]),
            param_dtype=jnp.float32,
            preferred_element_type=jnp.float32,
        )
        assert layer.kernel.value.dtype == jnp.float32, "weights must stay fp32"
        out = jax.jit(lambda m, x: m(x))(layer, jnp.ones((4, 8), jnp.float32))
        assert out.dtype == jnp.float32, "outputs must come back in fp32"


@pytest.mark.parametrize("name", ["bf16", "fp16"])
def test_half_precision_network_stays_close_to_fp32(name):
    """A half-precision forward pass must track fp32 closely and stay finite."""
    rngs = nnx.Rngs(0)
    dtype = jnp.dtype(PRECISION_PRESETS[name]["dtype"])
    kwargs = dict(param_dtype=jnp.float32, preferred_element_type=jnp.float32)
    ref = nnx.Linear(32, 32, rngs=nnx.Rngs(0), **kwargs)
    half = nnx.Linear(32, 32, rngs=rngs, dtype=dtype, **kwargs)
    nnx.update(half, nnx.state(ref))

    x = jax.random.normal(jax.random.PRNGKey(1), (16, 32), jnp.float32)
    out_ref = jax.jit(lambda m, v: m(v))(ref, x)
    out_half = jax.jit(lambda m, v: m(v))(half, x)

    assert np.all(np.isfinite(out_half))
    # bf16 carries ~8 mantissa bits, fp16 ~11; both stay well inside 5% here.
    np.testing.assert_allclose(out_half, out_ref, rtol=0.05, atol=0.05)


def test_theta_samples_are_cast_back_to_fp32():
    """Whatever the network emits, the sampler hands downstream code fp32.

    The stick-breaking bijection, the correctors and the simulator likelihood
    all assume fp32, so this cast is what makes half precision safe.
    """
    from dmri.eval.sampling_methods import build_base_theta_sample_fn

    class FakeSim:
        theta_dim = 4

    class FakeTokenizer:
        simulator = FakeSim

    class FakeModel:
        tokenizer = FakeTokenizer

        @staticmethod
        def sample_theta(
            key,
            acq,
            x,
            mask,
            num_steps=None,
            last_euler_step=True,
            t_min=None,
            t_max=None,
        ):
            # Emit half precision, as a bf16/fp16 network would.
            return jnp.zeros((FakeSim.theta_dim,), jnp.bfloat16) + x[0].astype(
                jnp.bfloat16
            )

    mask = jnp.ones((2, 3, 4), dtype=bool)
    fn = build_base_theta_sample_fn(
        3, FakeModel, acq=None, model_mask=mask, params={"num_steps": 1}
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    out = fn(keys, jnp.ones((2, 3), jnp.float32), mask)
    assert out.dtype == jnp.float32


def test_non_finite_warning_names_precision(caplog):
    from dmri.eval.eval_script import _warn_on_non_finite_thetas

    logger = __import__("logging").getLogger("test-precision")
    thetas = np.array([[1.0, np.nan], [2.0, 3.0]])

    with caplog.at_level("WARNING", logger="test-precision"):
        fraction = _warn_on_non_finite_thetas(thetas, "fp16", logger)
    assert fraction == pytest.approx(0.25)
    assert "fp16" in caplog.text and "precision=fp32" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="test-precision"):
        assert _warn_on_non_finite_thetas(np.ones((2, 2)), "fp32", logger) == 0.0
    assert caplog.text == ""


def test_sampler_and_corrector_are_separately_batchable():
    """The two stages are built independently so each gets its own batch size.

    The corrector works on theta rather than network activations, so it fits a
    much larger batch; fusing them held it to the sampler's limit.
    """
    from dmri.eval.sampling_methods import (
        build_base_theta_sample_fn,
        build_corrector_fn,
    )

    class FakeSim:
        theta_dim = 4

    class FakeTokenizer:
        simulator = FakeSim

    class FakeModel:
        tokenizer = FakeTokenizer

        @staticmethod
        def sample_theta(
            key,
            acq,
            x,
            mask,
            num_steps=None,
            last_euler_step=True,
            t_min=None,
            t_max=None,
        ):
            return jnp.zeros((FakeSim.theta_dim,), jnp.float32) + x[0]

    num_samples, num_voxels = 3, 5
    mask = jnp.ones((num_voxels, num_samples, 4), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(0), num_voxels)
    x = jnp.ones((num_voxels, 2), jnp.float32)

    base = build_base_theta_sample_fn(
        num_samples, FakeModel, acq=None, model_mask=mask, params={"num_steps": 1}
    )
    thetas = base(keys, x, mask)
    assert thetas.shape == (num_voxels, num_samples, FakeSim.theta_dim)
    assert thetas.dtype == jnp.float32

    corrector = build_corrector_fn("uncorrected", FakeModel, None, mask, FakeSim, {})
    corrected = corrector(keys, thetas, x, mask)
    assert corrected.shape == thetas.shape
    # The corrector consumes thetas produced by the sampler, so the two can run
    # as separate passes over the voxel axis.
    np.testing.assert_array_equal(corrected, thetas)


def test_network_evaluations_per_sample_counts_the_final_euler_step():
    """`num_steps` alone understates the cost: the solver also inits and corrects."""
    from dmri.eval.sampling_methods import network_evaluations_per_sample

    # 1 init + (num_steps - 1) intervals + 1 final Euler correction.
    assert network_evaluations_per_sample({"num_steps": 40}) == 41
    assert (
        network_evaluations_per_sample({"num_steps": 40, "last_euler_step": False})
        == 40
    )
    assert network_evaluations_per_sample({"num_steps": 1}) == 2
    assert network_evaluations_per_sample({"num_steps": 0}) == 0
    # Turning the correction off saves exactly one evaluation per sample.
    with_step = network_evaluations_per_sample({"num_steps": 10})
    without = network_evaluations_per_sample({
        "num_steps": 10,
        "last_euler_step": False,
    })
    assert with_step - without == 1


def test_fused_mask_pass_matches_separate_heads():
    """Sampling masks and scoring feasible masks share one encoding of the signal."""
    from dmri.eval.sampling_methods import build_mask_sample_fn

    calls = {"encode": 0}

    class FakeModel:
        def _encode_observations(self, acq, x):
            calls["encode"] += 1
            return x[:1], x

        def sample_mask(self, rng, acq, x, mask_prior=None, y_ctx=None, y=None):
            return (jax.random.uniform(rng, (4,)) > 0.5).astype(jnp.float32)

        def log_prob_mask(self, mask, acq, x, mask_prior=None, y_ctx=None, y=None):
            return -jnp.sum(mask) - x.sum()

    model = FakeModel()
    feasible = jnp.array([[True, False, True, True], [True, True, False, False]])
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jnp.ones((3, 5), jnp.float32)

    masks_only = build_mask_sample_fn("naive", 2, model, None, 0.3)(keys, x)
    masks, probs = build_mask_sample_fn(
        "naive", 2, model, None, 0.3, feasible_models=feasible
    )(keys, x)

    # Fusing must not disturb the masks themselves.
    np.testing.assert_array_equal(masks, masks_only)
    assert probs.shape == (3, len(feasible))
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)


def test_mask_heads_accept_a_precomputed_encoding():
    """The y/y_ctx hook is what lets one encoding serve both mask heads."""
    import inspect

    from dmri.nn.dmri_reconstruction_model import DMRIInferenceModel

    for name in ("sample_mask", "log_prob_mask", "sample_theta"):
        params = inspect.signature(getattr(DMRIInferenceModel, name)).parameters
        assert "y" in params and "y_ctx" in params, f"{name} cannot reuse an encoding"


def test_batch_cache_key_covers_everything_that_changes_the_graph():
    """A cached batch size is only valid for the graph it was measured on."""
    import numpy as np
    from omegaconf import OmegaConf

    from dmri.eval.eval_script import _batch_cache_key

    def make(num_steps=40, precision="fp32", corrector_steps=3, num_samples=8):
        return OmegaConf.create({
            "model_name": "m",
            "mask_sample": {"n_samples": 8, "method": "naive", "p_mask": 0.3},
            "theta_sample": {
                "num_samples": num_samples,
                "params": {
                    "num_steps": num_steps,
                    "t_max": 80,
                    "last_euler_step": True,
                },
                "corrector": {"name": "auto", "num_steps": corrector_steps},
            },
            "model_selection": {"name": "average"},
            "precision": precision,
            "sample_mask": True,
            "export_model_selection": None,
            "export": {"type": "ball3stick"},
        })

    data = np.zeros((100, 96), np.float32)

    def key(config, stage="theta_base"):
        return _batch_cache_key(config, stage, data, None)

    base = key(make())

    assert key(make()) == base, "an unchanged config must reuse its cached size"
    # Each of these changes the memory footprint of the compiled stage.
    assert key(make(num_steps=10)) != base
    assert key(make(precision="bf16")) != base
    assert key(make(corrector_steps=5)) != base
    assert key(make(num_samples=50)) != base
    # Different stages of the same run are independent.
    assert key(make(), stage="theta_corrector") != base

    # Solver settings must not invalidate a stage they cannot affect: changing
    # the ODE step count re-tunes theta sampling, not mask sampling.
    mask_key = key(make(), stage="mask_sample")
    assert key(make(num_steps=10), stage="mask_sample") == mask_key
    assert key(make(precision="bf16"), stage="mask_sample") != mask_key


def test_batch_stages_use_the_shared_stage_names():
    """The autotuner and the runner must agree on each stage's identity.

    They previously used separate string literals, so a mismatch silently made
    the runner re-resolve a size the autotuner had already measured.
    """
    from omegaconf import OmegaConf

    from dmri.eval import eval_script

    known = {
        eval_script.STAGE_MASK,
        eval_script.STAGE_THETA_BASE,
        eval_script.STAGE_THETA_CORRECTOR,
        eval_script.STAGE_FEASIBLE,
    }
    assert len(known) == 4, "stage names must be distinct"

    class FakeSim:
        theta_dim = 11
        model_types = (None,) * 4
        noise_types = (None,)

    cfg = OmegaConf.create({
        "sample_mask": True,
        "sample_theta": True,
        "mask_sample": {
            "method": "naive",
            "n_samples": 4,
            "p_mask": 0.3,
            "eval_batch_size": None,
        },
        "theta_sample": {
            "num_samples": 4,
            "params": {"num_steps": 2},
            "corrector": {"name": "mcmc_corrected"},
            "eval_batch_size": 1234,
        },
        "model_selection": {"name": "average"},
        "export_model_selection": None,
        "export": {"type": "ball3stick"},
    })

    stages = eval_script._batch_stages(cfg, model=None, acq=None, sim_type=FakeSim)
    names = [stage.name for stage in stages]

    assert set(names) <= known, f"unknown stage name in {names}"
    assert len(names) == len(set(names)), "stage names must be unique within a run"
    assert eval_script.STAGE_THETA_BASE in names
    assert eval_script.STAGE_THETA_CORRECTOR in names

    # Each stage reads its batch size from its own config section.
    by_name = {stage.name: stage for stage in stages}
    assert by_name[eval_script.STAGE_THETA_BASE].configured(cfg) == 1234
    assert by_name[eval_script.STAGE_MASK].configured(cfg) is None


def test_batch_stage_orders_its_probe_arguments():
    """The corrector takes thetas before the signal; the sampler does not."""
    import numpy as np

    from dmri.eval.eval_script import BatchStage

    data = np.zeros((4, 3), np.float32)
    mask = np.ones((1, 2, 5), bool)
    thetas = np.zeros((1, 2, 11), np.float32)

    sampler = BatchStage("s", "s", build=lambda: None, trailing_args=(mask,))
    assert [a.shape for a in sampler.probe_args(data)] == [data.shape, mask.shape]

    corrector = BatchStage(
        "c", "c", build=lambda: None, leading_args=(thetas,), trailing_args=(mask,)
    )
    assert [a.shape for a in corrector.probe_args(data)] == [
        thetas.shape,
        data.shape,
        mask.shape,
    ]
