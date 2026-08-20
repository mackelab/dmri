import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from dmri.nn.autoregressive import (
    BinaryAutoregressiveDecoder,
    DMRIModelSelectionAmortizedPriorConfig,
    DMRIModelSelectionConfig,
)
from dmri.nn.dmri_reconstruction_model import (
    DMRIInferenceModel,
    DMRIInferenceModelConfig,
    DMRIInferenceModelConfigMaskPriorAmortized,
    DMRIInferenceModelConfigMaskPriorAmortizedPPP,
)
from dmri.nn.embedding_net import (
    BvalBvecSignalEmbeddingNet,
    DMRIEmbeddingConfig,
    SSFPEmbeddingNet,
)
from dmri.nn.simformer import (
    DMRIThetaInferenceConfig,
    masked_standard_normal_log_prob,
)
from dmri.nn.tokenizer import DMRITokenizer
from dmri.simulators import Ball2Stick, Ball3Stick, BallStickZeppelinNoise


@pytest.fixture
def rng():
    return nnx.Rngs(0)


@pytest.fixture(params=[Ball2Stick, Ball3Stick, BallStickZeppelinNoise])
def simulator(request):
    return request.param


@pytest.fixture
def data(simulator):
    """Create test data for the simulator."""
    theta_dim = simulator.theta_dim
    num_components = len(simulator.model_types) + len(simulator.noise_types)
    p_mask = jnp.ones((100, 1))
    # Create model_mask with correct number of components
    model_mask = jnp.ones((100, num_components), dtype=jnp.bool_)
    # Create theta with correct dimension
    theta = jnp.zeros((100, theta_dim))
    # Create signal data
    x = jnp.zeros((100, 64))
    bvals = jnp.zeros((100, 64))
    bvecs = jnp.zeros((100, 64, 3))
    return model_mask, theta, x, bvals, bvecs, p_mask


def test_dmri_inference_model_amortized(rng, simulator, data):
    """Test instantiation of DMRIInferenceModel with amortized prior."""
    cfg = DMRIInferenceModelConfigMaskPriorAmortized(
        simulator=simulator,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionAmortizedPriorConfig(),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, rng)
    model_mask, theta, x, bvals, bvecs, p_mask = data

    # Create acquisition scheme
    from dmri.simulators.acquisition_scheme import acquisition_scheme

    acq = acquisition_scheme(bvals=bvals, bvecs=bvecs)

    output = model(model_mask, theta, x, acq, mask_prior=p_mask)
    assert isinstance(output, tuple)  # Model returns multiple outputs
    assert output[0].shape == model_mask.shape
    assert output[1].shape == theta.shape


def test_legacy_ppp_checkpoint_architecture_still_builds(rng):
    cfg = DMRIInferenceModelConfigMaskPriorAmortizedPPP(
        simulator=Ball3Stick,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionAmortizedPriorConfig(
            outlayer="mlp", outnorm=False
        ),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )

    model = DMRIInferenceModel(cfg, rng)

    assert type(model.tokenizer).__name__ == "DMRITokenizerPPP"
    assert type(model.model_decoder.output).__name__ == "MLP"


def test_ppp_decoder_inputs_do_not_leak_their_own_target_bit(rng):
    """`logits[i]` predicts `model_mask[i]`, so decoder position `i` must depend
    only on bits `< i`.

    `DMRITokenizerPPP` therefore overrides `embed_cfgs` to skip the base class's
    ``idx_tokens * model_mask`` masking: `embed_model_mask` packs cfg token `j`
    into position `j`, so masking there would write bit `j` into the very
    position that has to predict it. Without the override the model-selection
    head can read the answer off its own input.
    """
    cfg = DMRIInferenceModelConfigMaskPriorAmortizedPPP(
        simulator=Ball3Stick,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionAmortizedPriorConfig(
            outlayer="mlp", outnorm=False
        ),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, rng)
    tokenizer, decoder = model.tokenizer, model.model_decoder

    num_components = len(Ball3Stick.model_types) + len(Ball3Stick.noise_types)
    ones = jnp.ones((1, num_components), dtype=jnp.bool_)
    base = decoder._encode_model_mask(ones, tokenizer)

    for bit in range(num_components):
        flipped = ones.at[0, bit].set(False)
        delta = jnp.abs(decoder._encode_model_mask(flipped, tokenizer) - base)
        touched = jnp.where(delta.max(axis=-1)[0] > 1e-8)[0]
        assert touched.size > 0, f"bit {bit} does not reach the decoder at all"
        assert int(touched.min()) > bit, (
            f"mask bit {bit} leaks into position(s) {touched.tolist()}; "
            f"position {bit} predicts bit {bit} and must not see it"
        )


def test_binary_decoder_requires_mask_prior(simulator):
    """Decoder should enforce mask prior when the config requests it."""
    decoder = BinaryAutoregressiveDecoder(
        rngs=nnx.Rngs(10),
        model_dim=64,
        prior_params_embed_dim=8,
        mask_prior_dim=2,
        additional_context_dim=3,
    )
    tokenizer = DMRITokenizer(simulator=simulator, token_dim=64, rngs=nnx.Rngs(11))
    batch = 4
    num_components = tokenizer.num_models + tokenizer.num_noises
    model_mask = jnp.ones((batch, num_components), dtype=jnp.bool_)
    additional_context = jnp.zeros((batch, 3))

    with pytest.raises(ValueError, match="mask_prior must be provided"):
        decoder(model_mask, tokenizer, additional_context=additional_context)

    mask_prior = jnp.zeros((batch, 2))
    logits = decoder(
        model_mask,
        tokenizer,
        mask_prior=mask_prior,
        additional_context=additional_context,
    )
    assert logits.shape == model_mask.shape


def test_binary_decoder_fills_missing_additional_context(simulator):
    """Decoder should fall back to zeros when only extra context is expected."""
    decoder = BinaryAutoregressiveDecoder(
        rngs=nnx.Rngs(12),
        model_dim=64,
        prior_params_embed_dim=0,
        additional_context_dim=5,
    )
    tokenizer = DMRITokenizer(simulator=simulator, token_dim=64, rngs=nnx.Rngs(13))
    batch = 3
    num_components = tokenizer.num_models + tokenizer.num_noises
    model_mask = jnp.zeros((batch, num_components), dtype=jnp.bool_)

    logits = decoder(model_mask, tokenizer)
    assert logits.shape == model_mask.shape


def test_binary_decoder_prediction_matches_training_logits(simulator):
    """Prediction and teacher-forced training must use the same output head path."""
    decoder = BinaryAutoregressiveDecoder(
        rngs=nnx.Rngs(21),
        model_dim=32,
        num_heads=2,
        num_layers=2,
        prior_params_embed_dim=0,
    )
    tokenizer = DMRITokenizer(simulator=simulator, token_dim=32, rngs=nnx.Rngs(22))
    num_components = tokenizer.num_models + tokenizer.num_noises
    model_mask = jnp.array([True] * num_components)

    input_tokens = tokenizer.encode(model_mask=model_mask)
    output_tokens = decoder._forward_tokens(input_tokens, y=None)
    training_logits = decoder.output(output_tokens)[..., :-1, 0]

    prediction_logits = decoder(model_mask, tokenizer)
    assert jnp.allclose(prediction_logits, training_logits)


def test_dmri_inference_model_requires_mask_prior(simulator, data):
    """High-level model should surface mask-prior requirement errors."""
    cfg = DMRIInferenceModelConfigMaskPriorAmortized(
        simulator=simulator,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionAmortizedPriorConfig(),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, nnx.Rngs(14))
    model_mask, theta, x, bvals, bvecs, _ = data

    from dmri.simulators.acquisition_scheme import acquisition_scheme

    acq = acquisition_scheme(bvals=bvals, bvecs=bvecs)

    with pytest.raises(ValueError, match="mask_prior must be provided"):
        model(model_mask, theta, x, acq)


def test_dmri_inference_model_without_mask_prior(simulator, data):
    """Mask prior should be optional when prior embedding is disabled."""
    cfg = DMRIInferenceModelConfig(
        simulator=simulator,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionConfig(prior_params_embed_dim=0),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, nnx.Rngs(15))
    model_mask, theta, x, bvals, bvecs, _ = data

    from dmri.simulators.acquisition_scheme import acquisition_scheme

    acq = acquisition_scheme(bvals=bvals, bvecs=bvecs)
    logits, theta_pred = model(model_mask, theta, x, acq)

    assert logits.shape == model_mask.shape
    assert theta_pred.shape == theta.shape


def test_dmri_inference_model_mask_sampling_and_log_prob(rng, simulator, data):
    """Sampling / log-prob APIs should work end-to-end on the high-level model."""
    cfg = DMRIInferenceModelConfigMaskPriorAmortized(
        simulator=simulator,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionAmortizedPriorConfig(),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, rng)

    model_mask, _, x, bvals, bvecs, _ = data
    model_mask_single = model_mask[:1]
    x_single = x[:1]
    bvals_single = bvals[:1]
    bvecs_single = bvecs[:1]

    from dmri.simulators.acquisition_scheme import acquisition_scheme

    acq_single = acquisition_scheme(bvals=bvals_single, bvecs=bvecs_single)

    assert model.mask_prior_dim is not None
    mask_prior_sample = jnp.zeros((model.mask_prior_dim,))
    sampled_mask = model.sample_mask(
        rng.next(),
        acq_single,
        x_single,
        mask_prior=mask_prior_sample,
    )
    assert sampled_mask.shape == model_mask_single.shape[-1:]
    assert sampled_mask.dtype == jnp.bool_

    mask_prior_log_prob = jnp.zeros((model_mask_single.shape[0], model.mask_prior_dim))
    log_prob = model.log_prob_mask(
        model_mask_single,
        acq_single,
        x_single,
        mask_prior=mask_prior_log_prob,
    )
    assert log_prob.shape == (model_mask_single.shape[0],)
    assert jnp.all(jnp.isfinite(log_prob))


def test_dmri_inference_model_with_gated_simformer(simulator, data):
    """Ensure gated transformer configuration integrates without errors."""
    cfg = DMRIInferenceModelConfig(
        simulator=simulator,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionConfig(prior_params_embed_dim=0),
        theta_inference_cfg=DMRIThetaInferenceConfig(
            gate_attention=True,
            gate_mlp=True,
        ),
    )
    model = DMRIInferenceModel(cfg, nnx.Rngs(16))
    model_mask, theta, x, bvals, bvecs, _ = data

    from dmri.simulators.acquisition_scheme import acquisition_scheme

    acq = acquisition_scheme(bvals=bvals, bvecs=bvecs)
    logits, theta_pred = model(model_mask, theta, x, acq)

    assert logits.shape == model_mask.shape
    assert theta_pred.shape == theta.shape


def test_dmri_inference_model_custom_embedding_dims(rng, simulator, data):
    """Custom encoder output dims should propagate to decoders."""
    cfg = DMRIInferenceModelConfig(
        simulator=simulator,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(
            use_global_summary_token=True,
            y_seq_dim=32,
            y_glob_dim=16,
        ),
        model_selection_cfg=DMRIModelSelectionConfig(prior_params_embed_dim=0),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, rng)
    assert model.model_decoder.kv_in_features == 32
    assert model.y_ctx_dim == 16
    assert model.inference_decoder.net.kv_in_features == 32

    model_mask, theta, x, bvals, bvecs, _ = data

    from dmri.simulators.acquisition_scheme import acquisition_scheme

    acq = acquisition_scheme(bvals=bvals, bvecs=bvecs)
    logits, theta_pred = model(model_mask, theta, x, acq)

    assert logits.shape == model_mask.shape
    assert theta_pred.shape == theta.shape


@pytest.mark.parametrize("use_flashattn", [False])
def test_bval_bvec_signal_embedding_net(use_flashattn, rng, data):
    """Test instantiation and forward pass of BvalBvecSignalEmbeddingNet."""

    embedding_net = BvalBvecSignalEmbeddingNet(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        use_flash_attention=use_flashattn,
        rngs=rng,
    )

    # Get data from fixture
    _, _, signals, bvals, bvecs, _ = data

    # Create acquisition scheme
    from dmri.simulators.acquisition_scheme import acquisition_scheme

    acq = acquisition_scheme(bvals=bvals, bvecs=bvecs)

    # Test with full batch and default summary disabled
    global_summary, sequence_tokens = embedding_net(acq, signals)
    assert global_summary is None
    assert sequence_tokens.shape == (100, 64, 64)

    # Test with global summary enabled
    rng_summary = nnx.Rngs(1)
    embedding_net_summary = BvalBvecSignalEmbeddingNet(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        use_flash_attention=use_flashattn,
        use_global_summary_token=True,
        rngs=rng_summary,
    )
    summary_token, summarized_sequence = embedding_net_summary(acq, signals)
    assert summary_token.shape == (100, 64)
    assert summarized_sequence.shape == (100, 64, 64)


def test_signal_soft_squash_can_be_disabled():
    signals = jnp.array([0.0, 0.25, 1.0])
    embedding = BvalBvecSignalEmbeddingNet(
        model_dim=8,
        num_heads=1,
        num_layers=1,
        soft_squash_signals=False,
        rngs=nnx.Rngs(20),
    )

    assert jnp.allclose(embedding.transform_signals(signals), signals)


def test_masked_standard_normal_log_prob_ignores_inactive_values():
    mask = jnp.array([True, False, True])
    first = masked_standard_normal_log_prob(
        jnp.array([1.0, 100.0, -2.0]), jnp.array(1.0), mask
    )
    second = masked_standard_normal_log_prob(
        jnp.array([1.0, -100.0, -2.0]), jnp.array(1.0), mask
    )

    expected = jax.scipy.stats.norm.logpdf(jnp.array([1.0, -2.0])).sum()
    assert jnp.allclose(first, expected)
    assert jnp.allclose(second, expected)


@pytest.mark.parametrize("use_flashattn", [False])
def test_ssfp_embedding_net(use_flashattn, rng):
    """Test instantiation and forward pass of SSFPEmbeddingNet."""

    embedding_net = SSFPEmbeddingNet(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        use_flash_attention=use_flashattn,
        rngs=rng,
    )

    # Create SSFP acquisition scheme
    from dmri.simulators.acquisition_scheme import ssfp_acquisition_scheme

    # Create test data for SSFP
    batch_size = 100
    num_acquisitions = 64

    # Create SSFP acquisition parameters
    T1 = jnp.ones((batch_size, num_acquisitions)) * 1000  # ms
    T2 = jnp.ones((batch_size, num_acquisitions)) * 80  # ms
    B1 = jnp.ones((batch_size, num_acquisitions)) * 1.0  # unitless
    diffGradAmps = jnp.ones((batch_size, num_acquisitions)) * 50  # T/m
    flipAngles = jnp.ones((batch_size, num_acquisitions)) * 14.0  # degrees
    TRs = jnp.ones((batch_size, num_acquisitions)) * 0.021  # seconds
    diffGradDur = jnp.ones((batch_size, num_acquisitions)) * 0.01016  # seconds
    bvecs = jax.random.normal(rng.next(), (batch_size, num_acquisitions, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)

    acq = ssfp_acquisition_scheme(
        bvecs=bvecs,
        T1_raw=T1,
        T2_raw=T2,
        B1=B1,
        diffGradAmps_raw=diffGradAmps,
        flipAngles_raw=flipAngles,
        TRs=TRs,
        diffGradDur=diffGradDur,
    )

    # Create signal data
    signals = jax.random.normal(rng.next(), (batch_size, num_acquisitions))

    # Test with full batch and default summary disabled
    global_summary, sequence_tokens = embedding_net(acq, signals)
    assert global_summary is None
    assert sequence_tokens.shape == (batch_size, num_acquisitions, 64)

    # Test with global summary enabled
    rng_summary = nnx.Rngs(2)
    embedding_net_summary = SSFPEmbeddingNet(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        use_flash_attention=use_flashattn,
        use_global_summary_token=True,
        rngs=rng_summary,
    )
    summary_token, summarized_sequence = embedding_net_summary(acq, signals)
    assert summary_token.shape == (batch_size, 64)
    assert summarized_sequence.shape == (batch_size, num_acquisitions, 64)


def test_dmri_tokenizer(rng, simulator, data):
    """Test instantiation and functionality of DMRITokenizer."""
    tokenizer = DMRITokenizer(
        simulator=simulator,
        rngs=rng,
        token_dim=64,
    )

    # Get data from fixture
    model_mask, theta, _, _, _, _ = data

    # Test encode with model mask
    model_idx = tuple(range(len(simulator.model_types)))
    noise_idx = tuple(range(len(simulator.noise_types)))

    encoded_mask = tokenizer.encode(
        model_mask=model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert encoded_mask.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )

    # Test encode with theta
    encoded_theta = tokenizer.encode(
        theta=theta,
        model_mask=model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert encoded_theta.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )

    # Test decode
    decoded_theta = tokenizer.decode(
        encoded_theta,
        model_mask=model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert decoded_theta.shape == theta.shape

    # Test with alpha prior
    alpha_prior = jnp.ones((100, len(simulator.model_types)))
    encoded_with_prior = tokenizer.encode(
        theta=theta,
        model_mask=model_mask,
        alpha_prior=alpha_prior,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert encoded_with_prior.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )

    # Test mask coding
    mask2 = jnp.ones_like(model_mask)
    encoded_theta2 = tokenizer.encode(
        theta=theta,
        model_mask=mask2,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )

    # You only be different where mask is False in first encoding
    diff = jnp.all(encoded_theta == encoded_theta2, axis=-1)
    token_mask = tokenizer.theta_token_mask(
        model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert jnp.all(diff == token_mask), "Mask coding is not correct"


def test_dmri_tokenizer_pp(rng, simulator, data):
    """Test instantiation and functionality of DMRITokenizerPP."""
    from dmri.nn.tokenizer import DMRITokenizerPP

    tokenizer = DMRITokenizerPP(
        simulator=simulator,
        rngs=rng,
        token_dim=64,
    )

    # Get data from fixture
    model_mask, theta, x, bvals, bvecs, p_mask = data

    # Test encode with model mask
    model_idx = tuple(range(len(simulator.model_types)))
    noise_idx = tuple(range(len(simulator.noise_types)))

    encoded_mask = tokenizer.encode(
        model_mask=model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert encoded_mask.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )

    # Test encode with theta
    encoded_theta = tokenizer.encode(
        theta=theta,
        model_mask=model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert encoded_theta.shape == (
        100,
        len(simulator.model_types)
        + len(simulator.noise_types)
        + len(simulator.model_types)
        - 1,
        64,
    )

    # Test decode
    decoded_theta = tokenizer.decode(
        encoded_theta,
        model_mask=model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert decoded_theta.shape == theta.shape

    # Test mask coding
    mask2 = jnp.ones_like(model_mask)
    encoded_theta2 = tokenizer.encode(
        theta=theta,
        model_mask=mask2,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )

    # You only be different where mask is False in first encoding
    diff = jnp.all(encoded_theta == encoded_theta2, axis=-1)
    token_mask = tokenizer.theta_token_mask(
        model_mask,
        model_idx=model_idx,
        noise_idx=noise_idx,
    )
    assert jnp.all(diff == token_mask), "Mask coding is not correct"
