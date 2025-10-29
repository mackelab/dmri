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
)
from dmri.nn.embedding_net import (
    BvalBvecSignalEmbeddingNet,
    DMRIEmbeddingConfig,
    SSFPEmbeddingNet,
)
from dmri.nn.simformer import (
    DMRIThetaInferenceConfig,
    EDMSimformer,
    GaussianFourierEmbedding,
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
