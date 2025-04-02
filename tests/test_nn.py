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
from dmri.nn.embedding_net import BvalBvecSignalEmbeddingNet, DMRIEmbeddingConfig
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

    output = model(model_mask, theta, x, bvals, bvecs, p_mask)
    assert isinstance(output, tuple)  # Model returns multiple outputs
    assert output[0].shape == model_mask.shape
    assert output[1].shape == theta.shape


def c(rng):
    """Test instantiation of BvalBvecSignalEmbeddingNet."""
    embedding_net = BvalBvecSignalEmbeddingNet(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        rngs=rng,
    )

    # Test that the model can be called with dummy inputs
    bvals = jnp.zeros((32,))
    bvecs = jnp.zeros((32, 3))
    signals = jnp.zeros((32,))

    output = embedding_net(bvals, bvecs, signals)
    assert output.shape == (32, 64)


def test_bval_bvec_signal_embedding_net(rng, data):
    """Test instantiation and forward pass of BvalBvecSignalEmbeddingNet."""
    embedding_net = BvalBvecSignalEmbeddingNet(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        rngs=rng,
    )

    # Get data from fixture
    _, _, signals, bvals, bvecs, _ = data
    # Test with full batch
    output_batch = embedding_net(bvals, bvecs, signals)
    assert output_batch.shape == (100, 64, 64)


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
    encoded_mask = tokenizer.encode(model_mask=model_mask)
    assert encoded_mask.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )

    # Test encode with theta
    encoded_theta = tokenizer.encode(theta=theta, model_mask=model_mask)
    assert encoded_theta.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )

    # Test decode
    decoded_theta = tokenizer.decode(encoded_theta, model_mask=model_mask)
    assert decoded_theta.shape == theta.shape

    # Test with alpha prior
    alpha_prior = jnp.ones((100, len(simulator.model_types)))
    encoded_with_prior = tokenizer.encode(
        model_mask=model_mask,
        alpha_prior=alpha_prior,
    )
    assert encoded_with_prior.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )


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
    encoded_mask = tokenizer.encode(model_mask=model_mask)
    assert encoded_mask.shape == (
        100,
        1 + len(simulator.model_types) + len(simulator.noise_types),
        64,
    )

    # Test encode with theta
    encoded_theta = tokenizer.encode(theta=theta, model_mask=model_mask)
    assert encoded_theta.shape == (
        100,
        len(simulator.model_types) + len(simulator.noise_types) + len(simulator.model_types)-1,
        64,
    )

    # Test decode
    decoded_theta = tokenizer.decode(encoded_theta, model_mask=model_mask)
    assert decoded_theta.shape == theta.shape

  