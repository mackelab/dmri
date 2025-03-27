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
from dmri.simulators import MultiCompartment


@pytest.fixture
def rng():
    return nnx.Rngs(0)


def test_gaussian_fourier_embedding(rng):
    """Test instantiation of GaussianFourierEmbedding."""
    embedding = GaussianFourierEmbedding(1, 64, rngs=rng)
    t = jnp.array([0.5])
    output = embedding(t)
    assert output.shape == (64,)


def test_binary_autoregressive_decoder(rng):
    """Test instantiation of BinaryAutoregressiveDecoder."""
    decoder = BinaryAutoregressiveDecoder(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        dropout_rate=0.1,
        rngs=rng,
    )
    x = jnp.zeros((32, 64))
    output = decoder(x)
    assert output.shape == (32, 64)


def test_edm_simformer(rng):
    """Test instantiation of EDMSimformer."""
    simformer = EDMSimformer(
        rngs=rng,
        model_dim=64,
        context_dim=64,
        num_heads=4,
        num_layers=4,
        attn_size=16,
        widening_factor=3,
        dropout_rate=0.1,
        enable_cross_attention=True,
    )
    # Test that the model can be called with dummy inputs
    t = jnp.array([0.5])
    x = jnp.zeros((32, 64))
    output = simformer(
        t, x, None
    )  # None for tokenizer as it's not needed for this test
    assert output.shape == (32, 64)


def test_dmri_inference_model(rng):
    """Test instantiation of DMRIInferenceModel."""
    cfg = DMRIInferenceModelConfig(
        simulator=MultiCompartment,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionConfig(),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, rng)

    # Test that the model can be called with dummy inputs
    model_mask = jnp.zeros((32, 1))
    theta = jnp.zeros((32, 10))
    x = jnp.zeros((32, 64))
    bvals = jnp.zeros((32,))
    bvecs = jnp.zeros((32, 3))

    output = model(model_mask, theta, x, bvals, bvecs)
    assert isinstance(output, tuple)  # Model returns multiple outputs


def test_dmri_inference_model_amortized(rng):
    """Test instantiation of DMRIInferenceModel with amortized prior."""
    cfg = DMRIInferenceModelConfigMaskPriorAmortized(
        simulator=MultiCompartment,
        model_dim=64,
        embedding_cfg=DMRIEmbeddingConfig(),
        model_selection_cfg=DMRIModelSelectionAmortizedPriorConfig(),
        theta_inference_cfg=DMRIThetaInferenceConfig(),
    )
    model = DMRIInferenceModel(cfg, rng)

    # Test that the model can be called with dummy inputs
    model_mask = jnp.zeros((32, 1))
    theta = jnp.zeros((32, 10))
    x = jnp.zeros((32, 64))
    bvals = jnp.zeros((32,))
    bvecs = jnp.zeros((32, 3))

    output = model(model_mask, theta, x, bvals, bvecs)
    assert isinstance(output, tuple)  # Model returns multiple outputs


def test_embedding_net(rng):
    """Test instantiation of BvalBvecSignalEmbeddingNet."""
    embedding_net = BvalBvecSignalEmbeddingNet(
        model_dim=64,
        num_heads=4,
        num_layers=6,
        attn_size=16,
        widening_factor=3,
        dropout_rate=0.1,
        rngs=rng,
    )

    # Test that the model can be called with dummy inputs
    bvals = jnp.zeros((32,))
    bvecs = jnp.zeros((32, 3))
    signals = jnp.zeros((32,))

    output = embedding_net(bvals, bvecs, signals)
    assert output.shape == (32, 64)
